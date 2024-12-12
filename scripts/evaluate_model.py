from dataclasses import dataclass
from hmac import new
import json
from pathlib import Path
from typing import Optional, assert_never

import pandas as pd
import pyrallis
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from scripts.create_slurm_file import run_slurm
from src.consts import (
    MODEL_SIZES_PER_ARCH_TO_MODEL_ID,
    PATHS,
)
from src.datasets.download_dataset import load_dataset
from src.datasets.download_dataset import load_knowns_pd
from src.logit_utils import get_last_token_logits, logits_to_probs
from src.types import DATASETS
from src.types import MODEL_ARCH, SPLIT, DatasetArgs, TModelID
from src.utils.setup_models import get_tokenizer_and_model
from src.utils.slurm import submit_job


def generate_next_tokens(model, input_ids, num_tokens_to_generate, model_arch):
    """
    Generate the next `num_tokens_to_generate` tokens and collect their logits for each input in the batch.

    Args:
        model: The language model (e.g., LLaMA) used for token generation.
        input_ids: Tokenized input IDs (torch.Tensor) with shape [batch_size, sequence_length].
        num_tokens_to_generate: Number of tokens to generate for each input in the batch.
    Returns:
        all_logits: with shape [batch_size, num_tokens_to_generate, vocab_size].
        first_next_logits: with shape [batch_size, vocab_size].
        new_input_ids: with shape [batch_size, num_tokens_to_generate].
    """
    first_logits = None
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            out = model(input_ids)
        logits = _get_logits(out, model_arch)
        next_tokens = get_last_token_logits(logits)
        if first_logits is None:
            first_logits = next_tokens

        input_ids = torch.cat(
            [input_ids, torch.argmax(next_tokens, dim=-1, keepdim=True)], dim=-1
        )

    return logits, first_logits, input_ids[:, -num_tokens_to_generate:]


@dataclass
class Args:
    model_arch: MODEL_ARCH = MODEL_ARCH.LLAMA2
    model_size: str = "7B"
    # model_arch: MODEL_ARCH = MODEL_ARCH.LLAMA3_2
    # model_size: str = "1B"
    drop_subject: bool = False
    drop_subj_last_token: bool = False
    with_3_dots: bool = False
    new_max_tokens: int = 5
    top_k_tokens: int = 5
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.KNOWN_1000, splits="all"), is_mutable=True
    )
    _batch_size: int = 16  # Adjust based on GPU memory
    output_file: Optional[Path] = None
    with_slurm: bool = False

    @property
    def batch_size(self) -> int:
        return 1 if self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2 else self._batch_size

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]


def get_subj_idx(input: str, subj: str, tokenizer: AutoTokenizer, last: bool = True):
    prefix = input.split(subj)[0]
    sent2subj = prefix
    if last:
        sent2subj = prefix + subj

    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    return len(sent2subj_tokens) - 1


def _get_logits(out, model_arch: MODEL_ARCH):
    match model_arch:
        case MODEL_ARCH.MINIMAL_MAMBA1 | MODEL_ARCH.MINIMAL_MAMBA2:
            logits, _ = out
        case MODEL_ARCH.MAMBA1 | MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2 | MODEL_ARCH.MAMBA1 | MODEL_ARCH.MAMBA2:
            logits = out.logits
        case _:
            assert_never(model_arch)

    return logits


def trim_left_and_right_pad(tensor, trim_value=2, pad_value=0):
    """
    Trims leading specified values from each row and pads rows on the right to equalize their lengths.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        trim_value (int): The value to trim from the start of each row (default is 2).
        pad_value (int): The value to use for padding on the right (default is 0).
    
    Returns:
        torch.Tensor: The processed tensor with trimmed rows and right padding.
    """
    # Remove leading trim_value from each row
    trimmed_rows = [row[torch.nonzero(row != trim_value, as_tuple=True)[0][0]:] for row in tensor]
    
    # Determine the maximum length after trimming
    max_length = max(len(row) for row in trimmed_rows)
    
    # Pad each row from the right to make all rows equal to the max length
    padded_tensor = torch.stack([
        torch.cat([row, torch.full((max_length - len(row),), pad_value)]) for row in trimmed_rows
    ])
    
    return padded_tensor


def _generate_few_tokens(model, tokenizer, input_ids, new_max_tokens):
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.size(1) + new_max_tokens,
            num_return_sequences=1,
            top_k=1,
            temperature=1.0,
        )
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return "".join(generated_text)


def main_local(args: Args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame(load_dataset(args.dataset_args))

    tokenizer, model = get_tokenizer_and_model(args.model_arch, args.model_size)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    acc = 0

    df["model_correct"] = False
    df["model_top_output_confidence"] = 0.0
    df["target_rank"] = None
    df["model_top_outputs"] = None
    df["model_generation"] = None
    df["target_probs"] = None
    df["target_tokens"] = None
    if "target_true" in df.columns:
        df["attribute"] = df["target_true"]

    # pbar = tqdm(df.index, total=len(df))
    pbar = tqdm(range(0, len(df), args.batch_size), total=len(df) // args.batch_size)
    for start_idx in pbar:
        idx = df.index[start_idx : start_idx + args.batch_size]
        input_prompt = df.loc[idx, "prompt"]
        target = df.loc[idx, "attribute"]

        target_token_idx_padded = tokenizer(
            target.to_list(),
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )["input_ids"]

        if args.model_arch == MODEL_ARCH.LLAMA2:
            target_token_idx_padded = trim_left_and_right_pad(
                target_token_idx_padded,
                trim_value=29871,
                pad_value=tokenizer.pad_token_id,
            )

        target_first_token_idx = target_token_idx_padded[:, 0].unsqueeze(1)
        df.loc[idx, "target_tokens"] = list(
            map(
                json.dumps,
                map(
                    lambda x: tokenizer.batch_decode(x, skip_special_tokens=True),
                    [lst[:next((i for i in range(len(lst)-1, -1, -1) if lst[i] != tokenizer.pad_token_id), -1) + 1] for lst in target_token_idx_padded.tolist()],
                ),
            )
        )

        if args.with_3_dots:
            input_prompt += " ..."
        if args.drop_subject:
            input_prompt = input_prompt.replace(df.loc[idx, "subject"], "")
        elif args.drop_subj_last_token:
            subj_idx = get_subj_idx(input_prompt, df.loc[idx, "subject"], tokenizer)

        input_ids = tokenizer(
            input_prompt.to_list(), return_tensors="pt", padding=True
        )["input_ids"]

        if args.drop_subj_last_token:
            input_ids = input_ids[:subj_idx] + input_ids[subj_idx + 1 :]

        input_ids = input_ids.to(device)

        # TODO: the logits of the next token is different for different the amount token generated, understand why
        _, first_logits, new_input_ids = generate_next_tokens(
            model, input_ids, args.new_max_tokens, args.model_arch
        )

        # Get the next token probs
        next_probs = logits_to_probs(first_logits)

        # Get the top 5 outputs and their probs
        top_probs, top_indices = map(
            torch.Tensor.tolist, torch.topk(next_probs, args.top_k_tokens)
        )
        top_tokens = list(map(tokenizer.batch_decode, top_indices))
        top_outputs = list(
            map(
                list,
                map(
                    lambda x: zip(*x),
                    zip(
                        top_indices,
                        top_tokens,
                        top_probs,
                    ),
                ),
            )
        )

        target_probs = torch.gather(
            next_probs,
            1,
            target_first_token_idx.to(next_probs.device),
        )

        # Get the rank of the target token
        target_rank = (next_probs > target_probs).sum(dim=-1) + 1
        df.loc[idx, "target_rank"] = target_rank.tolist()
        df.loc[idx, "target_probs"] = target_probs.squeeze(1).tolist()
        df.loc[idx, "model_correct"] = (target_rank == 1).tolist()
        df.loc[idx, "model_output"] = list(map(lambda x: x[0], top_tokens))
        df.loc[idx, "model_top_output_confidence"] = list(
            map(lambda x: x[0], top_probs)
        )
        df.loc[idx, "model_top_outputs"] = list(map(json.dumps, top_outputs))
        df.loc[idx, "model_generation"] = tokenizer.batch_decode(new_input_ids)

        acc += df.loc[idx, "model_correct"].sum()

    print(acc / len(df))
    if not args.output_file:
        args.output_file = (
            PATHS.OUTPUT_DIR
            / args.model_id
            / "evaluate"
            / f"{args.dataset_args.dataset_name}.csv"
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file)


@pyrallis.wrap()
def main(args: Args):
    assert not (args.drop_subject and args.drop_subj_last_token)
    # args.with_slurm = True

    if args.with_slurm:
        gpu_type = "a100"
        # gpu_type = "titan_xp-studentrun"

        for model_arch, model_size in [
            # (MODEL_ARCH.MAMBA1, "130M"),
            (MODEL_ARCH.MAMBA1, "1.4B"),
            # (MODEL_ARCH.MAMBA1, "2.8B"),
            # (MODEL_ARCH.MAMBA2, "130M"),
            # (MODEL_ARCH.MAMBA2, "1.3B"),
            # (MODEL_ARCH.MAMBA2, "2.7B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2, "130M"),
            # (MODEL_ARCH.MINIMAL_MAMBA2, "2.7B"),
            # (MODEL_ARCH.LLAMA2, "7B"),
            # (MODEL_ARCH.LLAMA3_2, "1B"),
            # (MODEL_ARCH.LLAMA3_2, "3B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size
            # args._batch_size = 16

            # for i in range(5):
            #     args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"train{i+1}")
            # args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"train1")
            # args.dataset_args = DatasetArgs(name=DATASETS.KNOWN_1000, splits=f"all")
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"all")
            # args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"test")

            job_name = f"evaluate_model/{model_arch}_{model_size}_{args.dataset_args.dataset_name}"
            job = submit_job(
                main_local,
                args,
                log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                job_name=job_name,
                # timeout_min=1200,
                gpu_type=gpu_type,
                slurm_gpus_per_node=1,
            )

            print(f"{job}: {job_name}")
    else:
        main_local(args)


if __name__ == "__main__":
    main()
