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
    MODEL_IDS_TO_ARCH_AND_SIZE,
    MODEL_SIZES_PER_ARCH_TO_MODEL_ID,
    PATHS,
)
from src.datasets.download_dataset import load_dataset
from src.datasets.download_dataset import load_knowns_pd
from src.types import DATASETS
from src.types import MODEL_ARCH, SPLIT, DatasetArgs, TModelID
from src.utils.setup_models import get_tokenizer_and_model
from src.utils.slurm import submit_job


@dataclass
class Args:
    model_arch: MODEL_ARCH = MODEL_ARCH.LLAMA3_2
    model_size: str = "1B"
    drop_subject: bool = False
    drop_subj_last_token: bool = False
    with_3_dots: bool = False
    new_max_tokens: int = 5
    top_k_tokens: int = 5
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"), is_mutable=True
    )
    output_file: Optional[Path] = None
    with_slurm: bool = False

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
        case MODEL_ARCH.MAMBA1 | MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2:
            logits = out.logits
        case _:
            assert_never(model_arch)
            
    return logits


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame(load_dataset(args.dataset_args))

    tokenizer, model = get_tokenizer_and_model(args.model_arch, args.model_size, device)
    model.eval()

    acc = 0

    df["model_correct"] = False
    df["model_top_output_confidence"] = 0.0
    df["model_top_outputs"] = None
    df["model_generation"] = None
    if "target_true" in df.columns:
        df["attribute"] = df["target_true"]

    pbar = tqdm(df.index, total=len(df))
    for idx in pbar:
        input_prompt: str = df.loc[idx, "prompt"]
        target = df.loc[idx, "attribute"]

        if args.with_3_dots:
            input_prompt += " ..."
        if args.drop_subject:
            input_prompt = input_prompt.replace(df.loc[idx, "subject"], "")
        elif args.drop_subj_last_token:
            subj_idx = get_subj_idx(input_prompt, df.loc[idx, "subject"], tokenizer)

        input_ids = tokenizer(input_prompt)["input_ids"]

        if args.drop_subj_last_token:
            input_ids = input_ids[:subj_idx] + input_ids[subj_idx + 1 :]

        input_ids = torch.Tensor([input_ids]).long().to(device)

        with torch.no_grad():
            out = model(input_ids)

        logits = _get_logits(out, args.model_arch)

        # Get the last token logits
        logits = logits[:, -1, :]

        # Get the top 5 outputs and their confidence
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, args.top_k_tokens)
        top_outputs = [
            (tokenizer.decode([idx]), prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]

        # Get the top output confidence and generation
        decoded = tokenizer.decode(logits.argmax(dim=-1).squeeze()).strip()

        df.loc[idx, "model_correct"] = target.startswith(decoded)
        df.loc[idx, "model_output"] = decoded
        df.loc[idx, "model_top_output_confidence"] = top_probs[0][0].item()
        df.loc[idx, "model_top_outputs"] = json.dumps(top_outputs)
        df.loc[idx, "model_generation"] = _generate_few_tokens(
            model, tokenizer, input_ids, args.new_max_tokens
        )

        acc += float(target.startswith(decoded)) / len(df)

    print(acc)
    if not args.output_file:
        args.output_file = PATHS.OUTPUT_DIR / args.model_id / "evaluate"
        if args.dataset_args:
            pass

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file)


@pyrallis.wrap()
def main(args: Args):
    assert not (args.drop_subject and args.drop_subj_last_token)
    args.with_slurm = True

    if args.with_slurm:
        # gpu_type = "a100"
        gpu_type = "titan_xp-studentrun"

        for model_id in [
            # "state-spaces/mamba-130M-hf",
            # "state-spaces/mamba-2.8B-hf",
            # "state-spaces/mamba2-130M",
            # "state-spaces/mamba2-2.7B",
            # "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-3.2-1B",
            # "meta-llama/Llama-3.2-3B",
        ]:
            model_arch, model_size = MODEL_IDS_TO_ARCH_AND_SIZE[model_id]  # type: ignore
            args.model_arch = model_arch
            args.model_size = model_size

            job_name = f"evaluate_model_{args.model_id}"
            submit_job(
                main_local,
                args,
                log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                job_name=job_name,
                timeout_min=150,
                gpu_type=gpu_type,
                slurm_gpus_per_node=1,
            )
    else:
        main_local(args)


if __name__ == "__main__":
    main()

# python /home/yandex/DL20232024a/nirendy/repos/ssm_analysis/scripts/evaluate_model.py --model_arch mamba --model_size 2.8B --with_slurm
# python /home/yandex/DL20232024a/nirendy/repos/ssm_analysis/scripts/evaluate_model.py --model_arch minimal_mamba2 --model_size 2.8B --with_slurm
# python /home/yandex/DL20232024a/nirendy/repos/ssm_analysis/scripts/evaluate_model.py --model_arch llama2 --model_size 2-7b --with_slurm
