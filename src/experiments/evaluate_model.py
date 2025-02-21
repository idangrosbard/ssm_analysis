"""
EvaluateModelExperiment: Experiment for evaluating model performance on datasets

In this experiment implementation:
The sub-task is a batch of prompts
The inner loop is running the model on each prompt and collecting detailed metrics
The sub task result is a DataFrame with model predictions and metrics
The combined result is saved as a CSV file
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.consts import COLUMNS, COUNTER_FACT_2_KNOWN1000_COL_CONV
from src.experiment_infra.base_config import BaseConfig, create_mutable_field
from src.experiment_infra.model_interface import get_model_interface
from src.types import DATASETS, MODEL_ARCH, DatasetArgs
from src.utils.logits import get_last_token_logits, logits_to_probs


@dataclass
class EvaluateModelConfig(BaseConfig):
    """Configuration for model evaluation."""

    experiment_base_name: str = "evaluate"
    drop_subject: bool = False
    drop_subj_last_token: bool = False
    with_3_dots: bool = False
    new_max_tokens: int = 5
    top_k_tokens: int = 5

    dataset_args: DatasetArgs = create_mutable_field(
        lambda: DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"),
    )

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys[:-1]

    @property
    def output_result_path(self) -> Path:
        return self.outputs_path / f"{self.dataset_args.name}.csv"

    def get_outputs(self) -> pd.DataFrame:
        df = pd.read_csv(self.output_result_path, index_col=False)
        for counter_fact_col, known1000_col in COUNTER_FACT_2_KNOWN1000_COL_CONV.items():
            if counter_fact_col not in df.columns:
                assert known1000_col in df.columns
                df[counter_fact_col] = df[known1000_col]
            df = df.drop(columns=[known1000_col])
        return df

    def compute(self) -> None:
        run(self)


def get_subj_idx(
    input: str,
    subj: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    last: bool = True,
) -> int:
    prefix = input.split(subj)[0]
    sent2subj = prefix
    if last:
        sent2subj = prefix + subj

    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    return len(sent2subj_tokens) - 1  # type: ignore


def _get_logits(out, model_arch: MODEL_ARCH):
    match model_arch:
        case MODEL_ARCH.MAMBA2:
            logits, _ = out
        case MODEL_ARCH.MAMBA1 | MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2 | MODEL_ARCH.GPT2:
            logits = out.logits
        case _:
            assert_never(model_arch)

    return logits


def generate_next_tokens(model: Any, input_ids: torch.Tensor, num_tokens_to_generate: int, model_arch: MODEL_ARCH):
    """
    Generate the next `num_tokens_to_generate` tokens and collect their logits for each input in the batch.

    Args:
        model: The language model (e.g., LLaMA) used for token generation.
        input_ids: Tokenized input IDs (torch.Tensor) with shape [batch_size, sequence_length].
        num_tokens_to_generate: Number of tokens to generate for each input in the batch.
        model_arch: The architecture of the model.
    Returns:
        all_logits: with shape [batch_size, num_tokens_to_generate, vocab_size].
        first_next_logits: with shape [batch_size, vocab_size].
        new_input_ids: with shape [batch_size, num_tokens_to_generate].
    """
    first_logits = None
    logits = None
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            out = model(input_ids)
        logits = _get_logits(out, model_arch)
        next_tokens = get_last_token_logits(logits)
        if first_logits is None:
            first_logits = next_tokens

        input_ids = torch.cat([input_ids, torch.argmax(next_tokens, dim=-1, keepdim=True)], dim=-1)

    assert first_logits is not None
    return logits, first_logits, input_ids[:, -num_tokens_to_generate:]


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
    trimmed_rows = [row[torch.nonzero(row != trim_value, as_tuple=True)[0][0] :] for row in tensor]

    # Determine the maximum length after trimming
    max_length = max(len(row) for row in trimmed_rows)

    # Pad each row from the right to make all rows equal to the max length
    padded_tensor = torch.stack(
        [torch.cat([row, torch.full((max_length - len(row),), pad_value)]) for row in trimmed_rows]
    )

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


def run(args: EvaluateModelConfig):
    assert args.batch_size == 1, "Batch size must be 1, unless we debug the issue"
    print(args)
    if args.output_result_path.exists() and not args.overwrite_existing_outputs:
        print(f"Output file {args.output_result_path} already exists")
        return

    args.create_experiment_run_path()
    df = args.get_raw_data()

    model_interface = get_model_interface(args.model_arch, args.model_size)
    model = model_interface.model
    tokenizer = model_interface.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = model_interface.device

    acc = 0

    df[COLUMNS.MODEL_CORRECT] = False
    df[COLUMNS.MODEL_TOP_OUTPUT_CONFIDENCE] = 0.0
    df[COLUMNS.TARGET_RANK] = None
    df[COLUMNS.MODEL_TOP_OUTPUTS] = None
    df[COLUMNS.MODEL_GENERATION] = None
    df[COLUMNS.TARGET_PROBS] = 0.0
    df[COLUMNS.TARGET_TOKENS] = None
    for counter_fact_col, known1000_col in COUNTER_FACT_2_KNOWN1000_COL_CONV.items():
        if known1000_col in df.columns:
            df[counter_fact_col] = df[known1000_col]

    pbar = tqdm(range(0, len(df), args.batch_size), total=len(df) // args.batch_size)
    for start_idx in pbar:
        idx = df.index[start_idx : start_idx + args.batch_size]
        input_prompt = df.loc[idx, COLUMNS.PROMPT]
        target = df.loc[idx, COLUMNS.TARGET_TRUE]

        target_token_idx_padded = tokenizer(
            target.to_list(),
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="right",  # type: ignore
        )["input_ids"]

        if args.model_arch == MODEL_ARCH.LLAMA2:
            target_token_idx_padded = trim_left_and_right_pad(
                target_token_idx_padded,
                trim_value=29871,
                pad_value=tokenizer.pad_token_id,  # type: ignore
            )

        target_first_token_idx = target_token_idx_padded[:, 0].unsqueeze(1)  # type: ignore
        df.loc[idx, COLUMNS.TARGET_TOKENS] = list(
            map(
                json.dumps,
                map(
                    lambda x: tokenizer.batch_decode(x, skip_special_tokens=True),
                    [
                        lst[
                            : next(
                                (i for i in range(len(lst) - 1, -1, -1) if lst[i] != tokenizer.pad_token_id),
                                -1,
                            )
                            + 1
                        ]
                        for lst in target_token_idx_padded.tolist()  # type: ignore
                    ],
                ),
            )
        )

        if args.with_3_dots:
            input_prompt += " ..."
        if args.drop_subject:
            input_prompt = input_prompt.replace(df.loc[idx, COLUMNS.SUBJECT], "")
        elif args.drop_subj_last_token:
            subj_idx = get_subj_idx(input_prompt, df.loc[idx, COLUMNS.SUBJECT], tokenizer)  # type: ignore

        input_ids = tokenizer(input_prompt.to_list(), return_tensors="pt", padding=True)["input_ids"]

        if args.drop_subj_last_token:
            input_ids = input_ids[:subj_idx] + input_ids[subj_idx + 1 :]  # type: ignore

        input_ids = input_ids.to(device)  # type: ignore

        # TODO: the logits of the next token is different for different the amount token generated, understand why
        _, first_logits, new_input_ids = generate_next_tokens(model, input_ids, args.new_max_tokens, args.model_arch)

        # Get the next token probs
        next_probs = logits_to_probs(first_logits)

        # Get the top k outputs and their probs
        top_probs, top_indices = map(torch.Tensor.tolist, torch.topk(next_probs, args.top_k_tokens))
        top_tokens = list(map(tokenizer.batch_decode, top_indices))  # type: ignore
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
        df.loc[idx, COLUMNS.TARGET_RANK] = target_rank.tolist()
        df.loc[idx, COLUMNS.TARGET_PROBS] = target_probs.squeeze(1).tolist()
        df.loc[idx, COLUMNS.MODEL_CORRECT] = (target_rank == 1).tolist()
        df.loc[idx, COLUMNS.MODEL_OUTPUT] = list(map(lambda x: x[0], top_tokens))
        df.loc[idx, COLUMNS.MODEL_TOP_OUTPUT_CONFIDENCE] = list(map(lambda x: x[0], top_probs))
        df.loc[idx, COLUMNS.MODEL_TOP_OUTPUTS] = list(map(json.dumps, top_outputs))
        df.loc[idx, COLUMNS.MODEL_GENERATION] = tokenizer.batch_decode(new_input_ids)

        acc += df.loc[idx, COLUMNS.MODEL_CORRECT].sum()

    print(acc / len(df))
    df.to_csv(args.output_result_path, index=False)
