"""
DataConstructionExperiment:
Experiment for constructing datasets by running models on prompts and collecting their outputs
In this experiment implementation:
The sub-task is a batch of prompts
The inner loop is running the model on each prompt and collecting outputs
The sub task result is a DataFrame with model outputs and probabilities
The combined result is saved as a parquet file
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.experiment_infra.base_config import BaseConfig, OutputKey
from src.experiment_infra.model_interface import get_model_interface


@dataclass
class DataConstructionConfig(BaseConfig[pd.DataFrame]):
    """Configuration for data construction."""

    experiment_base_name: str = "data_construction"
    attention: bool = False

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [OutputKey[bool]("attention", key_display_name="attn=")]

    @property
    def output_result_path(self) -> Path:
        return self.outputs_path / f"entire_results_{self.attention}.csv"

    def get_outputs(self):
        return pd.read_csv(self.output_result_path, index_col=False)


def run(args: DataConstructionConfig):
    print(args)
    if args.output_result_path.exists() and not args.overwrite_existing_outputs:
        print(f"Output file {args.output_result_path} already exists")
        return

    args.create_output_path()
    original_data = args.get_raw_data(align_to_known=False)
    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    device = model_interface.device

    original_data["true_prob"] = 0.0
    original_data["max_prob"] = 0.0
    original_data["hit"] = False
    original_data["pred"] = ""

    def forward_eval(batch_start, batch_end, attention, print_period=10000):
        prompts = list(original_data.loc[batch_start : batch_end - 1, "prompt"].values)
        true_word = list(original_data.loc[batch_start : batch_end - 1, "target_true"].values)
        true_token = tokenizer(true_word, return_tensors="pt", padding=True)
        true_id = true_token.input_ids.to(device="cpu")
        tokens = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids.to(device=device)
        next_token_probs = model_interface.generate_logits(
            input_ids=input_ids,
            attention=attention,
        )
        max_idx = np.argmax(next_token_probs, axis=1)
        row_idx = np.arange(next_token_probs.shape[0])
        preds = [tokenizer.decode([t]) for t in max_idx]
        original_data.loc[batch_start : batch_end - 1, "true_prob"] = next_token_probs[
            row_idx, true_id[:, 0].numpy()
        ].tolist()
        original_data.loc[batch_start : batch_end - 1, "max_prob"] = next_token_probs[row_idx, max_idx].tolist()
        original_data.loc[batch_start : batch_end - 1, "hit"] = (
            original_data.loc[batch_start : batch_end - 1, "true_prob"].values
            == original_data.loc[batch_start : batch_end - 1, "max_prob"].values
        )
        original_data.loc[batch_start : batch_end - 1, "pred"] = preds
        if (batch_start + 1) % print_period == 0:
            print(f"Finished batch [{batch_start}:{batch_end - 1}]")
        torch.cuda.empty_cache()

    batch_size = args.batch_size
    N = len(original_data)
    batches = list(np.arange(0, N, batch_size)) + [N]

    for i in tqdm(range(len(batches) - 1)):
        forward_eval(
            batch_start=batches[i],
            batch_end=batches[i + 1],
            attention=args.attention,
        )

    print(original_data.pipe(lambda df: df[~df["hit"]]))
    print(original_data["hit"].mean())
    original_data.to_csv(args.output_result_path, index=False)
