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
from typing import cast

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.consts import FILTERATIONS, PATHS
from src.datasets.download_dataset import load_splitted_counter_fact
from src.experiment_infra.base_config import BaseConfig
from src.experiment_infra.model_interface import get_model_interface


@dataclass
class DataConstructionConfig(BaseConfig):
    """Configuration for data construction."""

    experiment_name: str = "data_construction"
    temperature: float = 1
    top_k: int = 0
    top_p: float = 1
    attention: bool = False

    @property
    def output_path(self) -> Path:
        return PATHS.OUTPUT_DIR / self.model_id / self.experiment_name / f"ds={self.dataset_args.dataset_name}"


def main_local(args: DataConstructionConfig):
    print(args)
    dataset = load_splitted_counter_fact(
        "all", align_to_known=False, filteration=getattr(FILTERATIONS, args.filteration)
    )
    original_data = pd.DataFrame(cast(dict, dataset))

    args.output_path.mkdir(parents=True, exist_ok=True)

    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    original_data["true_prob"] = 0.0
    original_data["max_prob"] = 0.0
    original_data["hit"] = False
    original_data["pred"] = ""

    def forward_eval(temperature, top_k, top_p, batch_start, batch_end, attention, print_period=10000):
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
        forward_eval(args.temperature, args.top_k, args.top_p, batches[i], batches[i + 1], args.attention)

    print(original_data.pipe(lambda df: df[~df["hit"]]))
    print(original_data["hit"].mean())
    attention_str = "attention" if args.attention else "original"
    original_data.to_parquet(args.output_path / f"entire_results_{attention_str}.parquet")
