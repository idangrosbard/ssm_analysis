"""
HeatmapExperiment: Single prompt-level experiment showing how different layers affect the model's token predictions

In this experiment implementation:
The sub-task is a prompt index ( notice - this experiment is not standard, we are not iterating over the dataset)
The inner loop is masking a sliding window over the model layers
The sub task result is a heatmap of the token probabilities for each layer in the window
The combined result is a dictionary of prompt index -> heatmap

"""

import functools
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.experiment_infra.base_config import (
    BASE_OUTPUT_KEYS,
    BaseConfig,
    create_mutable_field,
)
from src.experiment_infra.model_interface import get_model_interface
from src.plots.heatmaps import simple_diff_fixed
from src.utils.logits import Prompt, decode_tokens, get_prompt_row, get_prompt_row_index
from src.utils.setup_models import get_tokenizer


class HEATMAP_PLOT_FUNCS(StrEnum):
    _simple_diff_fixed_0_3 = "_simple_diff_fixed_0.3"


IHeatmap = pd.DataFrame

plot_suffix_to_function: dict[HEATMAP_PLOT_FUNCS, Callable] = {
    HEATMAP_PLOT_FUNCS._simple_diff_fixed_0_3: functools.partial(simple_diff_fixed, fixed_diff=0.3),
}


@dataclass
class HeatmapConfig(BaseConfig):
    """Configuration for heatmap generation."""

    experiment_base_name: str = "heatmap"
    window_size: int = 5
    prompt_indices_rows: list[int] = create_mutable_field(lambda: [1, 2, 3, 4, 5])
    prompt_original_indices: list[int] = create_mutable_field(lambda: [])

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [
            BASE_OUTPUT_KEYS.WINDOW_SIZE,
        ]

    def output_heatmap_path(self, prompt_idx: int) -> Path:
        return self.outputs_path / f"idx={prompt_idx}.csv"

    def get_prompt_original_idx_combined(self) -> list[int]:
        data = self.get_prompt_data()

        return list(
            set(
                [
                    *[get_prompt_row(data, idx).original_idx for idx in self.prompt_indices_rows],
                    *self.prompt_original_indices,
                ]
            )
        )

    def get_outputs(self) -> dict[int, IHeatmap]:
        return {idx: pd.read_csv(self.output_heatmap_path(idx)) for idx in self.get_prompt_original_idx_combined()}

    def get_plot_output_path(self, prompt_idx: int, plot_name: str) -> Path:
        return self.plots_path / f"idx={prompt_idx}{plot_name}.png"

    def plot(self) -> None:
        plot(self)

    def compute(self) -> None:
        run(self)


def plot(args: HeatmapConfig):
    data = args.get_prompt_data()
    tokenizer = get_tokenizer(args.model_arch, args.model_size)
    model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[args.model_arch][args.model_size]

    prob_mats = args.get_outputs()
    for prompt_idx, prob_mat in tqdm(prob_mats.items(), desc="Plotting heatmaps"):
        prompt = get_prompt_row_index(data, prompt_idx)
        input_ids = prompt.input_ids(tokenizer, "cpu")
        toks = cast(list[str], decode_tokens(tokenizer, input_ids[0]))
        last_tok = toks[-1]
        toks[-1] = toks[-1] + "*"

        fig, ax = simple_diff_fixed(
            prob_mat=prob_mat,
            model_id=model_id,
            window_size=args.window_size,
            last_tok=last_tok,
            base_prob=prompt.base_prob,
            true_word=prompt.true_word,
            toks=toks,
            fixed_diff=0.3,
        )
        output_path = args.get_plot_output_path(prompt.original_idx, HEATMAP_PLOT_FUNCS._simple_diff_fixed_0_3)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def run(args: HeatmapConfig):
    print(args)
    data = args.get_prompt_data()
    remaining_idx = [
        idx
        for idx in args.get_prompt_original_idx_combined()
        if not args.output_heatmap_path(idx).exists() or args.overwrite_existing_outputs
    ]
    if not remaining_idx:
        print("All heatmaps already exist")
        return

    args.create_experiment_run_path()
    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = model_interface.n_layers()

    def forward_eval(prompt: Prompt, window: list[int]):
        true_id = prompt.true_id(tokenizer, "cpu")
        input_ids = prompt.input_ids(tokenizer, device)

        last_idx = input_ids.shape[1] - 1
        probs = np.zeros((input_ids.shape[1]))

        for idx in range(input_ids.shape[1]):
            num_to_masks = {layer: [(last_idx, idx)] for layer in window}

            next_token_probs = model_interface.generate_logits(
                input_ids=input_ids,
                attention=True,
                num_to_masks=num_to_masks,
            )
            probs[idx] = next_token_probs[0, true_id[:, 0]]
            torch.cuda.empty_cache()
        return probs

    windows = [list(range(i, i + args.window_size)) for i in range(0, n_layers - args.window_size + 1)]

    for prompt_idx in tqdm(remaining_idx, desc="Prompts"):
        prob_mat = []
        prompt = get_prompt_row_index(data, prompt_idx)
        for window in windows:
            model_interface.setup(layers=window)
            prob_mat.append(forward_eval(prompt, window))

        prob_mat = np.array(prob_mat).T
        pd.DataFrame(prob_mat).to_csv(args.output_heatmap_path(prompt.original_idx), index=False)
