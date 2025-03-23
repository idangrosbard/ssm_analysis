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
from typing import Callable, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.analysis.plots.heatmaps import simple_diff_fixed
from src.core.names import EXPERIMENT_NAMES
from src.core.types import (
    FeatureCategory,
    TPromptOriginalIndex,
    TWindow,
    TWindowSize,
)
from src.data_ingestion.helpers.logits_utils import Prompt, decode_tokens, get_prompt_row_index
from src.experiments.infrastructure.base_config import (
    BASE_OUTPUT_KEYS,
    BaseRunner,
)
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams


class HEATMAP_PLOT_FUNCS(StrEnum):
    _simple_diff_fixed_0_3 = "_simple_diff_fixed_0.3"


IHeatmap = pd.DataFrame

plot_suffix_to_function: dict[HEATMAP_PLOT_FUNCS, Callable] = {
    HEATMAP_PLOT_FUNCS._simple_diff_fixed_0_3: functools.partial(simple_diff_fixed, fixed_diff=0.3),
}


@dataclass
class HeatmapParams:
    window_size: TWindowSize


class HeatmapDependencies(TypedDict):
    evaluate_model: EvaluateModelConfig


@dataclass
class HeatmapConfig(BaseRunner[HeatmapParams, dict[TPromptOriginalIndex, IHeatmap]]):
    """Configuration for heatmap generation."""

    runner_params: HeatmapParams

    @property
    def experiment_name(self):
        return EXPERIMENT_NAMES.HEATMAP

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [
            BASE_OUTPUT_KEYS.WINDOW_SIZE,
        ]

    def output_heatmap_path(self, prompt_idx: TPromptOriginalIndex):
        return self.variation_paths.outputs_path / f"idx={prompt_idx}.csv"

    def get_remaining_prompt_original_indices(self):
        return [
            idx
            for idx in self.prompt_ids
            if not self.output_heatmap_path(idx).exists() or self.run_params.overwrite_existing_outputs
        ]

    def get_outputs(self) -> dict[TPromptOriginalIndex, IHeatmap]:
        return {idx: pd.read_csv(self.output_heatmap_path(idx)) for idx in self.prompt_ids}

    def get_plot_output_path(self, prompt_idx: TPromptOriginalIndex, plot_name: HEATMAP_PLOT_FUNCS) -> Path:
        return self.variation_paths.plots_path / f"idx={prompt_idx}{plot_name}.png"

    def plot(self, plot_name: HEATMAP_PLOT_FUNCS) -> None:
        plot(self, plot_name)

    def compute(self) -> None:
        run(self)

    def is_computed(self) -> bool:
        return all(self.output_heatmap_path(idx).exists() for idx in self.prompt_ids)

    def get_runner_dependencies(self) -> HeatmapDependencies:  # type: ignore
        return HeatmapDependencies(
            evaluate_model=EvaluateModelConfig.init_from_config(
                self,
                runner_params=EvaluateModelParams(),
            ),
        )


def plot(args: HeatmapConfig, plot_name: HEATMAP_PLOT_FUNCS):
    data = args.get_runner_dependencies()["evaluate_model"].get_prompt_data()
    tokenizer = args.common_params.get_tokenizer
    model_id = args.common_params.model_id

    prob_mats = args.get_outputs()
    for prompt_idx, prob_mat in tqdm(prob_mats.items(), desc="Plotting heatmaps"):
        prompt = get_prompt_row_index(data, prompt_idx)
        input_ids = prompt.input_ids(tokenizer, "cpu")
        toks = cast(list[str], decode_tokens(tokenizer, input_ids[0]))
        last_tok = toks[-1]
        toks[-1] = toks[-1] + "*"

        fig, _ = simple_diff_fixed(
            prob_mat=prob_mat,
            model_id=model_id,
            window_size=args.runner_params.window_size,
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
    data = args.get_runner_dependencies()["evaluate_model"].get_prompt_data()
    remaining_idx = args.get_remaining_prompt_original_indices()
    if not remaining_idx:
        print("All heatmaps already exist")
        return

    args.create_experiment_run_path()
    model_interface = args.common_params.get_model_interface()
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = model_interface.n_layers()

    def forward_eval(prompt: Prompt, window: TWindow):
        true_id = prompt.true_id(tokenizer, "cpu")
        input_ids = prompt.input_ids(tokenizer, device)

        last_idx = input_ids.shape[1] - 1
        probs = np.zeros((input_ids.shape[1]))

        for idx in range(input_ids.shape[1]):
            num_to_masks = {layer: [(last_idx, idx)] for layer in window}

            next_token_probs = model_interface.generate_logits(
                input_ids=input_ids,
                num_to_masks=num_to_masks,
                feature_category=FeatureCategory.ALL,
            )
            probs[idx] = next_token_probs[0, true_id[:, 0]]
            torch.cuda.empty_cache()
        return probs

    windows = [
        TWindow(list(range(i, i + args.runner_params.window_size)))
        for i in range(0, n_layers - args.runner_params.window_size + 1)
    ]

    for prompt_idx in tqdm(remaining_idx, desc="Prompts"):
        prob_mat = []
        prompt = get_prompt_row_index(data, prompt_idx)
        for window in windows:
            model_interface.setup(layers=window)
            prob_mat.append(forward_eval(prompt, window))

        prob_mat = np.array(prob_mat).T
        pd.DataFrame(prob_mat).to_csv(args.output_heatmap_path(prompt.original_idx), index=False)
