"""
HeatmapExperiment: Single prompt-level experiment showing how different layers affect the model's token predictions

In this experiment implementation:
The sub-task is a prompt index ( notice - this experiment is not standard, we are not iterating over the dataset)
The inner loop is masking a sliding window over the model layers
The sub task result is a heatmap of the token probabilities for each layer in the window
The combined result is a dictionary of prompt index -> heatmap

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import torch
from tqdm import tqdm

from src.consts import PATHS
from src.experiment_infra.base_config import BaseConfig
from src.experiment_infra.base_experiment import BaseExperiment
from src.utils.logits import get_prompt_row


@dataclass
class HeatmapConfig(BaseConfig):
    """Configuration for heatmap generation."""

    experiment_name: str = "heatmap"
    window_size: int = 5
    prompt_indices: list[int] = pyrallis.field(default_factory=lambda: [1, 2, 3, 4, 5])

    @property
    def output_path(self) -> Path:
        return (
            PATHS.OUTPUT_DIR
            / self.model_id
            / self.experiment_name
            / f"ds={self.dataset_args.dataset_name}"
            / f"ws={self.window_size}"
        )


class HeatmapExperiment(
    BaseExperiment[
        HeatmapConfig,  # TConfig
        list[int],  # TInnerLoopData - window
        np.ndarray,  # TInnerLoopResult - probabilities of a window
        int,  # TSubTasksData - prompt_idx
        np.ndarray,  # TSubTasksResult - 2D array of probabilities representing a prompt heatmap
        dict[int, np.ndarray],  # TCombinedResult - prompt index -> heatmap
    ]
):
    @property
    def _n_layers(self) -> int:
        return len(self.model_interface.model.backbone.layers)

    def get_index_out_file(self, prompt_idx: int) -> Path:
        return self.config.output_path / f"idx={prompt_idx}.npy"

    def sub_tasks(self):
        """Get prompt indices as sub-tasks"""
        for prompt_idx in tqdm(self.config.prompt_indices, desc="Prompts"):
            yield prompt_idx

    def inner_loop(self, data: int):
        """Get windows for each prompt"""
        windows = [
            list(range(i, i + self.config.window_size)) for i in range(0, self._n_layers - self.config.window_size + 1)
        ]
        for window in tqdm(windows, desc="Windows"):
            yield window

    def run_single_inner_evaluation(self, data: tuple[int, list[int]]) -> np.ndarray:
        """Run evaluation for a single window"""
        prompt_idx, window = data
        prompt = get_prompt_row(self.dataset, prompt_idx)
        true_id = prompt.true_id(self.model_interface.tokenizer, "cpu")
        input_ids = prompt.input_ids(self.model_interface.tokenizer, self.model_interface.device)

        last_idx = input_ids.shape[1] - 1
        probs = np.zeros((input_ids.shape[1]))

        self.model_interface.setup(layers=window)

        for idx in range(input_ids.shape[1]):
            num_to_masks = {layer: [(last_idx, idx)] for layer in window}

            next_token_probs = self.model_interface.generate_logits(
                input_ids=input_ids,
                attention=True,
                num_to_masks=num_to_masks,
            )
            probs[idx] = next_token_probs[0, true_id[:, 0]]
            torch.cuda.empty_cache()
        return probs

    def combine_inner_results(self, results: list[tuple[list[int], np.ndarray]]) -> np.ndarray:
        """Combine results for a single prompt"""
        return np.array([result for _, result in results]).T

    def combine_sub_task_results(self, results: list[tuple[int, np.ndarray]]) -> dict[int, np.ndarray]:
        """Combine results from all prompts"""
        return {prompt_idx: prompt_results for prompt_idx, prompt_results in results}

    def save_results(self, results: dict[int, np.ndarray]):
        """Save final results"""
        for prompt_idx, prompt_results in results.items():
            np.save(self.get_index_out_file(prompt_idx), prompt_results)

    def save_sub_task_results(self, results: list[tuple[int, np.ndarray]]):
        """Save intermediate results for each prompt"""
        for prompt_idx, prompt_results in results:
            np.save(self.get_index_out_file(prompt_idx), np.array(prompt_results).T)

    def load_sub_task_result(self, data: int) -> Optional[np.ndarray]:
        """Load results for a single prompt if they exist"""
        prompt_idx = data

        prompt_path = self.get_index_out_file(prompt_idx)
        if prompt_path.exists():
            return np.load(prompt_path)
        return None
