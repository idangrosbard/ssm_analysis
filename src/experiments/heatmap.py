from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyrallis
import torch
from tqdm import tqdm

from src.consts import PATHS
from src.experiment_infra.base_experiment import BaseExperiment
from src.experiment_infra.config import BaseConfig
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


class HeatmapExperiment(BaseExperiment[HeatmapConfig, tuple[int, list[int]], np.ndarray, list[np.ndarray]]):
    @property
    def n_layers(self) -> int:
        return len(self.model_interface.model.backbone.layers)

    def evaluation_data(self):
        windows = [
            list(range(i, i + self.config.window_size)) for i in range(0, self.n_layers - self.config.window_size + 1)
        ]

        for prompt_idx in tqdm(self.config.prompt_indices, desc="Prompts"):
            for window in windows:
                yield (prompt_idx, window)

    def run_single_evaluation(self, data: tuple[int, list[int]]):
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

    def combine_results(self, results):
        prompt_prob_mats: list[np.ndarray] = []
        prev_prompt_idx = -1

        combined_results = []

        for (prompt_idx, _), probs in results:
            if prev_prompt_idx != prompt_idx:
                prompt_prob_mats = []
                combined_results.append(prompt_prob_mats)
            prompt_prob_mats.append(probs)
            prev_prompt_idx = prompt_idx

        return combined_results

    def save_results(self, results):
        for prompt_idx, prompt_results in enumerate(results):
            np.save(self.config.output_path / f"idx={prompt_idx}.npy", np.array(prompt_results).T)

    def save_single_results(self, results):
        pass

    def get_results(self):
        results = []
        for prompt_idx in self.config.prompt_indices:
            prompt_path = self.config.output_path / f"idx={prompt_idx}.npy"
            if prompt_path.exists():
                results.append(np.load(prompt_path))
        return results
