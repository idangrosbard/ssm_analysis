from pathlib import Path
from typing import List

import numpy as np
import torch

import src.zenml  # noqa
from src.datasets.download_dataset import get_hit_dataset
from src.experiments.heatmap import HeatmapConfig
from src.models.model_interface import ModelInterface, get_model_interface
from src.types import MODEL_ARCH, DatasetArgs, TPromptData
from src.utils.logits import get_prompt_row
from src.zenml.materializers.numpy_array_materializer import NumpyArrayMaterializer
from zenml import pipeline
from zenml.logger import get_logger
from zenml.steps import step

logger = get_logger(__name__)


@step
def get_dataset(
    model_id: str,
    dataset_args: DatasetArgs,
) -> TPromptData:
    """Get the dataset as an external artifact."""
    return get_hit_dataset(model_id=model_id, dataset_args=dataset_args)


@step
def setup_model_interface(model_arch: MODEL_ARCH, model_size: str) -> ModelInterface:
    """Setup the model interface."""
    return get_model_interface(model_arch, model_size)


@step(output_materializers=NumpyArrayMaterializer)
def process_window_step(
    window: List[int],
    prompt_idx: int,
    model_interface: ModelInterface,
    dataset: TPromptData,
) -> np.ndarray:
    """Process a single window of layers for a given prompt."""
    prompt = get_prompt_row(dataset, prompt_idx)
    true_id = prompt.true_id(model_interface.tokenizer, "cpu")
    input_ids = prompt.input_ids(model_interface.tokenizer, model_interface.device)

    last_idx = input_ids.shape[1] - 1
    probs = np.zeros((input_ids.shape[1]))

    model_interface.setup(layers=window)

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


@step
def save_prompt_result_step(
    base_output_path: Path,
    prompt_idx: int,
    prompt_result: np.ndarray,
) -> str:
    """Save the results for a single prompt as an artifact."""
    output_path = base_output_path / f"idx={prompt_idx}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, prompt_result)
    return str(output_path)


@step
def process_prompt_heatmap_step(
    prompt_idx: int,
    model_interface: ModelInterface,
    window_size: int,
    dataset: TPromptData,
) -> np.ndarray:
    n_layers = len(model_interface.model.backbone.layers)
    windows = [list(range(i, i + window_size)) for i in range(0, n_layers - window_size + 1)]
    window_results = []
    for window in windows:
        result = process_window_step(window, prompt_idx, model_interface, dataset)
        window_results.append(result)
        # prompt_result = combine_windows_step(window_results, model_interface)
    # save_prompt_result_step(config.output_path, prompt_idx, prompt_result)
    return np.array(window_results).T


@pipeline(enable_cache=True)
def heatmap_pipeline(config: HeatmapConfig):
    """Pipeline for running the heatmap experiment."""
    dataset = get_dataset(model_id=config.model_id, dataset_args=config.dataset_args)
    model_interface = setup_model_interface(config.model_arch, config.model_size)

    for prompt_idx in config.prompt_indices:
        process_prompt_heatmap_step(
            prompt_idx,
            model_interface,
            config.window_size,
            dataset,
        )


if __name__ == "__main__":

    def func():
        return heatmap_pipeline(
            HeatmapConfig(experiment_name="debug_with_zenml", window_size=10, prompt_indices=[3, 4, 5])
        )

    WITH_SLURM = False
    if WITH_SLURM:
        from src.consts import PATHS
        from src.utils.slurm import submit_job

        job_name = "debug_with_zenml"
        gpu_type = "titan_xp-studentrun"

        submit_job(
            func,
            log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
            job_name=job_name,
            gpu_type=gpu_type,
            slurm_gpus_per_node=1,
        )
    else:
        func()
