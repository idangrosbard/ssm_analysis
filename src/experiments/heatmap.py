import numpy as np
import pyrallis
import torch
from tqdm import tqdm

from src.config import HeatmapConfig
from src.consts import PATHS
from src.datasets.download_dataset import get_hit_dataset
from src.logit_utils import get_prompt_row
from src.models.model_interface import get_model_interface
from src.types import DATASETS, MODEL_ARCH, DatasetArgs
from src.utils.slurm import submit_job


def main_local(args: HeatmapConfig):
    print(args)
    data = get_hit_dataset(model_id=args.model_id, dataset_args=args.dataset_args)
    window_size = args.window_size

    if not args.output_file:
        args.output_file = (
            PATHS.OUTPUT_DIR
            / args.model_id
            / args.experiment_name
            / f"ds={args.dataset_args.dataset_name}"
            / f"ws={args.window_size}"
        )

    args.output_file.mkdir(parents=True, exist_ok=True)

    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = len(model_interface.model.backbone.layers)

    def forward_eval(prompt_idx, window):
        prompt = get_prompt_row(data, prompt_idx)
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

    windows = [list(range(i, i + window_size)) for i in range(0, n_layers - window_size + 1)]

    for prompt_idx in tqdm(args.prompt_indices, desc="Prompts"):
        prob_mat = []
        for window in windows:
            model_interface.setup(layers=window)
            prob_mat.append(forward_eval(prompt_idx, window))

        prob_mat = np.array(prob_mat).T
        np.save(args.output_file / f"idx={prompt_idx}.npy", prob_mat)


@pyrallis.wrap()
def main(args: HeatmapConfig):
    if args.with_slurm:
        gpu_type = "a100"
        # gpu_type = "titan_xp-studentrun"
        # window_sizes = [5, 9]
        experiment_name = "heatmap_debug_use_matrix"
        variation_name = ""
        args.experiment_name = experiment_name + variation_name
        # window_sizes = [1, 5]
        window_sizes = [9]

        for model_arch, model_size in [
            # (MODEL_ARCH.MAMBA1, "130M"),
            # (MODEL_ARCH.MAMBA1, "1.4B"),
            (MODEL_ARCH.MAMBA1, "2.8B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "130M"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "1.3B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "2.7B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all")
            for window_size in window_sizes:
                args.window_size = window_size

                job_name = (
                    f"{experiment_name}/{model_arch}_{model_size}_ws={window_size}_{args.dataset_args.dataset_name}"
                )
                job = submit_job(
                    main_local,
                    args,
                    log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                    job_name=job_name,
                    # timeout_min=1200,
                    gpu_type=(
                        "a100" if (model_size == "2.7B" and model_arch == MODEL_ARCH.MINIMAL_MAMBA2_new) else gpu_type
                    ),
                    slurm_gpus_per_node=(
                        3 if (model_size in ["2.8B", "2.7B"] and gpu_type == "titan_xp-studentrun") else 1
                    ),
                )

                print(f"{job}: {job_name}")
    else:
        args.experiment_name = "debug"
        args.prompt_indices = [1]
        main_local(args)


if __name__ == "__main__":
    args = HeatmapConfig()
    main(args)
