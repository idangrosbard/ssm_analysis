from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import torch
from tqdm import tqdm

from src.consts import FILTERATIONS, MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS
from src.datasets.download_dataset import get_hit_dataset
from src.logit_utils import decode_tokens, get_prompt_row
from src.models.model_interface import get_model_interface
from src.plots import plot_simple_heatmap
from src.types import DATASETS, MODEL_ARCH, DatasetArgs, TModelID
from src.utils.slurm import submit_job


@dataclass
class Args:
    # model_arch: MODEL_ARCH = MODEL_ARCH.MINIMAL_MAMBA2_new
    experiment_name: str = "heatmap"
    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    model_size: str = "130M"
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"), is_mutable=True
    )
    filteration: str = FILTERATIONS.all_correct
    _batch_size: int = 16  # Adjust based on GPU memory
    output_file: Optional[Path] = None
    with_slurm: bool = False
    window_size = 5
    prompt_indices = [1, 2, 3, 4, 5]

    output_dir: Optional[Path] = None

    @property
    def batch_size(self) -> int:
        return (
            1
            if (self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2 or self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2_new)
            else self._batch_size
        )

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]


def main_local(args: Args):
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
        prompt = data.loc[prompt_idx, "prompt"]
        true_word = data.loc[prompt_idx, "target_true"]
        base_prob = data.loc[prompt_idx, "true_prob"]
        tokens = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids.to(device=device)
        toks = decode_tokens(tokenizer, input_ids[0])
        last_tok = toks[-1]
        toks[-1] = toks[-1] + "*"

        np.save(args.output_file / f"idx={prompt_idx}.npy", prob_mat)
        for heatmap_func, heatmap_name in zip(
            [plot_simple_heatmap],
            ["diverging"],
        ):
            fig, _ = heatmap_func(
                prob_mat=prob_mat,
                model_id=args.model_id,
                window_size=window_size,
                last_tok=last_tok,
                base_prob=base_prob,
                true_word=true_word,
                toks=toks,
                fontsize=8,
            )

            # Save the figure
            output_path = args.output_file / f"idx={prompt_idx}_{heatmap_name}.png"
            fig.tight_layout()
            fig.savefig(output_path)


@pyrallis.wrap()
def main(args: Args):
    # args.with_slurm = True

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
    main()
