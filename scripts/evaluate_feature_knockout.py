import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyrallis
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, MambaForCausalLM

from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS
from src.datasets.download_dataset import get_hit_dataset
from src.knockout.knockout_evaluator import KnockoutEvaluator
from src.knockout.knockout_mode import KnockoutMode
from src.knockout.ssm_knockout import SSMKnockoutEvaluator
from src.knockout.ssm_knockout.ssm_classifier import (
    DecayNormClassifier,
)
from src.types import DATASETS, MODEL_ARCH, DatasetArgs, TModelID
from src.utils.setup_models import setup_mamba_model
from src.utils.slurm import submit_job


@dataclass
class Args:
    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    model_size: str = "2.8B"
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"), is_mutable=True
    )
    show_eval_progress: bool = False
    output_dir: Path = PATHS.RESULTS_DIR / "evaluate_context_interference"
    layer_checkpoint: Optional[Path] = None
    norm: str = "1"
    affected_output: str = "all"
    layer_window_length: int = 9
    with_slurm: bool = False

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]


def binary_search(
    evaluator: KnockoutEvaluator,
    dataset: pd.DataFrame,
    knockout_mode: KnockoutMode,
    start_layers: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    if start_layers is None:
        knockout_target_layers = list(range(len(evaluator.model.backbone.layers)))
    else:
        knockout_target_layers = start_layers

    _, acc = evaluator.knockout_eval(dataset, knockout_target_layers, knockout_mode)

    performance = {"acc": [], "layer": [], "start_layer": [], "end_layer": []}
    performance["layer"].append(str(knockout_target_layers))
    performance["start_layer"].append(min(knockout_target_layers))
    performance["end_layer"].append(max(knockout_target_layers))
    performance["acc"].append(acc)
    print(acc)

    # log(n) binary search
    n = len(knockout_target_layers)
    log_n = np.ceil(np.log2(n))

    with torch.no_grad():
        pbar = tqdm(range(int(log_n)), desc="Binary search for optimal layer")
        for _ in pbar:
            if (len(knockout_target_layers) // 2) < (len(knockout_target_layers) / 2):
                early = knockout_target_layers[: len(knockout_target_layers) // 2 + 1]
                late = knockout_target_layers[len(knockout_target_layers) // 2 :]
            else:
                early = knockout_target_layers[: len(knockout_target_layers) // 2]
                late = knockout_target_layers[len(knockout_target_layers) // 2 :]

            _, acc_early = evaluator.knockout_eval(dataset, early, knockout_mode)
            performance["layer"].append(str(early))
            performance["acc"].append(acc_early)
            performance["start_layer"].append(min(early))
            performance["end_layer"].append(max(early))

            _, acc_late = evaluator.knockout_eval(dataset, late, knockout_mode)
            performance["layer"].append(str(late))
            performance["acc"].append(acc_late)
            performance["start_layer"].append(min(late))
            performance["end_layer"].append(max(late))

            if acc_early < acc_late:
                knockout_target_layers = early
            else:
                knockout_target_layers = late
            pbar.set_description(f"Binary search for optimal layer. Curr acc: {min(acc_early, acc_late)}")

    df = pd.DataFrame(performance)

    return df


def layer_by_layer(
    evaluator: KnockoutEvaluator,
    dataset: pd.DataFrame,
    knockout_mode: KnockoutMode,
    n_layers: int,
) -> pd.DataFrame:
    performance = {"acc": [], "layer": []}
    n = len(evaluator.model.backbone.layers)

    with torch.no_grad():
        # evaluate every single layer
        for i in tqdm(range(n), desc="Iterating layers for knockout..."):
            _, acc = evaluator.knockout_eval(dataset, [j for j in range(i, min(n, i + n_layers))], knockout_mode)
            performance["layer"].append(i)
            performance["acc"].append(acc)

    df = pd.DataFrame(performance)

    return df


def ssm_knockout_evaluate(
    args: Args,
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    knowns_df: pd.DataFrame,
    norm: int | float,
    n_layers,
):
    ssm_classifier = DecayNormClassifier()
    out_df = None
    ssm_classifier.norm = norm

    categorized_ssms = ssm_classifier.classify_model(model.backbone)
    for category in categorized_ssms:
        evaluator = SSMKnockoutEvaluator(model, tokenizer, device, categorized_ssms[category], False)
        curr = layer_by_layer(evaluator, knowns_df, KnockoutMode.IGNORE_SSM, n_layers)
        curr["category"] = category
        curr["norm"] = norm
        out_df = pd.concat([out_df, curr])

        out_fname = args.output_dir / f"{args.model_size}_norm_{norm}_bin_search.csv"
        if out_fname.exists():
            os.remove(out_fname)
        out_df.to_csv(out_fname)

    out_df.to_csv(args.output_dir / f"{args.model_size}_norm_{norm}_output.csv")


def get_checkpoint(pth: Optional[Path]) -> Optional[pd.DataFrame]:
    if pth is not None:
        return pd.read_csv(pth)
    return None


def main_local(args: Args) -> None:
    print(args)
    model, tokenizer, device = setup_mamba_model(args.model_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # bin_search_checkpoint = get_checkpoint(args.bin_search_checkpoint)
    layer_checkpoint = get_checkpoint(args.layer_checkpoint)
    # print(bin_search_checkpoint)
    print(layer_checkpoint)

    knowns_df = get_hit_dataset(model_id=args.model_id, dataset_args=args.dataset_args)
    # drop the first character in the attribute string
    # knowns_df['attribute'] = knowns_df['attribute'].apply(lambda x: x[1:])
    if args.norm == "inf":
        norm = float("inf")
    else:
        norm = int(args.norm)

    ssm_knockout_evaluate(
        args,
        model,
        tokenizer,
        device,
        knowns_df,
        norm=norm,
        n_layers=args.layer_window_length,
    )


@pyrallis.wrap()
def main(args: Args):
    def get_experiment_configs():
        """Generate configurations for different experiment variations."""
        base_config = {
            "model_size": "2.8B",
            "output_dir": "",
        }

        # Standard configurations
        configs = [
            {**base_config, "interfere_mode": "ZERO_ATTENTION"},
            {**base_config, "interfere_mode": "IGNORE_SSM"},
            {
                **base_config,
                "interfere_mode": "IGNORE_SSM",
                "early_layers_ssm_knockout": True,
            },
        ]

        # Delta configurations
        delta_variations = [
            {"delta_factor_root": 0.5, "delta_start_layer": 40, "delta_end_layer": 48},
            {"delta_factor_root": 1.5, "delta_start_layer": 40, "delta_end_layer": 48},
            {"delta_factor_root": 0.5, "delta_start_layer": 56, "delta_end_layer": 64},
            {"delta_factor_root": 1.5, "delta_start_layer": 56, "delta_end_layer": 64},
        ]

        # Add delta configurations
        for delta_config in delta_variations:
            configs.append(
                {
                    **base_config,
                    "interfere_mode": "INCREASE_DELTA",
                    "increase_delta_target": "LAST",
                    **delta_config,
                }
            )

        return configs

    if args.with_slurm:
        gpu_type = "a100"
        # gpu_type = "titan_xp-studentkillable"

        for model_arch, model_size in [
            (MODEL_ARCH.MAMBA1, "130M"),
            (MODEL_ARCH.MAMBA1, "1.4B"),
            (MODEL_ARCH.MAMBA1, "2.8B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "130M"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "1.3B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "2.7B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all")

            job_name = f"evaluate_context_interference/{model_arch}_{model_size}_{args.dataset_args.dataset_name}"

            job = submit_job(
                main_local,
                args,
                log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                job_name=job_name,
                # timeout_min=1200,
                gpu_type=gpu_type,
                slurm_gpus_per_node=1,
            )

            print(f"{job}: {job_name}")
    else:
        main_local(args)


if __name__ == "__main__":
    main()
