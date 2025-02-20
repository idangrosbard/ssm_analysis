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

from src.consts import PATHS, is_falcon
from src.datasets.download_dataset import load_dataset, load_knowns_pd
from src.knockout.attention_knockout import (
    AttentionKnockoutEvaluator,
    KnockoutTarget,
    is_last_token_subj,
)
from src.knockout.increase_delta import IncreaseDeltaEvaluator
from src.knockout.knockout_evaluator import KnockoutEvaluator
from src.knockout.knockout_mode import KnockoutMode
from src.knockout.layer_knockout import LayerKnockoutEvaluator
from src.knockout.ssm_knockout import SSMKnockoutEvaluator
from src.knockout.ssm_knockout.ssm_classifier import (
    DecayNormClassifier,
    SSMClassifierStub,
)
from src.types import DATASETS, DatasetArgs
from src.utils.setup_models import setup_mamba_model
from src.utils.slurm import submit_job


@dataclass
class Args:
    model_size: str = "2.8B"
    interfere_mode: str = "INCREASE_DELTA"
    drop_subj_last: bool = False
    show_eval_progress: bool = False
    output_dir: Path = PATHS.RESULTS_DIR / "evaluate_context_interference"
    layer_checkpoint: Optional[Path] = None
    bin_search_checkpoint: Optional[Path] = None
    ignore_layer_by_layer: bool = False
    norm: str = "1"
    early_layers_ssm_knockout: bool = False
    affected_output: str = "all"
    delta_factor_root: float = 0.9
    delta_start_layer: int = 40
    delta_end_layer: int = 48
    non_selective_ssm: bool = False
    increase_delta_target: str = "LAST"
    with_slurm: bool = False
    split_name: str = "train1"
    search_mode: str = "binary"  # Options: "binary", "layer", "both"


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


def layer_by_layer(evaluator: KnockoutEvaluator, dataset: pd.DataFrame, knockout_mode: KnockoutMode) -> pd.DataFrame:
    performance = {"acc": [], "layer": []}

    with torch.no_grad():
        # evaluate every single layer
        for i in tqdm(
            range(len(evaluator.model.backbone.layers)),
            desc="Iterating layers for knockout...",
        ):
            _, acc = evaluator.knockout_eval(dataset, [i], knockout_mode)
            performance["layer"].append(i)
            performance["acc"].append(acc)

    df = pd.DataFrame(performance)

    return df


def get_last_token_stats(model_size: str = "130M"):
    knowns_df = load_knowns_pd()

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")

    pbar = tqdm(knowns_df.index, total=len(knowns_df), disable=False)
    stat = 0
    for idx in pbar:
        # Get relevant data
        input = knowns_df.loc[idx, "prompt"]
        subj = knowns_df.loc[idx, "subject"]

        val = is_last_token_subj(input, subj, tokenizer)

        stat += val / len(knowns_df)

    print(stat)


def attention_knockout_evaluate(
    args: Args,
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    knowns_df: pd.DataFrame,
    is_falcon: bool,
    layer_checkpoint: Optional[pd.DataFrame] = None,
    bin_search_checkpoint: Optional[pd.DataFrame] = None,
    affected_output: str = "all",
):
    evaluator = AttentionKnockoutEvaluator(
        model, tokenizer, device, -1, -1, is_falcon, args.drop_subj_last, args.show_eval_progress
    )

    bin_search_df = bin_search_checkpoint
    layer_df = layer_checkpoint

    if affected_output == "all":
        affected_outputs = [KnockoutTarget.LAST, KnockoutTarget.ENTIRE_SUBJ]
    elif affected_output == "last":
        affected_outputs = [KnockoutTarget.LAST]
    elif affected_output == "subj":
        affected_outputs = [KnockoutTarget.ENTIRE_SUBJ]

    specific_targets = {
        KnockoutTarget.LAST: [
            KnockoutTarget.LAST,
            KnockoutTarget.ENTIRE_SUBJ,
            KnockoutTarget.SUBJ_CONTEXT,
        ],
        KnockoutTarget.ENTIRE_SUBJ: [
            KnockoutTarget.ENTIRE_SUBJ,
            KnockoutTarget.SUBJ_CONTEXT,
        ],
    }

    for output in affected_outputs:
        for target in specific_targets[output]:
            evaluator.knockout_target = target
            evaluator.affected_target = output

            # Binary search evaluation
            if args.search_mode in ["binary", "both"]:
                cond = bin_search_df is None
                if not cond:
                    cond = (
                        len(
                            bin_search_df[
                                (bin_search_df["knockout_inputs"] == str(target))
                                & (bin_search_df["affected_outputs"] == str(output))
                            ]
                        )
                        == 0
                    )

                if cond:
                    print("Binary search for", target, output)
                    curr_df = binary_search(evaluator, knowns_df, KnockoutMode[args.interfere_mode])
                    curr_df["knockout_inputs"] = target
                    curr_df["affected_outputs"] = output
                    bin_search_df = pd.concat([bin_search_df, curr_df] if bin_search_df is not None else [curr_df])

                    # save to csv
                    out_fname = args.output_dir / f"{args.interfere_mode}_{args.model_size}_bin_search.csv"
                    if out_fname.exists():
                        os.remove(out_fname)
                    bin_search_df.to_csv(out_fname)
                else:
                    print(f"Skipping {target} {output} for binary search")

            # Layer by layer evaluation
            if args.search_mode in ["layer", "both"]:
                cond = layer_df is None
                if not cond:
                    cond = (
                        len(
                            layer_df[
                                (layer_df["knockout_inputs"] == str(target))
                                & (layer_df["affected_outputs"] == str(output))
                            ]
                        )
                        == 0
                    )

                if cond:
                    print("Layer iteration for", target, output)
                    curr_df = layer_by_layer(evaluator, knowns_df, KnockoutMode[args.interfere_mode])
                    curr_df["knockout_inputs"] = target
                    curr_df["affected_outputs"] = output
                    layer_df = pd.concat([layer_df, curr_df] if layer_df is not None else [curr_df])

                    # save to csv
                    out_fname = args.output_dir / f"{args.interfere_mode}_{args.model_size}_layer_by_layer.csv"
                    if out_fname.exists():
                        os.remove(out_fname)
                    layer_df.to_csv(out_fname)
                else:
                    print(f"Skipping {target} {output} for layer by layer")


def layer_knockout_evaluate(
    args: Args,
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    knowns_df: pd.DataFrame,
):
    evaluator = LayerKnockoutEvaluator(model, tokenizer, device, args.show_eval_progress)
    bin_search_df = binary_search(evaluator, knowns_df, KnockoutMode[args.interfere_mode])
    # layer_df = layer_by_layer(evaluator, knowns_df, KnockoutMode[args.interfere_mode])

    bin_search_df.to_csv(args.output_dir / f"{args.interfere_mode}_{args.model_size}_bin_search.csv")
    # layer_df.to_csv(args.output_dir / f"{args.interfere_mode}_{args.model_size}_layer_by_layer.csv")


def ssm_knockout_evaluate_early_layers(
    args: Args,
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    knowns_df: pd.DataFrame,
    norm: int | float,
):
    ssm_classifier = DecayNormClassifier()
    bin_search_df = None
    ssm_classifier.norm = norm

    categorized_ssms = ssm_classifier.classify_model(model.backbone)
    for category in categorized_ssms:
        evaluator = SSMKnockoutEvaluator(model, tokenizer, device, categorized_ssms[category], False)
        curr = binary_search(
            evaluator,
            knowns_df,
            KnockoutMode[args.interfere_mode],
            start_layers=list(range(len(model.backbone.layers) // 2)),
        )
        curr["category"] = category
        curr["norm"] = norm
        bin_search_df = pd.concat([bin_search_df, curr])

        out_fname = (
            args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_bin_search_early_layers_focus.csv"
        )
        if out_fname.exists():
            os.remove(out_fname)
        bin_search_df.to_csv(out_fname)

    bin_search_df.to_csv(
        args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_bin_search_early_layers_focus.csv"
    )


def ssm_knockout_evaluate(
    args: Args,
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    knowns_df: pd.DataFrame,
    norm: int | float,
    ignore_layer_by_layer: bool = True,
):
    ssm_classifier = DecayNormClassifier()
    bin_search_df = None
    layer_df = None
    ssm_classifier.norm = norm

    categorized_ssms = ssm_classifier.classify_model(model.backbone)
    for category in categorized_ssms:
        evaluator = SSMKnockoutEvaluator(model, tokenizer, device, categorized_ssms[category], False)

        # Binary search evaluation
        if args.search_mode in ["binary", "both"]:
            curr = binary_search(evaluator, knowns_df, KnockoutMode[args.interfere_mode])
            curr["category"] = category
            curr["norm"] = norm
            bin_search_df = pd.concat([bin_search_df, curr] if bin_search_df is not None else [curr])

            out_fname = args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_bin_search.csv"
            if out_fname.exists():
                os.remove(out_fname)
            bin_search_df.to_csv(out_fname)

        # Layer by layer evaluation
        if args.search_mode in ["layer", "both"] and not ignore_layer_by_layer:
            curr = layer_by_layer(evaluator, knowns_df, KnockoutMode[args.interfere_mode])
            curr["category"] = category
            curr["norm"] = norm
            layer_df = pd.concat([layer_df, curr] if layer_df is not None else [curr])

            out_fname = args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_layer_by_layer.csv"
            if out_fname.exists():
                os.remove(out_fname)
            layer_df.to_csv(out_fname)

    # Save final results
    if args.search_mode in ["binary", "both"]:
        bin_search_df.to_csv(args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_bin_search.csv")
    if args.search_mode in ["layer", "both"] and not ignore_layer_by_layer:
        layer_df.to_csv(args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_layer_by_layer.csv")


def get_checkpoint(pth: Optional[Path]) -> Optional[pd.DataFrame]:
    if pth is not None:
        return pd.read_csv(pth)
    return None


def increase_delta_evaluate(
    args: Args,
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    knowns_df: pd.DataFrame,
    root_factor: float,
    start_layer: int,
    end_layer: int,
    non_selective_ssm: bool,
    target: KnockoutTarget = KnockoutTarget.LAST,
):
    if args.model_size == "130M":
        layers_of_interest = [18, 19, 20, 21]
    else:
        layers_of_interest = [40, 41, 42, 43, 44, 45, 46, 47]
        layers_of_interest = sorted([63, 62, 61, 60, 59, 58, 57, 56])
        layers_of_interest = list(range(start_layer, end_layer))
    if non_selective_ssm:
        layer_classification = SSMClassifierStub().classify_model(model.backbone)
    else:
        layer_classification = DecayNormClassifier(norm=1).classify_model(model.backbone)

    performance = {"acc": [], "layers": [], "factor": [], "category": []}
    # target = KnockoutTarget.ENTIRE_SUBJ
    # target = KnockoutTarget.SUBJ_FIRST
    # target = KnockoutTarget.AFTER_SUBJ
    # target = KnockoutTarget.LAST

    for factor in [root_factor ** (i + 1) for i in range(6)]:
        for category in layer_classification:
            evaluator = IncreaseDeltaEvaluator(
                model,
                tokenizer,
                device,
                target,
                layer_classification[category],
                factor,
                args.show_eval_progress,
            )

            _, acc = evaluator.knockout_eval(knowns_df, layers_of_interest, KnockoutMode.INCREASE_DELTA)

            performance["layers"].append(str(layers_of_interest))
            performance["factor"].append(factor)
            performance["category"].append(category)
            performance["acc"].append(acc)

            # save to csv
            df = pd.DataFrame(performance)
            print(df)
            out_fname = args.output_dir / (
                f"{args.interfere_mode}_{args.model_size}_target_{target}_layer_neighborhood"
                f"_{max(layers_of_interest)}-{min(layers_of_interest)}_root_decay_factor_{root_factor}.csv"
            )
            if out_fname.exists():
                os.remove(out_fname)
            df.to_csv(out_fname)

    df = pd.DataFrame(performance)
    out_fname = args.output_dir / (
        f"{args.interfere_mode}_{args.model_size}_target_{target}"
        f"_layer_neighborhood_{max(layers_of_interest)}-{min(layers_of_interest)}_root_decay_factor_{root_factor}.csv"
    )
    if out_fname.exists():
        os.remove(out_fname)
    df.to_csv(out_fname)


def main_local(args: Args) -> None:
    print(args)
    get_last_token_stats(args.model_size)
    model, tokenizer, device = setup_mamba_model(args.model_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bin_search_checkpoint = get_checkpoint(args.bin_search_checkpoint)
    layer_checkpoint = get_checkpoint(args.layer_checkpoint)
    print(bin_search_checkpoint)
    print(layer_checkpoint)

    # If we do attention knockout:
    if KnockoutMode[args.interfere_mode] in {
        KnockoutMode.ZERO_ATTENTION,
        KnockoutMode.ZERO_DELTA,
    }:
        knowns_df = pd.DataFrame(load_dataset(DatasetArgs(name=DATASETS.KNOWN_1000, splits=[args.split_name])))
        knowns_df["attribute"] = knowns_df["attribute"].apply(lambda x: x[1:])
        attention_knockout_evaluate(
            args,
            model,
            tokenizer,
            device,
            knowns_df,
            is_falcon=is_falcon(args.model_size),
            layer_checkpoint=layer_checkpoint,
            bin_search_checkpoint=bin_search_checkpoint,
            affected_output=args.affected_output,
        )

    # Not done in our code
    # If we skip entire layer \ component
    elif KnockoutMode[args.interfere_mode] in {
        KnockoutMode.IGNORE_CONTEXT,
        KnockoutMode.IGNORE_LAYER,
        KnockoutMode.ONLY_CONTEXT,
    }:
        knowns_df = pd.DataFrame(load_dataset(DatasetArgs(name=DATASETS.KNOWN_1000, splits=[args.split_name])))
        knowns_df["attribute"] = knowns_df["attribute"].apply(lambda x: x[1:])
        layer_knockout_evaluate(args, model, tokenizer, device, knowns_df)

    # If we do SSM knockout
    elif KnockoutMode[args.interfere_mode] == KnockoutMode.IGNORE_SSM:
        knowns_df = pd.DataFrame(load_dataset(DatasetArgs(name=DATASETS.KNOWN_1000, splits=[args.split_name])))
        # drop the first character in the attribute string
        knowns_df["attribute"] = knowns_df["attribute"].apply(lambda x: x[1:])
        if args.norm == "inf":
            norm = float("inf")
        else:
            norm = int(args.norm)
        if args.early_layers_ssm_knockout:
            ssm_knockout_evaluate_early_layers(args, model, tokenizer, device, knowns_df, norm=norm)
        else:
            ssm_knockout_evaluate(
                args,
                model,
                tokenizer,
                device,
                knowns_df,
                norm=norm,
                ignore_layer_by_layer=args.ignore_layer_by_layer,
            )

    elif KnockoutMode[args.interfere_mode] == KnockoutMode.INCREASE_DELTA:
        knowns_df = pd.DataFrame(load_dataset(DatasetArgs(name=DATASETS.KNOWN_1000, splits=[args.split_name])))
        knowns_df["attribute"] = knowns_df["attribute"].apply(lambda x: x[1:])
        increase_delta_evaluate(
            args,
            model,
            tokenizer,
            device,
            knowns_df,
            args.delta_factor_root,
            args.delta_start_layer,
            args.delta_end_layer,
            args.non_selective_ssm,
            KnockoutTarget[args.increase_delta_target],
        )
    else:
        raise ValueError(f"Unknown knockout mode: {args.interfere_mode}")

    print("Done")


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
        args.model_size = "2.8B"

        # gpu_type = "titan_xp-studentrun"
        # gpu_type = "titan_xp-studentbatch"
        gpu_type = "titan_xp-studentkillable"
        # gpu_type = "a100"

        job_name1 = f"evaluate_context_interference_{args.model_size}"
        args.output_dir = args.output_dir / args.model_size / "split"

        for i in [
            1,
            # 2,
        ]:
            args.output_dir = args.output_dir.parent / f"split{i}"
            args.split_name = f"train{i}"
            job_name2 = job_name1 + f"_split={args.split_name}"

            for interfere_mode in [
                # 'ZERO_ATTENTION',
                # 'IGNORE_SSM',
                "INCREASE_DELTA",
            ]:
                args.interfere_mode = interfere_mode
                job_name3 = job_name2 + f"_{interfere_mode}"

                mods: list[dict] = [{}]

                if interfere_mode == "INCREASE_DELTA":
                    mods = [
                        {
                            "delta_factor_root": 0.5,
                            "delta_start_layer": 40,
                            "delta_end_layer": 48,
                            "increase_delta_target": "LAST",
                        },
                        {
                            "delta_factor_root": 1.5,
                            "delta_start_layer": 40,
                            "delta_end_layer": 48,
                            "increase_delta_target": "LAST",
                        },
                        {
                            "delta_factor_root": 0.5,
                            "delta_start_layer": 56,
                            "delta_end_layer": 64,
                            "increase_delta_target": "LAST",
                        },
                        {
                            "delta_factor_root": 1.5,
                            "delta_start_layer": 56,
                            "delta_end_layer": 64,
                            "increase_delta_target": "LAST",
                        },
                    ]
                    short_cuts = {
                        "delta_factor_root": "dfr",
                        "delta_start_layer": "dsl",
                        "delta_end_layer": "del",
                        "increase_delta_target": "idt",
                    }
                if interfere_mode == "IGNORE_SSM":
                    mods = [
                        {"early_layers_ssm_knockout": False},
                        {"early_layers_ssm_knockout": True},
                    ]

                    short_cuts = {
                        "early_layers_ssm_knockout": "elsk",
                    }

                # Update args with config
                for mod in mods:
                    prev_vals = {}
                    job_name = job_name3
                    for key in mod:
                        prev_vals[key] = getattr(args, key)
                        setattr(args, key, mod[key])
                        job_name += f"_{short_cuts[key]}={mod[key]}"

                    job = submit_job(
                        main_local,
                        args,
                        log_folder=str(PATHS.SLURM_DIR / job_name1 / job_name / "%j"),
                        job_name=job_name,
                        gpu_type=gpu_type,
                        slurm_gpus_per_node=1,
                    )

                    print(f"{job}: {job_name}")
                    # Restore args
                    for key in mod:
                        setattr(args, key, prev_vals[key])
    else:
        main_local(args)


if __name__ == "__main__":
    main()
