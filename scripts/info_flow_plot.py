from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyrallis
import torch
from matplotlib import pyplot as plt

from src.consts import FILTERATIONS, MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS
from src.types import DATASETS, MODEL_ARCH, DatasetArgs, TModelID, TokenType
from src.utils.slurm import submit_job


def get_top_outputs(probs, tokenizer, top_k):
    # Get the top 5 outputs and their probs
    top_probs, top_indices = map(torch.Tensor.tolist, torch.topk(torch.Tensor(probs), top_k))
    top_tokens = list(map(tokenizer.batch_decode, top_indices))
    return list(
        map(
            list,
            map(
                lambda x: zip(*x),
                zip(
                    top_indices,
                    top_tokens,
                    top_probs,
                ),
            ),
        )
    )


@dataclass
class Args:
    # model_arch: MODEL_ARCH = MODEL_ARCH.MINIMAL_MAMBA2_new
    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    experiment_name: str = "info_flow"
    model_size: str = "130M"
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"), is_mutable=True
    )
    filteration: str = FILTERATIONS.all_correct
    _batch_size: int = 16  # Adjust based on GPU memory
    output_file: Optional[Path] = None
    with_slurm: bool = False
    window_size = 9
    DEBUG_LAST_WINDOWS: Optional[int] = None
    overwrite: bool = False
    for_multi_plot: bool = False
    knockout_map = {
        TokenType.last: [
            TokenType.last,
            TokenType.first,
            TokenType.subject,
            TokenType.relation,
        ],
        TokenType.subject: [
            TokenType.context,
            TokenType.subject,
        ],
        TokenType.relation: [
            TokenType.context,
            TokenType.subject,
            TokenType.relation,
        ],
    }

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

    combined_results = defaultdict(lambda: defaultdict(dict))

    for key in args.knockout_map:
        for block in args.knockout_map[key]:
            print(f"Knocking out flow to {key} from {block}")
            metrics = [
                "acc",
                "diff",
                "wf_acc",
                "wf_diff",
                "wof_acc",
                "wof_diff",
                "diff_unnorm",
                "wf_diff_unnorm",
                "wof_diff_unnorm",
            ]
            block_outdir = args.output_file / f"block_{block}_target_{key}"
            block_outdir.mkdir(parents=True, exist_ok=True)

            # if (block_outdir/f"{metrics[0]}.parquet").exists():
            #     (block_outdir/f"{metrics[0]}.parquet").rename(block_outdir/f"{metrics[0]}.csv")

            res = {}
            if (block_outdir / f"{metrics[0]}.csv").exists():
                print("Reading from existing file")
                for metric in metrics:
                    res[metric] = pd.read_csv(block_outdir / f"{metric}.csv")

            if (block_outdir / f"{metrics[0]}.parquet").exists():
                print("Reading from existing file")
                for metric in metrics:
                    res[metric] = pd.read_parquet(block_outdir / f"{metric}.parquet")
                    (block_outdir / f"{metric}.parquet").unlink()

            for metric, value in res.items():
                df = pd.DataFrame(value)
                if len(df.columns) > 1:
                    df = df[df.columns[-1]]
                df.to_csv(block_outdir / f"{metric}.csv", index=False)
                combined_results[key][block][metric] = value

        layers = list(range(len(df)))

        colors = {
            "last": "#D2691E",
            "first": "blue",
            "subject": "green",
            "relation": "purple",
            "context": "red",
        }
        line_styles = {
            "last": "-.",
            "first": ":",
            "subject": "-",
            "relation": "--",
            "context": "--",
        }

        plots_meta_data = {
            "acc": {
                "title": f"Accuracy - knocking out flow to {key}",
                "ylabel": "% accuracy",
                "ylabel_loc": "center",
                "axhline_value": 100,
                "filename_suffix": "accuracy",
                "data_modifier": lambda x: x * 100,
            },
            "diff": {
                "title": "Normalized change in prediction probability",
                "ylabel": "% probability change",
                "ylabel_loc": "top",
                "axhline_value": 0,
                "filename_suffix": "norm_change",
                "data_modifier": lambda x: x,
            },
            "diff_unnorm": {
                "title": "Change in prediction probability",
                "ylabel": "Change",
                "ylabel_loc": "center",
                "axhline_value": 0,
                "filename_suffix": "unnorm_change",
                "data_modifier": lambda x: x,
            },
        }

        # fig, ax = plt.subplots(1, 3, figsize=(15, 3))

        for data_key, plot_metadata in plots_meta_data.items():
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            for block in args.knockout_map[key]:
                block_metrics = combined_results[key][block][data_key]
                ax.plot(
                    layers,
                    plot_metadata["data_modifier"](block_metrics),
                    label=block,
                    color=colors[block],
                    linestyle=line_styles[block],
                )
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Layers", fontsize=12)
            ax.legend(
                loc="upper center",  # Place legend above the plot
                bbox_to_anchor=(
                    0.4,
                    1.3,
                ),  # Anchor at the top-center, slightly above the plot
                ncol=len(colors),  # Arrange all legend items in a single horizontal row
                fontsize=12,  # Adjust font size
                frameon=False,  # Remove legend border for cleaner appearance
            )
            ax.axhline(plot_metadata["axhline_value"], color="gray", linewidth=1)
            ax.set_ylabel(plot_metadata["ylabel"], fontsize=12, loc=plot_metadata["ylabel_loc"])
            ax.tick_params(axis="both", which="major", labelsize=12)
            fig.subplots_adjust(top=0.4)

            if args.for_multi_plot:
                plt.suptitle(
                    # f"Knocking out flow to {key}"
                    # "\n"
                    f"{args.model_arch} - size {args.model_size}, window size={window_size}",
                    # "\n"
                    # f"{plot_metadata['title']}"
                    # "\n"
                    fontsize=12,
                )
            else:
                plt.suptitle(
                    f"Knocking out flow to {key}"
                    "\n"
                    f"{args.model_arch} - size {args.model_size}, window size={window_size}"
                    "\n"
                    f"{plot_metadata["title"]}",
                    # "\n"
                    fontsize=12,
                )

            results_dir_name = "results_for_multi_plot" if args.for_multi_plot else "results"
            key_results_path = args.output_file / results_dir_name / f"knockout_target={key}"
            key_results_path.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(key_results_path / f"{plot_metadata['filename_suffix']}.png")


@pyrallis.wrap()
def main(args: Args):
    # args.with_slurm = True

    if args.with_slurm:
        # gpu_type = "a100"
        gpu_type = "titan_xp-studentrun"
        args.experiment_name += "_v6"
        window_sizes = [9, 15]

        for model_arch, model_size in [
            (MODEL_ARCH.MAMBA1, "130M"),
            (MODEL_ARCH.MAMBA1, "1.4B"),
            (MODEL_ARCH.MAMBA1, "2.8B"),
            (MODEL_ARCH.MINIMAL_MAMBA2_new, "130M"),
            (MODEL_ARCH.MINIMAL_MAMBA2_new, "1.3B"),
            (MODEL_ARCH.MINIMAL_MAMBA2_new, "2.7B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all")
            # for window_size in [9, 15]:
            for window_size in window_sizes:
                args.window_size = window_size

                job_name = f"info_flow/{model_arch}_{model_size}_ws={window_size}_{args.dataset_args.dataset_name}"
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
        args.experiment_name += "_v6"
        main_local(args)


if __name__ == "__main__":
    main()
