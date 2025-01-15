from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt

from src.consts import PATHS
from src.experiments.info_flow import InfoFlowConfig


def main_local(args: InfoFlowConfig):
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
                    f"{args.model_arch} - size {args.model_size}, window size={window_size}",
                    fontsize=12,
                )
            else:
                plt.suptitle(
                    f"Knocking out flow to {key}"
                    "\n"
                    f"{args.model_arch} - size {args.model_size}, window size={window_size}"
                    "\n"
                    f"{plot_metadata["title"]}",
                    fontsize=12,
                )

            results_dir_name = "results_for_multi_plot" if args.for_multi_plot else "results"
            key_results_path = args.output_file / results_dir_name / f"knockout_target={key}"
            key_results_path.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(key_results_path / f"{plot_metadata['filename_suffix']}.png")
