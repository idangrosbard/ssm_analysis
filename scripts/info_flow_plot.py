from collections import defaultdict
from dataclasses import dataclass
from hmac import new
import json
from pathlib import Path
from typing import Optional, assert_never, Dict, List, Tuple
from torch import Tensor
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import pyrallis
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from scripts.create_slurm_file import run_slurm
from src.consts import (
    FILTERATIONS,
    MODEL_SIZES_PER_ARCH_TO_MODEL_ID,
    PATHS,
)
from src.datasets.download_dataset import load_dataset, load_splitted_counter_fact
from src.datasets.download_dataset import load_knowns_pd
from src.logit_utils import get_last_token_logits, logits_to_probs
from src.models.model_interface import get_model_interface
from src.types import DATASETS
from src.types import MODEL_ARCH, SPLIT, DatasetArgs, TModelID
from src.utils.setup_models import get_tokenizer_and_model
from src.utils.slurm import submit_job


@dataclass
class Args:
    model_arch: MODEL_ARCH = MODEL_ARCH.MINIMAL_MAMBA2_new
    # model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    model_size: str = "130M"
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"), is_mutable=True
    )
    filteration: str = FILTERATIONS.all_correct
    _batch_size: int = 16  # Adjust based on GPU memory
    output_file: Optional[Path] = None
    with_slurm: bool = False
    temperature = 1
    top_k = 0
    top_p = 1
    window_size = 15
    knockout_map = {'last': ['last', 'first', "subject", "relation"], 
                    'subject': ['context', 'subject']}

    output_dir: Optional[Path] = None

    @property
    def batch_size(self) -> int:
        return (
            1
            if (
                self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2
                or self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2_new
            )
            else self._batch_size
        )

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]


def main_local(args: Args):
    print(args)
    original_res, attn_res = [
        pd.read_parquet(
            PATHS.OUTPUT_DIR
            / args.model_id
            / "data_construction"
            / f"ds={args.dataset_args.dataset_name}"
            / f"entire_results_{"attention" if attention else "original"}.parquet"
        )
        for attention in [True, False]
    ]

    window_size = args.window_size

    if not args.output_file:
        args.output_file = (
            PATHS.OUTPUT_DIR
            / args.model_id
            / "info_flow"
            / f"ds={args.dataset_args.dataset_name}"
            / f"ws={args.window_size}"
        )

    args.output_file.mkdir(parents=True, exist_ok=True)

    combined_results = defaultdict(lambda: defaultdict(dict))

    for key in args.knockout_map:
        for block in args.knockout_map[key]:
            print(f"Knocking out flow to {key} from {block}")
            metrics = ['acc','diff','wf_acc','wf_diff','wof_acc','wof_diff', ]
            block_outdir = args.output_file / f"block_{block}_target_{key}"
            block_outdir.mkdir(parents=True, exist_ok=True)
            
            # if (block_outdir/f"{metrics[0]}.parquet").exists():
            #     (block_outdir/f"{metrics[0]}.parquet").rename(block_outdir/f"{metrics[0]}.csv")
            
            res = {}
            if (block_outdir/f"{metrics[0]}.csv").exists():
                print(f"Reading from existing file")
                for metric in metrics:
                    res[metric] = pd.read_csv(block_outdir / f"{metric}.csv")

            if (block_outdir/f"{metrics[0]}.parquet").exists():
                print(f"Reading from existing file")
                for metric in metrics:
                    res[metric] = pd.read_parquet(block_outdir / f"{metric}.parquet")
                    (block_outdir / f"{metric}.parquet").unlink()
            
            for metric, value in res.items():
                df = pd.DataFrame(value)
                if len(df.columns) >1:
                    df = df[df.columns[-1]]
                df.to_csv(block_outdir/f"{metric}.csv", index=False)
                combined_results[key][block][metric] = value

        layers = list(range(len(df)))
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        colors = {
            "last": "orange",
            "first": "blue",
            "subject": "green",
            "relation": "purple",
            "context": "cyan",
        }
        line_styles = {
            "last": ":",
            "first": "-.",
            "subject": "-",
            "relation": "--",
            "context": ":",
        }

        for block in args.knockout_map[key]:
            color = colors[block]
            line_style = line_styles[block]
            block_acc = combined_results[key][block]["acc"]
            block_diff = combined_results[key][block]["diff"]
            ax[0].plot(layers, block_acc * 100, label=block, color=color, linestyle=line_style)
            ax[1].plot(layers, block_diff, label=block, color=color, linestyle=line_style)

        ax[0].axhline(100, color="gray", linewidth=1)
        ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[0].set_xlabel("Layers")
        ax[0].set_ylabel("% accuracy")
        ax[0].set_title(f"Accuracy - knocking out flow to {key}", fontsize=10)
        ax[0].legend(loc="lower left", fontsize=8)

        ax[1].axhline(0, color="gray", linewidth=1)
        ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[1].set_xlabel("Layers")
        ax[1].set_ylabel("% change in prediction probability")
        ax[1].set_title(f"Change in prediction probability - knocking out flow to {key}", fontsize=10)
        ax[1].legend(loc="lower left", fontsize=8)

        plt.suptitle(f"Results with {args.model_id} and window size={window_size}")
        plt.tight_layout(pad=1, w_pad=3.0)
        plt.savefig(args.output_file /f"results_ws={window_size}_knockout_target={key}.png")
        plt.show()


@pyrallis.wrap()
def main(args: Args):
    # args.with_slurm = True

    if args.with_slurm:
        # gpu_type = "a100"
        gpu_type = "titan_xp-studentrun"

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
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"all")
            for window_size in [9, 15]:
            # for window_size in [9]:
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
        main_local(args)


if __name__ == "__main__":
    main()
