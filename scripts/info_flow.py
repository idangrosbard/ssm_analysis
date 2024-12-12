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
    # model_arch: MODEL_ARCH = MODEL_ARCH.MINIMAL_MAMBA2_new
    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
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
    window_size = 9
    overwrite: bool = False
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

    mask = (original_res["hit"] == attn_res["hit"]) & (attn_res["hit"] == True)
    data = attn_res[mask]

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    window_size = args.window_size

    if not args.output_file:
        args.output_file = (
            PATHS.OUTPUT_DIR
            / args.model_id
            / "info_flow_v4"
            / f"ds={args.dataset_args.dataset_name}"
            / f"ws={args.window_size}"
        )

    args.output_file.mkdir(parents=True, exist_ok=True)

    n_prompts = len(data)
    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = len(model_interface.model.backbone.layers)

    # Taken from https://github.com/google-research/google-research/blob/master/dissecting_factual_predictions/utils.py
    def decode_tokens(tokenizer, token_array):
        if hasattr(token_array, "shape") and len(token_array.shape) > 1:
            return [decode_tokens(tokenizer, row) for row in token_array]
        return [tokenizer.decode([t]) for t in token_array]

    def find_token_range(tokenizer, token_array, substring):
        """Find the tokens corresponding to the given substring in token_array."""
        toks = decode_tokens(tokenizer, token_array)
        whole_string = "".join(toks)
        char_loc = whole_string.index(substring)
        loc = 0
        tok_start, tok_end = None, None
        for i, t in enumerate(toks):
            loc += len(t)
            if tok_start is None and loc > char_loc:
                tok_start = i
            if tok_end is None and loc >= char_loc + len(substring):
                tok_end = i + 1
                break
        return (tok_start, tok_end)
    

    def get_num_to_masks(prompt_idx: int, window: List[int], knockout_src: str, knockout_target: str) -> Tuple[Dict[int, List[Tuple[int, int]]], bool]:
        prompt = data.loc[prompt_idx, 'prompt']
        tokens = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids.to(device=device)
        last_idx = input_ids.shape[1] - 1
        num_to_masks = {}
        first_token = False

        last_idx = input_ids.shape[1] - 1
        tok_start, tok_end = find_token_range(tokenizer, input_ids[0], data.loc[prompt_idx, 'subject'])
        subject_tokens = list(range(tok_start, tok_end))
        if 0 in subject_tokens:
            first_token = True
        if knockout_src == 'first':
            src_idx = [0]
        elif knockout_src == 'last':
            src_idx = [last_idx]
        elif knockout_src == 'subject':
            src_idx = subject_tokens
        elif knockout_src == 'relation':
            src_idx = [i for i in range(last_idx + 1) if i not in subject_tokens]
        elif knockout_src == 'context':
            src_idx = [i for i in range(subject_tokens[0])]
        else:
            src_idx = [last_idx]


        if knockout_target == 'last':
            target_idx = [last_idx]
        elif knockout_target == 'subject':
            target_idx = subject_tokens
        else:
            target_idx = [last_idx]
            
        for layer in window:
            for src in src_idx:
                for target in target_idx:
                    if layer not in num_to_masks:
                        num_to_masks[layer] = []
                    num_to_masks[layer].append((target, src))

        return num_to_masks, first_token


    def forward_eval(temperature, top_k, top_p, prompt_idx, window, knockout_src: str, knockout_target: str):
        prompt = data.loc[prompt_idx, "prompt"]
        true_word = data.loc[prompt_idx, "target_true"]
        base_prob = data.loc[prompt_idx, "true_prob"]
        true_token = tokenizer(true_word, return_tensors="pt", padding=True)
        true_id = true_token.input_ids.to(device="cpu")
        tokens = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids.to(device=device)

        num_to_masks, first_token = get_num_to_masks(prompt_idx, window, knockout_src, knockout_target)
    
        next_token_probs = model_interface.generate_logits(
            input_ids=input_ids,
            attention=True,
            num_to_masks=num_to_masks,
        )
        max_prob = np.max(next_token_probs, axis=1)[0]
        true_prob = next_token_probs[0, true_id[:, 0]]
        torch.cuda.empty_cache()
        return (
            true_prob == max_prob,
            (true_prob - base_prob) * 100.0 / base_prob,
            first_token,
        )

    def evaluate(
        temperature, top_k, top_p, prompt_indices, windows, knockout_src: str = None, knockout_target: str = None, print_period=100
    ):
        counts_w_first = np.zeros((len(windows)))
        counts_wo_first = np.zeros((len(windows)))
        diffs_w_first = np.zeros((len(windows)))
        diffs_wo_first = np.zeros((len(windows)))
        w_first = 0
        for i, window in enumerate(tqdm(windows, desc="Windows")):
            model_interface.setup(layers=window)
            for _, prompt_idx in enumerate(tqdm(prompt_indices, desc="Prompts", miniters=print_period)):
                hit, diff, first = forward_eval(
                    temperature, top_k, top_p, prompt_idx, window, knockout_src, knockout_target
                )
                if first:
                    if i == 0:
                        w_first += 1
                    counts_w_first[i] += hit
                    diffs_w_first[i] += diff
                else:
                    counts_wo_first[i] += hit
                    diffs_wo_first[i] += diff
        counts = counts_w_first + counts_wo_first
        diffs = diffs_w_first + diffs_wo_first
        return {
            f"acc": counts / n_prompts,
            f"diff": diffs / n_prompts,
            f"wf_acc": counts_w_first / w_first,
            f"wf_diff": diffs_w_first / w_first,
            f"wof_acc": counts_wo_first / (n_prompts - w_first),
            f"wof_diff": diffs_wo_first / (n_prompts - w_first),
        }

    # prompt_indices = list(data.index)
    # windows = [[]]
    # no_block_acc, no_block_diff, _, _, _, _ = evaluate(
    #     temperature, top_k, top_p, prompt_indices, windows
    # )

    # print(no_block_acc)
    # print(no_block_diff)

    # # Experiments - window size = 9
    prompt_indices = list(data.index)
    windows = [
        list(range(i, i + window_size)) for i in range(0, n_layers - window_size + 1)
    ]

    combined_results = defaultdict(lambda: defaultdict(dict))

    for key in args.knockout_map:
        for block in args.knockout_map[key]:
            print(f"Knocking out flow to {key} from {block}")
            metrics = ['acc','diff','wf_acc','wf_diff','wof_acc','wof_diff', ]
            block_outdir = args.output_file / f"block_{block}_target_{key}"
            block_outdir.mkdir(parents=True, exist_ok=True)
            
            res = {}
            if (block_outdir/f"{metrics[0]}.csv").exists() and not args.overwrite:
                print(f"Reading from existing file")
                for metric in metrics:
                    res[metric] = pd.read_csv(block_outdir / f"{metric}.csv")
            else:
                res = evaluate(temperature, top_k, top_p, prompt_indices, windows, knockout_src=block, knockout_target=key)
            
            for metric, value in res.items():
                df = pd.DataFrame(value)
                if len(df.columns) >1:
                    df = df[df.columns[-1]]
                df.to_csv(block_outdir/f"{metric}.csv", index=False)
                combined_results[key][block][metric] = value

        layers = list(range(n_layers - window_size + 1))
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
        gpu_type = "a100"
        # gpu_type = "titan_xp-studentrun"

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
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"all")
            # for window_size in [9, 15]:
            for window_size in [9]:
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
