import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pyrallis
import torch
from tqdm import tqdm

from src.consts import PATHS
from src.datasets.download_dataset import get_hit_dataset
from src.experiment_infra.config import BaseConfig
from src.models.model_interface import get_model_interface
from src.types import TokenType
from src.utils.logits import get_num_to_masks, get_prompt_row


@dataclass
class InfoFlowConfig(BaseConfig):
    """Configuration for information flow analysis."""

    experiment_name: str = "info_flow"
    window_size: int = 9
    DEBUG_LAST_WINDOWS: Optional[int] = None
    overwrite: bool = False
    for_multi_plot: bool = False
    knockout_map: dict[TokenType, list[TokenType]] = pyrallis.field(
        default_factory=lambda: {
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
    )


def main_local(args: InfoFlowConfig):
    print(args)
    data = get_hit_dataset(model_id=args.model_id, dataset_args=args.dataset_args)

    if not args.output_file:
        args.output_file = (
            PATHS.OUTPUT_DIR
            / args.model_id
            / args.experiment_name
            / f"ds={args.dataset_args.dataset_name}"
            / f"ws={args.window_size}"
        )

    args.output_file.mkdir(parents=True, exist_ok=True)

    n_prompts = len(data)
    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = len(model_interface.model.backbone.layers)

    def forward_eval(
        prompt_idx,
        window,
        knockout_src: TokenType,
        knockout_target: TokenType,
    ):
        prompt = get_prompt_row(data, prompt_idx)
        num_to_masks, first_token = get_num_to_masks(prompt, tokenizer, window, knockout_src, knockout_target, device)

        next_token_probs = model_interface.generate_logits(
            input_ids=prompt.input_ids(tokenizer, device),
            attention=True,
            num_to_masks=num_to_masks,
        )

        max_prob = np.max(next_token_probs, axis=1)[0]
        true_id = prompt.true_id(tokenizer, "cpu")
        base_prob = prompt.base_prob
        true_prob = next_token_probs[0, true_id[:, 0]]
        torch.cuda.empty_cache()
        return (
            true_prob == max_prob,
            ((true_prob - base_prob) / base_prob) * 100.0,
            first_token,
            (true_prob - base_prob),
            true_prob,
        )

    def evaluate(
        prompt_indices,
        windows,
        knockout_src: TokenType,
        knockout_target: TokenType,
        print_period=100,
    ):
        counts_w_first = np.zeros((len(windows)))
        counts_wo_first = np.zeros((len(windows)))
        diffs_w_first = np.zeros((len(windows)))
        diffs_wo_first = np.zeros((len(windows)))
        diffs_unnorm_w_first = np.zeros((len(windows)))
        diffs_unnorm_wo_first = np.zeros((len(windows)))
        windows_true_probs = defaultdict(lambda: defaultdict(list))
        w_first = 0
        for i, window in enumerate(tqdm(windows, desc="Windows")):
            windows_true_probs[i] = defaultdict(list)
            model_interface.setup(layers=window)
            for _, prompt_idx in enumerate(tqdm(prompt_indices, desc="Prompts", miniters=print_period)):
                hit, diff, first, diff_unnorm, true_prob = forward_eval(
                    prompt_idx,
                    window,
                    knockout_src,
                    knockout_target,
                )
                windows_true_probs[i]["hit"].append(bool(hit))
                windows_true_probs[i]["true_probs"].append(float(true_prob))
                windows_true_probs[i]["diffs"].append(float(diff))
                if first:
                    if i == 0:
                        w_first += 1
                    counts_w_first[i] += hit
                    diffs_w_first[i] += diff
                    diffs_unnorm_w_first[i] += diff_unnorm
                else:
                    counts_wo_first[i] += hit
                    diffs_wo_first[i] += diff
                    diffs_unnorm_wo_first[i] += diff_unnorm
        counts = counts_w_first + counts_wo_first
        diffs = diffs_w_first + diffs_wo_first
        diffs_unnorm = diffs_unnorm_w_first + diffs_unnorm_wo_first
        return {
            "acc": counts / n_prompts,
            "diff": diffs / n_prompts,
            "diff_unnorm": diffs_unnorm / n_prompts,
            "wf_acc": counts_w_first / w_first,
            "wf_diff": diffs_w_first / w_first,
            "wf_diff_unnorm": diffs_unnorm_w_first / w_first,
            "wof_acc": counts_wo_first / (n_prompts - w_first),
            "wof_diff": diffs_wo_first / (n_prompts - w_first),
            "wof_diff_unnorm": diffs_unnorm_wo_first / (n_prompts - w_first),
        }, windows_true_probs

    prompt_indices = list(data.index)
    windows = [list(range(i, i + args.window_size)) for i in range(0, n_layers - args.window_size + 1)]

    if args.DEBUG_LAST_WINDOWS:
        windows = windows[-args.DEBUG_LAST_WINDOWS :]

    combined_results: dict[str, dict] = defaultdict(lambda: defaultdict(dict))

    for key in args.knockout_map:
        for block in args.knockout_map[key]:
            print(f"Knocking out flow to {key} from {block}")
            block_outdir = args.output_file / f"block_{block}_target_{key}"
            block_outdir.mkdir(parents=True, exist_ok=True)

            output_file = block_outdir / "metrics.csv"
            if output_file.exists() and not args.overwrite:
                print("Reading from existing file")
                metrics_df = pd.read_csv(output_file)
                res = {metric: metrics_df[metric].values for metric in metrics_df.columns}
            else:
                res, window_outputs = evaluate(
                    prompt_indices,
                    windows,
                    knockout_src=block,
                    knockout_target=key,
                )
                if args.DEBUG_LAST_WINDOWS:
                    window_outputs = {
                        k + (n_layers - args.window_size + 1 - args.DEBUG_LAST_WINDOWS): v
                        for k, v in window_outputs.items()
                    }
                json.dump(window_outputs, (block_outdir / "outputs.json").open("w"))

                # Combine all metrics into a single DataFrame and save
                metrics_df = pd.DataFrame({metric: value for metric, value in res.items()})
                metrics_df.to_csv(output_file, index=False)

            combined_results[key][block] = res
