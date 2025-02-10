import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.experiment_infra.base_config import BASE_OUTPUT_KEYS, BaseConfig, OutputKey, create_mutable_field
from src.experiment_infra.model_interface import get_model_interface
from src.plots.info_flow_confidence import create_confidence_plot
from src.types import TokenType
from src.utils.logits import get_num_to_masks, get_prompt_row


@dataclass
class InfoFlowConfig(BaseConfig):
    """Configuration for information flow analysis."""

    experiment_base_name: str = "info_flow"
    window_size: int = 9
    DEBUG_LAST_WINDOWS: Optional[int] = None
    knockout_map: dict[TokenType, list[TokenType]] = create_mutable_field(
        lambda: {
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

    @property
    def experiment_output_keys(self):
        debug_last_windows_output_key = OutputKey[Optional[int]](
            "DEBUG_LAST_WINDOWS", key_display_name="debug_last_windows_count=", skip_condition=lambda x: x is None
        )
        return super().experiment_output_keys + [
            [BASE_OUTPUT_KEYS.WINDOW_SIZE, debug_last_windows_output_key],
        ]

    def output_block_target_path(self, target: TokenType) -> Path:
        return self.outputs_path / f"target={target}"

    def output_block_target_source_path(self, target: TokenType, source: TokenType) -> Path:
        return self.output_block_target_path(target) / f"source={source}.csv"

    def get_block_target_outputs(self, target: TokenType) -> dict[TokenType, dict[str, dict[str, list[float]]]]:
        return {
            source: json.load(self.output_block_target_source_path(target, source).open("r"))
            for source in self.knockout_map[target]
        }

    def get_outputs(self) -> dict[TokenType, dict[TokenType, dict[str, dict[str, list[float]]]]]:
        return {target: self.get_block_target_outputs(target) for target in self.knockout_map}

    def get_plot_output_path(self, target: TokenType, plot_name: str) -> Path:
        return self.plots_path / f"target={target}{plot_name}.png"

    def plot_block_target(self, target: TokenType, save: bool = False, confidence_level: float = 0.95):
        """Plot information flow from a target block to its source blocks.

        Args:
            target: The target TokenType to analyze flows from
            save: Whether to save the figure
        """
        data = self.get_block_target_outputs(target)

        fig = create_confidence_plot(
            targets_window_outputs=data,
            confidence_level=confidence_level,
            title=f"Knocking out flow to {target}",
        )
        if save:
            fig.savefig(self.get_plot_output_path(target, ""))
            plt.close(fig)
        return fig


def plot(args: InfoFlowConfig):
    knockout_map_outputs = args.get_outputs()
    for target in knockout_map_outputs:
        print(f"Plotting {target}")
        args.plot_block_target(target, save=True)


def run(args: InfoFlowConfig):
    print(args)
    remaining_knockout_map: list[tuple[TokenType, TokenType]] = [
        (target, source)
        for target in args.knockout_map
        for source in args.knockout_map[target]
        if (not args.output_block_target_source_path(target, source).exists() or args.overwrite_existing_outputs)
    ]
    if not remaining_knockout_map:
        print("All outputs already exist")
        return

    args.create_output_path()
    data = args.get_prompt_data()

    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = len(model_interface.model.backbone.layers)

    def forward_eval(
        prompt_idx,
        window,
        knockout_src: TokenType,
        knockout_source: TokenType,
    ):
        prompt = get_prompt_row(data, prompt_idx)
        num_to_masks, first_token = get_num_to_masks(prompt, tokenizer, window, knockout_src, knockout_source, device)

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
        knockout_source: TokenType,
        print_period=100,
    ):
        windows_true_probs: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for i, window in enumerate(tqdm(windows, desc="Windows")):
            windows_true_probs[i] = defaultdict(list)
            model_interface.setup(layers=window)
            for _, prompt_idx in enumerate(tqdm(prompt_indices, desc="Prompts", miniters=print_period)):
                hit, diff, first, diff_unnorm, true_prob = forward_eval(
                    prompt_idx,
                    window,
                    knockout_src,
                    knockout_source,
                )
                windows_true_probs[i]["hit"].append(bool(hit))
                windows_true_probs[i]["true_probs"].append(float(true_prob))
                windows_true_probs[i]["diffs"].append(float(diff))
        return windows_true_probs

    prompt_indices = list(data.index)
    windows = [list(range(i, i + args.window_size)) for i in range(0, n_layers - args.window_size + 1)]

    if args.DEBUG_LAST_WINDOWS:
        windows = windows[-args.DEBUG_LAST_WINDOWS :]

    for target, source in remaining_knockout_map:
        print(f"Knocking out flow to {target} from {source}")

        window_outputs = evaluate(
            prompt_indices,
            windows,
            knockout_src=source,
            knockout_source=target,
        )
        if args.DEBUG_LAST_WINDOWS:
            window_outputs = {
                k + (n_layers - args.window_size + 1 - args.DEBUG_LAST_WINDOWS): v for k, v in window_outputs.items()
            }
        args.output_block_target_source_path(target, source).parent.mkdir(parents=True, exist_ok=True)
        json.dump(window_outputs, (args.output_block_target_source_path(target, source)).open("w"))
