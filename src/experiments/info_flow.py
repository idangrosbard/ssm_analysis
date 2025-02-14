import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.consts import COLUMNS, is_mamba_arch
from src.experiment_infra.base_config import BASE_OUTPUT_KEYS, BaseConfig, create_mutable_field
from src.experiment_infra.model_interface import ModelInterface, get_model_interface
from src.experiment_infra.output_path import OutputKey
from src.plots.info_flow_confidence import create_confidence_plot
from src.types import MODEL_ARCH, FeatureCategory, TInfoFlowSource, TokenType, TTokenizer
from src.utils.logits import Prompt, get_num_to_masks, get_prompt_row_index

# Time in seconds between intermediate saves
SAVE_INTERVAL = 600  # 10 minutes


def skip_task(model_arch: MODEL_ARCH, source: TInfoFlowSource) -> bool:
    return not (is_mamba_arch(model_arch) or not isinstance(source, tuple))


@dataclass
class InfoFlowConfig(BaseConfig):
    """Configuration for information flow analysis."""

    experiment_base_name: str = "info_flow"
    window_size: int = 9
    DEBUG_LAST_WINDOWS: Optional[int] = None
    knockout_map: dict[TokenType, list[TInfoFlowSource]] = create_mutable_field(
        lambda: {
            TokenType.last: [
                TokenType.last,
                (TokenType.subject, FeatureCategory.SLOW_DECAY),
                (TokenType.subject, FeatureCategory.FAST_DECAY),
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

    def intermediate_outputs_path(self) -> Path:
        return self.experiment_variation_base_path / "intermediate_outputs"

    def get_intermediate_output_path(self, target: TokenType, source: TInfoFlowSource) -> Path:
        """Get the path for intermediate results for a specific target-source pair."""
        return self.intermediate_outputs_path() / f"intermediate_{target}_{source}.json"

    def save_intermediate_results(
        self, target: TokenType, source: TInfoFlowSource, window_outputs: dict, current_window: int
    ) -> None:
        """Save intermediate results to a temporary file."""
        path = self.get_intermediate_output_path(target, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save current progress and metadata
        data = {"window_outputs": window_outputs, "current_window": current_window, "timestamp": time.time()}
        json.dump(data, path.open("w"))

    def load_intermediate_results(
        self, target: TokenType, source: TInfoFlowSource
    ) -> tuple[Optional[dict[int, dict[str, list[float]]]], int]:
        """Load intermediate results if they exist."""
        path = self.get_intermediate_output_path(target, source)
        if path.exists():
            try:
                data = json.load(path.open("r"))
                return data["window_outputs"], data["current_window"]
            except Exception as e:
                print(f"Error loading intermediate results: {e}")
        return None, 0

    def cleanup_intermediate_results(self, target: TokenType, source: TInfoFlowSource) -> None:
        """Clean up intermediate results after successful completion."""
        path = self.get_intermediate_output_path(target, source)
        if path.exists():
            path.unlink()

    def output_block_target_path(self, target: TokenType, is_intermediate: bool) -> Path:
        output_path = self.intermediate_outputs_path() if is_intermediate else self.outputs_path
        return output_path / f"target={target}"

    def output_block_target_source_path(
        self, target: TokenType, source: TInfoFlowSource, is_intermediate: bool = False
    ) -> Path:
        if isinstance(source, tuple):
            feature_category_str = f"source={source[0]}_feature_category={source[1]}"
        else:
            feature_category_str = f"source={source}"
        return self.output_block_target_path(target, is_intermediate) / f"{feature_category_str}.csv"

    def get_block_target_outputs(
        self, target: TokenType, enforce_no_missing_outputs: bool
    ) -> dict[TInfoFlowSource, dict[str, dict[str, list[float]]]]:
        return {
            source: json.load(self.output_block_target_source_path(target, source).open("r"))
            for source in self.knockout_map[target]
            if not skip_task(self.model_arch, source)
            and (not enforce_no_missing_outputs or self.output_block_target_source_path(target, source).exists())
        }

    def get_outputs(
        self,
        enforce_no_missing_outputs: bool = True,
    ) -> dict[TokenType, dict[TInfoFlowSource, dict[str, dict[str, list[float]]]]]:
        return {
            target: self.get_block_target_outputs(target, enforce_no_missing_outputs) for target in self.knockout_map
        }

    def get_plot_output_path(self, target: TokenType, plot_name: str) -> Path:
        return self.plots_path / f"target={target}{plot_name}.png"

    def plot_block_target(
        self,
        target: TokenType,
        save: bool = False,
        confidence_level: float = 0.95,
        enforce_no_missing_outputs: bool = True,
    ):
        """Plot information flow from a target block to its source blocks.

        Args:
            target: The target TokenType to analyze flows from
            save: Whether to save the figure
        """
        data = self.get_block_target_outputs(target, enforce_no_missing_outputs)
        figs = {}
        for with_fixed_limits in [True, False]:
            sub_title = "_fixed_limits" if with_fixed_limits else ""
            figs[sub_title] = create_confidence_plot(
                targets_window_outputs=data,
                confidence_level=confidence_level,
                title=(
                    f"{self.model_arch} - {self.model_size} - window_size={self.window_size}"
                    + f"\nKnocking out flow to {target}"
                ),
                plots_meta_data={
                    "acc": {
                        "title": "Accuracy",
                        "ylabel": "% accuracy",
                        "ylabel_loc": "center",
                        "axhline_value": 100.0,
                        "ylim": (60.0, 105.0) if with_fixed_limits else None,
                    },
                    "diff": {
                        "title": "Normalized change in prediction probability",
                        "ylabel": "% probability change",
                        "ylabel_loc": "top",
                        "axhline_value": 0.0,
                        "ylim": (-50.0, 50.0) if with_fixed_limits else None,
                    },
                },
            )
            if save:
                p = self.get_plot_output_path(target, sub_title)
                p.parent.mkdir(parents=True, exist_ok=True)
                figs[sub_title].savefig(p)
                plt.close(figs[sub_title])
        return figs

    def plot(self, enforce_no_missing_outputs: bool = True) -> None:
        plot(self, enforce_no_missing_outputs)

    def compute(self) -> None:
        run(self)


def plot(args: InfoFlowConfig, enforce_no_missing_outputs: bool = True):
    knockout_map_outputs = args.get_outputs()
    for target in knockout_map_outputs:
        print(f"Plotting {target}")
        args.plot_block_target(target, save=True)


def forward_eval(
    prompt: Prompt,
    window: list[int],
    knockout_source: TInfoFlowSource,
    knockout_target: TokenType,
    model_interface: ModelInterface,
    tokenizer: TTokenizer,
    device,
):
    source, feature_category = (
        (
            knockout_source,
            FeatureCategory.ALL,
        )
        if isinstance(knockout_source, TokenType)
        else knockout_source
    )
    num_to_masks, first_token = get_num_to_masks(prompt, tokenizer, window, source, knockout_target, device)

    next_token_probs = model_interface.generate_logits(
        input_ids=prompt.input_ids(tokenizer, device),
        attention=True,
        num_to_masks=num_to_masks,
        feature_category=feature_category,
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


def run(args: InfoFlowConfig):
    print(args)

    remaining_knockout_map: list[tuple[TokenType, TInfoFlowSource]] = [
        (target, source)
        for target in args.knockout_map
        for source in args.knockout_map[target]
        if (
            (args.overwrite_existing_outputs or not args.output_block_target_source_path(target, source).exists())
            and not skip_task(args.model_arch, source)
        )
    ]
    if not remaining_knockout_map:
        print("All outputs already exist")
        return

    args.create_experiment_run_path()
    data = args.get_prompt_data()

    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = model_interface.n_layers()
    banned_prompt_indices: set[int] = set()

    def evaluate(
        prompt_indices: list[int],
        windows: list[list[int]],
        knockout_source: TInfoFlowSource,
        knockout_target: TokenType,
        print_period=100,
    ):
        # Try to load intermediate results
        windows_true_probs, start_window_idx = args.load_intermediate_results(knockout_target, knockout_source)
        if windows_true_probs is None:
            windows_true_probs = defaultdict(lambda: defaultdict(list))
            start_window_idx = 0
        else:
            print(f"Resuming from window {start_window_idx}")
            # Convert the loaded dict back to defaultdict
            windows_true_probs = defaultdict(lambda: defaultdict(list), windows_true_probs)

        last_save_time = time.time()

        for i, window in enumerate(
            tqdm(windows[start_window_idx:], desc="Windows", initial=float(start_window_idx)), start=start_window_idx
        ):
            windows_true_probs[i] = defaultdict(list)
            model_interface.setup(layers=window)
            for _, prompt_idx in enumerate(tqdm(prompt_indices, desc="Prompts", mininterval=print_period)):
                if prompt_idx in banned_prompt_indices:
                    continue
                try:
                    hit, diff, first, diff_unnorm, true_prob = forward_eval(
                        get_prompt_row_index(data, prompt_idx),
                        window,
                        knockout_source,
                        knockout_target,
                        model_interface,
                        tokenizer,
                        device,
                    )
                except Exception as e:
                    if "Test failure" in str(e):
                        # Test failure is expected, so we raise the error
                        raise e
                    print(f" Error evaluating {prompt_idx = } with {knockout_source = }, {knockout_target = }: {e}")
                    banned_prompt_indices.add(prompt_idx)
                    continue
                windows_true_probs[i][COLUMNS.IF_HIT].append(bool(hit))
                windows_true_probs[i][COLUMNS.IF_TRUE_PROBS].append(float(true_prob))
                windows_true_probs[i][COLUMNS.IF_DIFFS].append(float(diff))
                # Store original index for traceability
                windows_true_probs[i][COLUMNS.ORIGINAL_IDX].append(int(prompt_idx))

            # Check if it's time to save intermediate results
            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL:
                args.save_intermediate_results(knockout_target, knockout_source, dict(windows_true_probs), i)
                last_save_time = current_time
                print(f"\nSaved intermediate results at window {i}")

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
            knockout_source=source,
            knockout_target=target,
        )
        if args.DEBUG_LAST_WINDOWS:
            window_outputs = {
                k + (n_layers - args.window_size + 1 - args.DEBUG_LAST_WINDOWS): v for k, v in window_outputs.items()
            }
        args.output_block_target_source_path(target, source).parent.mkdir(parents=True, exist_ok=True)
        json.dump(window_outputs, args.output_block_target_source_path(target, source).open("w"))
        # Clean up intermediate results after successful completion
        args.cleanup_intermediate_results(target, source)
