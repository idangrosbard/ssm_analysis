import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.analysis.plots.info_flow_confidence import create_confidence_plot
from src.core.consts import is_mamba_arch
from src.core.names import COLS, EXPERIMENT_NAMES, INFO_FLOW_HP_COLS
from src.core.types import (
    MODEL_ARCH,
    FeatureCategory,
    TInfoFlowOutput,
    TInfoFlowOutputJSONOutput,
    TInfoFlowWindowValue,
    TLayerIndex,
    TokenType,
    TPromptOriginalIndex,
    TTokenizer,
    TWindow,
    TWindowSize,
    TWindowStartIndex,
)
from src.data_ingestion.helpers.logits_utils import Prompt, get_num_to_masks, get_prompt_row_index
from src.experiments.infrastructure.base_config import (
    BASE_OUTPUT_KEYS,
    BaseRunner,
)
from src.experiments.infrastructure.model_interface import ModelInterface
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams
from src.utils.infra.output_path import OutputKey
from src.utils.types_utils import first_dict_value

# Time in seconds between intermediate saves
SAVE_INTERVAL = 600  # 10 minutes


def skip_task(model_arch: MODEL_ARCH, feature_category: FeatureCategory) -> bool:
    return not (is_mamba_arch(model_arch) or feature_category == FeatureCategory.ALL)


@dataclass
class InfoFlowParams:
    window_size: TWindowSize
    source: TokenType
    feature_category: FeatureCategory
    target: TokenType


class InfoFlowDependencies(TypedDict):
    evaluate_model: EvaluateModelConfig


@dataclass
class InfoFlowConfig(BaseRunner[InfoFlowParams, TInfoFlowOutput]):
    """Configuration for information flow analysis."""

    runner_params: InfoFlowParams

    @property
    def experiment_name(self):
        return EXPERIMENT_NAMES.INFO_FLOW

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [
            BASE_OUTPUT_KEYS.WINDOW_SIZE,
            OutputKey[TokenType](INFO_FLOW_HP_COLS.target),
            OutputKey[TokenType](INFO_FLOW_HP_COLS.source),
            OutputKey[FeatureCategory](INFO_FLOW_HP_COLS.feature_category),
        ]

    def get_intermediate_output_path(self) -> Path:
        """Get the path for intermediate results for a specific target-source pair."""
        return self.variation_paths.intermediate_outputs / "intermediate.json"

    def save_intermediate_results(
        self,
        window_outputs: TInfoFlowOutput,
        current_window: TLayerIndex,
    ) -> None:
        """Save intermediate results to a temporary file."""
        path = self.get_intermediate_output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save current progress and metadata
        data = {
            "window_outputs": window_outputs,
            "current_window": current_window,
            "timestamp": time.time(),
        }
        json.dump(data, path.open("w"))

    def load_intermediate_results(self) -> tuple[TInfoFlowOutput, TWindowStartIndex]:
        """Load intermediate results if they exist."""
        path = self.get_intermediate_output_path()
        if path.exists():
            try:
                data = json.load(path.open("r"))
                print(f"Resuming from window {data['current_window']}")
                return self.convert_json_output_to_output(
                    cast(TInfoFlowOutputJSONOutput, data["window_outputs"])
                ), data["current_window"]
            except Exception as e:
                print(f"Error loading intermediate results: {e}")
        return cast(TInfoFlowOutput, defaultdict(lambda: defaultdict(list))), TWindowStartIndex(0)

    def cleanup_intermediate_results(self) -> None:
        """Clean up intermediate results after successful completion."""
        path = self.get_intermediate_output_path()
        if path.exists():
            path.unlink()

    def output_block_target_path(self, is_intermediate: bool) -> Path:
        output_path = (
            self.variation_paths.intermediate_outputs if is_intermediate else self.variation_paths.outputs_path
        )
        return output_path

    def output_block_target_source_path(self, is_intermediate: bool = False) -> Path:
        return self.output_block_target_path(is_intermediate) / "info_flow.csv"

    @staticmethod
    def convert_json_output_to_output(
        json_output: TInfoFlowOutputJSONOutput,
    ) -> TInfoFlowOutput:
        return {int(k): v for k, v in json_output.items()}

    @staticmethod
    def load_output(path: Path) -> TInfoFlowOutput:
        return InfoFlowConfig.convert_json_output_to_output(json.load(path.open("r")))

    def get_outputs(
        self,
        enforce_no_missing_outputs: bool = True,
    ):
        return self.load_output(self.output_block_target_source_path())

    def get_plot_output_path(self, plot_name: str) -> Path:
        return self.variation_paths.plots_path / f"{plot_name}.png"

    def plot_block_target(
        self,
        save: bool = False,
        confidence_level: float = 0.95,
        enforce_no_missing_outputs: bool = True,
    ):
        """Plot information flow from a target block to its source blocks.

        Args:
            target: The target TokenType to analyze flows from
            save: Whether to save the figure
        """
        data = self.get_outputs()
        figs = {}
        for with_fixed_limits in [True, False]:
            sub_title = "_fixed_limits" if with_fixed_limits else ""
            figs[sub_title] = create_confidence_plot(
                targets_window_outputs={str(self.runner_params.source): data},
                confidence_level=confidence_level,
                title=(
                    " - ".join(
                        [
                            self.common_params.model_arch,
                            self.common_params.model_size,
                            f"window_size={self.runner_params.window_size}",
                        ]
                    )
                    + f"\nKnocking out flow to {self.runner_params.target}"
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
                p = self.get_plot_output_path(sub_title)
                p.parent.mkdir(parents=True, exist_ok=True)
                figs[sub_title].savefig(p)
                plt.close(figs[sub_title])
        return figs

    def plot(self) -> None:
        plot(self)

    def compute(self) -> None:
        run(self)

    def remining_prompts_for_output_block_target_source(self) -> list[TPromptOriginalIndex]:
        remaining_prompts = self.prompt_ids
        output_path = self.output_block_target_source_path()
        if output_path.exists():
            output = self.load_output(output_path)
            first_windows_output = first_dict_value(output)
            existing_prompts = set(first_windows_output[COLS.ORIGINAL_IDX])
            remaining_prompts = [idx for idx in remaining_prompts if idx not in existing_prompts]

        return remaining_prompts

    def is_computed(self) -> bool:
        return len(self.remining_prompts_for_output_block_target_source()) == 0

    def get_runner_dependencies(self) -> InfoFlowDependencies:  # type: ignore
        return InfoFlowDependencies(
            evaluate_model=EvaluateModelConfig.init_from_config(
                self,
                runner_params=EvaluateModelParams(),
            ),
        )


def plot(args: InfoFlowConfig):
    knockout_map_outputs = args.get_outputs()
    for target in knockout_map_outputs:
        print(f"Plotting {target}")
        args.plot_block_target(save=True)


def forward_eval(
    prompt: Prompt,
    window: TWindow,
    knockout_source: TokenType,
    feature_category: FeatureCategory,
    knockout_target: TokenType,
    model_interface: ModelInterface,
    tokenizer: TTokenizer,
    device,
):
    num_to_masks, first_token = get_num_to_masks(prompt, tokenizer, window, knockout_source, knockout_target, device)

    next_token_probs = model_interface.generate_logits(
        input_ids=prompt.input_ids(tokenizer, device),
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

    remaining_knockout_map: list[tuple[TokenType, TokenType, FeatureCategory, list[TPromptOriginalIndex]]] = [
        (target, source, args.runner_params.feature_category, remaining_prompts)
        for target in [args.runner_params.target]
        for source in [args.runner_params.source]
        if (
            (
                len(remaining_prompts := args.remining_prompts_for_output_block_target_source()) > 0
                or args.run_params.overwrite_existing_outputs
            )
            and not skip_task(args.common_params.model_arch, args.runner_params.feature_category)
        )
    ]
    if not remaining_knockout_map:
        print("All outputs already exist")
        return

    args.create_experiment_run_path()
    data = args.get_runner_dependencies()["evaluate_model"].get_prompt_data()

    model_interface = args.common_params.get_model_interface()
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = model_interface.n_layers()
    banned_prompt_indices: set[TPromptOriginalIndex] = set()

    def evaluate(
        prompt_indices: list[TPromptOriginalIndex],
        windows: list[TWindow],
        knockout_source: TokenType,
        feature_category: FeatureCategory,
        knockout_target: TokenType,
        print_period=100,
    ):
        # Try to load intermediate results
        windows_true_probs, start_window_idx = args.load_intermediate_results()
        last_save_time = time.time()

        for i, window in enumerate(
            tqdm(windows[start_window_idx:], desc="Windows", initial=start_window_idx),
            start=start_window_idx,
        ):
            windows_true_probs[i] = cast(TInfoFlowWindowValue, defaultdict(list))
            model_interface.setup(layers=window)
            for prompt_idx in tqdm(prompt_indices, desc="Prompts", mininterval=print_period):
                if prompt_idx in banned_prompt_indices:
                    continue
                try:
                    hit, diff, first, diff_unnorm, true_prob = forward_eval(
                        get_prompt_row_index(data, prompt_idx),
                        window,
                        knockout_source,
                        feature_category,
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
                windows_true_probs[i][COLS.INFO_FLOW.HIT.value].append(bool(hit))
                windows_true_probs[i][COLS.INFO_FLOW.TRUE_PROBS.value].append(float(true_prob))
                windows_true_probs[i][COLS.INFO_FLOW.DIFFS.value].append(float(diff))
                # Store original index for traceability
                windows_true_probs[i][COLS.ORIGINAL_IDX].append(TPromptOriginalIndex(int(prompt_idx)))

            # Check if it's time to save intermediate results
            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL:
                args.save_intermediate_results(
                    window_outputs=dict(windows_true_probs),
                    current_window=i,
                )
                last_save_time = current_time
                print(f"\nSaved intermediate results at window {i}")

        return windows_true_probs

    prompt_indices: list[TPromptOriginalIndex] = list(data.index)
    windows: list[TWindow] = [
        TWindow(list(range(i, i + args.runner_params.window_size)))
        for i in range(0, n_layers - args.runner_params.window_size + 1)
    ]

    for target, source, feature_category, remaining_prompts in remaining_knockout_map:
        print(f"Knocking out flow to {target} from {source}")

        window_outputs = evaluate(
            prompt_indices,
            windows,
            knockout_source=source,
            feature_category=feature_category,
            knockout_target=target,
        )
        args.output_block_target_source_path().parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            window_outputs,
            args.output_block_target_source_path().open("w"),
        )
        # Clean up intermediate results after successful completion
        args.cleanup_intermediate_results()
