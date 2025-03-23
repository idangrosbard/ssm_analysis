"""
FullPipelineExperiment: Orchestrates all experiments in sequence

This experiment runs:
1. Data Construction - Creates the dataset
2. Model Evaluation - Evaluates model performance
3. Heatmap Analysis - Analyzes layer effects
4. Information Flow Analysis - Analyzes semantic information flow

The experiment ensures proper data flow between experiments and maintains
consistent configuration across all steps.
"""

from dataclasses import dataclass
from typing import TypedDict

from src.core.names import EXPERIMENT_NAMES
from src.core.types import FeatureCategory, TokenType, TWindowSize
from src.experiments.infrastructure.base_config import (
    BASE_OUTPUT_KEYS,
    BaseRunner,
    PromptFilteration,
)
from src.experiments.runners.heatmap import HEATMAP_PLOT_FUNCS, HeatmapConfig, HeatmapParams
from src.experiments.runners.info_flow import InfoFlowConfig, InfoFlowParams


class FullPipelineDependencies(TypedDict):
    heatmap: HeatmapConfig
    info_flow: dict[str, InfoFlowConfig]


@dataclass
class FullPipelineParam:
    knockout_map: dict[TokenType, list[tuple[TokenType, FeatureCategory]]]
    info_flow_window_size: TWindowSize

    heatmap_window_size: TWindowSize
    heatmap_prompts: PromptFilteration

    with_plotting: bool = False
    enforce_no_missing_outputs: bool = True
    with_generation: bool = True


@dataclass
class FullPipelineConfig(BaseRunner):
    """Configuration for the full experiment pipeline."""

    runner_params: FullPipelineParam

    @property
    def experiment_name(self):
        return EXPERIMENT_NAMES.FULL_PIPELINE

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [
            BASE_OUTPUT_KEYS.WINDOW_SIZE,
        ]

    def get_outputs(self) -> dict:
        """Get outputs from all experiments."""
        return {}

    def compute(self) -> None:
        main_local(self)

    def get_runner_dependencies(self) -> FullPipelineDependencies:  # type: ignore
        info_flow_deps: dict[str, InfoFlowConfig] = {}
        for target_token, source in self.runner_params.knockout_map.items():
            for source_token, feature_category in source:
                info_flow_deps[f"info_flow_{source_token}_{feature_category}->{target_token}"] = (
                    InfoFlowConfig.init_from_config(
                        config=self,
                        runner_params=InfoFlowParams(
                            window_size=self.runner_params.info_flow_window_size,
                            source=source_token,
                            feature_category=feature_category,
                            target=target_token,
                        ),
                    )
                )

        return FullPipelineDependencies(
            heatmap=HeatmapConfig.init_from_config(
                config=self,
                runner_params=HeatmapParams(
                    window_size=self.runner_params.heatmap_window_size,
                ),
                prompt_filteration=self.runner_params.heatmap_prompts,
            ),
            info_flow=info_flow_deps,
        )

    def is_computed(self) -> bool:
        return True


def main_local(args: FullPipelineConfig):
    """Run the full pipeline of experiments."""
    print("Starting Full Pipeline Experiment")
    print(
        " ".join(
            [
                f"{args.runner_params.with_generation=}",
                f"{args.runner_params.with_plotting=}",
                f"{args.runner_params.enforce_no_missing_outputs=}",
            ]
        )
    )
    print(args)

    if args.runner_params.with_plotting:
        print("\nPlotting all heatmaps...")
        try:
            args.get_runner_dependencies()["heatmap"].plot(HEATMAP_PLOT_FUNCS._simple_diff_fixed_0_3)
        except Exception as e:
            print(f"Error plotting heatmaps: {e}")

    if args.runner_params.with_plotting:
        print("\nPlotting all info flow blocks...")
        try:
            for info_flow_config in args.get_runner_dependencies()["info_flow"].values():
                info_flow_config.plot()
        except Exception as e:
            print(f"Error plotting info flow blocks: {e}")

    print("\nFull Pipeline Experiment Complete!")
