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
from typing import Optional

import src.experiments.evaluate_model as evaluate_model
import src.experiments.heatmap as heatmap
import src.experiments.info_flow as info_flow
from src.experiment_infra.base_config import (
    BASE_OUTPUT_KEYS,
    BaseConfig,
    create_mutable_field,
)
from src.experiments.evaluate_model import EvaluateModelConfig
from src.experiments.heatmap import HeatmapConfig
from src.experiments.info_flow import InfoFlowConfig
from src.types import TInfoFlowSource, TokenType


@dataclass
class FullPipelineConfig(BaseConfig):
    """Configuration for the full experiment pipeline."""

    experiment_base_name: str = "full_pipeline"

    with_plotting: bool = False
    enforce_no_missing_outputs: bool = True
    with_generation: bool = True

    # EvaluateModelConfig
    drop_subject: bool = EvaluateModelConfig.drop_subject
    drop_subj_last_token: bool = EvaluateModelConfig.drop_subj_last_token
    with_3_dots: bool = EvaluateModelConfig.with_3_dots
    new_max_tokens: int = EvaluateModelConfig.new_max_tokens
    top_k_tokens: int = EvaluateModelConfig.top_k_tokens

    # HeatmapConfig
    window_size: int = HeatmapConfig.window_size
    prompt_indices: list[int] = create_mutable_field(lambda: HeatmapConfig().prompt_indices)

    # InfoFlowConfig
    knockout_map: dict[TokenType, list[TInfoFlowSource]] = create_mutable_field(lambda: InfoFlowConfig().knockout_map)
    DEBUG_LAST_WINDOWS: Optional[int] = InfoFlowConfig.DEBUG_LAST_WINDOWS

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [
            BASE_OUTPUT_KEYS.WINDOW_SIZE,
        ]

    def evaluate_model_config(self) -> EvaluateModelConfig:
        return self.init_sub_config_from_full_pipeline_config(
            EvaluateModelConfig,
        )

    def heatmap_config(self) -> HeatmapConfig:
        return self.init_sub_config_from_full_pipeline_config(HeatmapConfig)

    def info_flow_config(self) -> InfoFlowConfig:
        return self.init_sub_config_from_full_pipeline_config(InfoFlowConfig)

    def get_outputs(self) -> dict:
        """Get outputs from all experiments."""
        return {}

    def run(self) -> None:
        main_local(self)


def main_local(args: FullPipelineConfig):
    """Run the full pipeline of experiments."""
    print("Starting Full Pipeline Experiment")
    print(f"{args.with_generation=} {args.with_plotting=} {args.enforce_no_missing_outputs=}")
    print(args)

    # Step 1: Model Evaluation
    if args.with_generation:
        print("\nRunning Model Evaluation Experiment...")
        evaluate_model.run(args.evaluate_model_config())

    # Step 2: Heatmap Analysis

    print("\nRunning Heatmap Analysis Experiment...")
    heatmap_config = args.heatmap_config()
    if args.with_generation:
        heatmap.run(heatmap_config)
    if args.with_plotting:
        print("\nPlotting all heatmaps...")
        try:
            heatmap.plot(heatmap_config)
        except Exception as e:
            print(f"Error plotting heatmaps: {e}")

    # Step 3: Information Flow Analysis
    info_flow_config = args.info_flow_config()
    if args.with_generation:
        print("\nRunning Information Flow Analysis Experiment...")
        info_flow.run(info_flow_config)
    if args.with_plotting:
        print("\nPlotting all info flow blocks...")
        try:
            info_flow.plot(info_flow_config, args.enforce_no_missing_outputs)
        except Exception as e:
            print(f"Error plotting info flow blocks: {e}")

    print("\nFull Pipeline Experiment Complete!")
