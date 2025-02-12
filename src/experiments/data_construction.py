"""
DataConstructionExperiment:
Experiment for constructing datasets by running models on prompts and collecting their outputs
In this experiment implementation:
The sub-task is a batch of prompts
The inner loop is running the model on each prompt and collecting outputs
The sub task result is a DataFrame with model outputs and probabilities
The combined result is saved as a parquet file
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.consts import COLUMNS, EVAL_MODEL_2_DATA_CONST_COL_CONV
from src.experiment_infra.base_config import BaseConfig
from src.experiment_infra.output_path import OutputKey


@dataclass
class DataConstructionConfig(BaseConfig[pd.DataFrame]):
    """Configuration for data construction."""

    experiment_base_name: str = "data_construction"
    attention: bool = False

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [OutputKey[bool]("attention", key_display_name="attn=")]

    @property
    def output_result_path(self) -> Path:
        return self.outputs_path / f"entire_results_{self.attention}.csv"

    def get_outputs(self):
        return pd.read_csv(self.output_result_path, index_col=False)


def run(args: DataConstructionConfig):
    print(args)
    if args.output_result_path.exists() and not args.overwrite_existing_outputs:
        print(f"Output file {args.output_result_path} already exists")
        return

    args.create_output_path()

    original_data = args.get_raw_data()

    from src.experiments.evaluate_model import EvaluateModelConfig

    eval_config = args.init_sub_config_from_full_pipeline_config(
        EvaluateModelConfig,
        drop_subject=False,
        drop_subj_last_token=False,
        with_3_dots=False,
        new_max_tokens=5,
        top_k_tokens=5,
    )

    eval_results = eval_config.get_outputs()

    filtered_data = pd.merge(
        original_data,
        eval_results[[COLUMNS.ORIGINAL_IDX] + list(EVAL_MODEL_2_DATA_CONST_COL_CONV.keys())],
        on=COLUMNS.ORIGINAL_IDX,
        how="inner",
    )

    renamed_data = filtered_data.rename(columns=EVAL_MODEL_2_DATA_CONST_COL_CONV)

    renamed_data.to_csv(args.output_result_path, index=False)
