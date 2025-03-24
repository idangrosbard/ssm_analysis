"""Tests for the full pipeline experiment."""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyrallis
import pytest
from datasets import DatasetDict

from src.analysis.experiment_results.helpers import serialize_result_bank
from src.analysis.experiment_results.results_bank import get_experiment_results_bank
from src.analysis.prompt_filterations import Correctness, ModelCorrectPromptFilteration, SelectivePromptFilteration
from src.core.consts import PathsConfig
from src.core.names import COLS
from src.core.types import (
    ALL_SPLITS_LITERAL,
    MODEL_ARCH,
    SPLIT,
    FeatureCategory,
    TModelSize,
    TokenType,
    TPromptOriginalIndex,
    TVariationName,
    TWindowSize,
)
from src.data_ingestion.datasets.download_dataset import DATASETS, load_splitted_counter_fact
from src.experiments.infrastructure.base_config import CommonParams
from src.experiments.runners.full_pipeline import FullPipelineConfig, FullPipelineParam
from src.experiments.runners.info_flow import forward_eval
from src.utils.types_utils import first_dict_value

HEATMAP_SIZE = 5
BASELINES_DIR = Path(__file__).parent / "baselines"
TEST_BASE_PATH = BASELINES_DIR / "full_pipeline"
ORIGINAL_IDS = cast(
    dict[SPLIT, list[TPromptOriginalIndex]],
    {
        SPLIT.TRAIN1: [
            53,
            59,
            74,
            90,
            93,
        ],
        SPLIT.TRAIN2: [
            10594,
            6410,
            140,
            148,
            159,
            182,
        ],
    },
)

# HARDCODED CODE PATHS FOR TESTS
PATHS_PROJECT_DIR_PATH = "src.core.consts.PATHS.PROJECT_DIR"
INFO_FLOW_FORWARD_EVAL_PATH = "src.experiments.runners.info_flow.forward_eval"
INFO_FLOW_SAVE_INTERVAL_PATH = "src.experiments.runners.info_flow.SAVE_INTERVAL"
GET_COMMIT_HASH_PATH = "src.experiments.infrastructure.base_config.get_git_commit_hash"
CREATE_RUN_ID_PATH = "src.experiments.infrastructure.base_config.create_run_id"


def get_config(variation_name: str, model_arch: MODEL_ARCH, model_size: str, with_plotting: bool) -> FullPipelineConfig:
    return FullPipelineConfig(
        variation=TVariationName(variation_name),
        runner_params=FullPipelineParam(
            knockout_map={
                TokenType.last: [
                    (TokenType.last, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.SLOW_DECAY),
                    (TokenType.subject, FeatureCategory.FAST_DECAY),
                    (TokenType.first, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.ALL),
                    (TokenType.relation, FeatureCategory.ALL),
                ],
                TokenType.subject: [
                    (TokenType.context, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.ALL),
                ],
                TokenType.relation: [
                    (TokenType.context, FeatureCategory.ALL),
                    (TokenType.subject, FeatureCategory.ALL),
                    (TokenType.relation, FeatureCategory.ALL),
                ],
            },
            info_flow_window_size=TWindowSize(15),
            heatmap_window_size=TWindowSize(15),
            heatmap_prompts=SelectivePromptFilteration(DATASETS.COUNTER_FACT, tuple(ORIGINAL_IDS[SPLIT.TRAIN1])),
            with_plotting=with_plotting,
            enforce_no_missing_outputs=True,
            with_generation=True,
        ),
        common_params=CommonParams(
            model_arch=model_arch,
            model_size=TModelSize(model_size),
        ),
        prompt_filteration=ModelCorrectPromptFilteration(
            DATASETS.COUNTER_FACT,
            model_arch=model_arch,
            model_size=TModelSize(model_size),
            correctness=Correctness.correct,
            variation=TVariationName(variation_name),
        ),
    )


def clean_and_generate_base_test_data(test_base_path: Path):
    test_paths = PathsConfig(PROJECT_DIR=test_base_path)

    # clean test base path
    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    test_base_path.mkdir(parents=True, exist_ok=True)

    # get sample of real data
    dataset = {
        split: load_splitted_counter_fact(
            ALL_SPLITS_LITERAL,
            align_to_known=False,
        ).filter(lambda x: x[COLS.ORIGINAL_IDX] in original_ids)
        for split, original_ids in ORIGINAL_IDS.items()
    }

    # save dataset to disk
    DatasetDict(dataset).save_to_disk(test_paths.dataset_dir(DATASETS.COUNTER_FACT) / "splitted")


def run_test_experiment(test_base_path: Path, normalizing_outputs: bool, with_plotting: bool):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(PATHS_PROJECT_DIR_PATH, test_base_path)
        if normalizing_outputs:
            mp.setattr(GET_COMMIT_HASH_PATH, lambda *args, **kwargs: "test_commit_hash")
            mp.setattr(CREATE_RUN_ID_PATH, lambda *args, **kwargs: "test_run_id")

        for model_arch, model_size in [
            (MODEL_ARCH.MAMBA1, "130M"),
            (MODEL_ARCH.MAMBA2, "130M"),
            (MODEL_ARCH.GPT2, "355M"),
        ]:
            config = get_config(
                variation_name="test_baseline",
                model_arch=model_arch,
                model_size=model_size,
                with_plotting=with_plotting,
            )
            config.compute_with_dependencies()

        if normalizing_outputs:
            (test_base_path / "serialized_results.json").write_text(
                serialize_result_bank(get_experiment_results_bank())
            )
        print(f"Baseline updated at: {test_base_path}")


def test_info_flow_intermediate_recovery(tmp_path: Path):
    """Test that info flow can save and recover from intermediate results correctly."""
    # Setup test environment
    clean_and_generate_base_test_data(tmp_path)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(PATHS_PROJECT_DIR_PATH, tmp_path)

        # Create a test config with minimal settings
        full_pipeline_config = get_config(
            variation_name="test_recovery",
            model_arch=MODEL_ARCH.MAMBA1,
            model_size="130M",
            with_plotting=True,
        )

        full_pipeline_config.runner_params.knockout_map = {
            TokenType.last: [
                (TokenType.last, FeatureCategory.ALL),
            ],
        }

        info_flow_config = first_dict_value(
            first_dict_value(full_pipeline_config.get_runner_dependencies()["info_flow"])
        )
        info_flow_config.get_runner_dependencies()["evaluate_model"].compute()

        # Mock the save interval to be very short for testing
        mp.setattr(INFO_FLOW_SAVE_INTERVAL_PATH, 1)  # 1 second for testing

        original_forward_eval = forward_eval

        global fail_after
        fail_after = len(info_flow_config.prompt_ids) * 3

        # Run the experiment and interrupt it
        def mock_forward_eval(*args, **kwargs):
            global fail_after
            # Simulate computation by sleeping
            time.sleep(0.1)
            fail_after -= 1
            if fail_after <= 0:
                raise Exception("Test failure")
            return original_forward_eval(*args, **kwargs)

        # Mock get prompt_data

        mp.setattr(INFO_FLOW_FORWARD_EVAL_PATH, mock_forward_eval)

        try:
            # First run - should create intermediate results
            info_flow_config.compute()
        except Exception:
            pass

        # Verify intermediate files were created and contain valid data
        intermediate_path = info_flow_config.get_intermediate_output_path()
        assert intermediate_path.exists(), "Intermediate file should exist"

        # Load and verify intermediate results
        data, window_idx = info_flow_config.load_intermediate_results()
        assert data is not None, "Should have valid intermediate data"
        assert window_idx >= 0, "Should have valid window index"

        # Verify the data structure
        assert isinstance(data, dict), "Data should be a dictionary"
        assert all(isinstance(k, (str, int)) for k in data.keys()), "Keys should be strings or ints"

        # Run again - should recover from intermediate results
        mp.setattr(INFO_FLOW_FORWARD_EVAL_PATH, original_forward_eval)
        info_flow_config.compute()

        # Verify final output exists and intermediate files are cleaned up
        final_output_path = info_flow_config.output_block_target_source_path()
        assert final_output_path.exists(), "Final output file should exist"
        assert not intermediate_path.exists(), "Intermediate file should be cleaned up"
        created_data = info_flow_config.get_outputs()

    _test_base_path = Path(__file__).parent / "baselines" / "full_pipeline"
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(PATHS_PROJECT_DIR_PATH, _test_base_path)
        info_flow_config.variation = TVariationName("test_baseline")
        baseline_data = info_flow_config.get_outputs()
        assert created_data == baseline_data, "Data should be the same"


def create_test_experiment(test_base_path: Path, resume: bool, normalizing_outputs: bool, with_plotting: bool):
    if not resume:
        clean_and_generate_base_test_data(test_base_path)
    # TODO: test why there was a change at commit of 7f0fdded984bca60686dd8586c365534aeffa009
    run_test_experiment(test_base_path, normalizing_outputs, with_plotting)


@dataclass
class CreateBaselineParams:
    resume: bool = False
    normalizing_outputs: bool = True
    with_plotting: bool = True


@pyrallis.wrap()
def main(params: CreateBaselineParams):
    create_test_experiment(
        TEST_BASE_PATH,
        resume=params.resume,
        normalizing_outputs=params.normalizing_outputs,
        with_plotting=params.with_plotting,
    )


if __name__ == "__main__":
    main()  # type: ignore
