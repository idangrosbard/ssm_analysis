"""Tests for the full pipeline experiment."""

import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from datasets import DatasetDict
from src.consts import COLUMNS, PathsConfig
from src.datasets.download_dataset import load_splitted_counter_fact
from src.experiments.full_pipeline import FullPipelineConfig, main_local
from src.experiments.info_flow import forward_eval
from src.types import FILTERATIONS, MODEL_ARCH, TokenType

HEATMAP_SIZE = 5


def get_config(variation_name: str, model_arch: MODEL_ARCH, model_size: str) -> FullPipelineConfig:
    return FullPipelineConfig(
        variation=variation_name,
        model_arch=model_arch,
        model_size=model_size,
        _batch_size=1,
        window_size=15,
        prompt_indices_rows=list(range(HEATMAP_SIZE)),
        with_plotting=True,
    )


def create_test_data(test_base_path: Path):
    test_paths = PathsConfig(PROJECT_DIR=test_base_path)

    # clean test base path
    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    test_base_path.mkdir(parents=True, exist_ok=True)
    test_paths.COUNTER_FACT_FILTERATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # get sample of real data
    dataset = load_splitted_counter_fact(
        "all",
        align_to_known=False,
        filteration=FILTERATIONS.all_correct,
    ).filter(
        lambda x: x[COLUMNS.ORIGINAL_IDX]
        in [
            53,
            59,
            74,
            90,
            93,
            10594,
            6410,
            140,
            148,
            159,
            182,
        ]
    )

    # save dataset to disk
    DatasetDict({"train1": dataset}).save_to_disk(test_paths.COUNTER_FACT_DIR / "splitted")

    # save filteration to disk
    (
        pd.DataFrame({COLUMNS.ORIGINAL_IDX: dataset[COLUMNS.ORIGINAL_IDX]}).to_csv(
            test_paths.COUNTER_FACT_FILTERATIONS_DIR / f"{FILTERATIONS.all_correct}.csv", index=False
        )
    )


def create_test_experiment(test_base_path: Path):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("src.consts.PATHS.PROJECT_DIR", test_base_path)
        for model_arch, model_size in [
            (MODEL_ARCH.MAMBA1, "130M"),
            (MODEL_ARCH.MAMBA2, "130M"),
            (MODEL_ARCH.GPT2, "355M"),
        ]:
            config = get_config(variation_name="test_baseline", model_arch=model_arch, model_size=model_size)
            main_local(config)
            print(f"Baseline updated at: {test_base_path}")


def test_info_flow_intermediate_recovery(tmp_path: Path):
    """Test that info flow can save and recover from intermediate results correctly."""
    # Setup test environment
    create_test_data(tmp_path)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("src.consts.PATHS.PROJECT_DIR", tmp_path)

        # Create a test config with minimal settings
        full_pipeline_config = get_config(
            variation_name="test_recovery",
            model_arch=MODEL_ARCH.MAMBA1,
            model_size="130M",
        )

        full_pipeline_config.knockout_map = {
            TokenType.last: [
                TokenType.last,
            ],
        }

        full_pipeline_config.evaluate_model_config().compute()

        info_flow_config = full_pipeline_config.info_flow_config()

        # Mock the save interval to be very short for testing
        mp.setattr("src.experiments.info_flow.SAVE_INTERVAL", 1)  # 1 second for testing

        original_forward_eval = forward_eval

        global fail_after
        fail_after = len(info_flow_config.get_prompt_data()) * 3

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

        mp.setattr("src.experiments.info_flow.forward_eval", mock_forward_eval)

        try:
            # First run - should create intermediate results
            info_flow_config.compute()
        except Exception:
            pass

        # Verify intermediate files were created and contain valid data
        intermediate_path = info_flow_config.get_intermediate_output_path(TokenType.last, TokenType.last)
        assert intermediate_path.exists(), "Intermediate file should exist"

        # Load and verify intermediate results
        data, window_idx = info_flow_config.load_intermediate_results(TokenType.last, TokenType.last)
        assert data is not None, "Should have valid intermediate data"
        assert window_idx >= 0, "Should have valid window index"

        # Verify the data structure
        assert isinstance(data, dict), "Data should be a dictionary"
        assert all(isinstance(k, (str, int)) for k in data.keys()), "Keys should be strings or ints"

        # Run again - should recover from intermediate results
        mp.setattr("src.experiments.info_flow.forward_eval", original_forward_eval)
        info_flow_config.compute()

        # Verify final output exists and intermediate files are cleaned up
        final_output_path = info_flow_config.output_block_target_source_path(TokenType.last, TokenType.last)
        assert final_output_path.exists(), "Final output file should exist"
        assert not intermediate_path.exists(), "Intermediate file should be cleaned up"
        created_data = info_flow_config.get_outputs()[TokenType.last][TokenType.last]

    _test_base_path = Path(__file__).parent / "baselines" / "full_pipeline"
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("src.consts.PATHS.PROJECT_DIR", _test_base_path)
        info_flow_config.variation = "test_baseline"
        baseline_data = info_flow_config.get_outputs()[TokenType.last][TokenType.last]
        assert created_data == baseline_data, "Data should be the same"


if __name__ == "__main__":
    # For updating baseline

    _test_base_path = Path(__file__).parent / "baselines" / "full_pipeline"
    # create_test_data(_test_base_path)
    # TODO: test why there was a change at commit of 7f0fdded984bca60686dd8586c365534aeffa009
    create_test_experiment(_test_base_path)
