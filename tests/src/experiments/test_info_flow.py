"""Tests for the full pipeline experiment."""

import time
from pathlib import Path

import pytest

from src.core.types import (
    MODEL_ARCH,
    FeatureCategory,
    TokenType,
    TVersionName,
)
from src.experiments.runners.info_flow import forward_eval
from src.utils.types_utils import first_dict_value
from tests.src.experiments.test_full_pipeline import (
    INFO_FLOW_FORWARD_EVAL_PATH,
    INFO_FLOW_PRINT_INTERVAL_PATH,
    PATHS_PROJECT_DIR_PATH,
    clean_and_generate_base_test_data,
    get_test_full_pipeline_config,
)

INFO_FLOW_SAVE_INTERVAL_PATH = "src.experiments.runners.info_flow.SAVE_INTERVAL"


def test_info_flow_recovery(tmp_path: Path):
    """Test that info flow can save and recover from intermediate results correctly."""
    # Setup test environment
    clean_and_generate_base_test_data(tmp_path)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(PATHS_PROJECT_DIR_PATH, tmp_path)
        mp.setattr(INFO_FLOW_PRINT_INTERVAL_PATH, 1)
        # Create a test config with minimal settings
        full_pipeline_config = get_test_full_pipeline_config(
            version_name="test_recovery",
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
        out_path = info_flow_config.output_file
        assert out_path.path.exists(), "Intermediate file should exist"

        before_len = len(out_path.get_existing_prompt_idx())

        # Run again - should recover from intermediate results
        mp.setattr(INFO_FLOW_FORWARD_EVAL_PATH, original_forward_eval)
        info_flow_config.compute()
        after_len = len(out_path.get_existing_prompt_idx())
        assert after_len > before_len, "Intermediate file should not be cleaned up"
        created_data = info_flow_config.get_outputs()

    _test_base_path = Path(__file__).parent / "baselines" / "full_pipeline"
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(PATHS_PROJECT_DIR_PATH, _test_base_path)
        info_flow_config.version = TVersionName("test_baseline")
        baseline_data = info_flow_config.get_outputs()
        assert created_data == baseline_data, "Data should be the same"
