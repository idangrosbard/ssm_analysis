"""Tests for the full pipeline experiment."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from datasets import DatasetDict
from src.consts import COLUMNS, PathsConfig
from src.datasets.download_dataset import load_splitted_counter_fact
from src.experiments.full_pipeline import FullPipelineConfig, main_local
from src.types import FILTERATIONS, MODEL_ARCH

DATA_SIZE = 10
HEATMAP_SIZE = 5


def get_config(variation_name: str, model_arch: MODEL_ARCH, model_size: str) -> FullPipelineConfig:
    return FullPipelineConfig(
        variation=variation_name,
        model_arch=model_arch,
        model_size=model_size,
        _batch_size=1,
        window_size=15,
        prompt_indices=list(range(HEATMAP_SIZE)),
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
    ).select(range(DATA_SIZE))

    # save dataset to disk
    DatasetDict({"train1": dataset}).save_to_disk(test_paths.COUNTER_FACT_DIR / "splitted")

    # save filteration to disk
    (
        pd.DataFrame({COLUMNS.ORIGINAL_IDX: dataset[COLUMNS.ORIGINAL_IDX]}).to_csv(
            test_paths.COUNTER_FACT_FILTERATIONS_DIR / f"{FILTERATIONS.all_correct}.csv", index=False
        )
    )


if __name__ == "__main__":
    # For updating baseline

    test_base_path = Path(__file__).parent / "baselines" / "full_pipeline"
    create_test_data(test_base_path)

    # Create test data and run pipeline with mocked paths
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
