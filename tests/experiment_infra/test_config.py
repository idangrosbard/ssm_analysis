from dataclasses import dataclass
from pathlib import Path

from src.experiment_infra.config import BaseConfig
from src.utils.tests_utils import run_mypy_check_on_function

EXPERIMENT_NAME_BEFORE = "simple"
EXPERIMENT_NAME_AFTER = "new_experiment_name"
NEW_FIELD_BEFORE = "new_field"
NEW_FIELD_AFTER = "new_field_2"


def _test_config(config: BaseConfig):
    assert config.experiment_name == EXPERIMENT_NAME_BEFORE
    config.experiment_name = EXPERIMENT_NAME_AFTER
    assert config.experiment_name == EXPERIMENT_NAME_AFTER

    assert config.new_field == NEW_FIELD_BEFORE  # type: ignore
    config.new_field = NEW_FIELD_AFTER  # type: ignore
    assert config.new_field == NEW_FIELD_AFTER  # type: ignore


def test_simple_config_no_dataclass():
    class SimpleConfig(BaseConfig):
        new_field: str = NEW_FIELD_BEFORE

        def __init__(self, experiment_name: str = EXPERIMENT_NAME_BEFORE):
            super().__init__(experiment_name)

        @property
        def output_path(self) -> Path:
            return Path("simple")

    config = SimpleConfig()

    _test_config(config)


def test_no_init_config_with_dataclass():
    @dataclass
    class NoInitConfig(BaseConfig):
        experiment_name: str = EXPERIMENT_NAME_BEFORE
        new_field: str = NEW_FIELD_BEFORE

        @property
        def output_path(self) -> Path:
            return Path("no_init")

    config = NoInitConfig()

    _test_config(config)


def test_heatmap_config_default_init():
    def test_heatmap_config_default_init():
        from src.experiments.heatmap import HeatmapConfig

        config = HeatmapConfig()

    mypy_errors = run_mypy_check_on_function(test_heatmap_config_default_init)

    assert len(mypy_errors) == 0
