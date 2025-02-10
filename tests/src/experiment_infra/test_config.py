from dataclasses import dataclass
from pathlib import Path

from src.experiment_infra.base_config import BaseConfig

EXPERIMENT_NAME_BEFORE = "simple"
EXPERIMENT_NAME_AFTER = "new_experiment_name"
NEW_FIELD_BEFORE = "new_field"
NEW_FIELD_AFTER = "new_field_2"


def _test_config(config: BaseConfig):
    assert config.experiment_base_name == EXPERIMENT_NAME_BEFORE
    config.experiment_base_name = EXPERIMENT_NAME_AFTER
    assert config.experiment_base_name == EXPERIMENT_NAME_AFTER

    assert config.new_field == NEW_FIELD_BEFORE  # type: ignore
    config.new_field = NEW_FIELD_AFTER  # type: ignore
    assert config.new_field == NEW_FIELD_AFTER  # type: ignore


def test_simple_config_no_dataclass():
    class SimpleConfig(BaseConfig):
        new_field: str = NEW_FIELD_BEFORE

        def __init__(self, experiment_base_name: str = EXPERIMENT_NAME_BEFORE):
            super().__init__(experiment_base_name)

        @property
        def output_path(self) -> Path:
            return Path("simple")

        @property
        def experiment_output_keys(self):
            return super().experiment_output_keys

        def get_outputs(self):
            return {}

    config = SimpleConfig()

    _test_config(config)


def test_no_init_config_with_dataclass():
    @dataclass
    class NoInitConfig(BaseConfig):
        experiment_base_name: str = EXPERIMENT_NAME_BEFORE
        new_field: str = NEW_FIELD_BEFORE

        @property
        def output_path(self) -> Path:
            return Path("no_init")

        @property
        def experiment_output_keys(self):
            return super().experiment_output_keys

        def get_outputs(self):
            return {}

    config = NoInitConfig()

    _test_config(config)
