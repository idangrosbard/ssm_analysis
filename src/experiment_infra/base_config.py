import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Type, TypeVar, Union, cast, final

import pandas as pd
import pyrallis

from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS
from src.datasets.download_dataset import load_splitted_counter_fact
from src.types import DATASETS, FILTERATIONS, MODEL_ARCH, DatasetArgs, TModelID, TPromptData
from src.utils.experiment_helper import create_run_id

_T = TypeVar("_T")
_TBaseConfig = TypeVar("_TBaseConfig", bound="BaseConfig")


def create_mutable_field(default_factory: Callable[[], _T]) -> _T:
    # Pyralis need mutable fields to be defined with field but it's typing is not complete.
    # This is a fix to make it work.

    return cast(_T, pyrallis.field(default_factory=default_factory, is_mutable=True))


class OutputKey(Generic[_T]):
    def __init__(
        self,
        key_name: str,
        convert_to_str: Callable[[_T], str] = str,
        key_display_name: Optional[str] = None,
        skip_condition: Optional[Callable[[_T], bool]] = None,
    ):
        """

        Args:
            key_name: key name in the config
            convert_to_str: function to convert the value to a string. Defaults to str.
            key_display_name: display name of the key. Defaults to None.
            skip_condition: condition to skip the key. Defaults to no skipping.
        """
        self.key_name = key_name
        self.convert_to_str = convert_to_str
        self.key_display_name = f"{key_name}=" if key_display_name is None else key_display_name
        self.skip_condition = skip_condition

    def should_skip(self, config: "BaseConfig") -> bool:
        if self.skip_condition is None:
            return False
        return self.skip_condition(self.get_value(config))

    def get_value(self, config: "BaseConfig") -> _T:
        assert hasattr(config, self.key_name)
        return cast(_T, getattr(config, self.key_name))

    def display(self, config: "BaseConfig") -> str:
        value = getattr(config, self.key_name)
        return f"{self.key_display_name}{self.convert_to_str(value)}"

    @staticmethod
    def combine_output_keys(
        config: "BaseConfig",
        keys: list[Union["OutputKey", list["OutputKey"]]],
        sep: str = "/",
        secondary_sep: str = "_",
    ) -> str:
        res = []
        for output_key in keys:
            if isinstance(output_key, list):
                res.append(secondary_sep.join(output_key.display(config) for output_key in output_key))
            else:
                res.append(output_key.display(config))
        return sep.join(res)


class BASE_OUTPUT_KEYS:
    MODEL_ID = OutputKey[TModelID]("model_id", key_display_name="")
    MODEL_ARCH = OutputKey[MODEL_ARCH]("model_arch", key_display_name="arch=")
    MODEL_SIZE = OutputKey[str]("model_size", key_display_name="size=")
    VARIATION = OutputKey[str]("variation", key_display_name="v=")
    EXPERIMENT_NAME = OutputKey[str]("experiment_name", key_display_name="")
    DATASET_NAME = OutputKey[str]("dataset_name", key_display_name="ds=")
    WINDOW_SIZE = OutputKey[int]("window_size", key_display_name="ws=")


_TConfigOutputs = TypeVar("_TConfigOutputs", bound=Any)


@dataclass
class BaseConfig(ABC, Generic[_TConfigOutputs]):
    """Base configuration class with common parameters across all scripts."""

    experiment_base_name: str
    variation: str = "v1"

    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    model_size: str = "130M"
    dataset_args: DatasetArgs = create_mutable_field(
        lambda: DatasetArgs(
            name=DATASETS.COUNTER_FACT,
            splits="all",
            filteration=FILTERATIONS.all_correct,
        ),
    )
    _batch_size: int = 1  # Adjust based on GPU memory
    with_slurm: bool = False
    overwrite_existing_outputs: bool = False

    @property
    def dataset_name(self) -> str:
        return self.dataset_args.display_name

    @property
    def batch_size(self) -> int:
        return 1 if (self.model_arch == MODEL_ARCH.MAMBA2) else self._batch_size

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]

    @final
    @property
    def experiment_name(self) -> str:
        name = f"{self.experiment_base_name}"
        if self.variation:
            name += f"/{self.variation}"
        return name

    @property
    @abstractmethod
    def experiment_output_keys(self) -> list[OutputKey | list[OutputKey]]:
        return [
            BASE_OUTPUT_KEYS.MODEL_ARCH,
            BASE_OUTPUT_KEYS.MODEL_SIZE,
            BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
            BASE_OUTPUT_KEYS.VARIATION,
            BASE_OUTPUT_KEYS.DATASET_NAME,
        ]

    @final
    @property
    def experiment_variation_base_path(self) -> Path:
        return PATHS.OUTPUT_DIR / OutputKey.combine_output_keys(
            self,
            self.experiment_output_keys,
            sep="/",
        )

    @property
    def job_name(self) -> str:
        return OutputKey.combine_output_keys(
            self,
            self.experiment_output_keys,
            sep="_",
        )

    @property
    def running_history_path(self) -> Path:
        return self.experiment_variation_base_path / "running_history"

    @property
    def plots_path(self) -> Path:
        return self.experiment_variation_base_path / "plots"

    @property
    def outputs_path(self) -> Path:
        return self.experiment_variation_base_path / "outputs"

    def running_history_json_path(self, run_id: str) -> Path:
        return self.running_history_path / f"{run_id}.json"

    def create_output_path(self) -> None:
        self.running_history_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        self.outputs_path.mkdir(parents=True, exist_ok=True)

        run_id = create_run_id(None)
        json.dump(asdict(self), self.running_history_json_path(run_id).open("w"), indent=4)

    def get_raw_data(self, align_to_known: bool = True) -> pd.DataFrame:
        dataset = load_splitted_counter_fact(
            "all", align_to_known=align_to_known, filteration=self.dataset_args.filteration
        )
        return pd.DataFrame(cast(dict, dataset))

    def init_sub_config_from_full_pipeline_config(
        self,
        sub_config_cls: Type[_TBaseConfig],
        **kwargs,
    ) -> _TBaseConfig:
        """Initialize a sub-config from this full pipeline config.

        Args:
            sub_config_cls: The class of the sub-config to initialize

        Returns:
            An instance of the sub-config with values copied from this config

        Raises:
            ValueError: If a required field in sub_config is missing from full_pipeline_config
        """
        # Get all fields from the sub-config class
        sub_config_field_names = {f.name for f in fields(sub_config_cls) if not f.name.startswith("_")}

        # Get all fields from this class
        full_config_fields = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}

        # Create kwargs for sub-config initialization
        init_kwargs = {}
        for field_name in sub_config_field_names:
            if field_name == "experiment_base_name":
                # Special case: use the sub-config's default experiment_base_name
                continue
            if field_name in kwargs:
                init_kwargs[field_name] = kwargs[field_name]
            elif field_name not in full_config_fields:
                raise ValueError(
                    f"Field '{field_name}' required by {sub_config_cls.__name__} "
                    f"is missing in {self.__class__.__name__}"
                )
            else:
                init_kwargs[field_name] = full_config_fields[field_name]

        # Initialize the sub-config
        return sub_config_cls(**init_kwargs)

    def get_prompt_data(self) -> TPromptData:
        from src.experiments.data_construction import DataConstructionConfig

        original_res, attn_res = [
            self.init_sub_config_from_full_pipeline_config(DataConstructionConfig, attention=attention).get_outputs()
            for attention in [True, False]
        ]

        mask = (original_res["hit"] == attn_res["hit"]) & (attn_res["hit"])
        return attn_res[mask]  # type: ignore

    @abstractmethod
    def get_outputs(self) -> _TConfigOutputs:
        pass
