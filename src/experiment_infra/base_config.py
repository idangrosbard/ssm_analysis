import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar, cast, final

import pandas as pd
import pyrallis

from src.consts import COLUMNS, MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS
from src.datasets.download_dataset import load_splitted_counter_fact
from src.experiment_infra.output_path import (
    _ATTRIBUTE_TYPE,
    OutputKey,
    combine_output_keys,
)
from src.types import (
    DATASETS,
    FILTERATIONS,
    MODEL_ARCH,
    DatasetArgs,
    TModelID,
    TPromptData,
)
from src.utils.experiment_helper import create_run_id

_TBaseConfig = TypeVar("_TBaseConfig", bound="BaseConfig")


def create_mutable_field(
    default_factory: Callable[[], _ATTRIBUTE_TYPE],
) -> _ATTRIBUTE_TYPE:
    # Pyralis need mutable fields to be defined with field but it's typing is not complete.
    # This is a fix to make it work.

    return cast(
        _ATTRIBUTE_TYPE,
        pyrallis.field(default_factory=default_factory, is_mutable=True),
    )


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
    variation: str = "v3"

    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    model_size: str = "130M"
    dataset_args: DatasetArgs = create_mutable_field(
        lambda: DatasetArgs(
            name=DATASETS.COUNTER_FACT,
            splits="all",
            filteration=FILTERATIONS.current_model_correct,
        ),
    )
    _batch_size: int = 1  # Adjust based on GPU memory
    with_slurm: bool = False
    # slurm_gpu_type: str = "titan_xp-studentrun"
    slurm_gpu_type: str = "l40s"
    slurm_gpus_per_node: int = 1
    overwrite_existing_outputs: bool = False

    @property
    def dataset_name(self) -> str:
        # return self.dataset_args.display_name
        # return 'counter_fact_all_correct'
        return self.dataset_args.name

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
        return name

    @property
    @abstractmethod
    def experiment_output_keys(self) -> list[OutputKey | list[OutputKey]]:
        return [
            BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
            BASE_OUTPUT_KEYS.VARIATION,
            BASE_OUTPUT_KEYS.MODEL_ARCH,
            BASE_OUTPUT_KEYS.MODEL_SIZE,
            BASE_OUTPUT_KEYS.DATASET_NAME,
        ]

    @final
    @property
    def experiment_variation_base_path(self) -> Path:
        return PATHS.OUTPUT_DIR / combine_output_keys(
            self,
            self.experiment_output_keys,
            sep="/",
        )

    @property
    def job_name(self) -> str:
        return combine_output_keys(
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

        params = asdict(self)
        params["run_id"] = run_id
        try:
            params["git_commit_hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            pass

        json.dump(params, self.running_history_json_path(run_id).open("w"), indent=4)

    def get_raw_data(self, align_to_known: bool = False) -> pd.DataFrame:
        dataset = load_splitted_counter_fact(
            "all",
            align_to_known=align_to_known,
            filteration=self.dataset_args.filteration,
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
        from src.experiments.evaluate_model import EvaluateModelConfig

        df = self.init_sub_config_from_full_pipeline_config(
            EvaluateModelConfig,
            drop_subject=EvaluateModelConfig.drop_subject,
            drop_subj_last_token=EvaluateModelConfig.drop_subj_last_token,
            with_3_dots=EvaluateModelConfig.with_3_dots,
            new_max_tokens=EvaluateModelConfig.new_max_tokens,
            top_k_tokens=EvaluateModelConfig.top_k_tokens,
        ).get_outputs()
        if self.dataset_args.filteration == FILTERATIONS.current_model_correct:
            return cast(
                TPromptData,
                df[df[COLUMNS.MODEL_CORRECT]].set_index(COLUMNS.ORIGINAL_IDX),
            )
        else:
            raise NotImplementedError(f"Filteration {self.dataset_args.filteration} not implemented")

    @abstractmethod
    def get_outputs(self) -> _TConfigOutputs:
        pass

    @abstractmethod
    def run(self) -> None:
        pass
