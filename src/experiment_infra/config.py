from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyrallis

from src.consts import FILTERATIONS, MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.types import DATASETS, MODEL_ARCH, DatasetArgs, TModelID


@dataclass
class BaseConfig:
    """Base configuration class with common parameters across all scripts."""

    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    model_size: str = "130M"
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"), is_mutable=True
    )
    filteration: str = FILTERATIONS.all_correct
    _batch_size: int = 16  # Adjust based on GPU memory
    output_file: Optional[Path] = None
    with_slurm: bool = False
    output_dir: Optional[Path] = None

    @property
    def batch_size(self) -> int:
        return (
            1
            if (self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2 or self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2_new)
            else self._batch_size
        )

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]

    @abstractmethod
    def get_output_path(self) -> Path:
        pass

    @abstractmethod
    def get_experiment_name(self) -> str:
        pass

    @property
    def output_path(self) -> Path:
        return self.get_output_path()

    @property
    def experiment_name(self) -> str:
        return self.get_experiment_name()
