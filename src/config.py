from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyrallis

from src.consts import FILTERATIONS, MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.types import DATASETS, MODEL_ARCH, DatasetArgs, TModelID, TokenType


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


@dataclass
class InfoFlowConfig(BaseConfig):
    """Configuration for information flow analysis."""

    experiment_name: str = "info_flow"
    window_size: int = 9
    DEBUG_LAST_WINDOWS: Optional[int] = None
    overwrite: bool = False
    for_multi_plot: bool = False
    knockout_map: dict[TokenType, list[TokenType]] = pyrallis.field(
        default_factory=lambda: {
            TokenType.last: [
                TokenType.last,
                TokenType.first,
                TokenType.subject,
                TokenType.relation,
            ],
            TokenType.subject: [
                TokenType.context,
                TokenType.subject,
            ],
            TokenType.relation: [
                TokenType.context,
                TokenType.subject,
                TokenType.relation,
            ],
        }
    )


@dataclass
class HeatmapConfig(BaseConfig):
    """Configuration for heatmap generation."""

    experiment_name: str = "heatmap"
    window_size: int = 5
    prompt_indices: list[int] = pyrallis.field(default_factory=lambda: [1, 2, 3, 4, 5])
