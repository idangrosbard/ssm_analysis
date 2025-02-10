import os
from dataclasses import dataclass
from pathlib import Path

from src.types import DATASETS, MODEL_ARCH, MODEL_SIZE_CAT, TDatasetID, TModelID, TokenType


@dataclass
class PathsConfig:
    """Configuration for project paths that can be easily mocked."""

    PROJECT_DIR: Path = Path(__file__).parent.parent.resolve()

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_DIR / "data"

    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def OTHER_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "other"

    @property
    def PREPROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "preprocessed"

    @property
    def COUNTER_FACT_DIR(self) -> Path:
        return self.PREPROCESSED_DATA_DIR / DATASETS.COUNTER_FACT

    @property
    def COUNTER_FACT_FILTERATIONS_DIR(self) -> Path:
        return self.COUNTER_FACT_DIR / "filterations"

    @property
    def DATA_SHARED_DIR(self) -> Path:
        return self.PROJECT_DIR / "shared"

    @property
    def RUNS_DIR(self) -> Path:
        return self.PROJECT_DIR / "runs"

    @property
    def TENSORBOARD_DIR(self) -> Path:
        return self.PROJECT_DIR / "tensorboard"

    @property
    def RESULTS_DIR(self) -> Path:
        return self.PROJECT_DIR / "results"

    @property
    def OUTPUT_DIR(self) -> Path:
        return self.PROJECT_DIR / "output"

    @property
    def SLURM_DIR(self) -> Path:
        return self.PROJECT_DIR / "slurm"


# Global instance
PATHS = PathsConfig()


class ENV_VARS:
    MASTER_PORT = "MASTER_PORT"
    MASTER_ADDR = "MASTER_ADDR"


class FORMATS:
    TIME = "%Y%m%d_%H-%M-%S"
    LOGGER_FORMAT = "%(asctime)s - %(message)s"


class DDP:
    MASTER_PORT = os.environ.get(ENV_VARS.MASTER_PORT, "12355")
    MASTER_ADDR = "localhost"
    BACKEND = "nccl"
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


MODEL_SIZES_PER_ARCH_TO_MODEL_ID: dict[MODEL_ARCH, dict[str, TModelID]] = {
    MODEL_ARCH.MAMBA1: {
        "130M": TModelID("state-spaces/mamba-130M-hf"),
        "1.4B": TModelID("state-spaces/mamba-1.4B-hf"),
        "2.8B": TModelID("state-spaces/mamba-2.8B-hf"),
        "7B": TModelID("TRI-ML/mamba-7b-rw"),
        "7B-falcon": TModelID("tiiuae/falcon-mamba-7b"),
        "7B-falcon-base": TModelID("tiiuae/Falcon3-Mamba-7B-Base"),
    },
    MODEL_ARCH.MAMBA2: {
        "130M": TModelID("state-spaces/mamba2-130M"),
        "1.3B": TModelID("state-spaces/mamba2-1.3b"),
        "2.7B": TModelID("state-spaces/mamba2-2.7B"),
        "8B": TModelID("nvidia/mamba2-8b-3t-4k"),
    },
    MODEL_ARCH.LLAMA2: {
        "7B": TModelID("meta-llama/Llama-2-7b-hf"),
    },
    MODEL_ARCH.LLAMA3_2: {
        "1B": TModelID("meta-llama/Llama-3.2-1B"),
        "3B": TModelID("meta-llama/Llama-3.2-3B"),
    },
}

GRAPHS_ORDER = [
    (MODEL_ARCH.MAMBA1, "130M"),
    (MODEL_ARCH.MAMBA2, "130M"),
    (MODEL_ARCH.MAMBA1, "1.4B"),
    (MODEL_ARCH.MAMBA2, "1.3B"),
    (MODEL_ARCH.MAMBA1, "2.8B"),
    (MODEL_ARCH.MAMBA2, "2.7B"),
    (MODEL_ARCH.MAMBA1, "7B"),
    (MODEL_ARCH.MAMBA1, "7B-falcon"),
    (MODEL_ARCH.MAMBA1, "7B-falcon-base"),
    (MODEL_ARCH.MAMBA2, "8B"),
]


def get_model_by_cat_size(cat_size: MODEL_SIZE_CAT) -> list[tuple[MODEL_ARCH, str]]:
    if isinstance(cat_size, str):
        cat_size = MODEL_SIZE_CAT[cat_size.upper()]

    for i, size in enumerate(MODEL_SIZE_CAT):
        if size == cat_size:
            return GRAPHS_ORDER[2 * i : 2 * i + 2]
    raise ValueError(f"Model size {cat_size} not found in GRAPHS_ORDER")


def reverse_model_id(model_id: str) -> tuple[MODEL_ARCH, str]:
    for arch, model_size in GRAPHS_ORDER:
        for model_id_prefix in ["", "state-spaces/", "tiiuae/"]:
            if MODEL_SIZES_PER_ARCH_TO_MODEL_ID[arch][model_size] == f"{model_id_prefix}{model_id}":
                return arch, model_size
    raise ValueError(f"Model id {model_id} not found in MODEL_SIZES_PER_ARCH_TO_MODEL_ID")


def is_falcon(model_size: str) -> bool:
    return "falcon" in model_size


DATASETS_IDS: dict[DATASETS, TDatasetID] = {DATASETS.COUNTER_FACT: TDatasetID("NeelNanda/counterfact-tracing")}  # type: ignore


class COLUMNS:
    ORIGINAL_IDX = "original_idx"


TOKEN_TYPE_COLORS: dict[TokenType, str] = {
    TokenType.last: "#D2691E",  # orange
    TokenType.first: "#0000FF",  # blue
    TokenType.subject: "#008000",  # green
    TokenType.relation: "#800080",  # purple
    TokenType.context: "#FF0000",  # red
}

TOKEN_TYPE_LINE_STYLES: dict[TokenType, str] = {
    TokenType.last: "-.",
    TokenType.first: ":",
    TokenType.subject: "-",
    TokenType.relation: "--",
    TokenType.context: "--",
}
