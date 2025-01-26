import os
from pathlib import Path
from typing import NamedTuple

from src.types import DATASETS, MODEL_ARCH, MODEL_SIZE_CAT, TDatasetID, TModelID


class PATHS:
    PROJECT_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = PROJECT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    OTHER_DATA_DIR = DATA_DIR / "other"
    PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
    RAW_KNOWN_1000_DIR = RAW_DATA_DIR / "spaced" / DATASETS.KNOWN_1000
    COUNTER_FACT_DIR = PREPROCESSED_DATA_DIR / DATASETS.COUNTER_FACT
    COUNTER_FACT_FILTERATIONS_DIR = COUNTER_FACT_DIR / "filterations"
    PROCESSED_KNOWN_DIR = PREPROCESSED_DATA_DIR / DATASETS.KNOWN_1000
    KNOWN_1000_FILTERATIONS_DIR = PROCESSED_KNOWN_DIR / "filterations"
    DATA_SHARED_DIR = PROJECT_DIR / "shared"
    RUNS_DIR = PROJECT_DIR / "runs"
    TENSORBOARD_DIR = PROJECT_DIR / "tensorboard"
    RESULTS_DIR = PROJECT_DIR / "results"
    OUTPUT_DIR = PROJECT_DIR / "output"
    SLURM_DIR = PROJECT_DIR / "slurm"


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


class ISlurmArgs(NamedTuple):
    with_parallel: bool
    partition: str = "gpu-a100-killable"
    time: int = 1200
    singal: str = "USR1@120"
    nodes: int = 1
    ntasks: int = 1
    mem: int = int(5e4)
    cpus_per_task: int = 1
    gpus: int = 1
    account: str = "gpu-research"
    workspace: Path = PATHS.PROJECT_DIR
    outputs_relative_path: Path = PATHS.TENSORBOARD_DIR.relative_to(PATHS.PROJECT_DIR)
    master_port: str = DDP.MASTER_PORT


MODEL_SIZES_PER_ARCH_TO_MODEL_ID: dict[MODEL_ARCH, dict[str, TModelID]] = {
    MODEL_ARCH.MAMBA1: {
        "130M": TModelID("state-spaces/mamba-130M-hf"),
        "1.4B": TModelID("state-spaces/mamba-1.4B-hf"),
        "2.8B": TModelID("state-spaces/mamba-2.8B-hf"),
    },
    MODEL_ARCH.MINIMAL_MAMBA1: {
        "130M": TModelID("state-spaces/mamba-130M"),
        "1.4B": TModelID("state-spaces/mamba-1.4B"),
        "2.8B": TModelID("state-spaces/mamba-2.8B"),
    },
    MODEL_ARCH.MINIMAL_MAMBA2: {
        "130M": TModelID("state-spaces/mamba2-130M"),
        "1.3B": TModelID("state-spaces/mamba2-1.3b"),
        "2.7B": TModelID("state-spaces/mamba2-2.7B"),
    },
    MODEL_ARCH.MINIMAL_MAMBA2_new: {
        "130M": TModelID("state-spaces/mamba2-130M"),
        "1.3B": TModelID("state-spaces/mamba2-1.3b"),
        "2.7B": TModelID("state-spaces/mamba2-2.7B"),
    },
    MODEL_ARCH.MAMBA2: {
        "130M": TModelID("state-spaces/mamba2-130M"),
        "1.3B": TModelID("state-spaces/mamba2-1.3b"),
        "2.7B": TModelID("state-spaces/mamba2-2.7B"),
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
    (MODEL_ARCH.MINIMAL_MAMBA2_new, "130M"),
    (MODEL_ARCH.MAMBA1, "1.4B"),
    (MODEL_ARCH.MINIMAL_MAMBA2_new, "1.3B"),
    (MODEL_ARCH.MAMBA1, "2.8B"),
    (MODEL_ARCH.MINIMAL_MAMBA2_new, "2.7B"),
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
        for model_id_prefix in ["", "state-spaces/"]:
            if MODEL_SIZES_PER_ARCH_TO_MODEL_ID[arch][model_size] == f"{model_id_prefix}{model_id}":
                return arch, model_size
    raise ValueError(f"Model id {model_id} not found in MODEL_SIZES_PER_ARCH_TO_MODEL_ID")


class FILTERATIONS:
    all_correct = "all_correct"
    all_any_correct = "all_any_correct"


DATASETS_IDS: dict[DATASETS, TDatasetID] = {DATASETS.COUNTER_FACT: TDatasetID("NeelNanda/counterfact-tracing")}  # type: ignore


class COLUMNS:
    ORIGINAL_IDX = "original_idx"
