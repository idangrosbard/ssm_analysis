import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from src.names import COLS, ResultBankParamNames
from src.types import (
    DATASETS,
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    MODEL_SIZE_CAT,
    SLURM_GPU_TYPE,
    TDatasetID,
    TModelID,
    TModelSize,
    TokenType,
)


class C_ACTIVE_USERS(StrEnum):
    nirendy = "nirendy"
    idangrosbard = "idangrosbard"
    other = "other"


ACTIVE_USER = C_ACTIVE_USERS.other
if env_user := os.environ.get("USER"):
    ACTIVE_USER = C_ACTIVE_USERS(env_user)


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

    @property
    def FINAL_PLOTS_DIR(self) -> Path:
        return self.PROJECT_DIR / "final_plots"


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


MODEL_SIZES_PER_ARCH_TO_MODEL_ID: dict[MODEL_ARCH, dict[TModelSize, TModelID]] = {
    MODEL_ARCH.MAMBA1: {
        TModelSize("130M"): TModelID("state-spaces/mamba-130M-hf"),
        TModelSize("1.4B"): TModelID("state-spaces/mamba-1.4B-hf"),
        TModelSize("2.8B"): TModelID("state-spaces/mamba-2.8B-hf"),
        TModelSize("7B"): TModelID("TRI-ML/mamba-7b-rw"),
        TModelSize("7B-falcon"): TModelID("tiiuae/falcon-mamba-7b"),
        TModelSize("7B-falcon-base"): TModelID("tiiuae/Falcon3-Mamba-7B-Base"),
    },
    MODEL_ARCH.MAMBA2: {
        TModelSize("130M"): TModelID("state-spaces/mamba2-130M"),
        TModelSize("1.3B"): TModelID("state-spaces/mamba2-1.3b"),
        TModelSize("2.7B"): TModelID("state-spaces/mamba2-2.7B"),
        # TModelSize("8B"): TModelID("nvidia/mamba2-8b-3t-4k"),
    },
    MODEL_ARCH.LLAMA2: {
        TModelSize("7B"): TModelID("meta-llama/Llama-2-7b-hf"),
    },
    MODEL_ARCH.LLAMA3_2: {
        TModelSize("1B"): TModelID("meta-llama/Llama-3.2-1B"),
        TModelSize("3B"): TModelID("meta-llama/Llama-3.2-3B"),
    },
    MODEL_ARCH.GPT2: {
        TModelSize("124M"): TModelID("openai-community/gpt2"),
        TModelSize("355M"): TModelID("openai-community/gpt2-medium"),
        TModelSize("774M"): TModelID("openai-community/gpt2-large"),
        TModelSize("1.5B"): TModelID("openai-community/gpt2-xl"),
    },
}


GRAPHS_ORDER: dict[MODEL_ARCH_AND_SIZE, MODEL_SIZE_CAT] = {
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, "124M"): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("130M")): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("130M")): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("355M")): MODEL_SIZE_CAT.SMALL,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("774M")): MODEL_SIZE_CAT.MEDIUM,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("1.4B")): MODEL_SIZE_CAT.MEDIUM,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("1.3B")): MODEL_SIZE_CAT.MEDIUM,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.GPT2, TModelSize("1.5B")): MODEL_SIZE_CAT.LARGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("2.8B")): MODEL_SIZE_CAT.LARGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, TModelSize("2.7B")): MODEL_SIZE_CAT.LARGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B")): MODEL_SIZE_CAT.HUGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B-falcon")): MODEL_SIZE_CAT.HUGE,
    MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA1, TModelSize("7B-falcon-base")): MODEL_SIZE_CAT.HUGE,
    # MODEL_ARCH_AND_SIZE(MODEL_ARCH.MAMBA2, "8B"): MODEL_SIZE_CAT.HUGE,
}


def get_model_by_cat_size(cat_size: MODEL_SIZE_CAT) -> list[MODEL_ARCH_AND_SIZE]:
    return [
        model_arch_and_size
        for model_arch_and_size, model_size_cat in GRAPHS_ORDER.items()
        if model_size_cat == cat_size
    ]


def reverse_model_id(model_id: TModelID) -> MODEL_ARCH_AND_SIZE:
    for model_arch_and_size in GRAPHS_ORDER.keys():
        for model_id_prefix in ["", "state-spaces/", "tiiuae/"]:
            if (
                MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch_and_size.arch][model_arch_and_size.size]
                == f"{model_id_prefix}{model_id}"
            ):
                return model_arch_and_size
    raise ValueError(f"Model id {model_id} not found in MODEL_SIZES_PER_ARCH_TO_MODEL_ID")


def model_and_size_to_slurm_gpu_type(
    model_arch_and_size: MODEL_ARCH_AND_SIZE,
) -> SLURM_GPU_TYPE:
    model_cat_size = GRAPHS_ORDER[model_arch_and_size]
    match model_cat_size:
        case MODEL_SIZE_CAT.SMALL | MODEL_SIZE_CAT.MEDIUM:
            return SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN
        case MODEL_SIZE_CAT.LARGE | MODEL_SIZE_CAT.HUGE:
            match ACTIVE_USER:
                case C_ACTIVE_USERS.nirendy:
                    return SLURM_GPU_TYPE.L40S
                case C_ACTIVE_USERS.idangrosbard:
                    return SLURM_GPU_TYPE.H100
                case _:
                    raise NotImplementedError(f"No SLURM GPU type for user {ACTIVE_USER}")
        case _:
            assert_never(model_cat_size)


def is_mamba_arch(model_arch: MODEL_ARCH) -> bool:
    return model_arch in [MODEL_ARCH.MAMBA1, MODEL_ARCH.MAMBA2]


def is_falcon(model_size: str) -> bool:
    return "falcon" in model_size


DATASETS_IDS: dict[DATASETS, TDatasetID] = {DATASETS.COUNTER_FACT: TDatasetID("NeelNanda/counterfact-tracing")}  # type: ignore

COUNTER_FACT_2_KNOWN1000_COL_CONV = {
    COLS.COUNTER_FACT.TARGET_TRUE: "attribute",
}


TOKEN_TYPE_COLORS: dict[TokenType, str] = {
    TokenType.last: "#D2691E",  # orange
    TokenType.first: "#0000FF",  # blue
    TokenType.subject: "#008000",  # green
    TokenType.relation: "#800080",  # purple
    TokenType.context: "#FF0000",  # red
    TokenType.all: "#000000",  # black
}

TOKEN_TYPE_LINE_STYLES: dict[TokenType, str] = {
    TokenType.last: "-.",
    TokenType.first: ":",
    TokenType.subject: "-",
    TokenType.relation: "--",
    TokenType.context: "--",
    TokenType.all: "-",
}
CONVERT_TO_PLOTLY_LINE_STYLE = {
    "-": "solid",
    ":": "dot",
    "--": "dash",
    "-.": "longdashdot",
    "-.-": "dashdot",
    "-.-.": "longdash",
}


def format_params_for_title(params: dict) -> str:
    """Format parameters for title display in a consistent order."""

    parts_remaining = set(params.keys())
    ordered_parts = []
    for param in ResultBankParamNames:
        if param in parts_remaining:
            parts_remaining.remove(param)
            match param:
                case ResultBankParamNames.experiment_name | ResultBankParamNames.variation:
                    ordered_parts.append(params[param])
                case ResultBankParamNames.model_arch:
                    if ResultBankParamNames.model_size in parts_remaining:
                        ordered_parts.append(
                            f"{params[ResultBankParamNames.model_arch]} {params[ResultBankParamNames.model_size]}"
                        )
                        parts_remaining.remove(ResultBankParamNames.model_size)
                    else:
                        ordered_parts.append(params[ResultBankParamNames.model_arch])
                case ResultBankParamNames.window_size:
                    ordered_parts.append(f"ws={params[ResultBankParamNames.window_size]}")
                case ResultBankParamNames.source:
                    base_str = f"From {params[ResultBankParamNames.source]}"
                    if ResultBankParamNames.feature_category in parts_remaining:
                        if params[ResultBankParamNames.feature_category] is not None:
                            base_str = f"{base_str} - {params[ResultBankParamNames.feature_category]}"
                        parts_remaining.remove(ResultBankParamNames.feature_category)
                    ordered_parts.append(base_str)
                case _:
                    ordered_parts.append(f"{param}={params[param]}")

    for param in parts_remaining:
        ordered_parts.append(f"{param}={params[param]}")

    return " | ".join(ordered_parts)
