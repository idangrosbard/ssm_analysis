import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.types import (
    DATASETS,
    MODEL_ARCH,
    MODEL_SIZE_CAT,
    FeatureCategory,
    TDatasetID,
    TInfoFlowSource,
    TModelID,
    TokenType,
)


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
        # "8B": TModelID("nvidia/mamba2-8b-3t-4k"),
    },
    MODEL_ARCH.LLAMA2: {
        "7B": TModelID("meta-llama/Llama-2-7b-hf"),
    },
    MODEL_ARCH.LLAMA3_2: {
        "1B": TModelID("meta-llama/Llama-3.2-1B"),
        "3B": TModelID("meta-llama/Llama-3.2-3B"),
    },
    MODEL_ARCH.GPT2: {
        "124M": TModelID("openai-community/gpt2"),
        "355M": TModelID("openai-community/gpt2-medium"),
        "774M": TModelID("openai-community/gpt2-large"),
        "1.5B": TModelID("openai-community/gpt2-xl"),
    },
}

GRAPHS_ORDER = [
    (MODEL_ARCH.GPT2, "124M", MODEL_SIZE_CAT.SMALL),
    (MODEL_ARCH.MAMBA1, "130M", MODEL_SIZE_CAT.SMALL),
    (MODEL_ARCH.MAMBA2, "130M", MODEL_SIZE_CAT.SMALL),
    (MODEL_ARCH.GPT2, "355M", MODEL_SIZE_CAT.SMALL),
    (MODEL_ARCH.GPT2, "774M", MODEL_SIZE_CAT.MEDIUM),
    (MODEL_ARCH.MAMBA1, "1.4B", MODEL_SIZE_CAT.MEDIUM),
    (MODEL_ARCH.MAMBA2, "1.3B", MODEL_SIZE_CAT.MEDIUM),
    (MODEL_ARCH.GPT2, "1.5B", MODEL_SIZE_CAT.MEDIUM),
    (MODEL_ARCH.MAMBA1, "2.8B", MODEL_SIZE_CAT.LARGE),
    (MODEL_ARCH.MAMBA2, "2.7B", MODEL_SIZE_CAT.LARGE),
    (MODEL_ARCH.MAMBA1, "7B", MODEL_SIZE_CAT.HUGE),
    (MODEL_ARCH.MAMBA1, "7B-falcon", MODEL_SIZE_CAT.HUGE),
    (MODEL_ARCH.MAMBA1, "7B-falcon-base", MODEL_SIZE_CAT.HUGE),
    # (MODEL_ARCH.MAMBA2, "8B", MODEL_SIZE_CAT.HUGE),
]


def get_model_cat_size(model_arch: MODEL_ARCH, model_size: str) -> MODEL_SIZE_CAT:
    for _model_arch, _model_size, model_size_cat in GRAPHS_ORDER:
        if model_arch == _model_arch and model_size == _model_size:
            return model_size_cat
    raise ValueError(f"Model {model_arch} {model_size} not found in GRAPHS_ORDER")


def get_model_by_cat_size(cat_size: MODEL_SIZE_CAT) -> list[tuple[MODEL_ARCH, str]]:
    return [(arch, model_size) for arch, model_size, model_size_cat in GRAPHS_ORDER if model_size_cat == cat_size]


def reverse_model_id(model_id: str) -> tuple[MODEL_ARCH, str]:
    for arch, model_size, _ in GRAPHS_ORDER:
        for model_id_prefix in ["", "state-spaces/", "tiiuae/"]:
            if MODEL_SIZES_PER_ARCH_TO_MODEL_ID[arch][model_size] == f"{model_id_prefix}{model_id}":
                return arch, model_size
    raise ValueError(f"Model id {model_id} not found in MODEL_SIZES_PER_ARCH_TO_MODEL_ID")


def is_mamba_arch(model_arch: MODEL_ARCH) -> bool:
    return model_arch in [MODEL_ARCH.MAMBA1, MODEL_ARCH.MAMBA2]


def is_falcon(model_size: str) -> bool:
    return "falcon" in model_size


DATASETS_IDS: dict[DATASETS, TDatasetID] = {DATASETS.COUNTER_FACT: TDatasetID("NeelNanda/counterfact-tracing")}  # type: ignore


class EXPERIMENT_NAMES(StrEnum):
    EVALUATE_MODEL = "evaluate_model"
    INFO_FLOW = "info_flow"
    HEATMAP = "heatmap"


class COLUMNS:
    # Preprocessing
    ORIGINAL_IDX = "original_idx"
    SPLIT = "split"

    # Counter Fact
    PROMPT = "prompt"
    TARGET_TRUE = "target_true"
    TARGET_FALSE = "target_false"
    SUBJECT = "subject"
    TARGET_FALSE_ID = "target_false_id"
    RELATION = "relation"
    RELATION_PREFIX = "relation_prefix"
    RELATION_SUFFIX = "relation_suffix"
    RELATION_ID = "relation_id"
    TARGET_TRUE_ID = "target_true_id"

    # KNOWN1000
    ATTRIBUTE = "attribute"

    # Evaluate Model
    TARGET_PROBS = "target_probs"
    MODEL_TOP_OUTPUT_CONFIDENCE = "model_top_output_confidence"
    MODEL_CORRECT = "model_correct"
    MODEL_OUTPUT = "model_output"
    TARGET_RANK = "target_rank"
    MODEL_TOP_OUTPUTS = "model_top_outputs"
    MODEL_GENERATION = "model_generation"
    TARGET_TOKENS = "target_tokens"

    # Data Construction
    HIT = "hit"
    MAX_PROB = "max_prob"
    TRUE_PROB = "true_prob"
    PRED = "pred"

    # Info Flow
    IF_HIT = "hit"
    IF_TRUE_PROBS = "true_probs"
    IF_DIFFS = "diffs"


COUNTER_FACT_2_KNOWN1000_COL_CONV = {
    COLUMNS.TARGET_TRUE: COLUMNS.ATTRIBUTE,
}

EVAL_MODEL_2_DATA_CONST_COL_CONV = {
    COLUMNS.TARGET_PROBS: COLUMNS.TRUE_PROB,
    COLUMNS.MODEL_TOP_OUTPUT_CONFIDENCE: COLUMNS.MAX_PROB,
    COLUMNS.MODEL_CORRECT: COLUMNS.HIT,
    COLUMNS.MODEL_OUTPUT: COLUMNS.PRED,
}

TOKEN_TYPE_COLORS: dict[TInfoFlowSource, str] = {
    TokenType.last: "#D2691E",  # orange
    TokenType.first: "#0000FF",  # blue
    TokenType.subject: "#008000",  # green
    TokenType.relation: "#800080",  # purple
    TokenType.context: "#FF0000",  # red
    TokenType.all: "#000000",  # black
    (TokenType.subject, FeatureCategory.FAST_DECAY): "#000000",  # black
    (TokenType.subject, FeatureCategory.SLOW_DECAY): "#000000",  # black
}

TOKEN_TYPE_LINE_STYLES: dict[TInfoFlowSource, str] = {
    TokenType.last: "-.",
    TokenType.first: ":",
    TokenType.subject: "-",
    TokenType.relation: "--",
    TokenType.context: "--",
    TokenType.all: "-",
    (TokenType.subject, FeatureCategory.FAST_DECAY): "--",
    (TokenType.subject, FeatureCategory.SLOW_DECAY): ":",
}


def to_model_name(model_arch: MODEL_ARCH, model_size: str) -> str:
    return f"{model_arch}-{model_size}"
