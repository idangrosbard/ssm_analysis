from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from git import Optional

from src.final_plots.results_bank import ParamNames
from src.types import SLURM_GPU_TYPE
from src.utils.streamlit_utils import SessionKey

DEFAULT_VARIATION = "v3"
DEFAULT_WINDOW_SIZE = 9


class AppSessionKeys:
    variation: SessionKey[str] = SessionKey.with_default("variation", DEFAULT_VARIATION)
    selected_gpu: SessionKey[SLURM_GPU_TYPE] = SessionKey.with_default("selected_gpu", SLURM_GPU_TYPE.L40S)
    window_size: SessionKey[int] = SessionKey.with_default("window_size", DEFAULT_WINDOW_SIZE)


class HeatmapSessionKeys:
    model_filters: SessionKey[dict[str, list[str]]] = SessionKey.with_default("model_filters", {})
    # selected_prompts: SessionKey[set[str]] = SessionKey.with_default("selected_prompts", set())
    selected_prompts: dict[int, SessionKey[bool]] = {}
    show_combination: SessionKey[Optional[int]] = SessionKey.with_default("show_combination", None)
    selected_combination: SessionKey[dict[str, dict[str, list[int]]]] = SessionKey("selected_combination")

    @staticmethod
    def selected_prompt(prompt_idx: Any) -> SessionKey[bool]:
        if prompt_idx not in HeatmapSessionKeys.selected_prompts:
            HeatmapSessionKeys.selected_prompts[prompt_idx] = SessionKey.with_default(
                f"select_prompt_{prompt_idx}", False
            )
        return HeatmapSessionKeys.selected_prompts[prompt_idx]

    @staticmethod
    def get_selected_combination_row() -> Optional[int]:
        if HeatmapSessionKeys.selected_combination.get() is None:
            return None
        selected_rows = HeatmapSessionKeys.selected_combination.get()["selection"]["rows"]
        assert len(selected_rows) <= 1
        if len(selected_rows) == 0:
            return None
        return selected_rows[0]

    @staticmethod
    def get_selected_prompts() -> set[int]:
        return {prompt_idx for prompt_idx, key in HeatmapSessionKeys.selected_prompts.items() if key.get()}

    @staticmethod
    def select_prompt(prompt_idx: Any) -> SessionKey[bool]:
        if prompt_idx not in HeatmapSessionKeys.selected_prompts:
            HeatmapSessionKeys.selected_prompts[prompt_idx] = SessionKey.with_default(
                f"select_prompt_{prompt_idx}", False
            )
        return HeatmapSessionKeys.selected_prompts[prompt_idx]


class ReqMetadataColumns:
    AvailableOptions = "Available Options"
    Options = "Options"
    CurrentOverride = "Current Override"
    Key = "Key"


# Heatmap Creation constants
FILTERING_MODLS_COLUMNS_N = 2

MINIMUM_COMBINATIONS_FOR_FILTERING = 100


class ModelFilterOption(StrEnum):
    CORRECT = "correct"
    ANY = "any"
    INCORRECT = "incorrect"


class HeatmapColumns:
    PROMPT_COUNT = "Prompt Count"
    SELECTED_PROMPT = "Selected Prompt"
    MODEL_CORRECT = "Model Correct"


MODEL_FILTER_OPTIONS: list[ModelFilterOption] = list(ModelFilterOption)

# Info Flow Plots constants
ParamRole = Literal["grid", "column", "row", "line", "fixed"]
PARAM_ROLES: list[ParamRole] = ["fixed", "grid", "column", "row", "line"]
DEFAULT_LINE_STYLES = ["-", "--", ":", "-."]
DEFAULT_PLOT_CONFIG = {
    "confidence_level": 0.95,
    "plot_height": 400,
    "plot_width": 600,
}


# Pagination constants
class PaginationConfig:
    RESULTS_BANK = {"default_page_size": 20, "key_prefix": "results_bank_"}
    DATA_REQS = {"default_page_size": 10, "key_prefix": "data_reqs_"}
    COMBINATIONS = {"default_page_size": 10, "key_prefix": "combinations_"}
    PROMPTS = {"default_page_size": 10, "key_prefix": "prompts_"}


# Preview constants
PREVIEW_LENGTH = 100

# Data Requirements filter columns
DATA_REQS_FILTER_COLUMNSs = [
    ReqMetadataColumns.AvailableOptions,
    ParamNames.experiment_name,
    ParamNames.model_arch,
    ParamNames.model_size,
    ParamNames.window_size,
    ParamNames.is_all_correct,
    ParamNames.source,
    ParamNames.target,
    ParamNames.prompt_idx,
]

DATA_REQS_DEFAULT_FILTER_VALUES = {
    ReqMetadataColumns.AvailableOptions: [0],
    ParamNames.is_all_correct: [False],
}


class DataReqsSessionKeys:
    overrides: SessionKey[dict[str, Path]] = SessionKey.with_default("overrides", {})
    selected_requirements: SessionKey[set[str]] = SessionKey.with_default("selected_requirements", set())

    @staticmethod
    def select_requirement(key: str) -> SessionKey[bool]:
        return SessionKey.with_default(f"select_{key}", False)

    @staticmethod
    def override_requirement(key: str) -> SessionKey[bool]:
        return SessionKey.with_default(f"override_{key}", False)


class InfoFlowSessionKeys:
    param_roles: SessionKey[dict[str, ParamRole]] = SessionKey.with_default("param_roles", {})

    @staticmethod
    def param_value(param: str) -> SessionKey[str]:
        return SessionKey.with_default(f"value_{param}", "")

    @staticmethod
    def param_role(param: str) -> SessionKey[ParamRole]:
        return SessionKey.with_default(f"role_{param}", "fixed")
