from enum import StrEnum
from typing import Any, Literal, Union

from git import Optional

from src.consts import ALL_MODEL_ARCH_AND_SIZES, model_and_size_to_slurm_gpu_type
from src.final_plots.results_bank import ParamNames
from src.types import MODEL_ARCH, SLURM_GPU_TYPE
from src.utils.streamlit_utils import SessionKey


# region Global App constants
class AppCols:
    pass


class GLOBAL_APP_CONSTS:
    DEFAULT_VARIATION = "v3"
    DEFAULT_WINDOW_SIZE = 9
    MODELS_COMBINATIONS = ALL_MODEL_ARCH_AND_SIZES

    class PaginationConfig:
        RESULTS_BANK = {"default_page_size": 20, "key_prefix": "results_bank_"}
        DATA_REQS = {"default_page_size": 10, "key_prefix": "data_reqs_"}
        COMBINATIONS = {"default_page_size": 10, "key_prefix": "combinations_"}
        PROMPTS = {"default_page_size": 10, "key_prefix": "prompts_"}


class _AppSessionKeys:
    variation: SessionKey[str] = SessionKey.with_default("variation", GLOBAL_APP_CONSTS.DEFAULT_VARIATION)
    _selected_gpu: SessionKey[Union[SLURM_GPU_TYPE, Literal["smart"]]] = SessionKey.with_default(
        "selected_gpu", "smart"
    )
    window_size: SessionKey[int] = SessionKey.with_default("window_size", GLOBAL_APP_CONSTS.DEFAULT_WINDOW_SIZE)

    def get_selected_gpu(self, model_arch: MODEL_ARCH, model_size: str) -> SLURM_GPU_TYPE:
        selected_gpu = self._selected_gpu.get()
        if selected_gpu == "smart":
            return model_and_size_to_slurm_gpu_type(model_arch, model_size)
        return selected_gpu


AppSessionKeys = _AppSessionKeys()


# Pagination constants


# endregion


# region Data Requirements
class DataReqCols:
    AvailableOptions = "Available Options"
    Options = "Options"
    CurrentOverride = "Current Override"
    Key = "Key"


class DataReqConsts:
    # Data Requirements filter columns
    DATA_REQS_FILTER_COLUMNS = [
        DataReqCols.AvailableOptions,
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
        DataReqCols.AvailableOptions: [0],
        ParamNames.is_all_correct: [False],
    }


class _DataReqsSessionKeys:
    selected_requirements: SessionKey[set[str]] = SessionKey.with_default("selected_requirements", set())

    @staticmethod
    def select_requirement(key: str) -> SessionKey[bool]:
        return SessionKey.with_default(f"select_{key}", False)

    @staticmethod
    def override_requirement(key: str) -> SessionKey[bool]:
        return SessionKey.with_default(f"override_{key}", False)


DataReqsSessionKeys = _DataReqsSessionKeys()
# endregion


# region Info Flow Plots
class InfoFlowCols:
    pass


class InfoFlowConsts:
    ParamRole = Literal["grid", "column", "row", "line", "fixed"]
    PARAM_ROLES: list[ParamRole] = ["fixed", "grid", "column", "row", "line"]
    DEFAULT_LINE_STYLES = ["-", "--", ":", "-."]
    DEFAULT_PLOT_CONFIG = {
        "confidence_level": 0.95,
        "plot_height": 400,
        "plot_width": 600,
    }


class _InfoFlowSessionKeys:
    param_roles: SessionKey[dict[str, InfoFlowConsts.ParamRole]] = SessionKey.with_default("param_roles", {})

    def param_value(self, param: str) -> SessionKey[str]:
        return SessionKey.with_default(f"value_{param}", "")

    def param_role(self, param: str) -> SessionKey[InfoFlowConsts.ParamRole]:
        return SessionKey.with_default(f"role_{param}", "fixed")


InfoFlowSessionKeys = _InfoFlowSessionKeys()

# endregion


# region Heatmap Creation


class HeatmapCols:
    PROMPT_COUNT = "Prompt Count"
    SELECTED_PROMPT = "Selected Prompt"
    MODEL_CORRECT = "Model Correct"


class HeatmapConsts:
    MINIMUM_COMBINATIONS_FOR_FILTERING = 30


class ModelFilterOption(StrEnum):
    CORRECT = "correct"
    ANY = "any"
    INCORRECT = "incorrect"


class _HeatmapSessionKeys:
    model_filters: SessionKey[dict[str, list[str]]] = SessionKey.with_default("model_filters", {})
    # selected_prompts: SessionKey[set[str]] = SessionKey.with_default("selected_prompts", set())
    selected_prompts: dict[int, SessionKey[bool]] = {}
    show_combination: SessionKey[Optional[int]] = SessionKey.with_default("show_combination", None)
    selected_combination: SessionKey[dict[str, dict[str, list[int]]]] = SessionKey("selected_combination")

    def selected_prompt(self, prompt_idx: Any) -> SessionKey[bool]:
        if prompt_idx not in self.selected_prompts:
            self.selected_prompts[prompt_idx] = SessionKey.with_default(f"select_prompt_{prompt_idx}", False)
        return self.selected_prompts[prompt_idx]

    def get_selected_combination_row(self) -> Optional[int]:
        if self.selected_combination.get() is None:
            return None
        selected_rows = self.selected_combination.get()["selection"]["rows"]
        assert len(selected_rows) <= 1
        if len(selected_rows) == 0:
            return None
        return selected_rows[0]

    def get_selected_prompts(self) -> set[int]:
        return {prompt_idx for prompt_idx, key in self.selected_prompts.items() if key.get()}

    def select_prompt(self, prompt_idx: Any) -> SessionKey[bool]:
        if prompt_idx not in self.selected_prompts:
            self.selected_prompts[prompt_idx] = SessionKey.with_default(f"select_prompt_{prompt_idx}", False)
        return self.selected_prompts[prompt_idx]


HeatmapSessionKeys = _HeatmapSessionKeys()

# endregion
