from enum import StrEnum
from typing import Any, ClassVar, Generic, Literal, Type, TypeVar, Union, cast

from src.consts import COLUMNS, GRAPHS_ORDER, model_and_size_to_slurm_gpu_type
from src.final_plots.results_bank import ParamNames
from src.types import MODEL_ARCH_AND_SIZE, SLURM_GPU_TYPE
from src.utils.streamlit_utils import SessionKey, SessionKeyDescriptor


# region Global App constants
class AppCols:
    pass


class GLOBAL_APP_CONSTS:
    DEFAULT_VARIATION = "v3"
    DEFAULT_WINDOW_SIZE = 9
    MODELS_COMBINATIONS = list(GRAPHS_ORDER.keys())
    PROMPT_RELATED_COLUMNS = [
        COLUMNS.PROMPT,
        COLUMNS.TARGET_TRUE,
        COLUMNS.TARGET_FALSE,
        COLUMNS.SUBJECT,
        COLUMNS.TARGET_FALSE_ID,
        COLUMNS.RELATION,
    ]

    class PaginationConfig:
        RESULTS_BANK = {"default_page_size": 20, "key_prefix": "results_bank_"}
        DATA_REQS = {"default_page_size": 10, "key_prefix": "data_reqs_"}
        COMBINATIONS = {"default_page_size": 10, "key_prefix": "combinations_"}
        PROMPTS = {"default_page_size": 10, "key_prefix": "prompts_"}


T = TypeVar("T", bound="SessionKeysBase[Any]")


class SessionKeysBase(Generic[T]):
    """Base class for session key containers that ensures singleton pattern."""

    _instance: ClassVar[dict[Type[Any], Any]] = {}

    def __new__(cls) -> T:
        if cls not in cls._instance:
            cls._instance[cls] = super().__new__(cls)
        return cast(T, cls._instance[cls])


class _AppSessionKeys(SessionKeysBase["_AppSessionKeys"]):
    # Each descriptor creates a SessionKey with the class name prefix
    variation = SessionKeyDescriptor[str](GLOBAL_APP_CONSTS.DEFAULT_VARIATION)
    _selected_gpu = SessionKeyDescriptor[Union[SLURM_GPU_TYPE, Literal["smart"]]]("smart")
    window_size = SessionKeyDescriptor[int](GLOBAL_APP_CONSTS.DEFAULT_WINDOW_SIZE)

    def get_selected_gpu(self, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> SLURM_GPU_TYPE:
        selected_gpu = self._selected_gpu.value
        if selected_gpu == "smart":
            return model_and_size_to_slurm_gpu_type(model_arch_and_size)
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


class _DataReqsSessionKeys(SessionKeysBase["_DataReqsSessionKeys"]):
    selected_requirements = SessionKeyDescriptor[set[str]](set())

    @staticmethod
    def select_requirement(key: str) -> SessionKey[bool]:
        return SessionKey(f"datareqs_select_{key}", False)

    @staticmethod
    def override_requirement(key: str) -> SessionKey[bool]:
        return SessionKey(f"datareqs_override_{key}", False)


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


class _InfoFlowSessionKeys(SessionKeysBase["_InfoFlowSessionKeys"]):
    param_roles = SessionKeyDescriptor[dict[str, InfoFlowConsts.ParamRole]]({})

    def param_value(self, param: str) -> SessionKey[str]:
        return SessionKey(f"infoflow_value_{param}", "")

    def param_role(self, param: str) -> SessionKey[InfoFlowConsts.ParamRole]:
        return SessionKey(f"infoflow_role_{param}", "fixed")


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


class _HeatmapSessionKeys(SessionKeysBase["_HeatmapSessionKeys"]):
    pass


HeatmapSessionKeys = _HeatmapSessionKeys()

# endregion
