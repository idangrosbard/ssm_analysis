from enum import StrEnum
from typing import Literal, Union

from src.core.consts import GRAPHS_ORDER, model_and_size_to_slurm_gpu_type
from src.core.names import COLS, ResultBankParamNames
from src.core.types import MODEL_ARCH_AND_SIZE, TVariationName, TWindowSize
from src.utils.infra.slurm import SLURM_GPU_TYPE
from src.utils.streamlit.helpers.session_keys import SessionKeysBase
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor


# region Global App constants
class AppCols:
    pass


class GLOBAL_APP_CONSTS:
    DEFAULT_VARIATION = TVariationName("v3")
    DEFAULT_WINDOW_SIZE = TWindowSize(9)
    MODELS_COMBINATIONS = list(GRAPHS_ORDER.keys())
    PROMPT_RELATED_COLUMNS = [
        COLS.COUNTER_FACT.PROMPT,
        COLS.COUNTER_FACT.TARGET_TRUE,
        COLS.COUNTER_FACT.TARGET_FALSE,
        COLS.COUNTER_FACT.SUBJECT,
        COLS.COUNTER_FACT.TARGET_FALSE_ID,
        COLS.COUNTER_FACT.RELATION,
    ]

    MODEL_EVALS_COLUMNS = [
        COLS.ORIGINAL_IDX,
        COLS.COUNTER_FACT.PROMPT,
        COLS.COUNTER_FACT.TARGET_TRUE,
        COLS.EVALUATE_MODEL.TARGET_PROBS,
        COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUT_CONFIDENCE,
        COLS.EVALUATE_MODEL.MODEL_CORRECT,
        COLS.EVALUATE_MODEL.MODEL_OUTPUT,
        COLS.EVALUATE_MODEL.TARGET_RANK,
        COLS.EVALUATE_MODEL.MODEL_GENERATION,
        COLS.EVALUATE_MODEL.TARGET_TOKENS,
        COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS,
    ]

    class PaginationConfig:
        RESULTS_BANK = {"default_page_size": 20}
        DATA_REQS = {"default_page_size": 20}
        COMBINATIONS = {"default_page_size": 10}
        PROMPTS = {"default_page_size": 10}


class _AppSessionKeys(SessionKeysBase["_AppSessionKeys"]):
    # Each descriptor creates a SessionKey with the class name prefix
    variation = SessionKeyDescriptor[TVariationName](GLOBAL_APP_CONSTS.DEFAULT_VARIATION)
    _selected_gpu = SessionKeyDescriptor[Union[SLURM_GPU_TYPE, Literal["smart"]]]("smart")
    window_size = SessionKeyDescriptor[TWindowSize](GLOBAL_APP_CONSTS.DEFAULT_WINDOW_SIZE)

    def get_selected_gpu(self, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> SLURM_GPU_TYPE:
        selected_gpu = self._selected_gpu.value
        if selected_gpu == "smart":
            return model_and_size_to_slurm_gpu_type(model_arch_and_size)
        return selected_gpu


AppSessionKeys = _AppSessionKeys()


# Pagination constants


# endregion


# region Data Requirements
class SummarizedDataFulfilledReqsCols:
    AvailableOptions = "Available Options"
    Options = "Options"
    CurrentOverride = "Current Override"
    Key = "Key"


class DataReqConsts:
    # Data Requirements filter columns
    DATA_REQS_FILTER_COLUMNS = [
        SummarizedDataFulfilledReqsCols.AvailableOptions,
        ResultBankParamNames.experiment_name,
        ResultBankParamNames.model_arch,
        ResultBankParamNames.model_size,
        ResultBankParamNames.window_size,
        # ResultBankParamNames.is_all_correct,
        ResultBankParamNames.source,
        ResultBankParamNames.target,
        ResultBankParamNames.feature_category,
        ResultBankParamNames.prompt_idx,
    ]

    DATA_REQS_DEFAULT_FILTER_VALUES = {
        SummarizedDataFulfilledReqsCols.AvailableOptions: [0],
        ResultBankParamNames.is_all_correct: [False],
    }


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


# endregion


# region Heatmap Creation


class HeatmapConsts:
    MINIMUM_COMBINATIONS_FOR_FILTERING = 30


class ModelFilterOption(StrEnum):
    CORRECT = "correct"
    ANY = "any"
    INCORRECT = "incorrect"


# endregion
