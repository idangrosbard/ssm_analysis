from enum import Enum, auto
from typing import Literal, Union, assert_never

from src.app.texts import (
    DATA_REQUIREMENTS_TEXTS,
    FINAL_PLOTS_TEXTS,
    HEATMAP_TEXTS,
    HOME_TEXTS,
    INFO_FLOW_ANALYSIS_TEXTS,
    RESULTS_BANK_TEXTS,
)
from src.core.consts import GRAPHS_ORDER, model_and_size_to_slurm_gpu_type
from src.core.names import COLS, ResultBankParamNames, SummarizedDataFulfilledReqsCols
from src.core.types import MODEL_ARCH_AND_SIZE, TVariationName, TWindowSize
from src.utils.infra.slurm import SLURM_GPU_TYPE
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor, SessionKeysBase

# region Global App constants


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


# region Data Requirements


class DataReqConsts:
    # Data Requirements filter columns
    DATA_REQS_FILTER_COLUMNS = [
        SummarizedDataFulfilledReqsCols.AvailableOptions,
        ResultBankParamNames.experiment_name,
        ResultBankParamNames.model_arch,
        ResultBankParamNames.model_size,
        ResultBankParamNames.window_size,
        ResultBankParamNames.source,
        ResultBankParamNames.target,
        ResultBankParamNames.feature_category,
        ResultBankParamNames.prompt_idx,
    ]


# endregion


# region Heatmap Creation


class HeatmapConsts:
    MINIMUM_COMBINATIONS_FOR_FILTERING = 30


# endregion


class PAGE_ORDER(Enum):
    HOME = auto()
    RESULTS_BANK = auto()
    DATA_REQUIREMENTS = auto()
    HEATMAP = auto()
    INFO_FLOW_ANALYSIS = auto()
    FINAL_PLOTS = auto()

    @property
    def page_details(self) -> tuple[str, str]:
        match self:
            case PAGE_ORDER.HOME:
                return (HOME_TEXTS.title, HOME_TEXTS.icon)
            case PAGE_ORDER.HEATMAP:
                return (HEATMAP_TEXTS.title, HEATMAP_TEXTS.icon)
            case PAGE_ORDER.RESULTS_BANK:
                return (RESULTS_BANK_TEXTS.title, RESULTS_BANK_TEXTS.icon)
            case PAGE_ORDER.DATA_REQUIREMENTS:
                return (DATA_REQUIREMENTS_TEXTS.title, DATA_REQUIREMENTS_TEXTS.icon)
            case PAGE_ORDER.FINAL_PLOTS:
                return (FINAL_PLOTS_TEXTS.title, FINAL_PLOTS_TEXTS.icon)
            case PAGE_ORDER.INFO_FLOW_ANALYSIS:
                return (INFO_FLOW_ANALYSIS_TEXTS.title, INFO_FLOW_ANALYSIS_TEXTS.icon)
            case _:
                assert_never(self)

    @property
    def icon(self) -> str:
        return self.page_details[1]

    @property
    def title(self) -> str:
        return self.page_details[0]
