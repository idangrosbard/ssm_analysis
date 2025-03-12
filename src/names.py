from enum import StrEnum
from typing import cast

from src.types import FinalPlotsPlanOrientation


class EXPERIMENT_NAMES(StrEnum):
    EVALUATE_MODEL = "evaluate"
    INFO_FLOW = "info_flow"
    HEATMAP = "heatmap"
    FULL_PIPELINE = "full_pipeline"


class COLS:
    # Preprocessing
    ORIGINAL_IDX = "original_idx"
    SPLIT = "split"

    # Counter Fact
    class COUNTER_FACT(StrEnum):
        PROMPT = "prompt"
        TARGET_TRUE = "target_true"
        RELATION = "relation"
        SUBJECT = "subject"
        TARGET_FALSE = "target_false"
        RELATION_PREFIX = "relation_prefix"
        RELATION_SUFFIX = "relation_suffix"
        TARGET_TRUE_ID = "target_true_id"
        TARGET_FALSE_ID = "target_false_id"
        RELATION_ID = "relation_id"

    # Evaluate Model
    class EVALUATE_MODEL(StrEnum):
        TARGET_PROBS = "target_probs"
        MODEL_TOP_OUTPUT_CONFIDENCE = "model_top_output_confidence"
        MODEL_CORRECT = "model_correct"
        MODEL_OUTPUT = "model_output"
        TARGET_RANK = "target_rank"
        MODEL_TOP_OUTPUTS = "model_top_outputs"
        MODEL_GENERATION = "model_generation"
        TARGET_TOKENS = "target_tokens"

    # Info Flow
    class INFO_FLOW(StrEnum):
        HIT = "hit"
        TRUE_PROBS = "true_probs"
        DIFFS = "diffs"


class ResultBankParamNames(StrEnum):
    experiment_name = "experiment_name"
    variation = "variation"
    model_arch = "model_arch"
    model_size = "model_size"
    window_size = "window_size"
    is_all_correct = "is_all_correct"
    source = "source"
    feature_category = "feature_category"
    target = "target"
    prompt_idx = "prompt_idx"
    path = "path"


class ExperimentHyperParams(StrEnum):
    model_arch = ResultBankParamNames.model_arch
    model_size = ResultBankParamNames.model_size
    model_arch_and_size = "model_arch_and_size"
    window_size = ResultBankParamNames.window_size
    source = ResultBankParamNames.source
    feature_category = ResultBankParamNames.feature_category
    target = ResultBankParamNames.target
    prompt_idx = ResultBankParamNames.prompt_idx


class HeatmapCols:
    PROMPT_COUNT = "Prompt Count"
    SELECTED_PROMPT = "Selected Prompt"
    MODEL_CORRECT = "Model Correct"


class PlotType(StrEnum):
    ARCHITECTURE_KNOCKOUT = "architecture_knockout"
    MODEL_SIZE_KNOCKOUT = "model_size_knockout"
    WINDOW_SIZE_KNOCKOUT = "window_size_knockout"
    FEATURE_KNOCKOUT = "feature_knockout"
    SHARED_KNOCKOUT = "shared_knockout"
    HEATMAP = "heatmap"


def map_final_plots_plan_orientation_to_options(orientation: FinalPlotsPlanOrientation) -> "PlotPlanOptionCols":
    return cast("PlotPlanOptionCols", f"{orientation}_options")


class PlotPlanCols(StrEnum):
    TITLE = "title"
    description = "description"
    plot_type = "plot_type"
    is_appendix = "is_appendix"
    order = "order"
    experiment_name = "experiment_name"
    rows = FinalPlotsPlanOrientation.ROWS
    cols = FinalPlotsPlanOrientation.COLS
    grids = FinalPlotsPlanOrientation.GRIDS
    lines = FinalPlotsPlanOrientation.LINES
    output_path = "output_path"
    rows_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.ROWS)
    cols_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.COLS)
    grids_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.GRIDS)
    lines_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.LINES)


class PlotPlanOptionCols(StrEnum):
    rows_options = PlotPlanCols.rows_options
    cols_options = PlotPlanCols.cols_options
    grids_options = PlotPlanCols.grids_options
    lines_options = PlotPlanCols.lines_options
