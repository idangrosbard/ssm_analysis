from enum import StrEnum
from typing import Literal, cast

from src.utils.types_utils import class_values, literal_guard


class DATASETS(StrEnum):
    COUNTER_FACT = "counter_fact"


class BASE_CONFIG_HP_COLS(StrEnum):
    experiment_name = "experiment_name"
    model_arch = "model_arch"
    model_size = "model_size"
    code_version = "code_version"
    dataset_name = "dataset_name"
    prompt_filteration = "prompt_filteration"


class WINDOW_SIZE_HP_COLS(StrEnum):
    window_size = "window_size"


class INFO_FLOW_HP_COLS(StrEnum):
    window_size = WINDOW_SIZE_HP_COLS.window_size
    source = "source"
    feature_category = "feature_category"
    target = "target"


class HEATMAP_HP_COLS(StrEnum):
    window_size = WINDOW_SIZE_HP_COLS.window_size


class EXPERIMENT_NAMES(StrEnum):
    EVALUATE_MODEL = "evaluate_model"
    INFO_FLOW = "info_flow"
    HEATMAP = "heatmap"
    FULL_PIPELINE = "full_pipeline"

    @staticmethod
    def get_hp_cols(col: "EXPERIMENT_NAMES") -> list[str]:
        base_cols = list(BASE_CONFIG_HP_COLS)
        match col:
            case EXPERIMENT_NAMES.INFO_FLOW:
                return base_cols + class_values(INFO_FLOW_HP_COLS)
            case EXPERIMENT_NAMES.HEATMAP:
                return base_cols + class_values(HEATMAP_HP_COLS)
            case EXPERIMENT_NAMES.EVALUATE_MODEL:
                return cast(list[str], base_cols)
            case _:
                raise ValueError(f"Experiment name {col} is not implemented")

    @classmethod
    def get_experiment_name_by_str(cls, name: str) -> "EXPERIMENT_NAMES":
        match name:
            case "evaluate":
                return cls.EVALUATE_MODEL
            case "info_flow":
                return cls.INFO_FLOW
            case "heatmap":
                return cls.HEATMAP
            case "full_pipeline":
                return cls.FULL_PIPELINE
            case _:
                raise ValueError(f"Experiment name {name} is not implemented")


class COLS:
    # Preprocessing
    ORIGINAL_IDX: Literal["original_idx"] = "original_idx"
    SPLIT: Literal["split"] = "split"

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


class InfoFlowCols:
    hit: Literal["hit"] = literal_guard(COLS.INFO_FLOW.HIT, "hit")
    diffs: Literal["diffs"] = literal_guard(COLS.INFO_FLOW.DIFFS, "diffs")
    true_probs: Literal["true_probs"] = literal_guard(COLS.INFO_FLOW.TRUE_PROBS, "true_probs")


class DataReqCols(StrEnum):
    experiment_name = BASE_CONFIG_HP_COLS.experiment_name
    model_arch = BASE_CONFIG_HP_COLS.model_arch
    model_size = BASE_CONFIG_HP_COLS.model_size
    # prompt_filteration = BASE_CONFIG_HP_COLS.prompt_filteration
    window_size = WINDOW_SIZE_HP_COLS.window_size
    source = INFO_FLOW_HP_COLS.source
    feature_category = INFO_FLOW_HP_COLS.feature_category
    target = INFO_FLOW_HP_COLS.target

    @classmethod
    def get_cols_by_experiment_name(cls, experiment_name: EXPERIMENT_NAMES) -> list[str]:
        this_cols = set(class_values(cls))
        return [col for col in experiment_name.get_hp_cols(experiment_name) if col in this_cols]


class ResultBankParamNames(StrEnum):
    experiment_name = DataReqCols.experiment_name
    model_arch = DataReqCols.model_arch
    model_size = DataReqCols.model_size
    window_size = DataReqCols.window_size
    source = DataReqCols.source
    feature_category = DataReqCols.feature_category
    target = DataReqCols.target
    code_version = BASE_CONFIG_HP_COLS.code_version
    path = "path"


class ExperimentHyperParams(StrEnum):
    model_arch = DataReqCols.model_arch
    model_size = DataReqCols.model_size
    model_arch_and_size = "model_arch_and_size"
    window_size = DataReqCols.window_size
    source = DataReqCols.source
    feature_category = DataReqCols.feature_category
    target = DataReqCols.target
    prompt_idx = "prompt_idx"


class HeatmapCols:
    PROMPT_COUNT = "Prompt Count"
    SELECTED_PROMPT = "Selected Prompt"
    MODEL_CORRECT = "Model Correct"


class PlotType(StrEnum):
    ARCHITECTURE_KNOCKOUT = "ARCHITECTURE_KNOCKOUT"
    HEATMAP = "heatmap"


class FinalPlotsPlanOrientation(StrEnum):
    grids = "grids"
    rows = "rows"
    cols = "cols"
    lines = "lines"


def map_final_plots_plan_orientation_to_options(orientation: FinalPlotsPlanOrientation) -> "PlotPlanOptionCols":
    return cast("PlotPlanOptionCols", f"{orientation}_options")


class PlotPlanCols(StrEnum):
    plot_id = "plot_id"
    TITLE = "title"
    description = "description"
    plot_type = "plot_type"
    is_appendix = "is_appendix"
    order = "order"
    experiment_name = "experiment_name"
    rows = FinalPlotsPlanOrientation.rows
    cols = FinalPlotsPlanOrientation.cols
    grids = FinalPlotsPlanOrientation.grids
    lines = FinalPlotsPlanOrientation.lines
    rows_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.rows)
    cols_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.cols)
    grids_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.grids)
    lines_options = map_final_plots_plan_orientation_to_options(FinalPlotsPlanOrientation.lines)


class PlotPlanOptionCols(StrEnum):
    rows_options = PlotPlanCols.rows_options
    cols_options = PlotPlanCols.cols_options
    grids_options = PlotPlanCols.grids_options
    lines_options = PlotPlanCols.lines_options


class SummarizedDataFulfilledReqsCols:
    AvailableOptions = "Available Options"
    Options = "Options"
    Key = "Key"
    filters_requested = "filters_requested"


class ModelCombinationCols(StrEnum):
    correct_models = "correct_models"
    incorrect_models = "incorrect_models"
    prompts = "prompts"
    chosen_prompt = "chosen_prompt"


class SlurmStatus(StrEnum):
    NOT_SUBMITTED = "NOT_SUBMITTED"
    RUNNING = "RUNNING"
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def scheduled(self) -> bool:
        return self in [self.PENDING, self.RUNNING]


class RunningHistoryCols(StrEnum):
    run_id = "run_id"
    git_commit_hash = "git_commit_hash"
