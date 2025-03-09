from enum import StrEnum

from src.utils.types_utils import class_values


class EXPERIMENT_NAMES(StrEnum):
    EVALUATE_MODEL = "evaluate"
    INFO_FLOW = "info_flow"
    HEATMAP = "heatmap"
    FULL_PIPELINE = "full_pipeline"


class COLUMNS:
    # Preprocessing
    ORIGINAL_IDX = "original_idx"
    SPLIT = "split"

    # Counter Fact
    class COUNTER_FACT_COLS(StrEnum):
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

    PROMPT_DATA_COLS = class_values(COUNTER_FACT_COLS)
    PROMPT = COUNTER_FACT_COLS.PROMPT
    TARGET_TRUE = COUNTER_FACT_COLS.TARGET_TRUE
    TARGET_FALSE = COUNTER_FACT_COLS.TARGET_FALSE
    SUBJECT = COUNTER_FACT_COLS.SUBJECT
    TARGET_FALSE_ID = COUNTER_FACT_COLS.TARGET_FALSE_ID
    RELATION = COUNTER_FACT_COLS.RELATION
    RELATION_ID = COUNTER_FACT_COLS.RELATION_ID
    RELATION_PREFIX = "relation_prefix"
    RELATION_SUFFIX = "relation_suffix"
    RELATION_ID = "relation_id"
    TARGET_TRUE_ID = "target_true_id"

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


class HeatmapCols:
    PROMPT_COUNT = "Prompt Count"
    SELECTED_PROMPT = "Selected Prompt"
    MODEL_CORRECT = "Model Correct"
