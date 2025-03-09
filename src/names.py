from enum import StrEnum


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


class HeatmapCols:
    PROMPT_COUNT = "Prompt Count"
    SELECTED_PROMPT = "Selected Prompt"
    MODEL_CORRECT = "Model Correct"
