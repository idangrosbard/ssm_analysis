from typing import Tuple as PyTuple  # Rename to avoid conflict
from typing import cast

import pandas as pd
import streamlit as st
from streamlit import cache_data as st_cache_data

from src.consts import COLUMNS, GRAPHS_ORDER, MODEL_ARCH, get_model_cat_size
from src.experiments.evaluate_model import EvaluateModelConfig
from src.experiments.heatmap import HeatmapConfig
from src.final_plots.app.app_consts import ReqMetadataColumns
from src.final_plots.app.utils import (
    cache_data,
    format_path_for_display,
    load_experiment_data,
)
from src.final_plots.data_reqs import (
    get_data_fullfment_options,
)
from src.final_plots.results_bank import (
    ParamNames,
    get_results_bank,
)
from src.types import MODEL_SIZE_CAT

# Constants
PROMPT_RELATED_COLUMNS = [
    COLUMNS.PROMPT,
    COLUMNS.TARGET_TRUE,
    COLUMNS.TARGET_FALSE,
    COLUMNS.SUBJECT,
    COLUMNS.TARGET_FALSE_ID,
    COLUMNS.RELATION,
]

DEFAULT_VARIATION = "v3"
DEFAULT_WINDOW_SIZE = 9


class HeatmapState:
    """Class to hold the state for heatmap creation."""

    def __init__(self):
        self.models_evaluations: dict[PyTuple[MODEL_ARCH, str], pd.DataFrame] = {}
        self.model_combinations: list[PyTuple[MODEL_ARCH, str]] = []

    def initialize(self):
        """Initialize the state with evaluation data."""
        self.models_evaluations = {
            k: v for k, v in load_evaluation_data().items() if get_model_cat_size(k[0], k[1]) != MODEL_SIZE_CAT.HUGE
        }
        self.model_combinations = list(self.models_evaluations.keys())


def get_heatmap_state() -> HeatmapState:
    """Get the heatmap state from session state."""
    if "heatmap_state" not in st.session_state:
        st.session_state.heatmap_state = HeatmapState()
    return st.session_state.heatmap_state


# Results Bank hooks
@cache_data
def load_results() -> pd.DataFrame:
    """Load and process results with caching"""
    results = get_results_bank()
    results_data = []
    for result in results:
        result_dict = {param: getattr(result, param, None) for param in ParamNames}
        result_dict[ParamNames.path] = format_path_for_display(result_dict[ParamNames.path])
        results_data.append(result_dict)

    return pd.DataFrame(results_data)


# Data Requirements hooks
@cache_data
def load_data() -> pd.DataFrame:
    """Load requirements and options data with caching"""
    options = get_data_fullfment_options()

    data = []
    for req, opts in options.items():
        override = st.session_state.overrides.get(str(req))

        row = {
            **{
                param: getattr(req, param, None)
                for param in ParamNames
                if param not in [ParamNames.path, ParamNames.variation]
            },
            ReqMetadataColumns.AvailableOptions: len(opts),
            ReqMetadataColumns.Options: opts,
            ReqMetadataColumns.CurrentOverride: override,
            ReqMetadataColumns.Key: str(req),
        }
        data.append(row)

    return pd.DataFrame(data)


# Info Flow Plots hooks
@cache_data
def load_info_flow_data() -> pd.DataFrame:
    """Load info flow requirements and their fulfillment data"""
    return load_experiment_data("info_flow")


# Heatmap Creation hooks
@st_cache_data
def load_evaluation_data() -> dict[PyTuple[MODEL_ARCH, str], pd.DataFrame]:
    """Load evaluation data for all models with caching"""
    return {
        (model_arch, model_size): EvaluateModelConfig(
            model_arch=model_arch, model_size=model_size, variation=DEFAULT_VARIATION
        ).get_outputs()
        for model_arch, model_size, _ in GRAPHS_ORDER
    }


@st_cache_data
def get_model_combinations_prompts() -> pd.DataFrame:
    """Get all possible model combinations and their corresponding prompts.

    Returns:
        DataFrame with columns for each model combination and the count of prompts
        that match that combination pattern.
    """
    state = get_heatmap_state()
    # Get all prompts
    all_prompts = set(state.models_evaluations[state.model_combinations[0]].index)

    # Create a DataFrame with correctness for each model
    correctness_df = pd.DataFrame(index=sorted(all_prompts))
    for model_arch, model_size in state.model_combinations:
        model_key = f"{model_arch}-{model_size}"
        model_df = state.models_evaluations[(model_arch, model_size)]
        correctness_df[model_key] = [
            model_df.at[idx, COLUMNS.MODEL_CORRECT] if idx in model_df.index else False for idx in correctness_df.index
        ]

    # Generate all possible combinations
    combinations = []

    # Convert to numpy for faster operations
    correctness_matrix = correctness_df.values
    model_names = correctness_df.columns.tolist()

    # For each possible combination of models
    for i in range(1, 2 ** len(state.model_combinations) + 1):
        # Convert number to binary to get combination
        binary = format(i - 1, f"0{len(state.model_combinations)}b")
        active_models = [model_names[j] for j, bit in enumerate(binary) if bit == "1"]

        if not active_models:
            continue

        # Find prompts that are correct for all active models
        mask = correctness_matrix[:, [j for j, bit in enumerate(binary) if bit == "1"]].all(axis=1)
        matching_prompts = correctness_df.index[mask].tolist()

        if matching_prompts:
            combinations.append(
                {
                    "active_models": active_models,
                    "binary_pattern": binary,
                    "prompt_count": len(matching_prompts),
                    "prompts": matching_prompts,
                }
            )

    # Convert to DataFrame and sort by count
    result_df = pd.DataFrame(combinations)
    result_df = result_df.sort_values("prompt_count", ascending=False)
    return result_df


@st_cache_data
def get_merged_evaluations(prompt_idx: int) -> PyTuple[pd.Series, pd.DataFrame]:
    """Get merged evaluations for a specific prompt.

    Args:
        prompt_idx: The prompt index to get evaluations for

    Returns:
        tuple of:
            - Series with prompt-specific data
            - DataFrame with model-specific evaluations merged
    """
    state = get_heatmap_state()
    # Get the prompt data from first model (prompt data is the same for all models)
    first_model_df = state.models_evaluations[state.model_combinations[0]]
    prompt_data = cast(pd.Series, first_model_df.loc[prompt_idx])

    # Create list to hold each model's evaluation
    model_evals = []

    for model_arch, model_size in state.model_combinations:
        model_df = state.models_evaluations[(model_arch, model_size)]
        if prompt_idx not in model_df.index:
            continue

        row = model_df.loc[prompt_idx]

        # Filter out prompt-related columns (they're the same for all models)
        model_specific_data = {col: val for col, val in row.items() if col not in PROMPT_RELATED_COLUMNS}

        model_specific_data["model_arch"] = model_arch
        model_specific_data["model_size"] = model_size

        model_evals.append(model_specific_data)

    return prompt_data, pd.DataFrame(model_evals)


@st_cache_data
def get_models_is_heatmap_available(prompt_idx: int) -> dict[PyTuple[MODEL_ARCH, str], bool]:
    """Check if a model has a heatmap for a given prompt index."""
    state = get_heatmap_state()
    return {
        (model_arch, model_size): HeatmapConfig(
            model_arch=model_arch,
            model_size=model_size,
            window_size=DEFAULT_WINDOW_SIZE,
            prompt_indices_rows=[],
            prompt_original_indices=[prompt_idx],
        )
        .output_heatmap_path(prompt_idx)
        .exists()
        for model_arch, model_size in state.model_combinations
    }
