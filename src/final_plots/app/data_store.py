from typing import Literal, cast
from typing import Tuple as PyTuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from streamlit import cache_data as st_cache_data

from src.consts import COLUMNS, GRAPHS_ORDER, MODEL_ARCH, to_model_name
from src.experiments.evaluate_model import EvaluateModelConfig
from src.experiments.heatmap import HeatmapConfig
from src.final_plots.app.app_consts import DataReqsSessionKeys, ReqMetadataColumns
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
from src.plots.info_flow_confidence import PlotMetadata, create_confidence_plot, load_window_outputs

# Constants
PROMPT_RELATED_COLUMNS = [
    COLUMNS.PROMPT,
    COLUMNS.TARGET_TRUE,
    COLUMNS.TARGET_FALSE,
    COLUMNS.SUBJECT,
    COLUMNS.TARGET_FALSE_ID,
    COLUMNS.RELATION,
]


@st_cache_data
def get_model_evaluations(variation: str) -> dict[PyTuple[MODEL_ARCH, str], pd.DataFrame]:
    """Get the model evaluations from session state."""
    return load_evaluation_data(variation)


@st_cache_data
def get_model_combinations(variation: str) -> list[PyTuple[MODEL_ARCH, str]]:
    """Get the model combinations from session state."""
    return list(get_model_evaluations(variation).keys())


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
def load_evaluation_data(variation: str) -> dict[PyTuple[MODEL_ARCH, str], pd.DataFrame]:
    """Load evaluation data for all models with caching"""
    return {
        (model_arch, model_size): EvaluateModelConfig(
            model_arch=model_arch, model_size=model_size, variation=variation
        ).get_outputs()
        for model_arch, model_size, _ in GRAPHS_ORDER
    }


@st_cache_data
def get_model_combinations_prompts(variation: str) -> pd.DataFrame:
    """Get all possible model combinations and their corresponding prompts.
    Each combination specifies which models should be correct and which should be incorrect.

    Returns:
        DataFrame with columns for each model combination and the count of prompts
        that match that combination pattern.
    """
    model_evaluations = get_model_evaluations(variation)
    model_combinations = get_model_combinations(variation)
    # Get all prompts
    all_prompts = set(model_evaluations[model_combinations[0]].index)

    # Create a DataFrame with correctness for each model
    correctness_df = pd.DataFrame(index=sorted(all_prompts))
    for model_arch, model_size in model_combinations:
        model_name = to_model_name(model_arch, model_size)
        model_df = model_evaluations[(model_arch, model_size)]
        correctness_df[model_name] = [
            model_df.at[idx, COLUMNS.MODEL_CORRECT] if idx in model_df.index else False for idx in correctness_df.index
        ]

    # Generate all possible combinations
    combinations = []

    # Convert to numpy for faster operations
    correctness_matrix = correctness_df.values
    model_names = correctness_df.columns.tolist()

    # For each possible combination of models being correct/incorrect
    for i in range(2 ** len(model_combinations)):
        # Convert number to binary to get combination of correct models
        binary = format(i, f"0{len(model_combinations)}b")

        # Get models that should be correct and incorrect
        correct_models = [model_names[j] for j, bit in enumerate(binary) if bit == "1"]
        incorrect_models = [model_names[j] for j, bit in enumerate(binary) if bit == "0"]

        # Find prompts that are correct for all correct_models AND incorrect for all incorrect_models
        correct_mask = correctness_matrix[:, [j for j, bit in enumerate(binary) if bit == "1"]].all(axis=1)
        incorrect_mask = ~correctness_matrix[:, [j for j, bit in enumerate(binary) if bit == "0"]].any(axis=1)

        # Combined mask for prompts meeting both conditions
        mask = correct_mask & incorrect_mask
        matching_prompts = correctness_df.index[mask].tolist()

        if matching_prompts:
            combinations.append(
                {
                    "correct_models": correct_models,
                    "incorrect_models": incorrect_models,
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
def get_merged_evaluations(prompt_idx: int, variation: str) -> PyTuple[pd.Series, pd.DataFrame]:
    """Get merged evaluations for a specific prompt.

    Args:
        prompt_idx: The prompt index to get evaluations for

    Returns:
        tuple of:
            - Series with prompt-specific data
            - DataFrame with model-specific evaluations merged
    """
    model_evaluations = get_model_evaluations(variation)
    model_combinations = get_model_combinations(variation)
    # Get the prompt data from first model (prompt data is the same for all models)
    first_model_df = model_evaluations[model_combinations[0]]
    prompt_data = cast(pd.Series, first_model_df.loc[prompt_idx])

    # Create list to hold each model's evaluation
    model_evals = []

    for model_arch, model_size in model_combinations:
        model_df = model_evaluations[(model_arch, model_size)]
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
def get_models_is_heatmap_available(
    prompt_idx: int, variation: str, window_size: int
) -> dict[PyTuple[MODEL_ARCH, str], bool]:
    """Check if a model has a heatmap for a given prompt index."""
    model_combinations = get_model_combinations(variation)
    return {
        (model_arch, model_size): HeatmapConfig(
            model_arch=model_arch,
            model_size=model_size,
            window_size=window_size,
            variation=variation,
            prompt_indices_rows=[],
            prompt_original_indices=[prompt_idx],
        )
        .output_heatmap_path(prompt_idx)
        .exists()
        for model_arch, model_size in model_combinations
    }


def empty_selected_requirements() -> None:
    """Empty the selected requirements set in session state."""
    DataReqsSessionKeys.selected_requirements.get().clear()


def create_info_flow_plots(
    df: pd.DataFrame,
    grid_param: str,
    row_param: str,
    col_param: str,
    line_param: str,
    plot_width: int,
    plot_height: int,
    confidence_level: float,
    custom_colors: dict[str, str],
    custom_styles: dict[str, str],
) -> tuple[dict[str, Figure], list[str]]:
    """Create info flow plots for the given data.

    Args:
        df: The DataFrame containing the data
        grid_param: The parameter used for grid
        row_param: The parameter used for rows
        col_param: The parameter used for columns
        line_param: The parameter used for lines
        plot_width: Width of each plot
        plot_height: Height of each plot
        confidence_level: Confidence level for error bars
        custom_colors: Custom colors for each line
        custom_styles: Custom line styles for each line

    Returns:
        A tuple of:
            - Dictionary mapping plot keys to figures
            - List of error messages for failed plots
    """
    grid_values = sorted(df[grid_param].unique())
    plots = {}
    failed_plots = []

    for grid_val in grid_values:
        try:
            grid_df = df[df[grid_param] == grid_val]

            # Get unique values for rows and columns
            row_values = sorted(grid_df[row_param].unique())
            col_values = sorted(grid_df[col_param].unique())

            # Create figure with subplots
            fig, axes = plt.subplots(
                len(row_values), len(col_values), figsize=(plot_width / 100, plot_height / 100), squeeze=False
            )

            # Create plots for each combination
            for i, row_val in enumerate(row_values):
                for j, col_val in enumerate(col_values):
                    subplot_df = grid_df[(grid_df[row_param] == row_val) & (grid_df[col_param] == col_val)]

                    if subplot_df.empty:
                        continue

                    # Group by line parameter and create plot
                    targets_window_outputs = {}
                    paths = []
                    load_errors = []

                    for _, row in subplot_df.iterrows():
                        line_val = row[line_param]
                        if pd.isna(line_val):
                            continue

                        try:
                            window_outputs = load_window_outputs(row["data_path"])
                            targets_window_outputs[line_val] = window_outputs
                            paths.append(format_path_for_display(row["data_path"]))
                        except Exception as e:
                            load_errors.append(f"Error loading data for {line_val}: {e}")
                            continue

                    if not targets_window_outputs:
                        if load_errors:
                            failed_plots.append(
                                f"Failed to load any data for {grid_param}={grid_val}, "
                                f"{row_param}={row_val}, {col_param}={col_val}:\n"
                                + "\n".join(f"  - {err}" for err in load_errors)
                            )
                        continue

                    plots_meta_data: dict[Literal["acc", "diff"], PlotMetadata] = {
                        "acc": {
                            "title": "Accuracy",
                            "ylabel": "% accuracy",
                            "ylabel_loc": "center",
                            "axhline_value": 100.0,
                            "ylim": (60.0, 105.0),
                        },
                        "diff": {
                            "title": "Normalized change in prediction probability",
                            "ylabel": "% probability change",
                            "ylabel_loc": "top",
                            "axhline_value": 0.0,
                            "ylim": (-50.0, 50.0),
                        },
                    }

                    try:
                        fig = create_confidence_plot(
                            targets_window_outputs=targets_window_outputs,
                            confidence_level=confidence_level,
                            title=f"{grid_param}={grid_val}\n{row_param}={row_val}, {col_param}={col_val}",
                            plots_meta_data=plots_meta_data,
                            # colors=custom_colors,
                            # line_styles=custom_styles,
                        )

                        plots[f"{grid_val}_{row_val}_{col_val}"] = fig

                    except Exception:
                        failed_plots.append(
                            "Failed to create plot for one of the paths: "
                            # f"{paths}\n{row['data_path']}:\nError: {str(e)}"
                        )
                        continue

        except Exception as e:
            failed_plots.append(f"Failed to process grid {grid_param}={grid_val}: {str(e)}")
            continue

    return plots, failed_plots


@st_cache_data
def get_models_remaining_prompts(
    model_combinations: list[tuple[MODEL_ARCH, str]],
    window_size: int,
    variation: str,
    prompt_original_indices: list[int],
) -> dict[tuple[MODEL_ARCH, str], HeatmapConfig]:
    """Get the remaining prompts for each model."""
    res = {}
    for model_arch, model_size in model_combinations:
        config = HeatmapConfig(
            model_arch=model_arch,
            model_size=model_size,
            window_size=window_size,
            variation=variation,
            prompt_original_indices=prompt_original_indices,
        )
        remaining_prompt_original_indices = config.get_remaining_prompt_original_indices()
        if remaining_prompt_original_indices:
            config.prompt_original_indices = remaining_prompt_original_indices
            res[(model_arch, model_size)] = config
    return res
