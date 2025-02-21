from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pygwalker.api.streamlit import StreamlitRenderer
from streamlit import cache_data, cache_resource

from src.experiments.heatmap import HeatmapConfig
from src.final_plots.app.app_consts import (
    GLOBAL_APP_CONSTS,
    DataReqCols,
    DataReqsSessionKeys,
)
from src.final_plots.app.utils import (
    format_path_for_display,
    load_experiment_data,
    my_cache_data,
)
from src.final_plots.data_reqs import (
    ModelCombination,
    get_data_fullfment_options,
    get_model_combinations_prompts,
    get_model_evaluations,
    load_data_fulfilled_overides,
)
from src.final_plots.results_bank import (
    ParamNames,
    get_experiment_results_bank,
)
from src.plots.info_flow_confidence import PlotMetadata, create_confidence_plot, load_window_outputs
from src.types import MODEL_ARCH_AND_SIZE


# Constants
@cache_data
def load_model_evaluations(variation: str) -> dict[MODEL_ARCH_AND_SIZE, pd.DataFrame]:
    """Load evaluation data for all models with caching"""
    return get_model_evaluations(variation, GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)


@cache_resource
def merge_model_evaluations_streamlit_rendered(variation: str) -> StreamlitRenderer:
    """Load evaluation data for all models with caching"""
    return StreamlitRenderer(
        pd.concat(
            [
                df.assign(model_arch=key.arch, model_size=key.size)
                for key, df in load_model_evaluations(variation).items()
            ]
        ),
        spec=f"model_evals_{variation}.csv",
        spec_io_mode="rw",
    )


# Results Bank hooks
@my_cache_data
def load_experiment_results() -> pd.DataFrame:
    """Load and process results with caching"""
    results = get_experiment_results_bank()
    results_data = []
    for result in results:
        result_dict = {param: getattr(result, param, None) for param in ParamNames}
        result_dict[ParamNames.path] = format_path_for_display(result_dict[ParamNames.path])
        results_data.append(result_dict)

    return pd.DataFrame(results_data)


# Data Requirements hooks
@my_cache_data
def load_data() -> pd.DataFrame:
    """Load requirements and options data with caching"""
    options = get_data_fullfment_options()
    data_fulfilled_overides = load_data_fulfilled_overides()

    data = []
    for req, opts in options.items():
        override = data_fulfilled_overides.get(req)

        row = {
            **{
                param: getattr(req, param, None)
                for param in ParamNames
                if param not in [ParamNames.path, ParamNames.variation]
            },
            DataReqCols.AvailableOptions: len(opts),
            DataReqCols.Options: opts,
            DataReqCols.CurrentOverride: override,
            DataReqCols.Key: str(req),
        }
        data.append(row)

    return pd.DataFrame(data)


# Info Flow Plots hooks
@my_cache_data
def load_info_flow_data() -> pd.DataFrame:
    """Load info flow requirements and their fulfillment data"""
    return load_experiment_data("info_flow")


@cache_data
def get_merged_evaluations(prompt_idx: int, variation: str) -> pd.DataFrame:
    """Get merged evaluations for a specific prompt.

    Args:
        prompt_idx: The prompt index to get evaluations for

    Returns:
        tuple of:
            - DataFrame with model-specific evaluations merged
    """
    model_evaluations = load_model_evaluations(variation)

    # Create list to hold each model's evaluation
    model_evals = []

    for model_combination in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
        model_df = model_evaluations[model_combination]
        if prompt_idx not in model_df.index:
            continue

        row = model_df.loc[prompt_idx]

        # Filter out prompt-related columns (they're the same for all models)
        model_specific_data = {
            col: val for col, val in row.items() if col not in GLOBAL_APP_CONSTS.PROMPT_RELATED_COLUMNS
        }

        model_specific_data["model_arch"] = model_combination.arch
        model_specific_data["model_size"] = model_combination.size

        model_evals.append(model_specific_data)

    return pd.DataFrame(model_evals)


@cache_data
def get_models_is_heatmap_available(
    prompt_idx: int, variation: str, window_size: int
) -> dict[MODEL_ARCH_AND_SIZE, Path]:
    """Check if a model has a heatmap for a given prompt index."""
    return {
        model_combination: HeatmapConfig(
            model_arch=model_combination.arch,
            model_size=model_combination.size,
            window_size=window_size,
            variation=variation,
            prompt_indices_rows=[],
            prompt_original_indices=[prompt_idx],
        ).output_heatmap_path(prompt_idx)
        for model_combination in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
    }


def empty_selected_requirements() -> None:
    """Empty the selected requirements set in session state."""
    DataReqsSessionKeys.selected_requirements.value.clear()


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


@cache_data
def get_models_remaining_prompts(
    model_combinations: list[MODEL_ARCH_AND_SIZE],
    window_size: int,
    variation: str,
    prompt_original_indices: list[int],
) -> dict[MODEL_ARCH_AND_SIZE, HeatmapConfig]:
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


@cache_data
def load_model_combinations_prompts(
    variation: str, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
) -> list[ModelCombination]:
    """Get all possible model combinations and their corresponding prompts."""
    return get_model_combinations_prompts(variation, model_arch_and_sizes)
