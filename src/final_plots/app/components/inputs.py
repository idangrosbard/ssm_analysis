from typing import NamedTuple

import pandas as pd
import streamlit as st

from src.experiments.heatmap import HEATMAP_PLOT_FUNCS
from src.final_plots.app.app_consts import AppSessionKeys
from src.final_plots.app.texts import AppGlobalText
from src.final_plots.app.utils import format_path_for_display
from src.types import MODEL_ARCH_AND_SIZE, SLURM_GPU_TYPE


def select_gpu_type():
    options = ["smart"] + [value for value in SLURM_GPU_TYPE]
    st.selectbox(
        AppGlobalText.gpu_type,
        options=options,
        key=AppSessionKeys._selected_gpu.key,
        index=options.index(AppSessionKeys._selected_gpu.value),
    )


def select_variation():
    st.text_input(
        AppGlobalText.variation,
        key=AppSessionKeys.variation.key,
        value=AppSessionKeys.variation.value,
    )


def select_window_size():
    options = [1, 3, 5, 7, 9, 12, 15]
    st.selectbox(
        AppGlobalText.window_size,
        options=options,
        key=AppSessionKeys.window_size.key,
        index=options.index(AppSessionKeys.window_size.value),
    )


def select_models_and_sizes(available_models: list[MODEL_ARCH_AND_SIZE]) -> list[MODEL_ARCH_AND_SIZE]:
    """Display a multi-select widget for choosing model architectures and sizes.

    Args:
        available_models: List of (model_arch, model_size) tuples to choose from

    Returns:
        List of selected (model_arch, model_size) tuples
    """
    # Create display names for models
    model_options = [model_arch_and_size for model_arch_and_size in available_models]
    model_display_names = [model_arch_and_size.model_name for model_arch_and_size in model_options]

    # Create mapping from display name back to tuple
    name_to_model = dict(zip(model_display_names, model_options))

    with st.expander("Filter Models", expanded=False):
        selected_names = st.pills(
            "Select Models",
            options=model_display_names,
            default=model_display_names,
            key="model_multiselect",
            selection_mode="multi",
        )

        # Convert selected names back to model tuples
        selected_models = [name_to_model[name] for name in selected_names]

    return selected_models


def info_flow_data_tree(
    df: pd.DataFrame,
    grid_param: str,
    row_param: str,
    col_param: str,
    line_param: str,
) -> None:
    """Display a tree view of data sources for info flow plots.

    Args:
        df: The DataFrame containing the data
        grid_param: The parameter used for grid
        row_param: The parameter used for rows
        col_param: The parameter used for columns
        line_param: The parameter used for lines
    """
    # Get all unique values for each parameter
    grid_values = sorted(df[grid_param].unique())

    # Display tree structure
    for grid_val in grid_values:
        grid_df = df[df[grid_param] == grid_val]

        with st.expander(f"🗂 {grid_param} = {grid_val}", expanded=True):
            row_values = sorted(grid_df[row_param].unique())

            for row_val in row_values:
                row_df = grid_df[grid_df[row_param] == row_val]
                st.markdown(f"**└── {row_param} = {row_val}**")

                col_values = sorted(row_df[col_param].unique())
                for col_val in col_values:
                    col_df = row_df[row_df[col_param] == col_val]
                    st.markdown(f"{'&nbsp;' * 4}**└── {col_param} = {col_val}**")

                    for _, row in col_df.iterrows():
                        line_val = row[line_param]
                        if pd.notna(line_val):
                            st.markdown(f"{'&nbsp;' * 7}└── {line_param} = {line_val}")
                            st.markdown(f"{'&nbsp;' * 10}└── `{format_path_for_display(row['data_path'])}`")


class HeatmapPlotsParams(NamedTuple):
    plot_name: HEATMAP_PLOT_FUNCS


def choose_heatmap_parms():
    return HeatmapPlotsParams(
        plot_name=st.selectbox(
            "Plot Name",
            options=list(HEATMAP_PLOT_FUNCS),
            index=0,
        )
    )
