import pandas as pd
import streamlit as st

from src.final_plots.app.app_consts import AppSessionKeys
from src.final_plots.app.texts import AppGlobalText
from src.final_plots.app.utils import format_path_for_display
from src.types import SLURM_GPU_TYPE


def select_gpu_type():
    st.selectbox(
        AppGlobalText.gpu_type,
        options=[value for value in SLURM_GPU_TYPE],
        key=AppSessionKeys.selected_gpu.key,
    )


def select_variation():
    st.text_input(
        AppGlobalText.variation,
        key=AppSessionKeys.variation.key,
    )


def select_window_size():
    st.selectbox(
        AppGlobalText.window_size,
        options=[1, 3, 5, 7, 9, 12, 15],
        key=AppSessionKeys.window_size.key,
    )


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
