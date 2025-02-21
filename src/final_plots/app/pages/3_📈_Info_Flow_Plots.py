# Purpose: Create and display information flow plots with customizable parameters and visualization options
# High Level Outline:
# 1. Page setup and configuration
# 2. Parameter configuration and role assignment
# 3. Data source management and display
# 4. Plot creation and customization
# Outline Issues:
# - Add export functionality for generated plots
# - Consider adding more plot customization options
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly

from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.consts import TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.final_plots.app.app_consts import InfoFlowConsts
from src.final_plots.app.data_store import load_info_flow_data
from src.final_plots.app.texts import COMMON_TEXTS, INFO_FLOW_TEXTS
from src.final_plots.app.utils import (
    format_path_for_display,
    get_param_values,
)
from src.final_plots.results_bank import ParamNames
from src.utils.streamlit_utils import StreamlitPage


# region Type Definitions
class DataRow(TypedDict):
    experiment_name: str
    model_arch: str
    model_size: str
    window_size: int
    is_all_correct: bool
    source: str
    feature_category: str
    target: str
    data_path: Path


class InfoFlowPlotsPage(StreamlitPage):
    def render(self):
        # region Page Configuration
        st.set_page_config(page_title=INFO_FLOW_TEXTS.title, page_icon=INFO_FLOW_TEXTS.icon, layout="wide")
        st.title(f"{INFO_FLOW_TEXTS.title} {INFO_FLOW_TEXTS.icon}")
        # endregion

        # endregion

        # region Data Loading and Parameter Setup
        # Load the data
        df = pd.DataFrame(load_info_flow_data())

        # Available parameters
        available_params = [
            ParamNames.model_arch,
            ParamNames.model_size,
            ParamNames.window_size,
            ParamNames.is_all_correct,
            ParamNames.source,
            ParamNames.target,
        ]

        # Initialize session state for parameter roles if not exists
        if "param_roles" not in st.session_state:
            st.session_state.param_roles = {
                param: cast(InfoFlowConsts.ParamRole, "fixed") for param in available_params
            }
            # Set default roles
            st.session_state.param_roles[ParamNames.model_arch] = cast(InfoFlowConsts.ParamRole, "grid")
            st.session_state.param_roles[ParamNames.model_size] = cast(InfoFlowConsts.ParamRole, "column")
            st.session_state.param_roles[ParamNames.window_size] = cast(InfoFlowConsts.ParamRole, "row")
            st.session_state.param_roles[ParamNames.source] = cast(InfoFlowConsts.ParamRole, "line")
        # endregion

        # region Parameter Configuration
        st.sidebar.header(INFO_FLOW_TEXTS.plot_config_title)
        st.sidebar.subheader("Parameter Configuration")

        # Store selected values for each parameter
        param_values = {}

        # Create parameter controls
        for param in available_params:
            st.sidebar.markdown(f"**{param}:**")
            col1, col2 = st.sidebar.columns([2, 1])

            with col1:
                # If parameter is fixed, show value selector
                unique_values = get_param_values(df, param)
                if st.session_state.param_roles[param] == "fixed":
                    param_values[param] = st.selectbox(
                        f"Value for {param}", unique_values, key=f"value_{param}", label_visibility="collapsed"
                    )
                else:
                    # Show available values but disabled
                    values_str = ", ".join(str(v) for v in unique_values)
                    st.text_input(
                        f"Available values for {param}", value=values_str, disabled=True, label_visibility="collapsed"
                    )

            with col2:
                # current_role = st.session_state.param_roles[param]
                selected_role = st.selectbox(
                    f"Role for {param}",
                    options=InfoFlowConsts.PARAM_ROLES,
                    key=f"role_{param}",
                    label_visibility="collapsed",
                )
                st.session_state.param_roles[param] = cast(InfoFlowConsts.ParamRole, selected_role)
        # endregion

        # region Role Validation
        # Validate and update roles
        role_counts = {role: 0 for role in ["grid", "column", "row", "line"]}
        for param, role in st.session_state.param_roles.items():
            if role != "fixed":
                role_counts[role] += 1

        # Check if we have exactly one parameter for each role
        roles_valid = all(count == 1 for count in role_counts.values())
        if not roles_valid:
            st.sidebar.error("Please select exactly one parameter for each role (grid, column, row, line)")
            st.stop()

        # Get parameters for each role
        grid_param = next(param for param, role in st.session_state.param_roles.items() if role == "grid")
        col_param = next(param for param, role in st.session_state.param_roles.items() if role == "column")
        row_param = next(param for param, role in st.session_state.param_roles.items() if role == "row")
        line_param = next(param for param, role in st.session_state.param_roles.items() if role == "line")
        # endregion

        # region Data Filtering
        # Filter dataframe based on fixed parameters
        for param, value in param_values.items():
            df = df[df[param] == value]

        if df.empty:
            st.sidebar.error("No data available for the selected parameter values")
            st.stop()
        # endregion

        # region Plot Customization
        st.sidebar.header("Plot Customization")
        # confidence_level = st.sidebar.slider(
        #     "Confidence Level",
        #     0.8,
        #     0.99,
        #     InfoFlowConsts.DEFAULT_PLOT_CONFIG["confidence_level"],
        #     0.01,
        # )
        # plot_height = st.sidebar.slider(
        #     "Plot Height",
        #     300,
        #     1000,
        #     InfoFlowConsts.DEFAULT_PLOT_CONFIG["plot_height"],
        # )
        # plot_width = st.sidebar.slider(
        #     "Plot Width",
        #     400,
        #     1200,
        #     InfoFlowConsts.DEFAULT_PLOT_CONFIG["plot_width"],
        # )

        # Color customization
        st.sidebar.header("Color Customization")
        use_custom_colors = st.sidebar.checkbox("Use Custom Colors", False)

        if use_custom_colors:
            custom_colors = {}
            custom_styles = {}
            unique_values = df[line_param].unique()

            for value in unique_values:
                if pd.notna(value):
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        custom_colors[value] = st.color_picker(
                            f"Color for {value}", TOKEN_TYPE_COLORS.get(value, "#000000")
                        )
                    with col2:
                        custom_styles[value] = st.selectbox(
                            f"Style for {value}", InfoFlowConsts.DEFAULT_LINE_STYLES, index=0
                        )
        else:
            custom_colors = TOKEN_TYPE_COLORS
            custom_styles = TOKEN_TYPE_LINE_STYLES
        # endregion

        # region Data Source Display
        st.header("Data Sources")

        def display_tree():
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

        # Display data source tree
        col1, col2 = st.columns([1, 3])
        with col1:
            show_tree = st.checkbox(INFO_FLOW_TEXTS.show_data_sources, value=True)
            if show_tree:
                st.info(INFO_FLOW_TEXTS.total_experiments(len(df)))

        with col2:
            if show_tree:
                display_tree()
        # endregion

        # region Plot Creation
        def create_grid_plots():
            grid_values = sorted(df[grid_param].unique())
            plots = {}
            failed_plots = []

            for grid_val in grid_values:
                try:
                    pass
                    # grid_df = df[df[grid_param] == grid_val]

                    # # Get unique values for rows and columns
                    # row_values = sorted(grid_df[row_param].unique())
                    # col_values = sorted(grid_df[col_param].unique())

                    # Create figure with subplots
                    # fig, axes = plt.subplots(
                    #     len(row_values), len(col_values), figsize=(plot_width / 100, plot_height / 100), squeeze=False
                    # )
                    # TODO: Add plots
                    # Create plots for each combination
                    # for (i, row_val), (j, col_val) in product(enumerate(row_values), enumerate(col_values)):
                    #     try:
                    #         paths = []
                    #         subplot_df = grid_df[(grid_df[row_param] == row_val) & (grid_df[col_param] == col_val)]

                    #         if subplot_df.empty:
                    #             continue

                    #         # Group by line parameter and create plot
                    #         targets_window_outputs = {}

                    #         load_errors = []
                    #         for _, row in subplot_df.iterrows():
                    #             line_val = row[line_param]
                    #             if pd.isna(line_val):
                    #                 continue

                    #             try:
                    #                 window_outputs = load_window_outputs(row["data_path"])
                    #                 targets_window_outputs[line_val] = window_outputs
                    #                 paths.append(format_path_for_display(row["data_path"]))
                    #             except Exception as e:
                    #                 load_errors.append(f"Error loading data for {line_val}: {e}")
                    #                 continue

                    #         if not targets_window_outputs:
                    #             if load_errors:
                    #                 failed_plots.append(
                    #                     f"Failed to load any data for {grid_param}={grid_val}, "
                    #                     f"{row_param}={row_val}, {col_param}={col_val}:\n"
                    #                     + "\n".join(f"  - {err}" for err in load_errors)
                    #                 )
                    #             continue

                    #         plots_meta_data: dict[Literal["acc", "diff"], PlotMetadata] = {
                    #             "acc": {
                    #                 "title": "Accuracy",
                    #                 "ylabel": "% accuracy",
                    #                 "ylabel_loc": "center",
                    #                 "axhline_value": 100.0,
                    #                 "ylim": (60.0, 105.0),
                    #             },
                    #             "diff": {
                    #                 "title": "Normalized change in prediction probability",
                    #                 "ylabel": "% probability change",
                    #                 "ylabel_loc": "top",
                    #                 "axhline_value": 0.0,
                    #                 "ylim": (-50.0, 50.0),
                    #             },
                    #         }

                    #         fig = create_confidence_plot(
                    #             targets_window_outputs=targets_window_outputs,
                    #             confidence_level=confidence_level,
                    #             title=f"{grid_param}={grid_val}\n{row_param}={row_val}, {col_param}={col_val}",
                    #             plots_meta_data=plots_meta_data,
                    #         )

                    #         plots[f"{grid_val}_{row_val}_{col_val}"] = fig

                    #     except Exception as e:
                    #         failed_plots.append(
                    #             "Failed to create plot for one of the paths:"
                    #             f"{paths}\n{row['data_path']}:\nError: {str(e)}"
                    #         )
                    #         continue

                except Exception as e:
                    failed_plots.append(f"Failed to process grid {grid_param}={grid_val}: {str(e)}")
                    continue

            return plots, failed_plots

        # Create plots button
        if st.button(INFO_FLOW_TEXTS.generate_plots):
            with st.spinner(INFO_FLOW_TEXTS.generating_plots):
                plots, failed_plots = create_grid_plots()

                if failed_plots:
                    st.warning("Some plots failed to generate:")
                    with st.expander(COMMON_TEXTS.error_details):
                        for error in failed_plots:
                            st.error(error)

                if plots:
                    st.success(INFO_FLOW_TEXTS.plots_generated(len(plots)))
                    # Display plots
                    for plot_key, fig in plots.items():
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up
                else:
                    st.error(INFO_FLOW_TEXTS.no_plots_generated)


if __name__ == "__main__":
    InfoFlowPlotsPage().render()
