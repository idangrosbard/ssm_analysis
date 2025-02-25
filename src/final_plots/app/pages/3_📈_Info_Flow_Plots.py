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

import itertools
from pathlib import Path
from typing import Literal, TypedDict, Union, cast

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from src.consts import EXPERIMENT_NAMES, TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.final_plots.app.app_consts import InfoFlowConsts
from src.final_plots.app.data_store import load_experiment_fulfilled_reqs_df
from src.final_plots.app.texts import COMMON_TEXTS, INFO_FLOW_TEXTS
from src.final_plots.app.utils import (
    format_path_for_display,
    get_param_values,
)
from src.final_plots.results_bank import ParamNames
from src.plots.info_flow_confidence import PlotMetadata, create_confidence_plot, load_window_outputs
from src.utils.streamlit_utils import StreamlitComponent, StreamlitPage


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


# region Page Configuration
st.set_page_config(page_title=INFO_FLOW_TEXTS.title, page_icon=INFO_FLOW_TEXTS.icon, layout="wide")
st.title(f"{INFO_FLOW_TEXTS.title} {INFO_FLOW_TEXTS.icon}")
# endregion


class ParameterConfiguration(StreamlitComponent):
    def __init__(self, df: pd.DataFrame, available_params: list[ParamNames]):
        self.df = df
        self.available_params = available_params

    def render(self):
        st.sidebar.header(INFO_FLOW_TEXTS.plot_config_title)
        st.sidebar.subheader("Parameter Configuration")

        # Store selected values for each parameter
        param_values = {}

        # Create parameter controls
        for param in self.available_params:
            st.sidebar.markdown(f"**{param}:**")
            col1, col2 = st.sidebar.columns([2, 1])

            with col1:
                # If parameter is fixed, show value selector
                unique_values = get_param_values(self.df, param)
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
                selected_role = st.selectbox(
                    f"Role for {param}",
                    options=InfoFlowConsts.PARAM_ROLES,
                    key=f"role_{param}",
                    label_visibility="collapsed",
                )
                st.session_state.param_roles[param] = cast(InfoFlowConsts.ParamRole, selected_role)

        # Validate and update roles
        role_counts = {role: 0 for role in ["grid", "column", "row", "line"]}
        for param, role in st.session_state.param_roles.items():
            if role != "fixed":
                role_counts[role] += 1

        # Check if we have exactly one parameter for each role
        roles_valid = all(count == 1 for count in role_counts.values())
        if not roles_valid:
            st.sidebar.error("Please select exactly one parameter for each role (grid, column, row, line)")
            return None, None

        # Get parameters for each role
        param_roles = {
            "grid": next(param for param, role in st.session_state.param_roles.items() if role == "grid"),
            "column": next(param for param, role in st.session_state.param_roles.items() if role == "column"),
            "row": next(param for param, role in st.session_state.param_roles.items() if role == "row"),
            "line": next(param for param, role in st.session_state.param_roles.items() if role == "line"),
        }

        return param_values, param_roles


class PlotCustomization(StreamlitComponent):
    def __init__(self, df: pd.DataFrame, line_param: str):
        self.df = df
        self.line_param = line_param

    def render(self):
        st.header("Plot Customization")
        plot_config = {
            "confidence_level": st.slider(
                "Confidence Level",
                0.8,
                0.99,
                InfoFlowConsts.DEFAULT_PLOT_CONFIG["confidence_level"],
                0.01,
            ),
            "plot_height": st.slider(
                "Plot Height",
                300,
                1000,
                InfoFlowConsts.DEFAULT_PLOT_CONFIG["plot_height"],
            ),
            "plot_width": st.slider(
                "Plot Width",
                400,
                1200,
                InfoFlowConsts.DEFAULT_PLOT_CONFIG["plot_width"],
            ),
        }

        # Color customization
        st.header("Color Customization")
        use_custom_colors = st.checkbox("Use Custom Colors", False)

        if use_custom_colors:
            custom_colors = {}
            custom_styles = {}
            unique_values = self.df[self.line_param].unique()

            for value in unique_values:
                if pd.notna(value):
                    col1, col2 = st.columns(2)
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

        return plot_config, custom_colors, custom_styles


class DataSourceDisplay(StreamlitComponent):
    def __init__(self, df: pd.DataFrame, param_roles: dict[str, str]):
        self.df = df
        self.param_roles = param_roles

    def render(self):
        st.header("Data Sources")

        grid_items: list[Union[str, dict, sac.TreeItem]] = []
        for grid_val in sorted(self.df[self.param_roles["grid"]].unique()):
            row_items: list[sac.TreeItem] = []
            grid_items.append(sac.TreeItem(label=f"{self.param_roles['grid']} = {grid_val}", children=row_items))
            for row_val in sorted(self.df[self.param_roles["row"]].unique()):
                col_items: list[sac.TreeItem] = []
                row_items.append(sac.TreeItem(label=f"{self.param_roles['row']} = {row_val}", children=col_items))
                for col_val in sorted(self.df[self.param_roles["column"]].unique()):
                    line_items: list[sac.TreeItem] = []
                    col_items.append(
                        sac.TreeItem(label=f"{self.param_roles['column']} = {col_val}", children=line_items)
                    )
                    for _, row in self.df[
                        (self.df[self.param_roles["grid"]] == grid_val)
                        & (self.df[self.param_roles["row"]] == row_val)
                        & (self.df[self.param_roles["column"]] == col_val)
                    ].iterrows():
                        line_items.append(
                            sac.TreeItem(
                                label=f"{self.param_roles['line']} = {row[self.param_roles['line']]}",
                                children=[],
                            )
                        )

        sac.tree(
            items=grid_items,
            label="Data Sources",
            # align="center",
            width=500,
            size="lg",
            open_all=True,
            key="data_sources_tree",
            height=500,
        )


class PlotCreation(StreamlitComponent):
    def __init__(
        self,
        df: pd.DataFrame,
        param_roles: dict[str, str],
        plot_config: dict,
        custom_colors: dict,
        custom_styles: dict,
    ):
        self.df = df
        self.param_roles = param_roles
        self.plot_config = plot_config
        self.custom_colors = custom_colors
        self.custom_styles = custom_styles

    def render(self):
        def create_grid_plots():
            grid_values = sorted(self.df[self.param_roles["grid"]].unique())
            plots = {}
            failed_plots = []

            for grid_val in grid_values:
                try:
                    grid_df = self.df[self.df[self.param_roles["grid"]] == grid_val]

                    # Get unique values for rows and columns
                    row_values = sorted(grid_df[self.param_roles["row"]].unique())
                    col_values = sorted(grid_df[self.param_roles["column"]].unique())

                    # Create fixture with subplots
                    fig, axes = plt.subplots(
                        len(row_values),
                        len(col_values),
                        figsize=(self.plot_config["plot_width"] / 100, self.plot_config["plot_height"] / 100),
                        squeeze=False,
                    )

                    # Create plots for each combination
                    for (i, row_val), (j, col_val) in itertools.product(enumerate(row_values), enumerate(col_values)):
                        paths = []
                        try:
                            subplot_df = grid_df[
                                (grid_df[self.param_roles["row"]] == row_val)
                                & (grid_df[self.param_roles["column"]] == col_val)
                            ]

                            if subplot_df.empty:
                                continue

                            # Group by line parameter and create plot
                            targets_window_outputs = {}

                            load_errors = []
                            for _, row in subplot_df.iterrows():
                                line_val = row[self.param_roles["line"]]
                                if pd.isna(line_val):
                                    continue

                                try:
                                    window_outputs = load_window_outputs(row["data_path"])
                                    targets_window_outputs[line_val] = window_outputs
                                    paths.append(format_path_for_display(row["data_path"]))
                                except Exception as e:
                                    load_errors.append(f"Error loading data for {row['data_path']}{line_val}:\n {e}")
                                    continue

                            if not targets_window_outputs:
                                if load_errors:
                                    failed_plots.append(
                                        f"Failed to load any data for {self.param_roles['grid']}={grid_val}, "
                                        f"{self.param_roles['row']}={row_val},"
                                        f" {self.param_roles['column']}={col_val}:\n"
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

                            fig = create_confidence_plot(
                                targets_window_outputs=targets_window_outputs,
                                confidence_level=self.plot_config["confidence_level"],
                                title=(
                                    f"{self.param_roles['grid']}={grid_val}\n"
                                    f"{self.param_roles['row']}={row_val}, "
                                    f"{self.param_roles['column']}={col_val}"
                                ),
                                plots_meta_data=plots_meta_data,
                            )

                            plots[f"{grid_val}_{row_val}_{col_val}"] = fig

                        except Exception as e:
                            failed_plots.append(f"Failed to create plot for one of the paths:{paths}:\nError: {str(e)}")
                            continue

                except Exception as e:
                    failed_plots.append(f"Failed to process grid {self.param_roles['grid']}={grid_val}: {str(e)}")
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


class InfoFlowPlotsPage(StreamlitPage):
    def render(self):
        # Load the data
        df = pd.DataFrame(load_experiment_fulfilled_reqs_df(EXPERIMENT_NAMES.INFO_FLOW))

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

        # Configure parameters and get roles
        param_values, param_roles = ParameterConfiguration(df, available_params).render()
        if param_values is None or param_roles is None:
            st.stop()

        # Filter dataframe based on fixed parameters
        for param, value in param_values.items():
            df = df[df[param] == value]

        if df.empty:
            st.sidebar.error("No data available for the selected parameter values")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            # Configure plot customization
            plot_config, custom_colors, custom_styles = PlotCustomization(df, param_roles["line"]).render()

        with col2:
            # Display data sources
            DataSourceDisplay(df, param_roles).render()

        # Create and display plots
        PlotCreation(df, param_roles, plot_config, custom_colors, custom_styles).render()


if __name__ == "__main__":
    InfoFlowPlotsPage().render()
