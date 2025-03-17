import itertools
from pathlib import Path
from typing import Literal, Union, cast

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from matplotlib import pyplot as plt

from src.consts import GRAPHS_ORDER
from src.experiments.heatmap import HeatmapConfig
from src.experiments.info_flow import InfoFlowConfig
from src.final_plots.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys, InfoFlowConsts
from src.final_plots.app.components.inputs import choose_heatmap_parms
from src.final_plots.app.texts import COMMON_TEXTS, INFO_FLOW_TEXTS
from src.final_plots.app.utils import format_path_for_display, get_param_values
from src.final_plots.image_combiner import ImageGridParams, combine_image_grid
from src.names import ResultBankParamNames
from src.plots.info_flow_confidence import PlotMetadata, create_confidence_plot
from src.types import MODEL_SIZE_CAT, TPromptOriginalIndex
from src.utils.extended_streamlit_pydantic import pydantic_input
from src.utils.streamlit_utils import StreamlitComponent


class ParameterConfiguration(StreamlitComponent):
    def __init__(self, df: pd.DataFrame, available_params: list[ResultBankParamNames]):
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


class HeatmapPlotGenerationComponent(StreamlitComponent):
    def __init__(self, prompt_idx: TPromptOriginalIndex):
        self.prompt_idx = prompt_idx

    def render(self):
        assert self.prompt_idx is not None

        # Apply model filters to get qualifying prompts
        with st.sidebar:
            heatmap_parms = choose_heatmap_parms()
            image_grid_params = pydantic_input(key="my_form", model=ImageGridParams)

        N = 3
        M = 4
        # Create a 3x3 grid where rows are size categories and columns are architectures
        grid: list[list[Path | None]] = [[None for _ in range(N)] for _ in range(M)]  # 3x3 grid of None values
        size_cats = [MODEL_SIZE_CAT.SMALL, MODEL_SIZE_CAT.MEDIUM, MODEL_SIZE_CAT.LARGE, MODEL_SIZE_CAT.HUGE]

        i = 0
        for model_arch_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
            model_arch, model_size = model_arch_and_size
            config = HeatmapConfig(
                model_arch=model_arch,
                model_size=model_size,
                window_size=AppSessionKeys.window_size.value,
                variation=AppSessionKeys.variation.value,
                prompt_indices_rows=[],
                prompt_original_indices=[self.prompt_idx],
            )

            if not config.output_heatmap_path(self.prompt_idx).exists():
                continue
            plots_path = config.get_plot_output_path(self.prompt_idx, heatmap_parms.plot_name)
            if not plots_path.exists():
                config.plot(heatmap_parms.plot_name)

            # Add plot path to its position in the grid
            size_cat = GRAPHS_ORDER[model_arch_and_size]
            if size_cat in size_cats:
                grid[i // N][i % N] = plots_path
                i += 1

        # Create the combined image
        if any(any(row) for row in grid):  # Only show if we have any images
            # Cast grid to list[list[Path]] by filtering out None values
            non_none_grid = [[p for p in row if p is not None] for row in grid]
            if image_grid_params is not None:
                combined_image = combine_image_grid(non_none_grid, ImageGridParams(**image_grid_params))
                if combined_image is not None:
                    st.image(combined_image)


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
                    fig, _ = plt.subplots(
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
                                    window_outputs = InfoFlowConfig.load_output(row["data_path"])
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
