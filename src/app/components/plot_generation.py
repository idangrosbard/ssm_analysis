# Purpose: Provide components for generating plots based on plot plans
# High Level Outline:
# 1. Plot generation components for different plot types
# 2. Utility functions for loading and processing data
# 3. Plot rendering and saving functionality
# Outline Issues:
# - Consider adding more customization options for plots
# - Add support for interactive plots
# Outline Compatibility Issues:
# - New file, outline will be implemented

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, assert_never, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis.experiment_results.helpers import get_model_evaluations
from src.analysis.experiment_results.plot_plan import PlotPlan, PlotType, get_hyper_param_definition
from src.analysis.plots.heatmaps import simple_diff_fixed
from src.analysis.plots.info_flow_confidence import create_confidence_plot
from src.app.texts import FINAL_PLOTS_TEXTS
from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.core.names import SummarizedDataFulfilledReqsCols
from src.core.types import MODEL_ARCH_AND_SIZE, TInfoFlowOutput, TPromptData
from src.data_ingestion.data_defs import DataReqs, FulfilledReqs, PlotPlans, ResultBank
from src.data_ingestion.helpers.logits_utils import decode_tokens, get_prompt_row_index
from src.experiments.infrastructure.setup_models import get_tokenizer
from src.experiments.runners.heatmap import HeatmapConfig
from src.experiments.runners.info_flow import InfoFlowConfig
from src.utils.streamlit.helpers.component import StreamlitComponent


def load_info_flow_data(data: TInfoFlowOutput, idx: bool = False):
    df = {"Depth": [], "Probability diff": [], "Correct": []}

    if idx:
        df["original_idx"] = []

    n_layers = len(data)

    for layer in range(n_layers):
        curr_layer = data[layer]
        if len(curr_layer) == 0:
            continue

        if "correct" in curr_layer:
            n_samples = len(curr_layer["correct"])
        else:
            assert "hit" in curr_layer
            n_samples = len(curr_layer["hit"])

        for i in range(n_samples):
            df["Depth"] += [layer / (n_layers - 1)]
            if "probability_diff" in curr_layer:
                df["Probability diff"] += [curr_layer["probability_diff"][i]]
            elif "diffs" in curr_layer:
                df["Probability diff"] += [curr_layer["diffs"][i]]
            else:
                raise ValueError(f"No probability diff in layer: {list(curr_layer.keys())}")

            if "correct" in curr_layer:
                df["Correct"] += [curr_layer["correct"][i]]
            elif "hit" in curr_layer:
                df["Correct"] += [curr_layer["hit"][i]]
            else:
                raise ValueError(f'No "Correct" in layer: {list(curr_layer.keys())}')

            if idx:
                df["original_idx"] += [curr_layer["original_idx"][i]]

            # df['original_idx'] += [curr_layer['original_idx'][i]]

    return pd.DataFrame(df)


# Constants for plotting
COLORS = {
    "Mamba-1 2.8B": (31, 119, 180),
    "Mamba-2 2.7B": (255, 127, 14),
    "Falcon-Mamba 7B": (44, 160, 44),
    "GPT-2 1.5B": (214, 39, 40),
    "gpt2-355M": (148, 103, 189),
    "gpt2-774M": (140, 86, 75),
    "gpt2-1.5B": (227, 119, 194),
    "mamba-130M": (127, 127, 127),
    "mamba-370M": (188, 189, 34),
    "mamba-790M": (23, 190, 207),
    "mamba-1.4B": (31, 119, 180),
    "mamba-2.8B": (255, 127, 14),
    "mamba2-370M": (44, 160, 44),
    "mamba2-790M": (214, 39, 40),
    "mamba2-1.4B": (148, 103, 189),
    "mamba2-2.7B": (140, 86, 75),
    "All": (31, 119, 180),
    "Context dependent": (255, 127, 14),
    "Context independent": (44, 160, 44),
    "first": (31, 119, 180),
    "relation": (255, 127, 14),
    "subject": (44, 160, 44),
    "last": (214, 39, 40),
}


@dataclass
class PlotDimensions:
    height_per_row: int = 400
    width_per_col: int = 400
    margin_top: int = 100
    margin_bottom: int = 100


@dataclass
class PlotFontSettings:
    size: int = 36


@dataclass
class PlotColors:
    paper_bgcolor: str = "#FFFFFF"
    plot_bgcolor: str = "#FFFFFF"
    grid_color: str = "#D3D3D3"
    zero_line_color: str = "#D3D3D3"
    axis_line_color: str = "#000000"


@dataclass
class PlotLineSettings:
    line_width: int = 2
    grid_width: int = 1
    zero_line_width: int = 2
    fill_opacity: float = 0.2


@dataclass
class PlotLegendSettings:
    orientation: Literal["h", "v"] = "h"
    y_anchor: Literal["top", "bottom", "middle"] = "bottom"
    y: float = 1.02
    x_anchor: Literal["left", "right", "center"] = "right"
    x: float = 1


@dataclass
class PlotParams:
    dimensions: PlotDimensions = field(default_factory=PlotDimensions)
    font: PlotFontSettings = field(default_factory=PlotFontSettings)
    colors: PlotColors = field(default_factory=PlotColors)
    line: PlotLineSettings = field(default_factory=PlotLineSettings)
    legend: PlotLegendSettings = field(default_factory=PlotLegendSettings)


class PlotStyle:
    """Class to manage plot styling parameters."""

    def __init__(self):
        self.params = PlotParams()

    def show_style_form(self):
        """Display a form to control plot styling parameters in the sidebar."""
        with st.sidebar:
            st.subheader("Plot Style Settings")

            with st.form("plot_style_settings"):
                with st.expander("Figure Dimensions", expanded=False):
                    self.params.dimensions.height_per_row = st.number_input(
                        "Height per row", min_value=100, value=self.params.dimensions.height_per_row
                    )
                    self.params.dimensions.width_per_col = st.number_input(
                        "Width per column", min_value=100, value=self.params.dimensions.width_per_col
                    )
                    self.params.dimensions.margin_top = st.number_input(
                        "Top margin", min_value=0, value=self.params.dimensions.margin_top
                    )
                    self.params.dimensions.margin_bottom = st.number_input(
                        "Bottom margin", min_value=0, value=self.params.dimensions.margin_bottom
                    )

                with st.expander("Font Settings", expanded=False):
                    self.params.font.size = st.number_input("Font size", min_value=8, value=self.params.font.size)

                with st.expander("Colors", expanded=False):
                    self.params.colors.paper_bgcolor = st.color_picker(
                        "Paper background color", self.params.colors.paper_bgcolor
                    )
                    self.params.colors.plot_bgcolor = st.color_picker(
                        "Plot background color", self.params.colors.plot_bgcolor
                    )
                    self.params.colors.grid_color = st.color_picker("Grid color", self.params.colors.grid_color)
                    self.params.colors.zero_line_color = st.color_picker(
                        "Zero line color", self.params.colors.zero_line_color
                    )
                    self.params.colors.axis_line_color = st.color_picker(
                        "Axis line color", self.params.colors.axis_line_color
                    )

                with st.expander("Lines Settings", expanded=False):
                    self.params.line.line_width = st.number_input(
                        "Line width", min_value=1, value=self.params.line.line_width
                    )
                    self.params.line.grid_width = st.number_input(
                        "Grid width", min_value=1, value=self.params.line.grid_width
                    )
                    self.params.line.zero_line_width = st.number_input(
                        "Zero line width", min_value=1, value=self.params.line.zero_line_width
                    )
                    self.params.line.fill_opacity = st.slider(
                        "Fill opacity", min_value=0.0, max_value=1.0, value=self.params.line.fill_opacity
                    )

                with st.expander("Legend Settings", expanded=False):
                    self.params.legend.orientation = st.selectbox(
                        "Legend orientation", ["h", "v"], index=0 if self.params.legend.orientation == "h" else 1
                    )
                    self.params.legend.y = st.number_input("Legend Y position", value=self.params.legend.y)
                    self.params.legend.x = st.number_input("Legend X position", value=self.params.legend.x)
                    self.params.legend.y_anchor = st.selectbox(
                        "Legend Y anchor",
                        ["top", "bottom", "middle"],
                        index=["top", "bottom", "middle"].index(self.params.legend.y_anchor),
                    )
                    self.params.legend.x_anchor = st.selectbox(
                        "Legend X anchor",
                        ["left", "right", "center"],
                        index=["left", "right", "center"].index(self.params.legend.x_anchor),
                    )

                return st.form_submit_button("Apply Style")

    def apply_to_figure(self, fig: go.Figure, n_cols: int, n_rows: int = 1) -> go.Figure:
        """Apply the current style settings to a figure."""
        fig.update_layout(
            height=self.params.dimensions.height_per_row * n_rows,
            width=self.params.dimensions.width_per_col * n_cols,
            paper_bgcolor=self.params.colors.paper_bgcolor,
            plot_bgcolor=self.params.colors.plot_bgcolor,
            margin=dict(t=self.params.dimensions.margin_top, b=self.params.dimensions.margin_bottom),
            font=dict(size=self.params.font.size),
            legend=dict(
                orientation=self.params.legend.orientation,
                yanchor=self.params.legend.y_anchor,
                y=self.params.legend.y,
                xanchor=self.params.legend.x_anchor,
                x=self.params.legend.x,
                font=dict(size=self.params.font.size),
            ),
        )

        # Update all axes
        for i in range(1, n_cols + 1):
            for j in range(1, n_rows + 1):
                # X axis
                fig.update_xaxes(
                    title_text="Relative depth (%)" if j == n_rows else None,
                    showline=True,
                    linewidth=self.params.line.line_width,
                    linecolor=self.params.colors.axis_line_color,
                    showgrid=True,
                    gridwidth=self.params.line.grid_width,
                    gridcolor=self.params.colors.grid_color,
                    zeroline=True,
                    zerolinewidth=self.params.line.zero_line_width,
                    zerolinecolor=self.params.colors.zero_line_color,
                    tickfont=dict(size=self.params.font.size),
                )

                # Y axis
                fig.update_yaxes(
                    title_text="Probability diff" if i == 1 else None,
                    showline=True,
                    linewidth=self.params.line.line_width,
                    linecolor=self.params.colors.axis_line_color,
                    showgrid=True,
                    gridwidth=self.params.line.grid_width,
                    gridcolor=self.params.colors.grid_color,
                    zeroline=True,
                    zerolinewidth=self.params.line.zero_line_width,
                    zerolinecolor=self.params.colors.zero_line_color,
                    tickfont=dict(size=self.params.font.size),
                )

        return fig


def plot_trend(
    fig: go.Figure,
    joined: pd.DataFrame,
    model: str,
    rgb: Tuple[int, int, int],
    column: str = "Model",
    line_dash: str = "solid",
    col: int = 1,
    row: int = 1,
    style: PlotStyle = PlotStyle(),
) -> go.Figure:
    """Add a trend line to a plotly figure."""
    filtered = joined[joined[column] == model]

    upper = filtered["Probability diff_mean"] + filtered["Probability diff_ci95"]
    lower = filtered["Probability diff_mean"] - filtered["Probability diff_ci95"]

    name_concode_version = {
        "Mamba-1 2.8B": "mamba-2.8B",
        "Mamba-2 2.7B": "mamba2-2.7B",
        "Falcon-Mamba 7B": "falcon-mamba-7B",
        "GPT-2 1.5B": "gpt2-1.5B",
    }

    if model in name_concode_version:
        title = name_concode_version[model]
    else:
        title = model

    fig.add_trace(
        go.Scatter(
            x=100 * filtered["Depth"],
            y=filtered["Probability diff_mean"],
            line=dict(color=f"rgb{rgb}", dash=line_dash, width=style.params.line.line_width),
            mode="lines",
            name=title,
            showlegend=((col == 1) & (row == 1)),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=(100 * filtered["Depth"]).tolist() + (100 * filtered["Depth"][::-1]).tolist(),  # x, then x reversed
            y=upper.tolist() + lower[::-1].tolist(),  # upper, then lower reversed
            fill="toself",
            fillcolor=f"rgba{rgb + (style.params.line.fill_opacity,)}",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
    )
    return fig


def format_fig(fig: go.Figure, n_cols: int, n_rows: int = 1, style: PlotStyle = PlotStyle()) -> go.Figure:
    """Format a plotly figure with consistent styling."""
    return style.apply_to_figure(fig, n_cols, n_rows)


class PlotGenerator(StreamlitComponent[Optional[str]]):
    """Component for generating plots based on plot plans."""

    def __init__(self, plot_plan: PlotPlan, result_bank: ResultBank):
        self.plot_plan = plot_plan
        self.result_bank = result_bank
        self.style = PlotStyle()

    def _get_model_display_name(self, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> str:
        return model_arch_and_size.model_name

    def _generate_architecture_knockout_plot(self) -> Optional[go.Figure]:
        """Generate an architecture knockout plot."""
        # Get data requirements for the plot
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).choose_latest_fulfilled(self.result_bank)

        # Group data by source
        source_groups = {}
        for req, _ in fulfilled_reqs.to_rows():
            if req.source not in source_groups:
                source_groups[req.source] = []
            source_groups[req.source].append(req)

        if not source_groups:
            st.error("No data available for architecture knockout plot.")
            return None

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=len(source_groups),
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=[f"({chr(97 + i)}) {src}" for i, src in enumerate(source_groups.keys())],
        )

        # Plot each source as a separate subplot
        for i, (source, reqs) in enumerate(source_groups.items()):
            # Load data for each model
            dfs = []
            for req in reqs:
                df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0].path))
                if df is not None:
                    model_name = self._get_model_display_name(req.model_arch_and_size)
                    df["Model"] = model_name
                    dfs.append(df)

            if not dfs:
                continue

            # Combine data
            combined_df = pd.concat(dfs)

            # Calculate statistics
            means = (
                combined_df.groupby(["Depth", "Model"])
                .mean()
                .reset_index()
                .rename(columns={"Probability diff": "Probability diff_mean"})
            )
            ci95 = (
                combined_df.groupby(["Depth", "Model"])
                .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
                .reset_index(name="Probability diff_ci95")
            )
            joined = means.merge(ci95, on=["Depth", "Model"])

            # Plot each model
            for model in joined["Model"].unique():
                if model in COLORS:
                    color = COLORS[model]
                else:
                    # Generate a random color if not in the predefined colors
                    color = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    )

                fig = plot_trend(fig, joined, model, color, col=i + 1, row=1)

        # Format the figure
        fig = format_fig(fig, len(source_groups))

        return fig

    def _generate_model_size_knockout_plot(self) -> Optional[go.Figure]:
        """Generate a model size knockout plot."""
        # Get data requirements for the plot
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).choose_latest_fulfilled(self.result_bank)

        # Group data by source
        source_groups = {}
        for req, _ in fulfilled_reqs.to_rows():
            if req.source not in source_groups:
                source_groups[req.source] = []
            source_groups[req.source].append(req)

        if not source_groups:
            st.error("No data available for model size knockout plot.")
            return None

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=len(source_groups),
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=[f"({chr(97 + i)}) {src}" for i, src in enumerate(source_groups.keys())],
        )

        # Plot each source as a separate subplot
        for i, (source, reqs) in enumerate(source_groups.items()):
            # Group by model architecture
            arch_groups = {}
            for req in reqs:
                if req.model_arch not in arch_groups:
                    arch_groups[req.model_arch] = []
                arch_groups[req.model_arch].append(req)

            # Load data for each model size within architecture
            for arch, arch_reqs in arch_groups.items():
                dfs = []
                for req in arch_reqs:
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0].path))
                    if df is not None:
                        model_name = f"{arch}-{req.model_size}"
                        df["Model"] = model_name
                        dfs.append(df)

                if not dfs:
                    continue

                # Combine data
                combined_df = pd.concat(dfs)

                # Calculate statistics
                means = (
                    combined_df.groupby(["Depth", "Model"])
                    .mean()
                    .reset_index()
                    .rename(columns={"Probability diff": "Probability diff_mean"})
                )
                ci95 = (
                    combined_df.groupby(["Depth", "Model"])
                    .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
                    .reset_index(name="Probability diff_ci95")
                )
                joined = means.merge(ci95, on=["Depth", "Model"])

                # Plot each model size
                for model in joined["Model"].unique():
                    if model in COLORS:
                        color = COLORS[model]
                    else:
                        # Generate a random color if not in the predefined colors
                        color = (
                            np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255),
                        )

                    fig = plot_trend(fig, joined, model, color, col=i + 1, row=1)

        # Format the figure
        fig = format_fig(fig, len(source_groups))

        return fig

    def _generate_window_size_knockout_plot(self) -> Optional[go.Figure]:
        """Generate a window size knockout plot."""
        # Get data requirements for the plot
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).choose_latest_fulfilled(self.result_bank)

        # Group data by source
        source_groups = {}
        for req, _ in fulfilled_reqs.to_rows():
            if req.source not in source_groups:
                source_groups[req.source] = []
            source_groups[req.source].append(req)

        if not source_groups:
            st.error("No data available for window size knockout plot.")
            return None

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=len(source_groups),
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=[f"({chr(97 + i)}) {src}" for i, src in enumerate(source_groups.keys())],
        )

        # Plot each source as a separate subplot
        for i, (source, reqs) in enumerate(source_groups.items()):
            # Group by window size
            ws_groups = {}
            for req in reqs:
                if req.window_size not in ws_groups:
                    ws_groups[req.window_size] = []
                ws_groups[req.window_size].append(req)

            # Load data for each window size
            dfs = []
            for ws, ws_reqs in ws_groups.items():
                for req in ws_reqs:
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0].path))
                    if df is not None:
                        df["ws"] = ws
                        dfs.append(df)

            if not dfs:
                continue

            # Combine data
            combined_df = pd.concat(dfs)

            # Calculate statistics
            means = (
                combined_df.groupby(["Depth", "ws"])
                .mean()
                .reset_index()
                .rename(columns={"Probability diff": "Probability diff_mean"})
            )
            ci95 = (
                combined_df.groupby(["Depth", "ws"])
                .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
                .reset_index(name="Probability diff_ci95")
            )
            joined = means.merge(ci95, on=["Depth", "ws"])

            # Plot each window size
            for ws in sorted(joined["ws"].unique()):
                # Generate a color based on window size
                brightness = 0.5 + 0.5 * (ws / max(joined["ws"].unique()))
                color = (
                    int(31 * brightness),
                    int(119 * brightness),
                    int(180 * brightness),
                )

                fig = plot_trend(fig, joined, ws, color, column="ws", col=i + 1, row=1)

        # Format the figure
        fig = format_fig(fig, len(source_groups))

        return fig

    def _generate_feature_knockout_plot(self) -> Optional[go.Figure]:
        """Generate a feature knockout plot."""
        # Get data requirements for the plot
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).choose_latest_fulfilled(self.result_bank)

        # Group data by model
        model_groups = {}
        for req, _ in fulfilled_reqs.to_rows():
            model_key = self._get_model_display_name(req.model_arch_and_size)
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(req)

        if not model_groups:
            st.error("No data available for feature knockout plot.")
            return None

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=len(model_groups),
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=[f"({chr(97 + i)}) {model}" for i, model in enumerate(model_groups.keys())],
        )

        # Plot each model as a separate subplot
        for i, (model, reqs) in enumerate(model_groups.items()):
            # Group by feature category
            feature_groups = {}
            for req in reqs:
                feature_key = "All" if req.feature_category is None else req.feature_category
                if feature_key not in feature_groups:
                    feature_groups[feature_key] = []
                feature_groups[feature_key].append(req)

            # Load data for each feature category
            dfs = []
            for feature, feature_reqs in feature_groups.items():
                for req in feature_reqs:
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0].path))
                    if df is not None:
                        feature_name = (
                            "All"
                            if feature == "All"
                            else ("Context dependent" if feature == "FAST_DECAY" else "Context independent")
                        )
                        df["Feature"] = feature_name
                        dfs.append(df)

            if not dfs:
                continue

            # Combine data
            combined_df = pd.concat(dfs)

            # Calculate statistics
            means = (
                combined_df.groupby(["Depth", "Feature"])
                .mean()
                .reset_index()
                .rename(columns={"Probability diff": "Probability diff_mean"})
            )
            ci95 = (
                combined_df.groupby(["Depth", "Feature"])
                .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
                .reset_index(name="Probability diff_ci95")
            )
            joined = means.merge(ci95, on=["Depth", "Feature"])

            # Plot each feature category
            for feature in joined["Feature"].unique():
                if feature in COLORS:
                    color = COLORS[feature]
                else:
                    # Generate a random color if not in the predefined colors
                    color = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    )

                fig = plot_trend(fig, joined, feature, color, column="Feature", col=i + 1, row=1)

        # Format the figure
        fig = format_fig(fig, len(model_groups))

        return fig

    def _generate_shared_knockout_plot(self) -> Optional[go.Figure]:
        """Generate a shared knockout plot."""
        # Get data requirements for the plot
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).choose_latest_fulfilled(self.result_bank)

        # Group data by model
        model_groups = {}
        for req, _ in fulfilled_reqs.to_rows():
            model_key = self._get_model_display_name(req.model_arch_and_size)
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(req)

        if not model_groups:
            st.error("No data available for shared knockout plot.")
            return None

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=len(model_groups),
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=[f"({chr(97 + i)}) {model}" for i, model in enumerate(model_groups.keys())],
        )

        # Plot each model as a separate subplot
        for i, (model, reqs) in enumerate(model_groups.items()):
            # Group by source
            source_groups = {}
            for req in reqs:
                if req.source not in source_groups:
                    source_groups[req.source] = []
                source_groups[req.source].append(req)

            # Load data for each source
            dfs = []
            for source, source_reqs in source_groups.items():
                for req in source_reqs:
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0].path))
                    if df is not None:
                        df["Source"] = source
                        dfs.append(df)

            if not dfs:
                continue

            # Combine data
            combined_df = pd.concat(dfs)

            # Calculate statistics
            means = (
                combined_df.groupby(["Depth", "Source"])
                .mean()
                .reset_index()
                .rename(columns={"Probability diff": "Probability diff_mean"})
            )
            ci95 = (
                combined_df.groupby(["Depth", "Source"])
                .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
                .reset_index(name="Probability diff_ci95")
            )
            joined = means.merge(ci95, on=["Depth", "Source"])

            # Plot each source
            for source in joined["Source"].unique():
                if source in COLORS:
                    color = COLORS[source]
                else:
                    # Generate a random color if not in the predefined colors
                    color = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    )

                fig = plot_trend(fig, joined, source, color, column="Source", col=i + 1, row=1)

        # Format the figure
        fig = format_fig(fig, len(model_groups))

        return fig

    def _get_cell_cache_path(self, grid_name: Any, row_name: Any, col_name: Any) -> Path:
        """Generate a unique cache path for a cell's plot."""
        cache_dir = PlotPlans.get_cache_dir(self.plot_plan.plot_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique identifier for the cell
        cell_id = f"{grid_name}_{row_name}_{col_name}".replace(" ", "_")
        return cache_dir / f"{cell_id}.png"

    def _plot_cell(
        self,
        data_reqs: DataReqs,
        grid_name: Any,
        row_name: Any,
        col_name: Any,
        recreate: bool = False,
        with_plotly: bool = False,
    ) -> None:
        """Plot a single cell with caching."""
        cache_path = self._get_cell_cache_path(grid_name, row_name, col_name)

        if not recreate and cache_path.exists():
            # Load and display cached plot
            st.image(str(cache_path))
            return

        # Get fulfilled requirements
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank)

        # Create the plot based on plot type
        fig = None
        match self.plot_plan.plot_type:
            case PlotType.ARCHITECTURE_KNOCKOUT:
                fig = self._generate_cell_architecture_knockout(fulfilled_reqs)
            case PlotType.MODEL_SIZE_KNOCKOUT:
                fig = self._generate_cell_model_size_knockout(fulfilled_reqs)
            case PlotType.WINDOW_SIZE_KNOCKOUT:
                fig = self._generate_cell_window_size_knockout(fulfilled_reqs)
            case PlotType.FEATURE_KNOCKOUT:
                fig = self._generate_cell_feature_knockout(fulfilled_reqs)
            case PlotType.SHARED_KNOCKOUT:
                fig = self._generate_cell_shared_knockout(fulfilled_reqs)
            case PlotType.HEATMAP:
                fulfilled_reqs = fulfilled_reqs.choose_latest_fulfilled(self.result_bank)
                configs = list(fulfilled_reqs.get_config().values())
                assert len(configs) == 1
                config = configs[-1]
                assert isinstance(config, HeatmapConfig)
                prompt_idx = config.get_remaining_prompt_original_indices()
                assert len(prompt_idx) == 1
                prompt_id = prompt_idx[0]
                model_arch_and_size = MODEL_ARCH_AND_SIZE(
                    config.variant_params.model_arch, config.variant_params.model_size
                )
                data = cast(
                    TPromptData,
                    get_model_evaluations(config.metadata_params.code_version, [model_arch_and_size])[
                        model_arch_and_size
                    ],
                )
                tokenizer = get_tokenizer(config.variant_params.model_arch, config.variant_params.model_size)
                model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[config.variant_params.model_arch][
                    config.variant_params.model_size
                ]
                prob_mat = config.get_outputs()[prompt_id]
                prompt = get_prompt_row_index(data, prompt_id)
                input_ids = prompt.input_ids(tokenizer, "cpu")
                toks = cast(list[str], decode_tokens(tokenizer, input_ids[0]))
                last_tok = toks[-1]
                toks[-1] = toks[-1] + "*"

                fig, _ = simple_diff_fixed(
                    prob_mat=prob_mat,
                    model_id=model_id,
                    window_size=config.variant_params.window_size,
                    last_tok=last_tok,
                    base_prob=prompt.base_prob,
                    true_word=prompt.true_word,
                    toks=toks,
                    fixed_diff=0.3,
                )

            case _:
                assert_never(self.plot_plan.plot_type)

        if fig is not None:
            if isinstance(fig, go.Figure):
                fig.write_image(str(cache_path), scale=4)
            else:
                # Save the plot
                plt.savefig(str(cache_path), bbox_inches="tight")
                plt.close(fig)

            # Display the plot
            if with_plotly:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.image(str(cache_path))

    def _generate_cell_architecture_knockout(self, fulfilled_reqs: FulfilledReqs):
        """Generate architecture knockout plot for a single cell."""
        # Create the base figure
        configs = list(fulfilled_reqs.get_config().values())
        data = []
        title = "-".join(
            [
                # config.common_params.model_arch,
                # config.common_params.model_size,
                # str(config.runner_params.window_size),
            ]
        )
        for config in configs:
            assert isinstance(config, InfoFlowConfig)

            data.append(
                {
                    "label": f"{config.variant_params.source} - {config.variant_params.feature_category}",
                    "color": TOKEN_TYPE_COLORS.get(config.variant_params.source, "#000000"),
                    "linestyle": TOKEN_TYPE_LINE_STYLES.get(config.variant_params.feature_category, "-"),
                    "data": config.get_outputs(),
                }
            )
        with_fixed_limits = False
        fig = create_confidence_plot(
            lines_metadata=data,
            confidence_level=0.95,
            title=title,
            plots_meta_data={
                "acc": {
                    "title": "Accuracy",
                    "ylabel": "% accuracy",
                    "ylabel_loc": "center",
                    "axhline_value": 100.0,
                    "ylim": (60.0, 105.0) if with_fixed_limits else None,
                },
                "diff": {
                    "title": "Normalized change in prediction probability",
                    "ylabel": "% probability change",
                    "ylabel_loc": "top",
                    "axhline_value": 0.0,
                    "ylim": (-50.0, 50.0) if with_fixed_limits else None,
                },
            },
        )

        return fig

    def _generate_cell_model_size_knockout(self, fulfilled_reqs) -> Optional[go.Figure]:
        """Generate model size knockout plot for a single cell."""
        # Similar implementation to architecture knockout but grouped by model size
        # For now, return None to indicate not implemented
        return None

    def _generate_cell_window_size_knockout(self, fulfilled_reqs) -> Optional[go.Figure]:
        """Generate window size knockout plot for a single cell."""
        # Similar implementation to architecture knockout but grouped by window size
        # For now, return None to indicate not implemented
        return None

    def _generate_cell_feature_knockout(self, fulfilled_reqs) -> Optional[go.Figure]:
        """Generate feature knockout plot for a single cell."""
        # Similar implementation to architecture knockout but grouped by feature
        # For now, return None to indicate not implemented
        return None

    def _generate_cell_shared_knockout(self, fulfilled_reqs) -> Optional[go.Figure]:
        """Generate shared knockout plot for a single cell."""
        # Similar implementation to architecture knockout but grouped by source
        # For now, return None to indicate not implemented
        return None

    def render(self) -> Optional[str]:
        """Generate and display a plot based on the plot plan."""
        st.subheader(f"{FINAL_PLOTS_TEXTS.generating_plot(self.plot_plan.title)}")

        # Show style form in sidebar
        with st.sidebar:
            st.markdown("### Plot Settings")
            form_submitted = self.style.show_style_form()
            if form_submitted:
                st.success("Style settings applied!")

        # Check if we have all the required data
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        summarized_fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).summarize()
        reqs = summarized_fulfilled_reqs.to_data_reqs().to_rows()

        missing_reqs = [
            req
            for summary, req in zip(summarized_fulfilled_reqs.to_rows(), reqs)
            if summary[SummarizedDataFulfilledReqsCols.AvailableOptions] == 0
        ]

        if missing_reqs:
            st.error(f"Missing data for {len(missing_reqs)} requirements. Please run the missing requirements first.")
            return None

        # Generate the plot based on the plot type
        data_reqs_per_cell = self.plot_plan.get_data_requirements_per_cell(self.result_bank)
        grid_row_col: dict[Optional[str], dict[Optional[str], dict[Optional[str], DataReqs]]] = {}
        for cell, data_reqs in data_reqs_per_cell.items():
            grid_row_col.setdefault(cell.grids, {}).setdefault(cell.rows, {}).setdefault(cell.cols, data_reqs)

        # Add checkbox for plot recreation
        recreate_plots = st.checkbox("Recreate all plots", value=False)

        grids = list(grid_row_col.keys())
        # Create tabs for different plot views
        if grids[0] is None:
            tabs = [st.empty()]
        else:
            grid_options = self.plot_plan.grids
            assert grid_options is not None
            grid_param_definition = get_hyper_param_definition(grid_options)
            tabs = st.tabs([grid_param_definition.get_display_name(option) for option in grids])

        for grid_name, tab in zip(grids, tabs):
            # Grid layout settings
            with tab:
                rows = list(grid_row_col[grid_name].keys())
                # Create a grid of plots
                cols = list(grid_row_col[grid_name][rows[0]].keys())

                is_row_labels = rows[0] is not None
                is_col_labels = cols[0] is not None

                if is_col_labels:
                    cols_options = self.plot_plan.cols
                    assert cols_options is not None
                    cols_param_definition = get_hyper_param_definition(cols_options)
                    col_names = [cols_param_definition.get_display_name(option) for option in cols]
                    # Add an empty column for row labels
                    col_cols = st.columns(([0.2] if is_row_labels else []) + [1] * len(col_names))
                    # Skip the first column (row labels) when writing column headers
                    for col_name, col_col in zip(col_names, col_cols[1:]):
                        with col_col:
                            st.write(col_name)

                for i, row_name in enumerate(rows):
                    row_display_name = row_name
                    # Create columns for this row, including the label column
                    cols = grid_row_col[grid_name][row_name].keys()
                    cols_cols = st.columns(([0.2] if is_col_labels else []) + [1] * len(cols))

                    # Add row label in the first column if applicable
                    if rows[0] is not None:
                        with cols_cols[0]:
                            rows_options = self.plot_plan.rows
                            assert rows_options is not None
                            rows_param_definition = get_hyper_param_definition(rows_options)
                            row_display_name = rows_param_definition.get_display_name(row_name)
                            st.write(f"**{row_display_name}**")

                    # Add plots in the remaining columns
                    for col_name, col_col in zip(cols, cols_cols[1:]):
                        data_reqs = grid_row_col[grid_name][row_name][col_name]
                        with col_col:
                            self._plot_cell(data_reqs, grid_name, row_display_name, col_name, recreate_plots)

        return None
