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

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_defs import ResultBank
from src.experiments.info_flow import InfoFlowConfig
from src.final_plots.app.app_consts import SummarizedDataFulfilledReqsCols
from src.final_plots.app.texts import FINAL_PLOTS_TEXTS
from src.final_plots.plot_plan import PlotPlan, PlotType
from src.types import MODEL_ARCH_AND_SIZE, TInfoFlowOutput
from src.utils.streamlit_utils import StreamlitComponent


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

SOURCES = ["first", "relation", "subject", "last"]


def load_data(input_path: Path, idx: bool = False) -> pd.DataFrame:
    """Load data from a CSV file."""
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)

    if idx:
        df = df.set_index("Unnamed: 0")

    return df


class PlotStyle:
    """Class to manage plot styling parameters."""

    def __init__(self):
        # Figure dimensions
        self.height_per_row = 400
        self.width_per_col = 400
        self.margin_top = 100
        self.margin_bottom = 100

        # Font settings
        self.font_size = 36

        # Colors
        self.paper_bgcolor = "#FFFFFF"  # white
        self.plot_bgcolor = "#FFFFFF"  # white
        self.grid_color = "#D3D3D3"  # lightgray
        self.zero_line_color = "#D3D3D3"  # lightgray
        self.axis_line_color = "#000000"  # black

        # Line settings
        self.line_width = 2
        self.grid_width = 1
        self.zero_line_width = 2
        self.fill_opacity = 0.2

        # Legend settings
        self.legend_orientation = "h"
        self.legend_y_anchor = "bottom"
        self.legend_y = 1.02
        self.legend_x_anchor = "right"
        self.legend_x = 1

    def show_style_form(self):
        """Display a form to control plot styling parameters."""
        st.subheader("Plot Style Settings")

        with st.expander("Figure Dimensions"):
            self.height_per_row = st.number_input("Height per row", min_value=100, value=self.height_per_row)
            self.width_per_col = st.number_input("Width per column", min_value=100, value=self.width_per_col)
            self.margin_top = st.number_input("Top margin", min_value=0, value=self.margin_top)
            self.margin_bottom = st.number_input("Bottom margin", min_value=0, value=self.margin_bottom)

        with st.expander("Font Settings"):
            self.font_size = st.number_input("Font size", min_value=8, value=self.font_size)

        with st.expander("Colors"):
            self.paper_bgcolor = st.color_picker("Paper background color", self.paper_bgcolor)
            self.plot_bgcolor = st.color_picker("Plot background color", self.plot_bgcolor)
            self.grid_color = st.color_picker("Grid color", self.grid_color)
            self.zero_line_color = st.color_picker("Zero line color", self.zero_line_color)
            self.axis_line_color = st.color_picker("Axis line color", self.axis_line_color)

        with st.expander("Line Settings"):
            self.line_width = st.number_input("Line width", min_value=1, value=self.line_width)
            self.grid_width = st.number_input("Grid width", min_value=1, value=self.grid_width)
            self.zero_line_width = st.number_input("Zero line width", min_value=1, value=self.zero_line_width)
            self.fill_opacity = st.slider("Fill opacity", min_value=0.0, max_value=1.0, value=self.fill_opacity)

        with st.expander("Legend Settings"):
            self.legend_orientation = st.selectbox("Legend orientation", ["h", "v"], index=0)
            self.legend_y = st.number_input("Legend Y position", value=self.legend_y)
            self.legend_x = st.number_input("Legend X position", value=self.legend_x)
            self.legend_y_anchor = st.selectbox("Legend Y anchor", ["top", "bottom", "middle"], index=1)
            self.legend_x_anchor = st.selectbox("Legend X anchor", ["left", "right", "center"], index=1)

    def apply_to_figure(self, fig: go.Figure, n_cols: int, n_rows: int = 1) -> go.Figure:
        """Apply the current style settings to a figure."""
        fig.update_layout(
            height=self.height_per_row * n_rows,
            width=self.width_per_col * n_cols,
            # paper_bgcolor=self.paper_bgcolor,
            # plot_bgcolor=self.plot_bgcolor,
            margin=dict(t=self.margin_top, b=self.margin_bottom),
            font=dict(size=self.font_size),
            legend=dict(
                orientation=self.legend_orientation,
                yanchor=self.legend_y_anchor,
                y=self.legend_y,
                xanchor=self.legend_x_anchor,
                x=self.legend_x,
                font=dict(size=self.font_size),
            ),
        )

        # Update all axes
        for i in range(1, n_cols + 1):
            for j in range(1, n_rows + 1):
                # X axis
                fig.update_xaxes(
                    title_text="Relative depth (%)" if j == n_rows else None,
                    showline=True,
                    linewidth=self.line_width,
                    linecolor=self.axis_line_color,
                    showgrid=True,
                    gridwidth=self.grid_width,
                    gridcolor=self.grid_color,
                    zeroline=True,
                    zerolinewidth=self.zero_line_width,
                    zerolinecolor=self.zero_line_color,
                    tickfont=dict(size=self.font_size),
                    col=i,
                    row=j,
                )

                # Y axis
                fig.update_yaxes(
                    title_text="Probability diff" if i == 1 else None,
                    showline=True,
                    linewidth=self.line_width,
                    linecolor=self.axis_line_color,
                    showgrid=True,
                    gridwidth=self.grid_width,
                    gridcolor=self.grid_color,
                    zeroline=True,
                    zerolinewidth=self.zero_line_width,
                    zerolinecolor=self.zero_line_color,
                    tickfont=dict(size=self.font_size),
                    col=i,
                    row=j,
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

    name_conversion = {
        "Mamba-1 2.8B": "mamba-2.8B",
        "Mamba-2 2.7B": "mamba2-2.7B",
        "Falcon-Mamba 7B": "falcon-mamba-7B",
        "GPT-2 1.5B": "gpt2-1.5B",
    }

    if model in name_conversion:
        title = name_conversion[model]
    else:
        title = model

    fig.add_trace(
        go.Scatter(
            x=100 * filtered["Depth"],
            y=filtered["Probability diff_mean"],
            line=dict(color=f"rgb{rgb}", dash=line_dash, width=style.line_width),
            mode="lines",
            name=title,
            showlegend=((col == 1) & (row == 1)),
        ),
        col=col,
        row=row,
    )
    fig.add_trace(
        go.Scatter(
            x=(100 * filtered["Depth"]).tolist() + (100 * filtered["Depth"][::-1]).tolist(),  # x, then x reversed
            y=upper.tolist() + lower[::-1].tolist(),  # upper, then lower reversed
            fill="toself",
            fillcolor=f"rgba{rgb + (style.fill_opacity,)}",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        col=col,
        row=row,
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
        for req in fulfilled_reqs.to_rows():
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
                df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0]))
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
        for req in fulfilled_reqs.to_rows():
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
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0]))
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
        for req in fulfilled_reqs.to_rows():
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
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0]))
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
        for req in fulfilled_reqs.to_rows():
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
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0]))
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
        for req in fulfilled_reqs.to_rows():
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
                    df = load_info_flow_data(InfoFlowConfig.load_output(fulfilled_reqs._raw[req][0]))
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

    def _generate_heatmap_plot(self) -> Optional[go.Figure]:
        """Generate a heatmap plot."""
        # Heatmap plots are more complex and would require additional implementation
        st.warning("Heatmap plot generation is not yet implemented.")
        return None

    def render(self) -> Optional[str]:
        """Generate and display a plot based on the plot plan."""
        st.subheader(f"{FINAL_PLOTS_TEXTS.generating_plot(self.plot_plan.title)}")

        # Show style form
        self.style.show_style_form()

        # Check if we have all the required data
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        summarized_fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).summarize(None)
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
        fig = None
        if self.plot_plan.plot_type == PlotType.ARCHITECTURE_KNOCKOUT:
            fig = self._generate_architecture_knockout_plot()
        elif self.plot_plan.plot_type == PlotType.MODEL_SIZE_KNOCKOUT:
            fig = self._generate_model_size_knockout_plot()
        elif self.plot_plan.plot_type == PlotType.WINDOW_SIZE_KNOCKOUT:
            fig = self._generate_window_size_knockout_plot()
        elif self.plot_plan.plot_type == PlotType.FEATURE_KNOCKOUT:
            fig = self._generate_feature_knockout_plot()
        elif self.plot_plan.plot_type == PlotType.SHARED_KNOCKOUT:
            fig = self._generate_shared_knockout_plot()
        elif self.plot_plan.plot_type == PlotType.HEATMAP:
            fig = self._generate_heatmap_plot()

        if fig is None:
            st.error("Failed to generate plot.")
            return None

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Save the plot if an output path is specified
        if self.plot_plan.output_path:
            output_path = Path(self.plot_plan.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                fig.write_image(str(output_path), scale=4)
                st.success(f"Plot saved to {output_path}")
                return str(output_path)
            except Exception as e:
                st.error(f"Error saving plot: {e}")
                return None

        return None
