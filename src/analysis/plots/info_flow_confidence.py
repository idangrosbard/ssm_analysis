from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict, cast

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color
from scipy import stats

from src.core.consts import CONVERT_TO_PLOTLY_LINE_STYLE
from src.core.names import COLS
from src.core.types import FeatureCategory, TInfoFlowOutput, TLineStyle, TokenType
from src.utils.types_utils import first_dict_value


class InfoFlowPlotConfig(BaseModel):
    """Configuration for Info Flow confidence plots."""

    with_fixed_limits: bool = Field(default=False, description="Use fixed limits for y-axis")
    acc_ylim_min: float = Field(default=60.0, description="Minimum y value for accuracy plot", ge=0.0, le=100.0)
    acc_ylim_max: float = Field(default=105.0, description="Maximum y value for accuracy plot", ge=0.0, le=110.0)
    diff_ylim_min: float = Field(default=-50.0, description="Minimum y value for difference plot", ge=-100.0, le=0.0)
    diff_ylim_max: float = Field(default=50.0, description="Maximum y value for difference plot", ge=0.0, le=100.0)
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals", ge=0.5, le=0.99)
    title: str = Field(default="", description="Custom plot title (leave empty for default)")
    metrics_to_show: List[Literal["acc", "diff"]] = Field(
        default=["acc", "diff"], description="Which metrics to show in the plot"
    )

    # Figure and Layout options
    figure_width: float = Field(default=12.0, description="Figure width in inches", ge=4.0, le=24.0)
    figure_height: float = Field(default=5.0, description="Figure height in inches", ge=3.0, le=18.0)
    title_fontsize: int = Field(default=12, description="Font size for title", ge=8, le=24)
    axis_fontsize: int = Field(default=10, description="Font size for axis labels", ge=8, le=20)

    # Color and style options
    custom_colors: Dict[TokenType, Color] = Field(
        default=None, description="Custom colors mapping token types to hex color codes"
    )
    custom_line_styles: Dict[FeatureCategory, TLineStyle] = Field(
        default=None, description="Custom line styles mapping feature categories to styles"
    )
    alpha: float = Field(default=0.2, description="Alpha (transparency) for confidence intervals", ge=0.0, le=1.0)

    # X-axis options
    x_axis_as_percentage: bool = Field(
        default=True, description="Show X-axis (layer positions) as percentages instead of indices"
    )
    x_tick_count: int = Field(default=6, description="Number of tick marks on the x-axis", ge=2, le=12)

    # Column sizing - adjusts subplot widths based on metrics shown
    subplot_width_ratio: Dict[str, float] = Field(
        default={"acc": 1.0, "diff": 1.0}, description="Width ratio for each subplot type (proportional sizing)"
    )


class MetricData(TypedDict):
    mean: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]


class Confidence(TypedDict):
    mean: float
    ci_lower: float
    ci_upper: float


class MetricsDict(TypedDict):
    acc: MetricData
    diff: MetricData


class PlotMetadata(TypedDict):
    title: str
    ylabel: str
    ylabel_loc: Literal["bottom", "center", "top"]
    axhline_value: float
    ylim: Optional[tuple[float, float]]


class LinePlotMetadata(TypedDict):
    label: str
    color: str
    linestyle: str
    data: TInfoFlowOutput


# region Confidence Calculation
def calculate_ci(data: NDArray[np.float64], confidence_level: float = 0.95) -> Confidence:
    """Calculate confidence intervals for a given data set using standard error."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n_samples = len(data)

    if std == 0 or n_samples == 1:
        # If no variance or single sample, CI is just the mean
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # Calculate standard error and confidence interval
    se = std / np.sqrt(n_samples)
    ci = stats.t.interval(confidence_level, df=n_samples - 1, loc=mean, scale=se)

    return {
        "mean": float(mean),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def calculate_pi(data: NDArray[np.float64], confidence_level: float = 0.95) -> Confidence:
    """Calculate prediction intervals for a given data set."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n_samples = len(data)

    if std == 0 or n_samples == 1:
        # If no variance or single sample, PI is just the mean
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # For prediction interval, we need to account for both
    # the uncertainty in the mean and the spread of future observations
    pi_scale = std * np.sqrt(1 + 1 / n_samples)
    pi = stats.t.interval(confidence_level, df=n_samples - 1, loc=mean, scale=pi_scale)

    return {
        "mean": float(mean),
        "ci_lower": float(pi[0]),
        "ci_upper": float(pi[1]),
    }


def calculate_bootstrap(
    data: NDArray[np.float64], confidence_level: float = 0.95, n_bootstrap: int = 10000
) -> Confidence:
    """Calculate bootstrap confidence intervals for a given data set."""
    mean = np.mean(data)
    n_samples = len(data)

    if n_samples == 1:
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # Generate bootstrap samples
    rng = np.random.default_rng()
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = rng.choice(data, size=n_samples, replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # Calculate percentile confidence intervals
    alpha = (1 - confidence_level) / 2
    ci_lower, ci_upper = np.percentile(bootstrap_means, [100 * alpha, 100 * (1 - alpha)])

    return {
        "mean": float(mean),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def calculate_se(data: NDArray[np.float64], confidence_level: float = 0.95) -> Confidence:
    """Calculate standard error based intervals for a given data set."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n_samples = len(data)

    if std == 0 or n_samples == 1:
        return {
            "mean": float(mean),
            "ci_lower": float(mean),
            "ci_upper": float(mean),
        }

    # Calculate standard error
    se = std / np.sqrt(n_samples)

    # Use normal distribution (simpler than t-distribution)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_score * se

    return {
        "mean": float(mean),
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
    }


def calculate_confidence(
    confidence_method: Literal["CI", "PI", "bootstrap", "SE"],
    data: NDArray[np.float64],
    confidence_level: float = 0.95,
) -> Confidence:
    """Calculate confidence intervals for a given data set."""
    if confidence_method == "CI":
        return calculate_ci(data, confidence_level)
    elif confidence_method == "PI":
        return calculate_pi(data, confidence_level)
    elif confidence_method == "bootstrap":
        return calculate_bootstrap(data, confidence_level)
    elif confidence_method == "SE":
        return calculate_se(data, confidence_level)
    else:
        raise ValueError(f"Invalid confidence method: {confidence_method}")


# endregion


def calculate_metrics_with_confidence(
    window_outputs: TInfoFlowOutput,
    metric_types: list[Literal["acc", "diff"]],
    confidence_level: float = 0.95,
    confidence_method: Literal["CI", "PI", "bootstrap", "SE"] = "CI",
) -> MetricsDict:
    """
    Calculate metrics with confidence intervals from raw window outputs.

    Returns:
        Dictionary with keys 'acc' and 'diff', each containing:
            - 'mean': mean values per window
            - 'ci_lower': lower confidence bound
            - 'ci_upper': upper confidence bound
    """
    metrics: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    metric_to_name = {
        "acc": COLS.INFO_FLOW.HIT,
        "diff": COLS.INFO_FLOW.DIFFS,
    }

    for window_idx in window_outputs.keys():
        window_data = window_outputs[window_idx]

        for metric_type in metric_types:
            confidence = calculate_confidence(
                confidence_method=confidence_method,
                data=np.array(window_data[metric_to_name[metric_type].value]),
                confidence_level=confidence_level,
            )
            for key in confidence:
                metrics[metric_type][key].append(float(confidence[key]))  # type: ignore

    return cast(
        MetricsDict,
        {
            metric_type: {key: np.array(value) for key, value in metrics[metric_type].items()}
            for metric_type in metric_types
        },
    )


def plot_with_confidence(
    metrics: MetricsDict,
    metric_type: Literal["acc", "diff"],
    label: str,
    color: str,
    linestyle: str,
    ax: Axes,
    alpha: float = 0.2,
):
    """Plot a single metric with confidence intervals."""
    layers = np.arange(len(metrics[metric_type]["mean"]))

    # Plot mean line
    ax.plot(
        layers,
        metrics[metric_type]["mean"] * (100 if metric_type == "acc" else 1),
        label=label,
        color=color,
        linestyle=linestyle,
    )

    # Plot confidence interval
    ax.fill_between(
        layers,
        metrics[metric_type]["ci_lower"] * (100 if metric_type == "acc" else 1),
        metrics[metric_type]["ci_upper"] * (100 if metric_type == "acc" else 1),
        color=color,
        alpha=alpha,
    )


def create_confidence_plot(
    lines_metadata: list[LinePlotMetadata],
    confidence_level: float,
    title: str,
    plots_meta_data: dict[Literal["acc", "diff"], PlotMetadata],
    config: Optional[InfoFlowPlotConfig] = None,
) -> Figure:
    """Create plots with confidence intervals for all metrics.

    Args:
        lines_metadata: List of metadata for each line to plot
        confidence_level: Confidence level for intervals (0-1)
        title: Title for the plot
        plots_meta_data: Dictionary mapping metric types to their plot metadata
        config: Optional plot configuration

    Returns:
        The matplotlib figure containing the plots
    """
    # Use default config if none provided
    if config is None:
        config = InfoFlowPlotConfig()

    # Determine subplot width ratios based on the metrics shown
    if len(plots_meta_data) > 1:
        width_ratios = [config.subplot_width_ratio.get(metric_type, 1.0) for metric_type in plots_meta_data.keys()]
    else:
        width_ratios = None

    # Create figure with subplots side by side, with possibly different widths
    fig, axes = plt.subplots(
        1,
        len(plots_meta_data),
        figsize=(config.figure_width, config.figure_height),
        gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
    )

    # If only one subplot, wrap in list
    if len(plots_meta_data) == 1:
        axes = [axes]

    # Dictionary to store unique handles by their properties
    unique_handles = {}

    # Get number of points from first window of first block
    first_window = first_dict_value(lines_metadata[0]["data"])
    n_points = len(first_window[COLS.INFO_FLOW.HIT.value])

    max_layers = max(len(line_metadata["data"]) for line_metadata in lines_metadata)
    # Process each metric type (accuracy and diff)
    for i, (metric_type, plot_metadata) in enumerate(plots_meta_data.items()):
        ax = axes[i]

        if config.x_axis_as_percentage:
            x_axis_label = "Layer Depth (%)"
            x_normalized_steps = np.linspace(0, 100, max_layers)

            percentages = np.linspace(0, 100, config.x_tick_count)
            x_ticks = [int(p / 100 * (max_layers - 1)) for p in percentages]
            x_ticks_labels = [str(int(x)) for x in percentages]

        else:
            x_axis_label = "Layer Depth"
            x_ticks = np.linspace(0, max_layers - 1, config.x_tick_count, dtype=int)
            x_ticks_labels = [str(int(x)) for x in x_ticks]

        # Plot data for each block
        for line_metadata in lines_metadata:
            # Apply custom colors if provided
            color = line_metadata["color"]
            if config.custom_colors is not None:
                for token_type, custom_color in config.custom_colors.items():
                    if token_type.value in line_metadata["label"]:
                        color = str(custom_color)
                        break

            # Apply custom line styles if provided
            linestyle = line_metadata["linestyle"]
            if config.custom_line_styles is not None:
                for feature_cat, custom_style in config.custom_line_styles.items():
                    if feature_cat.value in line_metadata["label"]:
                        linestyle = custom_style.value
                        break

            metrics = calculate_metrics_with_confidence(
                line_metadata["data"], list(plots_meta_data.keys()), confidence_level
            )

            # Get layer indices and convert to percentage if requested

            num_layers = len(metrics[metric_type]["mean"])

            if config.x_axis_as_percentage:
                # Convert layer indices to percentages (0-100%)
                def interpolate_to_percentage(values):
                    # interpolate to get the x_ticks
                    return np.interp(x_normalized_steps, np.linspace(0, 100, num_layers), values)

                metrics[metric_type]["mean"] = interpolate_to_percentage(metrics[metric_type]["mean"])
                metrics[metric_type]["ci_lower"] = interpolate_to_percentage(metrics[metric_type]["ci_lower"])
                metrics[metric_type]["ci_upper"] = interpolate_to_percentage(metrics[metric_type]["ci_upper"])

            # Draw the plot
            plot_with_confidence(
                metrics=metrics,
                metric_type=metric_type,
                label=line_metadata["label"],
                color=color,
                linestyle=linestyle,
                ax=ax,
                alpha=config.alpha,
            )

            # Only collect handles and labels from the first subplot
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    # Create a unique key based on the handle's visual properties
                    key = (label, handle.get_color(), handle.get_linestyle())
                    if key not in unique_handles:
                        unique_handles[key] = (handle, label)

        # Customize subplot
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Set X-axis label and ticks
        ax.set_xlabel(x_axis_label, fontsize=config.axis_fontsize)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels, fontsize=config.axis_fontsize)

        ax.axhline(plot_metadata["axhline_value"], color="gray", linewidth=1)
        ax.set_ylabel(plot_metadata["ylabel"], fontsize=config.axis_fontsize, loc=plot_metadata["ylabel_loc"])
        if plot_metadata["ylim"]:
            ax.set_ylim(plot_metadata["ylim"])
        ax.tick_params(axis="both", which="major", labelsize=config.axis_fontsize)
        ax.set_title(plot_metadata["title"], fontsize=config.title_fontsize)

        # Adjust tick parameters
        ax.tick_params(axis="both", which="both", length=0, labelsize=config.axis_fontsize)

    # Extract unique handles and labels
    all_handles, all_labels = zip(*unique_handles.values()) if unique_handles else ([], [])

    # Create a single legend for the entire figure
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=len(all_handles),
        fontsize=config.axis_fontsize,
        frameon=False,
    )

    # Set overall title
    custom_title = config.title if hasattr(config, "title") and config.title else title
    fig.suptitle(
        f"{custom_title} ({n_points = })",
        fontsize=config.title_fontsize,
    )

    fig.tight_layout()
    return fig


def combine_confidence_plots(
    figs: dict[str, Figure],
    output_path: Optional[Path] = None,
    suptitle: Optional[str] = None,
    figsize: tuple[float, float] = (15, 20),
    show_fig: bool = True,
) -> Figure:
    """
    Combine multiple confidence plots into a single comparison figure.

    Args:
        figs: Dictionary mapping model names to their figures
        output_path: Optional path to save the combined figure
        suptitle: Optional super title for the combined figure
        figsize: Size of the combined figure (width, height)
        show_fig: Whether to display the figure

    Returns:
        Combined matplotlib figure
    """
    n_models = len(figs)
    fig, axes = plt.subplots(n_models, 2, figsize=figsize)

    # If only one model, wrap axes in a list to make it 2D
    if n_models == 1:
        axes = np.array([axes])

    for i, (model_name, model_fig) in enumerate(figs.items()):
        # Extract the subplots from the original figure
        for j, ax_orig in enumerate(model_fig.axes):
            # Copy the plot data to the new axes
            ax_new = axes[i, j]

            # Copy lines (main plots and confidence intervals)
            for line in ax_orig.lines:
                ax_new.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    label=line.get_label(),
                    alpha=line.get_alpha() if line.get_alpha() is not None else 1.0,
                )

            # Copy filled regions (confidence intervals)
            for collection in ax_orig.collections:
                if isinstance(collection, PolyCollection):
                    # Get the vertices of the filled region
                    path = collection.get_paths()[0]
                    verts = np.asarray(path.vertices)
                    codes = path.codes

                    # Find the indices where the path moves
                    if codes is not None:
                        move_idx = np.where(codes == path.MOVETO)[0]
                        if len(move_idx) > 0:
                            split_idx = int(move_idx[1]) if len(move_idx) > 1 else len(verts)
                            lower = verts[:split_idx]
                            upper = verts[split_idx:][::-1] if len(move_idx) > 1 else verts[::-1]

                            # Extract coordinates as numpy arrays
                            x = np.asarray(lower[:, 0])
                            y1 = np.asarray(lower[:, 1])
                            y2 = np.asarray(upper[:, 1])

                            ax_new.fill_between(
                                x,
                                y1,
                                y2,
                                color=collection.get_facecolor()[0],
                                alpha=collection.get_alpha(),
                            )

            # Copy axes properties
            ax_new.set_xlabel(ax_orig.get_xlabel())
            ax_new.set_ylabel(ax_orig.get_ylabel())
            ax_new.set_title(ax_orig.get_title())
            ax_new.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Copy limits
            ax_new.set_xlim(ax_orig.get_xlim())
            ax_new.set_ylim(ax_orig.get_ylim())

            # Copy legend
            if ax_orig.get_legend() is not None:
                handles, _ = ax_orig.get_legend_handles_labels()
                ax_new.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.2),
                    ncol=len(handles),
                    fontsize=10,
                    frameon=False,
                )

            # Add model name to the left of the row
            if j == 0:
                ax_new.text(
                    -0.2,
                    0.5,
                    model_name,
                    transform=ax_new.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=12,
                )

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=14)

    # Adjust layout to prevent overlapping
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return fig


def process_info_flow_files(
    from_blocks: dict[TokenType, tuple[dict, Path]],
    target_block: TokenType,
    plots_meta_data: dict[Literal["acc", "diff"], PlotMetadata],
    confidence_level: float = 0.95,
    save_fig: bool = True,
    show_fig: bool = True,
) -> Figure:
    """
    Process information flow files for multiple from_blocks and create plots with confidence intervals.

    Args:
        from_blocks: Dictionary mapping TokenTypes to their details and file paths
        target_block: TokenType to analyze flows to
        confidence_level: Confidence level for intervals (default: 0.95)
        colors: Dictionary mapping TokenTypes to colors (optional)
        line_styles: Dictionary mapping TokenTypes to line styles (optional)
        save_fig: Whether to save the figure (default: True)
        show_fig: Whether to show the figure (default: True)

    Returns:
        The matplotlib figure containing the plots
    """
    from src.experiments.runners.info_flow import InfoFlowRunner

    targets_window_outputs = {
        target: InfoFlowRunner.load_output(file_path) for target, (_, file_path) in from_blocks.items()
    }
    # Create plots
    first_details = next(iter(from_blocks.values()))[0]
    model_id = first_details["model_id"]
    window_size = first_details["window_size"]
    title = f"Knocking out flow to {target_block}\n{model_id}, window size={window_size}"

    fig = create_confidence_plot(
        lines_metadata=targets_window_outputs,  # type: ignore
        confidence_level=confidence_level,
        title=title,
        plots_meta_data=plots_meta_data,
    )

    if save_fig:
        # Get the output directory from the first file path
        first_file_path = next(iter(from_blocks.values()))[1]
        output_dir = first_file_path.parent.parent
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            results_dir / f"knockout_target={target_block}_with_confidence.png",
            bbox_inches="tight",
        )

    if show_fig:
        # plt.show()
        pass
    else:
        plt.close(fig)

    return fig


def create_plotly_confidence_chart(
    targets_window_outputs: list[TInfoFlowOutput],
    metric_type: Literal["acc", "diff"],
    colors: list[str],
    line_styles: list[str],
    legend_labels: list[str],
    confidence_level: float = 0.95,
) -> go.Figure:
    """
    Create a Plotly figure with confidence intervals for the specified metric.

    Args:
        targets_window_outputs: Dictionary mapping sources to their window outputs
        metric_type: Type of metric to plot ('acc' for accuracy or 'diff' for probability difference)
        confidence_level: Confidence level for intervals (0-1)
        title: Title for the plot
        custom_colors: Optional dictionary mapping sources to custom colors
        custom_line_styles: Optional dictionary mapping sources to custom line styles

    Returns:
        Plotly figure with confidence intervals
    """
    fig = go.Figure()

    # Set up y-axis parameters based on metric type
    if metric_type == "acc":
        y_title = "Accuracy (%)"
        y_range = [0, 100]
        axhline_value = 100
        multiplier = 100  # Convert to percentage
    else:  # diff
        y_title = "Probability Difference"
        y_range = None
        axhline_value = 0
        multiplier = 1

    # Add horizontal reference line
    fig.add_hline(
        y=axhline_value,
        line_dash="dash",
        line_color="gray",
        line_width=1,
    )

    max_layers = max(len(info_flow.keys()) for info_flow in targets_window_outputs)

    # Process each source
    for i, window_outputs in enumerate(targets_window_outputs):
        # Calculate metrics with confidence intervals
        metrics = calculate_metrics_with_confidence(window_outputs, [metric_type], confidence_level)

        # Get color and line style
        x_values = list(window_outputs.keys())

        # Create a legend group for this source
        legend_group = f"group_{i}"

        # Add main line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=metrics[metric_type]["mean"] * multiplier,
                mode="lines",
                line=dict(
                    color=colors[i],
                    dash=CONVERT_TO_PLOTLY_LINE_STYLE[line_styles[i]],
                    width=2,
                ),
                name=legend_labels[i],
                legendgroup=legend_group,
                showlegend=True,
            )
        )

        # Add confidence interval as a filled area
        fig.add_trace(
            go.Scatter(
                x=x_values + x_values[::-1],
                y=list(metrics[metric_type]["ci_upper"] * multiplier)
                + list(metrics[metric_type]["ci_lower"] * multiplier)[::-1],
                fill="toself",
                fillcolor=colors[i],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                legendgroup=legend_group,
                showlegend=False,
                opacity=0.2,
            )
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Layers",
        yaxis_title=y_title,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(
        tickmode="linear",
        dtick=max(1, max_layers // 10),  # Adjust tick spacing based on max layer
    )

    # Only show legend if there are multiple flows
    if len(targets_window_outputs) <= 1:
        fig.update_layout(showlegend=False)

    # Set y-axis range if specified
    if y_range:
        fig.update_yaxes(range=y_range)

    return fig
