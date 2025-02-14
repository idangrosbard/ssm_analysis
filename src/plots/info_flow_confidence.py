import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Literal, Optional, TypedDict, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats

from src.consts import COLUMNS, TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.types import TInfoFlowSource, TokenType


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


def load_window_outputs(file_path: Path) -> dict[str, dict[str, list[float]]]:
    """Load the raw window outputs from json file."""
    with open(file_path) as f:
        return json.load(f)


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


def calculate_metrics_with_confidence(
    window_outputs: dict[str, dict[str, list[float]]],
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
        "acc": COLUMNS.IF_HIT,
        "diff": COLUMNS.IF_DIFFS,
    }

    for window_idx in window_outputs.keys():
        window_data = window_outputs[window_idx]

        for metric_type in metric_types:
            confidence = calculate_confidence(
                confidence_method=confidence_method,
                data=np.array(window_data[metric_to_name[metric_type]]),
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
    block: TInfoFlowSource,
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
        label=f"{block[0]} feature={block[1]}" if isinstance(block, tuple) else block,
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
    targets_window_outputs: dict[TInfoFlowSource, dict[str, dict[str, list[float]]]],
    confidence_level: float,
    title: str,
    plots_meta_data: dict[Literal["acc", "diff"], PlotMetadata],
) -> Figure:
    """Create plots with confidence intervals for all metrics.

    Args:
        target_key: The target TokenType being analyzed
        knockout_map: Dictionary mapping TokenTypes to their details and file paths
        colors: Dictionary mapping TokenTypes to their plot colors
        line_styles: Dictionary mapping TokenTypes to their line styles
        confidence_level: Confidence level for intervals (0-1)
        for_multi_plot: Whether this is part of a multi-plot figure

    Returns:
        The matplotlib figure containing the plots
    """

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dictionary to store unique handles by their properties
    unique_handles = {}

    # Get number of points from first window of first block
    first_block = next(iter(targets_window_outputs.values()))
    first_window = next(iter(first_block.values()))
    n_points = len(first_window[COLUMNS.IF_HIT])

    # Process each metric type (accuracy and diff)
    for i, (metric_type, plot_metadata) in enumerate(plots_meta_data.items()):
        ax = axes[i]

        # Plot data for each block
        for block, window_outputs in targets_window_outputs.items():
            metrics = calculate_metrics_with_confidence(window_outputs, list(plots_meta_data.keys()), confidence_level)

            plot_with_confidence(
                metrics=metrics,
                metric_type=metric_type,
                block=block,
                color=TOKEN_TYPE_COLORS[block],
                linestyle=TOKEN_TYPE_LINE_STYLES[block],
                ax=ax,
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
        ax.set_xlabel("Layers", fontsize=12)
        ax.axhline(plot_metadata["axhline_value"], color="gray", linewidth=1)
        ax.set_ylabel(plot_metadata["ylabel"], fontsize=12, loc=plot_metadata["ylabel_loc"])
        if plot_metadata["ylim"]:
            ax.set_ylim(plot_metadata["ylim"])
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_title(plot_metadata["title"])

    # Extract unique handles and labels
    all_handles, all_labels = zip(*unique_handles.values()) if unique_handles else ([], [])

    # Create a single legend for the entire figure
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=len(all_handles),
        fontsize=10,
        frameon=False,
    )

    # Set overall title
    fig.suptitle(
        f"{title} ({n_points = })",
        fontsize=12,
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
                handles, labels = ax_orig.get_legend_handles_labels()
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
    targets_window_outputs = {target: load_window_outputs(file_path) for target, (_, file_path) in from_blocks.items()}
    # Create plots
    first_details = next(iter(from_blocks.values()))[0]
    model_id = first_details["model_id"]
    window_size = first_details["window_size"]
    title = f"Knocking out flow to {target_block}\n{model_id}, window size={window_size}"

    fig = create_confidence_plot(
        targets_window_outputs=targets_window_outputs,  # type: ignore
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
        fig.savefig(results_dir / f"knockout_target={target_block}_with_confidence.png", bbox_inches="tight")

    if show_fig:
        # plt.show()
        pass
    else:
        plt.close(fig)

    return fig
