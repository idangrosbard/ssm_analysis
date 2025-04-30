from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from pydantic import BaseModel, Field

from src.core.consts import reverse_model_id


class HeatmapPlotConfig(BaseModel):
    """Configuration for heatmap plots."""

    # Figure layout and positioning
    figure_width: float = Field(
        default=4.0,
        description="Figure width in inches",
        ge=2.0,
        le=12.0,
    )
    figure_height: float = Field(default=3.0, description="Figure height in inches", ge=2.0, le=10.0)
    minimal_title: bool = Field(default=False, description="Use minimal title without extra details")
    title_position: Tuple[float, float] = Field(default=(0.45, 0.95), description="Position of the title (x, y)")
    fontsize: int = Field(default=12, description="Font size for labels and title", ge=8, le=20)
    title_fontsize: int = Field(default=12, description="Font size for the title", ge=8, le=24)
    is_tight_layout: bool = Field(default=True, description="Use tight layout")

    # Color settings
    colormap: str = Field(default="RdYlGn", description="Colormap name (e.g., 'RdYlGn', 'coolwarm', 'viridis')")
    reverse_colormap: bool = Field(default=False, description="Reverse the colormap direction")

    # Normalization and scaling
    with_fixed_diff: bool = Field(default=True, description="Use fixed difference value for colormap scaling")
    fixed_diff: float = Field(default=0.3, description="Fixed difference value for colormap scaling", ge=0.01, le=1.0)
    is_diff_probs: bool = Field(default=True, description="Normalize values by subtracting the base probability")
    two_slopes_normalization: bool = Field(default=False, description="Use two slopes normalization")
    is_robust_normalization: bool = Field(default=False, description="Use robust normalization")

    # Axis ticks and labels
    tick_fontsize: int = Field(default=10, description="Font size for tick labels", ge=6, le=18)
    is_base_prob_in_title: bool = Field(default=False, description="Show the base probability in the title")
    x_axis_label: str = Field(default="Depth %", description="Label for x-axis")
    y_axis_label: str = Field(default="", description="Label for y-axis")
    colorbar_nbins: int = Field(default=5, description="Number of bins in the colorbar", ge=3, le=10)

    # X-axis options
    x_axis_as_percentage: bool = Field(default=True, description="Show X-axis as percentages")
    x_tick_count: int = Field(default=6, description="Number of tick marks on the x-axis", ge=2, le=12)


def simple_diff_fixed(
    prob_mat,
    model_id,
    window_size,
    last_tok,
    base_prob,
    true_word,
    toks,
    config: Optional[HeatmapPlotConfig] = None,
):
    """
    Creates a diverging heatmap with a specified baseline value and returns the plot object.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fixed_diff: Fixed difference value for colormap scaling.
    - fontsize: Font size for labels and title.
    - minimal_title: Whether to use minimal title.
    - config: Optional HeatmapPlotConfig with additional customization options.

    Returns:
    - fig, ax: The matplotlib figure and axis objects.
    """
    # Use provided config or default parameters
    if config is None:
        config = HeatmapPlotConfig()

    # Create figure with specified dimensions
    fig, ax = plt.subplots(figsize=(config.figure_width, config.figure_height))
    center = base_prob if config.is_diff_probs else 0
    # Normalize values if requested
    plot_data = prob_mat - center

    # Set scaling values for the colormap
    fixed_diff_value = config.fixed_diff

    model_arch, model_size = reverse_model_id(model_id)

    sub_params = {}

    if config.two_slopes_normalization:
        sub_params["norm"] = TwoSlopeNorm(vmin=-fixed_diff_value, vmax=fixed_diff_value, vcenter=center)
    elif config.with_fixed_diff:
        sub_params["vmin"] = -fixed_diff_value
        sub_params["vmax"] = fixed_diff_value

    # Select colormap
    cmap = config.colormap
    if config.reverse_colormap:
        cmap = f"{config.colormap}_r"

    # Plot the heatmap
    sns.heatmap(
        plot_data,
        cbar=True,
        cmap=cmap,
        robust=config.is_robust_normalization,
        ax=ax,
        **sub_params,
    )

    # Set title with appropriate formatting
    plt.suptitle(
        (
            f"{model_arch} - size {model_size}"
            + (
                ""
                if config.minimal_title
                else (f" - Window Size: {window_size}" "\n" "Knockout to last token '" r"$\bf{" f"{last_tok}" r"}$" "'")
            )
            + (f"\nbase probability: {round(base_prob, 4)}" if config.is_base_prob_in_title else "")
        ),
        position=config.title_position,
        fontsize=config.title_fontsize,
    )

    # Customize X-axis ticks
    n_cols = plot_data.shape[1]

    if config.x_axis_as_percentage:
        # Calculate positions for percentages based on the specified number of ticks
        percentages = np.linspace(0, 100, config.x_tick_count)

        # Convert percentages to positions in the matrix
        x_ticks = np.array([int((p / 100) * (n_cols - 1)) for p in percentages])
        x_ticks_labels = [str(int(i)) for i in percentages]
        x_axis_label = config.x_axis_label or "Depth %"
    else:
        # If not using percentages, use indices
        x_ticks = np.linspace(0, n_cols - 1, config.x_tick_count, dtype=int)
        x_ticks_labels = [str(i) for i in x_ticks]
        x_axis_label = config.x_axis_label or "Layer"

    # Set axis labels
    ax.set_xlabel(x_axis_label, fontsize=config.fontsize)
    ax.set_ylabel(config.y_axis_label, fontsize=config.fontsize)

    # Set ticks and labels
    ax.set_xticks(x_ticks + 0.5)
    ax.set_xticklabels(x_ticks_labels, rotation=0, fontsize=config.tick_fontsize)

    # Set Y-axis ticks
    ax.set_yticks(np.arange(plot_data.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=config.tick_fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0, labelsize=config.tick_fontsize)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.locator = MaxNLocator(nbins=config.colorbar_nbins)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=config.tick_fontsize)

    # fig.subplots_adjust(top=0.8)

    if config.is_tight_layout:
        fig.tight_layout()

    return fig, ax
