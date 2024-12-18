import matplotlib.pyplot as plt
from matplotlib.text import Text
import seaborn as sns
import numpy as np
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors
from matplotlib.colors import Normalize
from torch import mode

from src.consts import reverse_model_id


def plot_simple_heatmap(
    prob_mat, model_id, window_size, last_tok, base_prob, true_word, toks, fontsize=8
):
    """
    Creates a diverging heatmap with a specified baseline value and returns the plot object.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - baseline_value: The baseline value to set as white in the colormap.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fontsize: Font size for labels and title.

    Returns:
    - fig, ax: The matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    # Create a diverging colormap
    norm = TwoSlopeNorm(vmin=prob_mat.min(), vcenter=base_prob, vmax=prob_mat.max())

    # Plot the heatmap
    sns.heatmap(
        prob_mat,
        cbar=True,
        cmap="RdYlGn",  # Red for lower, white for baseline, green for higher
        norm=norm,
        ax=ax,
    )

    # Add title and labels
    ax.set_title(
        f"{model_id} - Window Size: {window_size}"
        + f"\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}",
        fontsize=fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f"p({true_word[1:]})", labelpad=10, fontsize=fontsize)
    cbar.locator = plt.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fontsize)
    plt.tight_layout()

    return fig, ax


def simple_diff_fixed(
    prob_mat,
    model_id,
    window_size,
    last_tok,
    base_prob,
    true_word,
    toks,
    fixed_diff,
    fontsize=12,
    minimal_title=False,
):
    """
    Creates a diverging heatmap with a specified baseline value and returns the plot object.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - baseline_value: The baseline value to set as white in the colormap.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fontsize: Font size for labels and title.

    Returns:
    - fig, ax: The matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    prob_mat = prob_mat - base_prob

    model_arch, model_size = reverse_model_id(model_id)

    # Create a diverging colormap
    # norm = TwoSlopeNorm(vmin=-fixed_diff, vcenter=0, vmax=fixed_diff)

    # Plot the heatmap
    sns.heatmap(
        prob_mat,
        cbar=True,
        cmap="RdYlGn",  # Red for lower, white for baseline, green for higher
        # norm=norm,
        vmin=-fixed_diff,
        vmax=fixed_diff,
        ax=ax,
    )
    plt.suptitle(
        (
            f"{model_arch} - size {model_size}"
            + (
                ""
                if minimal_title
                else (
                    f" - Window Size: {window_size}"
                    "\n"
                    "Knockout to last token '"
                    r"$\bf{"
                    f"{last_tok}"
                    r"}$"
                    "'"
                )
            )
        )
        # f"'base probability: {round(base_prob, 4)}"
        ,
        position=(0.45, 0.95),
        fontsize=12,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Customize ticks
    tick_size = int(prob_mat.shape[1] // 9)
    x_pos = list(range(0, prob_mat.shape[1], tick_size))
    x_pos = list(range(0, prob_mat.shape[1], tick_size))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], tick_size)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0, labelsize=fontsize)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    ax.set_xlabel(
        f"p_knockout({true_word[1:]}) - p_base({true_word[1:]})[={round(base_prob, 4)}]",
        labelpad=5,
        fontsize=fontsize,
        # loc='left'
    )
    cbar.locator = plt.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fontsize)

    fig.subplots_adjust(top=0.8)

    # plt.tight_layout()

    return fig, ax


def plot_percentile_normalized_heatmap(
    prob_mat,
    model_id,
    window_size,
    last_tok,
    base_prob,
    true_word,
    toks,
    fontsize=8,
    low_percentile=5,
    high_percentile=95,
):
    """
    Plots a heatmap normalized to a fixed percentile range for consistent comparison.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fontsize: Font size for labels and title.
    - low_percentile: Lower percentile for normalization.
    - high_percentile: Upper percentile for normalization.
    """
    # Compute percentiles
    vmin = np.percentile(prob_mat, low_percentile)
    vmax = np.percentile(prob_mat, high_percentile)

    # Create normalization object
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        prob_mat, cbar=True, cmap="RdYlGn", norm=norm, ax=ax  # Diverging colormap
    )

    # Add title and labels
    ax.set_title(
        f"{model_id} - Window Size: {window_size}"
        + f"\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}",
        fontsize=fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f"p({true_word[1:]})", labelpad=10, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax


def plot_heatmap_robust(
    prob_mat,
    base_prob,
    model_id,
    window_size,
    last_tok,
    true_word,
    toks,
    fontsize=8,
    low_percentile=5,
    high_percentile=95,
):
    """
    Plots a heatmap normalized to a fixed percentile range for consistent comparison.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fontsize: Font size for labels and title.
    - low_percentile: Lower percentile for normalization.
    - high_percentile: Upper percentile for normalization.
    """
    # Compute percentiles
    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # # Create normalization object
    # norm = Normalize(vmin=vmin, vmax=vmax)

    # prob_mat = prob_mat - base_prob

    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=prob_mat.min(), vmax=prob_mat.max())

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        prob_mat,
        cbar=True,
        cmap="RdYlGn",  # Diverging colormap
        robust=True,
        # norm=norm,
        ax=ax,
    )

    # Add title and labels
    ax.set_title(
        f"{model_id} - Window Size: {window_size}"
        + f"\nKnockout tokens to: {last_tok}, base probability: {round(base_prob, 4)}",
        fontsize=fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f"p({true_word[1:]})", labelpad=15, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax


def plot_heatmap_robust_diff(
    prob_mat,
    base_prob,
    model_id,
    window_size,
    last_tok,
    true_word,
    toks,
    fontsize=8,
    low_percentile=5,
    high_percentile=95,
):
    """
    Plots a heatmap normalized to a fixed percentile range for consistent comparison.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fontsize: Font size for labels and title.
    - low_percentile: Lower percentile for normalization.
    - high_percentile: Upper percentile for normalization.
    """
    # Compute percentiles
    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # # Create normalization object
    # norm = Normalize(vmin=vmin, vmax=vmax)

    prob_mat = prob_mat - base_prob

    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=prob_mat.min(), vmax=prob_mat.max())

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        prob_mat,
        cbar=True,
        cmap="RdYlGn",  # Diverging colormap
        robust=True,
        # norm=norm,
        ax=ax,
    )

    # Add title and labels
    ax.set_title(
        f"{model_id} - Window Size: {window_size}"
        + f"\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}",
        fontsize=fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(
        f"p({true_word[1:]}) - base_prob", labelpad=15, fontsize=fontsize
    )
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax


def plot_heatmap_diff_symlog(
    prob_mat,
    base_prob,
    model_id,
    window_size,
    last_tok,
    true_word,
    toks,
    fontsize=8,
    low_percentile=5,
    high_percentile=95,
):
    """
    Plots a heatmap normalized to a fixed percentile range for consistent comparison.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fontsize: Font size for labels and title.
    - low_percentile: Lower percentile for normalization.
    - high_percentile: Upper percentile for normalization.
    """
    # Compute percentiles
    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # # Create normalization object
    # norm = Normalize(vmin=vmin, vmax=vmax)

    prob_mat = prob_mat - base_prob

    norm = SymLogNorm(linthresh=0.01, vmin=-0.15, vmax=0.1, base=10)
    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=prob_mat.min(), vmax=prob_mat.max())

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        prob_mat,
        cbar=True,
        cmap="RdYlGn",  # Diverging colormap
        # robust=True,
        norm=norm,
        ax=ax,
    )

    # Add title and labels
    ax.set_title(
        f"{model_id} - Window Size: {window_size}"
        + f"\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}",
        fontsize=fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(
        f"p({true_word[1:]}) - base_prob", labelpad=15, fontsize=fontsize
    )
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax


def plot_heatmap(
    prob_mat,
    base_prob,
    model_id,
    window_size,
    last_tok,
    true_word,
    toks,
    fontsize=8,
    low_percentile=5,
    high_percentile=95,
):
    """
    Plots a heatmap normalized to a fixed percentile range for consistent comparison.

    Parameters:
    - prob_mat: 2D numpy array of probabilities to visualize.
    - model_id: Identifier for the model (for the title).
    - window_size: Window size parameter (for the title).
    - last_tok: Token intervened upon (for the title).
    - base_prob: Base probability (for the title).
    - true_word: The true word label for the colorbar.
    - toks: List of tokens for y-axis labels.
    - fontsize: Font size for labels and title.
    - low_percentile: Lower percentile for normalization.
    - high_percentile: Upper percentile for normalization.
    """
    # Compute percentiles
    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # # Create normalization object
    # norm = Normalize(vmin=vmin, vmax=vmax)

    prob_mat = prob_mat - base_prob

    norm = SymLogNorm(linthresh=0.01, vmin=-0.15, vmax=0.1, base=10)
    # vmin = np.percentile(prob_mat, low_percentile)
    # vmax = np.percentile(prob_mat, high_percentile)

    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=prob_mat.min(), vmax=prob_mat.max())

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        prob_mat,
        cbar=True,
        cmap="RdYlGn",  # Diverging colormap
        # robust=True,
        norm=norm,
        ax=ax,
    )

    # Add title and labels
    ax.set_title(
        f"{model_id} - Window Size: {window_size}"
        + f"\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}",
        fontsize=fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(
        f"p({true_word[1:]}) - base_prob", labelpad=15, fontsize=fontsize
    )
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax
