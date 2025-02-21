import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src.consts import reverse_model_id


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
                else (f" - Window Size: {window_size}" "\n" "Knockout to last token '" r"$\bf{" f"{last_tok}" r"}$" "'")
            )
        ),
        # f"'base probability: {round(base_prob, 4)}"
        position=(0.45, 0.95),
        fontsize=12,
    )

    ax.set_xlabel("Depth %")
    ax.set_ylabel("")

    # Customize ticks
    n_cols = prob_mat.shape[1]
    # Calculate positions for 0, 20, 40, 60, 80, 100 percent
    percentages = np.linspace(0, 100, 6)
    x_positions = [(p / 100) * (n_cols - 1) for p in percentages]
    ax.set_xticks(np.array(x_positions) + 0.5)
    ax.set_xticklabels([f"{int(p)}" for p in percentages], rotation=0, fontsize=fontsize)

    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="both", length=0, labelsize=fontsize)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=fontsize)

    fig.subplots_adjust(top=0.8)

    return fig, ax
