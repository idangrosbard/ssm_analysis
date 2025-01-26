import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import SymLogNorm, TwoSlopeNorm

def create_diverging_heatmap(
    prob_mat, 
    model_id, 
    window_size, 
    last_tok, 
    base_prob, 
    true_word, 
    toks, 
    fontsize=8
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
        cmap='RdYlGn',  # Red for lower, white for baseline, green for higher
        norm=norm,
        ax=ax
    )

    # Add title and labels
    ax.set_title(
        f'{model_id} - Window Size: {window_size}' +
        f'\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}', 
        fontsize=fontsize
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f'p({true_word[1:]})', labelpad=10, fontsize=fontsize)
    cbar.locator = plt.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fontsize)

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors


from matplotlib.colors import Normalize

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
    high_percentile=95
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
        prob_mat, 
        cbar=True, 
        cmap='RdYlGn',  # Diverging colormap
        norm=norm, 
        ax=ax
    )

    # Add title and labels
    ax.set_title(
        f'{model_id} - Window Size: {window_size}' +
        f'\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}', 
        fontsize=fontsize
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f'p({true_word[1:]})', labelpad=10, fontsize=fontsize)
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
    high_percentile=95
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
        cmap='RdYlGn',  # Diverging colormap
        robust=True,
        # norm=norm,
        ax=ax
    )

    # Add title and labels
    ax.set_title(
        f'{model_id} - Window Size: {window_size}' +
        f'\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}', 
        fontsize=fontsize
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f'p({true_word[1:]})', labelpad=15, fontsize=fontsize)
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
    high_percentile=95
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
        cmap='RdYlGn',  # Diverging colormap
        robust=True,
        # norm=norm,
        ax=ax
    )

    # Add title and labels
    ax.set_title(
        f'{model_id} - Window Size: {window_size}' +
        f'\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}', 
        fontsize=fontsize
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f'p({true_word[1:]})', labelpad=15, fontsize=fontsize)
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
    high_percentile=95
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
        cmap='RdYlGn',  # Diverging colormap
        # robust=True,
        norm=norm,
        ax=ax
    )

    # Add title and labels
    ax.set_title(
        f'{model_id} - Window Size: {window_size}' +
        f'\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}', 
        fontsize=fontsize
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f'p({true_word[1:]}) - base_prob', labelpad=15, fontsize=fontsize)
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
    high_percentile=95
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
        cmap='RdYlGn',  # Diverging colormap
        # robust=True,
        norm=norm,
        ax=ax
    )

    # Add title and labels
    ax.set_title(
        f'{model_id} - Window Size: {window_size}' +
        f'\nIntervening on flow to: {last_tok}, base probability: {round(base_prob, 4)}', 
        fontsize=fontsize
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Customize ticks
    x_pos = list(range(0, prob_mat.shape[1], 5))
    ax.set_xticks(np.array(range(0, prob_mat.shape[1], 5)) + 0.5)
    ax.set_xticklabels([str(x) for x in x_pos], rotation=0, fontsize=fontsize)
    ax.set_yticks(np.arange(prob_mat.shape[0]) + 0.5)
    ax.set_yticklabels(toks, rotation=0, fontsize=fontsize)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='both', length=0)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xlabel(f'p({true_word[1:]}) - base_prob', labelpad=15, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax
    