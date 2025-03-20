from typing import NamedTuple

import streamlit as st

from src.app.app_consts import AppSessionKeys
from src.app.texts import AppGlobalText
from src.core.types import MODEL_ARCH_AND_SIZE
from src.experiments.runners.heatmap import HEATMAP_PLOT_FUNCS
from src.utils.infra.slurm import SLURM_GPU_TYPE


def select_gpu_type():
    options = ["smart"] + [value for value in SLURM_GPU_TYPE]
    st.selectbox(
        AppGlobalText.gpu_type,
        options=options,
        key=AppSessionKeys._selected_gpu.key,
    )


def select_variation():
    st.text_input(
        AppGlobalText.variation,
        key=AppSessionKeys.variation.key,
    )


def select_window_size():
    options = [1, 3, 5, 7, 9, 12, 15]
    st.selectbox(
        AppGlobalText.window_size,
        options=options,
        key=AppSessionKeys.window_size.key,
        index=options.index(AppSessionKeys.window_size.value),
    )


def select_models_and_sizes(available_models: list[MODEL_ARCH_AND_SIZE]) -> list[MODEL_ARCH_AND_SIZE]:
    """Display a multi-select widget for choosing model architectures and sizes.

    Args:
        available_models: List of (model_arch, model_size) tuples to choose from

    Returns:
        List of selected (model_arch, model_size) tuples
    """
    # Create display names for models
    model_options = [model_arch_and_size for model_arch_and_size in available_models]
    model_display_names = [model_arch_and_size.model_name for model_arch_and_size in model_options]

    # Create mapping from display name back to tuple
    name_to_model: dict[str, MODEL_ARCH_AND_SIZE] = dict(zip(model_display_names, model_options))

    with st.expander("Filter Models", expanded=False):
        selected_names = st.pills(
            "Select Models",
            options=model_display_names,
            default=model_display_names,
            key="model_multiselect",
            selection_mode="multi",
        )

        # Convert selected names back to model tuples
        selected_models = [name_to_model[name] for name in selected_names]

    return selected_models


class HeatmapPlotsParams(NamedTuple):
    plot_name: HEATMAP_PLOT_FUNCS


def choose_heatmap_parms():
    return HeatmapPlotsParams(
        plot_name=st.selectbox(
            "Plot Name",
            options=list(HEATMAP_PLOT_FUNCS),
            index=0,
        )
    )
