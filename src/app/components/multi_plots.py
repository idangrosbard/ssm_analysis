from pathlib import Path

import streamlit as st

from src.analysis.plots.image_combiner import ImageGridParams, combine_image_grid
from src.analysis.prompt_filterations import SelectivePromptFilteration
from src.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys
from src.app.components.inputs import choose_heatmap_parms
from src.core.consts import GRAPHS_ORDER
from src.core.names import DATASETS
from src.core.types import MODEL_SIZE_CAT, TPromptOriginalIndex
from src.experiments.infrastructure.base_config import CommonParams
from src.experiments.runners.heatmap import HeatmapConfig, HeatmapParams
from src.utils.streamlit.components.extended_streamlit_pydantic import pydantic_input
from src.utils.streamlit.helpers.component import StreamlitComponent


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

        rows_count = len(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)
        progress_bar = st.progress(0, text="Plotting...")
        for i, model_arch_and_size in enumerate(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS):
            model_arch, model_size = model_arch_and_size
            config = HeatmapConfig(
                code_version=AppSessionKeys.code_version.value,
                common_params=CommonParams(
                    model_arch=model_arch,
                    model_size=model_size,
                ),
                prompt_filteration=SelectivePromptFilteration(
                    dataset_name=DATASETS.COUNTER_FACT,
                    prompt_ids=(self.prompt_idx,),
                ),
                runner_params=HeatmapParams(
                    window_size=AppSessionKeys.window_size.value,
                ),
            )

            # Check if the prompt exists in the HDF5 file
            prompt_exists = False
            existing_prompts = config.output_hdf5_path.get_existing_prompt_idx()
            if self.prompt_idx in existing_prompts:
                prompt_exists = True

            if not prompt_exists:
                continue

            plots_path = config.get_plot_output_path(self.prompt_idx, heatmap_parms.plot_name)
            if not plots_path.exists():
                config.plot(heatmap_parms.plot_name)

            progress = min((i + 1) / rows_count, 1.0)
            progress_bar.progress(progress, text=f"Plotting {i + 1}/{rows_count}")

            # Add plot path to its position in the grid
            size_cat = GRAPHS_ORDER[model_arch_and_size]
            if size_cat in size_cats:
                grid[i // N][i % N] = plots_path
                i += 1
        progress_bar.empty()

        # Create the combined image
        if any(any(row) for row in grid):  # Only show if we have any images
            # Cast grid to list[list[Path]] by filtering out None values
            non_none_grid = [[p for p in row if p is not None] for row in grid]
            if image_grid_params is not None:
                combined_image = combine_image_grid(non_none_grid, ImageGridParams(**image_grid_params))
                if combined_image is not None:
                    st.image(combined_image)
