# Purpose: Create and manage heatmaps for model analysis with filtering and batch processing capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Model combinations analysis
# 3. Prompt filtering and selection
# 4. Heatmap generation and execution
# Outline Issues:
# - Add comparison view for multiple heatmaps
# - Consider adding heatmap export functionality
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly


from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from pandas import DataFrame

from src.consts import COLUMNS, GRAPHS_ORDER
from src.experiments.heatmap import HeatmapConfig
from src.final_plots.app.app_consts import (
    GLOBAL_APP_CONSTS,
    AppSessionKeys,
    HeatmapCols,
    HeatmapSessionKeys,
)
from src.final_plots.app.components.inputs import (
    choose_heatmap_parms,
    select_gpu_type,
    select_models_and_sizes,
    select_variation,
    select_window_size,
)
from src.final_plots.app.components.prompt import show_prompt
from src.final_plots.app.data_store import (
    get_merged_evaluations,
    get_models_remaining_prompts,
    load_model_combinations_prompts,
    load_model_evaluations,
)
from src.final_plots.app.texts import COMMON_TEXTS, HEATMAP_TEXTS
from src.final_plots.app.utils import (
    filter_combinations,
    get_steamlit_dataframe_selected_row,
)
from src.final_plots.data_reqs import save_model_combinations_prompts
from src.final_plots.image_combiner import ImageGridParams, combine_image_grid
from src.types import MODEL_ARCH, MODEL_ARCH_AND_SIZE, MODEL_SIZE_CAT
from src.utils.extended_streamlit_pydantic import pydantic_input
from src.utils.logits import Prompt

# region Data Loading
st.set_page_config(layout="wide", page_icon=HEATMAP_TEXTS.icon, page_title=HEATMAP_TEXTS.title)

st.header(HEATMAP_TEXTS.MODEL_COMBINATIONS_HEADER)


with st.sidebar:
    selected_models = select_models_and_sizes(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)

with st.spinner(COMMON_TEXTS.LOADING("data"), show_time=True):
    # Get combinations data
    model_evaluations = load_model_evaluations(AppSessionKeys.variation.value)
    representative_model_evaluations = next(iter(model_evaluations.values()))
    # Get combinations using selected models
    combinations_df = load_model_combinations_prompts(AppSessionKeys.variation.value, selected_models)
    combinations_df = sorted(combinations_df, key=lambda x: len(x.prompts), reverse=True)

# region Create table data
table_data = []
for row in combinations_df:
    # Create row with model correctness
    table_row = {}

    # Add prompt count and selected prompt first
    table_row[HeatmapCols.PROMPT_COUNT] = len(row.prompts)
    table_row[HeatmapCols.SELECTED_PROMPT] = row.chosen_prompt

    # Add model columns at the end
    for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
        model_name = model_name_and_size.model_name
        if model_name_and_size in row.correct_models:
            table_row[model_name] = "✅"
        elif model_name_and_size in row.incorrect_models:
            table_row[model_name] = "❌"
        else:
            table_row[model_name] = "-"
    table_data.append(table_row)

# endregion
# region Create DataFrame for display
display_df = pd.DataFrame(table_data)

assert display_df[HeatmapCols.PROMPT_COUNT].sum() == len(representative_model_evaluations), (
    "Display df prompt count mismatch, "
    f"{display_df[HeatmapCols.PROMPT_COUNT].sum()} != {len(representative_model_evaluations)}"
)

with st.sidebar.expander(HEATMAP_TEXTS.MODEL_COMBINATIONS_FILTERING, expanded=True):
    filtered_df = filter_combinations(
        display_df,
        [model_name_and_size.model_name for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS],
    )

selected_combination_row = get_steamlit_dataframe_selected_row(
    st.dataframe(
        filtered_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            HeatmapCols.PROMPT_COUNT: st.column_config.NumberColumn(pinned=True),
            HeatmapCols.SELECTED_PROMPT: st.column_config.TextColumn(pinned=True),
            **{
                model_name_and_size.model_name: st.column_config.TextColumn()
                for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
            },
        },
    )
)

# endregion
# endregion

if selected_combination_row is None:
    st.write(HEATMAP_TEXTS.NO_SELECTED_COMBINATION)
else:
    tab = sac.tabs(
        [
            sac.TabsItem(label=HEATMAP_TEXTS.TAB_SELECT_COMBINATION),
            sac.TabsItem(label=HEATMAP_TEXTS.TAB_HEATMAP_PLOTS_GENERATION),
        ]
    )
    combination_row = combinations_df[selected_combination_row]
    # region Show prompt selection for selected row
    if tab == HEATMAP_TEXTS.TAB_SELECT_COMBINATION:
        possible_prompts = representative_model_evaluations.loc[combination_row.prompts]
        selected_prompt_idx = combination_row.chosen_prompt
        selected_row_idx = get_steamlit_dataframe_selected_row(
            st.dataframe(
                possible_prompts[COLUMNS.PROMPT_DATA_COLS],
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    COLUMNS.PROMPT: st.column_config.TextColumn(
                        pinned=True,
                    )
                },
                use_container_width=True,
            )
        )
        if selected_row_idx is not None:
            selected_prompt_idx_new = int(cast(pd.Index, possible_prompts.iloc[selected_row_idx]).name)
            if selected_prompt_idx_new != selected_prompt_idx:
                if st.button(HEATMAP_TEXTS.BUT_SAVE_NEW_SELECTION(selected_prompt_idx, selected_prompt_idx_new)):
                    selected_prompt = possible_prompts.iloc[selected_row_idx]  # type: ignore
                    raise NotImplementedError("Saving is not implemented yet")
                    save_model_combinations_prompts(combinations_df)

                # Update the selected prompt index
                selected_prompt_idx = selected_prompt_idx_new

        if selected_prompt_idx is not None:
            show_prompt(Prompt(possible_prompts.loc[selected_prompt_idx]))
            model_evals: DataFrame = get_merged_evaluations(selected_prompt_idx, AppSessionKeys.variation.value)
            # renderer = merge_model_evaluations_streamlit_rendered(AppSessionKeys.variation.value)
            # renderer.explorer()

            st.dataframe(
                (model_evals.pipe(lambda df: df[[col for col in df.columns if col not in COLUMNS.PROMPT_DATA_COLS]])),
                hide_index=True,
                column_config={
                    "model_arch": st.column_config.TextColumn(pinned=True),
                    "model_size": st.column_config.TextColumn(pinned=True),
                    COLUMNS.MODEL_TOP_OUTPUTS: st.column_config.ListColumn(),
                },
            )
    # endregion

    # region Heatmap Plots Generation
    if tab == HEATMAP_TEXTS.TAB_HEATMAP_PLOTS_GENERATION:
        prompt_idx = combination_row.chosen_prompt

        assert prompt_idx is not None

        # Apply model filters to get qualifying prompts
        with st.sidebar:
            heatmap_parms = choose_heatmap_parms()
            image_grid_params = pydantic_input(key="my_form", model=ImageGridParams)

        archs = {arch: i for i, arch in enumerate([MODEL_ARCH.MAMBA1, MODEL_ARCH.MAMBA2, MODEL_ARCH.GPT2])}
        cols = st.columns(3, gap="small")
        N = 3
        M = 4
        # Create a 3x3 grid where rows are size categories and columns are architectures
        grid: list[list[Path | None]] = [[None for _ in range(N)] for _ in range(M)]  # 3x3 grid of None values
        size_cats = [MODEL_SIZE_CAT.SMALL, MODEL_SIZE_CAT.MEDIUM, MODEL_SIZE_CAT.LARGE, MODEL_SIZE_CAT.HUGE]
        # size_cats = [MODEL_SIZE_CAT.LARGE]
        size_cat_to_row = {cat: i for i, cat in enumerate(size_cats)}

        st.write(prompt_idx)
        # prompt_idx = 5088
        # prompt_idx = 19389
        # for prompt_idx in [140]:
        i = 0
        for model_arch_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
            model_arch, model_size = model_arch_and_size
            config = HeatmapConfig(
                model_arch=model_arch,
                model_size=model_size,
                window_size=AppSessionKeys.window_size.value,
                variation=AppSessionKeys.variation.value,
                prompt_indices_rows=[],
                prompt_original_indices=[prompt_idx],
            )

            if not config.output_heatmap_path(prompt_idx).exists():
                continue
            plots_path = config.get_plot_output_path(prompt_idx, heatmap_parms.plot_name)
            if not plots_path.exists():
                config.plot(heatmap_parms.plot_name)

            # Add plot path to its position in the grid
            size_cat = GRAPHS_ORDER[model_arch_and_size]
            if size_cat in size_cats:
                grid[i // N][i % N] = plots_path
                i += 1

        # Create the combined image
        if any(any(row) for row in grid):  # Only show if we have any images
            # Cast grid to list[list[Path]] by filtering out None values
            non_none_grid = [[p for p in row if p is not None] for row in grid]
            if image_grid_params is not None:
                combined_image = combine_image_grid(non_none_grid, ImageGridParams(**image_grid_params))
                if combined_image is not None:
                    st.image(combined_image)
    # endregion

# region Heatmap Generation
# Add SLURM configuration in sidebar
with st.sidebar:
    with st.expander(HEATMAP_TEXTS.run_selected_prompts_button(len(filtered_df))):
        # Show count of selected prompts
        selected_count = len(HeatmapSessionKeys.get_selected_prompts())

        # SLURM configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            select_window_size()
        with col2:
            select_variation()
        with col3:
            select_gpu_type()

        test_existing_prompts = st.checkbox("Test existing prompts", value=False)

        if test_existing_prompts:
            if st.button("reset remaining prompts"):
                get_models_remaining_prompts.clear()  # type: ignore
            prompt_original_indices = [int(x) for x in filtered_df[HeatmapCols.SELECTED_PROMPT]]
            with st.spinner("Calculating remaining prompts to run...", show_time=True):
                models_remaining_prompts = get_models_remaining_prompts(
                    GLOBAL_APP_CONSTS.MODELS_COMBINATIONS,
                    AppSessionKeys.window_size.value,
                    AppSessionKeys.variation.value,
                    [140],
                )

            table_data = []
            for model_arch_and_size, heatmap_config in models_remaining_prompts.items():
                model_name = model_arch_and_size.model_name
                table_data.append(
                    {
                        "Model": model_name,
                        "Prompt Count": len(heatmap_config.prompt_original_indices),
                        "Running": heatmap_config.is_running(),
                        "GPU": AppSessionKeys.get_selected_gpu(model_arch_and_size),
                    }
                )
            st.table(table_data)

            skip_running = st.checkbox("Skip running", value=True)
            # Run button
            if models_remaining_prompts and st.button(
                HEATMAP_TEXTS.run_selected_prompts(len(models_remaining_prompts))
            ):
                success_count = 0
                failed_count = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                # Get all rows from filtered_df that match selected prompts
                for i, heatmap_config in enumerate(models_remaining_prompts.values()):
                    try:
                        if skip_running and heatmap_config.is_running():
                            st.warning(
                                HEATMAP_TEXTS.skipping_running(heatmap_config.model_arch, heatmap_config.model_size)
                            )
                            continue

                        # Set running parameters
                        heatmap_config.set_running_params(
                            with_slurm=True,
                            slurm_gpu_type=AppSessionKeys.get_selected_gpu(
                                MODEL_ARCH_AND_SIZE(heatmap_config.model_arch, heatmap_config.model_size)
                            ),
                        )

                        # Submit job
                        heatmap_config.run()
                        success_count += 1

                    except Exception as e:
                        st.error(HEATMAP_TEXTS.submit_failed(heatmap_config.prompt_original_indices, e))
                        failed_count += 1

                    # Update progress
                    progress = (i + 1) / len(models_remaining_prompts)
                    progress_bar.progress(progress)
                    status_text.text(HEATMAP_TEXTS.processing_status(i + 1, len(models_remaining_prompts)))

                # Show final status
                if success_count > 0:
                    st.success(HEATMAP_TEXTS.success_status(success_count))
                if failed_count > 0:
                    st.warning(HEATMAP_TEXTS.error_status(failed_count))
# endregion
