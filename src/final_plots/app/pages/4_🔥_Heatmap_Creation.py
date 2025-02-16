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


import pandas as pd
import streamlit as st

from src.consts import COLUMNS, to_model_name
from src.final_plots.app.app_consts import (
    GLOBAL_APP_CONSTS,
    AppSessionKeys,
    HeatmapCols,
    HeatmapSessionKeys,
    ModelFilterOption,
)
from src.final_plots.app.components.inputs import (
    select_gpu_type,
    select_models_and_sizes,
    select_variation,
    select_window_size,
)
from src.final_plots.app.data_store import (
    get_merged_evaluations,
    get_models_is_heatmap_available,
    get_models_remaining_prompts,
    load_model_combinations_prompts,
    load_model_evaluations,
)
from src.final_plots.app.texts import HEATMAP_TEXTS
from src.final_plots.app.utils import (
    apply_pagination,
    create_pagination_config,
    filter_combinations,
)
from src.final_plots.data_reqs import save_model_combinations_prompts

# region Data Loading
st.set_page_config(layout="wide", page_icon=HEATMAP_TEXTS.icon, page_title=HEATMAP_TEXTS.title)

# region Model Combinations Analysis
st.header("Model Combinations Analysis")

with st.sidebar:
    selected_models = select_models_and_sizes(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)

with st.spinner("Loading data...", show_time=True):
    # Get combinations data
    model_evaluations = load_model_evaluations(AppSessionKeys.variation.get())

    # Get combinations using selected models
    combinations_df = load_model_combinations_prompts(AppSessionKeys.variation.get(), selected_models)
    combinations_df = sorted(combinations_df, key=lambda x: len(x.prompts), reverse=True)

# Create table data
table_data = []
for row in combinations_df:
    # Create row with model correctness
    table_row = {}

    # Add prompt count and selected prompt first
    table_row[HeatmapCols.PROMPT_COUNT] = len(row.prompts)
    table_row[HeatmapCols.SELECTED_PROMPT] = row.chosen_prompt

    # Add model columns at the end
    for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
        model_name = to_model_name(*model_name_and_size)
        if model_name_and_size in row.correct_models:
            table_row[model_name] = "✅"
        elif model_name_and_size in row.incorrect_models:
            table_row[model_name] = "❌"
        else:
            table_row[model_name] = "-"
    table_data.append(table_row)

# Create DataFrame for display
display_df = pd.DataFrame(table_data)


# Apply pagination and filtering
with st.sidebar.expander("Model Combinations Filtering", expanded=True):
    filtered_df = filter_combinations(
        display_df,
        [to_model_name(*model_name_and_size) for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS],
    )

st.write(display_df[HeatmapCols.PROMPT_COUNT].sum())
# Display table
st.dataframe(
    filtered_df,
    key=HeatmapSessionKeys.selected_combination.key,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        HeatmapCols.PROMPT_COUNT: st.column_config.TextColumn(pinned=True),
        HeatmapCols.SELECTED_PROMPT: st.column_config.TextColumn(pinned=True),
        **{
            to_model_name(*model_name_and_size): st.column_config.TextColumn()
            for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
        },
    },
)


# Handle row selection
selected_combination_row = HeatmapSessionKeys.get_selected_combination_row()
if selected_combination_row is not None and len((row := combinations_df[selected_combination_row]).prompts) > 0:
    # Show prompt selection for selected row
    st.write("### Selected Combination")
    selected_prompt = st.selectbox(
        "Choose prompt",
        options=row.prompts,
        index=row.prompts.index(row.chosen_prompt) if row.chosen_prompt in row.prompts else 0,
        key="prompt_selector",
    )

    if st.button("Save Selection"):
        # Save selection to CSV
        row.chosen_prompt = selected_prompt

        save_model_combinations_prompts(combinations_df)

    # endregion

    # region Prompt Filtering
    st.markdown("---")

    # Apply model filters to get qualifying prompts
    filtered_prompts = set(next(iter(model_evaluations.values())).index)  # Start with all prompts

    for model_arch, model_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
        model_name = to_model_name(model_arch, model_size)
        model_filter = st.session_state.model_filters.get(model_name, ModelFilterOption.ANY)

        if model_filter != ModelFilterOption.ANY:
            model_df = model_evaluations[(model_arch, model_size)]

            if model_filter == ModelFilterOption.CORRECT:
                filtered_prompts &= set(model_df[model_df[COLUMNS.MODEL_CORRECT]].index)
            else:  # INCORRECT
                filtered_prompts &= set(model_df[~model_df[COLUMNS.MODEL_CORRECT]].index)

    # Create DataFrame with filtered prompts
    prompts_df = pd.DataFrame({COLUMNS.ORIGINAL_IDX: sorted(filtered_prompts)})

    # Add correctness columns for each model
    for model_arch, model_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
        model_name = to_model_name(model_arch, model_size)
        model_df = model_evaluations[(model_arch, model_size)]
        prompts_df[f"{model_name}_correct"] = prompts_df[COLUMNS.ORIGINAL_IDX].isin(
            set(model_df[model_df[COLUMNS.MODEL_CORRECT]].index)
        )

    # Display results count
    st.write(HEATMAP_TEXTS.matching_counts(len(filtered_prompts), len(next(iter(model_evaluations.values())))))
    # endregion

    # region Prompt Display
    # Add pagination for prompts
    prompts_pagination_config = create_pagination_config(
        total_items=len(prompts_df),
        default_page_size=GLOBAL_APP_CONSTS.PaginationConfig.PROMPTS["default_page_size"],
        key_prefix=GLOBAL_APP_CONSTS.PaginationConfig.PROMPTS["key_prefix"],
    )

    # Apply pagination to prompts
    paginated_prompts_df = apply_pagination(prompts_df, prompts_pagination_config)
    if HeatmapSessionKeys.show_combination.get() is not None:
        # Display prompts
        for _, row in paginated_prompts_df.iterrows():
            prompt_idx = row[COLUMNS.ORIGINAL_IDX]

            # Get prompt text for preview
            first_model_df = next(iter(model_evaluations.values()))
            prompt_text = str(first_model_df.at[prompt_idx, COLUMNS.PROMPT])

            # Create a container for each prompt
            prompt_container = st.container()

            with prompt_container:
                col1, col2 = st.columns([0.9, 0.1])

                with col1:
                    st.write(f"**Prompt {prompt_idx}**: {prompt_text}")

                with col2:
                    st.checkbox(
                        " ",
                        key=HeatmapSessionKeys.selected_prompt(prompt_idx).key,
                        label_visibility="hidden",
                    )

                # Get prompt evaluations
                prompt_data, model_evals = get_merged_evaluations(prompt_idx, AppSessionKeys.variation.get())

                # Check which models have heatmaps available
                heatmaps_available = get_models_is_heatmap_available(
                    prompt_idx, AppSessionKeys.variation.get(), AppSessionKeys.window_size.get()
                )

                # Display model results
                for _, model_eval in model_evals.iterrows():
                    model_arch = model_eval["model_arch"]
                    model_size = model_eval["model_size"]
                    model_name = to_model_name(model_arch, model_size)

                    st.write(
                        f"- **{model_name}**: {'✅' if row[f'{model_name}_correct'] else '❌'} "
                        f"{'🔥' if heatmaps_available.get((model_arch, model_size), False) else ''}"
                    )
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
                    AppSessionKeys.window_size.get(),
                    AppSessionKeys.variation.get(),
                    prompt_original_indices,
                )

            table_data = []
            for (model_arch, model_size), heatmap_config in models_remaining_prompts.items():
                model_name = to_model_name(model_arch, model_size)
                table_data.append(
                    {
                        "Model": model_name,
                        "Prompt Count": len(heatmap_config.prompt_original_indices),
                        "Running": heatmap_config.is_running(),
                        "GPU": AppSessionKeys.get_selected_gpu(model_arch, model_size),
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
                                heatmap_config.model_arch, heatmap_config.model_size
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
