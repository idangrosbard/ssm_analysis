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
    PREVIEW_LENGTH,
    AppSessionKeys,
    HeatmapColumns,
    HeatmapSessionKeys,
    ModelFilterOption,
    PaginationConfig,
)
from src.final_plots.app.components.inputs import select_gpu_type, select_variation, select_window_size
from src.final_plots.app.data_store import (
    get_merged_evaluations,
    get_model_combinations,
    get_model_combinations_prompts,
    get_model_evaluations,
    get_models_is_heatmap_available,
    get_models_remaining_prompts,
)
from src.final_plots.app.texts import HEATMAP_TEXTS
from src.final_plots.app.utils import (
    apply_pagination,
    create_pagination_config,
    filter_combinations,
)
from src.final_plots.data_reqs import PROMPT_SELECTION_PATH

# region Data Loading
# Load evaluation data
with st.spinner("Loading evaluation data...", show_time=True):
    model_evaluations = get_model_evaluations(AppSessionKeys.variation.get())
    model_combinations = get_model_combinations(AppSessionKeys.variation.get())

model_names = [to_model_name(model_arch, model_size) for model_arch, model_size in model_combinations]
# endregion

# region Model Combinations Analysis
st.header("Model Combinations Analysis")


# Get combinations data
combinations_df = get_model_combinations_prompts(AppSessionKeys.variation.get())

# Load saved selections from CSV if exists
if PROMPT_SELECTION_PATH.exists():
    saved_selections = pd.read_csv(PROMPT_SELECTION_PATH)
    selected_prompts_dict = saved_selections.set_index("combination_id")["selected_prompt"].to_dict()
else:
    selected_prompts_dict = {}

# Create table data
table_data = []
for idx, row in combinations_df.iterrows():
    # Create row with model correctness
    table_row = {}

    # Add prompt count and selected prompt first
    table_row["Prompt Count"] = row["prompt_count"]

    # Add selected prompt
    if row["prompt_count"] > 0:
        combination_id = row["binary_pattern"]
        default_prompt = row["prompts"][0] if row["prompts"] else None
        selected_prompt = selected_prompts_dict.get(combination_id, default_prompt)
        table_row["Selected Prompt"] = selected_prompt
    else:
        table_row["Selected Prompt"] = None

    # Add model columns at the end
    for model_name in model_names:
        if model_name in row["correct_models"]:
            table_row[model_name] = "✅"
        elif model_name in row["incorrect_models"]:
            table_row[model_name] = "❌"
        else:
            table_row[model_name] = "-"

    table_row["combination_id"] = row["binary_pattern"]
    table_row["prompts"] = row["prompts"]
    table_data.append(table_row)

# Create DataFrame for display
display_df = pd.DataFrame(table_data)


# Apply pagination and filtering
with st.sidebar.expander("Model Combinations Filtering", expanded=True):
    filtered_df = filter_combinations(display_df, model_names)

# Display table
st.dataframe(
    filtered_df.drop(columns=["combination_id", "prompts"]),
    key=HeatmapSessionKeys.selected_combination.key,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)

# Handle row selection
if HeatmapSessionKeys.get_selected_combination_row() is not None:
    selected_idx = HeatmapSessionKeys.get_selected_combination_row()
    if selected_idx is not None:
        row = filtered_df.iloc[selected_idx]

        # Show prompt selection for selected row
        if row[HeatmapColumns.PROMPT_COUNT] > 0:
            st.write("### Selected Combination")
            selected_prompt = st.selectbox(
                "Choose prompt",
                options=row["prompts"],
                index=row["prompts"].index(row["Selected Prompt"]) if row["Selected Prompt"] in row["prompts"] else 0,
                key="prompt_selector",
            )

            if st.button("Save Selection"):
                # Save selection to CSV
                selections_df = pd.DataFrame(
                    {"combination_id": [row["combination_id"]], "selected_prompt": [selected_prompt]}
                )

                # Update or append to existing CSV
                if PROMPT_SELECTION_PATH.exists():
                    existing_df = pd.read_csv(PROMPT_SELECTION_PATH)
                    existing_df = existing_df[existing_df["combination_id"] != row["combination_id"]]
                    selections_df = pd.concat([existing_df, selections_df])

                PROMPT_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
                selections_df.to_csv(PROMPT_SELECTION_PATH, index=False)

# endregion

# region Prompt Filtering
st.markdown("---")

# Apply model filters to get qualifying prompts
filtered_prompts = set(model_evaluations[model_combinations[0]].index)  # Start with all prompts

for model_arch, model_size in model_combinations:
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
for model_arch, model_size in model_combinations:
    model_name = to_model_name(model_arch, model_size)
    model_df = model_evaluations[(model_arch, model_size)]
    prompts_df[f"{model_name}_correct"] = prompts_df[COLUMNS.ORIGINAL_IDX].isin(
        set(model_df[model_df[COLUMNS.MODEL_CORRECT]].index)
    )

# Display results count
st.write(HEATMAP_TEXTS.matching_counts(len(filtered_prompts), len(model_evaluations[model_combinations[0]])))
# endregion

# region Prompt Display
# Add pagination for prompts
prompts_pagination_config = create_pagination_config(
    total_items=len(prompts_df),
    default_page_size=PaginationConfig.PROMPTS["default_page_size"],
    key_prefix=PaginationConfig.PROMPTS["key_prefix"],
)

# Apply pagination to prompts
paginated_prompts_df = apply_pagination(prompts_df, prompts_pagination_config)
if HeatmapSessionKeys.show_combination.get() is not None:
    # Display prompts
    for _, row in paginated_prompts_df.iterrows():
        prompt_idx = row[COLUMNS.ORIGINAL_IDX]

        # Get prompt text for preview
        first_model_df = model_evaluations[model_combinations[0]]
        prompt_text = str(first_model_df.at[prompt_idx, COLUMNS.PROMPT])
        prompt_preview = f"{prompt_text[:PREVIEW_LENGTH]}..." if len(prompt_text) > PREVIEW_LENGTH else prompt_text

        # Create a container for each prompt
        prompt_container = st.container()

        with prompt_container:
            col1, col2 = st.columns([0.9, 0.1])

            with col1:
                st.write(f"**Prompt {prompt_idx}**: {prompt_preview}")

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
        selected_count = len(selected_prompts_dict)

        # SLURM configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            select_window_size()
        with col2:
            select_variation()
        with col3:
            select_gpu_type()

        test_existing_prompts = st.checkbox("Test existing prompts", value=True)

        if test_existing_prompts:
            prompt_original_indices = [int(x) for x in filtered_df[HeatmapColumns.SELECTED_PROMPT]]
            with st.spinner("Calculating remaining prompts to run...", show_time=True):
                models_remaining_prompts = get_models_remaining_prompts(
                    model_combinations,
                    AppSessionKeys.window_size.get(),
                    AppSessionKeys.variation.get(),
                    prompt_original_indices,
                )
            for (model_arch, model_size), heatmap_config in models_remaining_prompts.items():
                model_name = to_model_name(model_arch, model_size)
                st.write(f"{model_name}: {len(heatmap_config.prompt_original_indices)}")

            # Run button
            if models_remaining_prompts and st.button(
                HEATMAP_TEXTS.run_selected_prompts(len(models_remaining_prompts))
            ):
                st.info(HEATMAP_TEXTS.preparing_to_run(len(models_remaining_prompts)))

                success_count = 0
                failed_count = 0

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Get all rows from filtered_df that match selected prompts
                for i, heatmap_config in enumerate(models_remaining_prompts.values()):
                    try:
                        # Set running parameters
                        heatmap_config.set_running_params(
                            with_slurm=True,
                            slurm_gpu_type=AppSessionKeys.selected_gpu.get(),
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
