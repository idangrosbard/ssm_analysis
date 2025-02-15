from enum import StrEnum

import pandas as pd
import streamlit as st

from src.consts import COLUMNS
from src.final_plots.app.data_store import (
    DEFAULT_WINDOW_SIZE,
    get_heatmap_state,
    get_merged_evaluations,
    get_model_combinations_prompts,
    get_models_is_heatmap_available,
)
from src.final_plots.app.texts import HEATMAP_TEXTS
from src.final_plots.app.utils import (
    apply_pagination,
    create_pagination_config,
)
from src.types import SLURM_GPU_TYPE

st.set_page_config(page_title=HEATMAP_TEXTS.title, page_icon=HEATMAP_TEXTS.icon, layout="wide")

st.title(HEATMAP_TEXTS.title)

FILTERING_MODLS_COLUMNS_N = 2

# Initialize session state for model filters if not exists
if "model_filters" not in st.session_state:
    st.session_state.model_filters = {}

# Initialize session state for selected prompts
if "selected_prompts" not in st.session_state:
    st.session_state.selected_prompts = set()

# Initialize heatmap state
state = get_heatmap_state()


class ModelFilterOption(StrEnum):
    CORRECT = "correct"
    ANY = "any"
    INCORRECT = "incorrect"


MODEL_FILTER_OPTIONS: list[ModelFilterOption] = list(ModelFilterOption)

# Load evaluation data
with st.spinner("Loading evaluation data...", show_time=True):
    state.initialize()

# Add Model Combinations Section at the top
st.header("Model Combinations Analysis")

# Get combinations data
combinations_df = get_model_combinations_prompts()

# Add pagination for combinations
combinations_pagination_config = create_pagination_config(
    total_items=len(combinations_df),
    default_page_size=10,
    key_prefix="combinations_",
)

# Apply pagination to combinations
paginated_combinations_df = apply_pagination(combinations_df, combinations_pagination_config)

# Create a container for the combinations table
combinations_table = st.container()

with combinations_table:
    # Display combinations in a table with action buttons
    for idx, row in paginated_combinations_df.iterrows():
        col1, col2, col3 = st.columns([0.6, 0.2, 0.2])

        # Format active models list
        active_models_str = " ∩ ".join(row["active_models"])

        with col1:
            st.write(f"**{active_models_str}**")

        with col2:
            st.write(f"{row['prompt_count']} prompts")

        with col3:
            if st.button("Show Prompts", key=f"show_combination_{idx}"):
                # Update model filters to show this combination
                for model_key in st.session_state.model_filters:
                    if model_key in row["active_models"]:
                        st.session_state.model_filters[model_key] = ModelFilterOption.CORRECT
                    else:
                        st.session_state.model_filters[model_key] = ModelFilterOption.ANY
                st.rerun()

st.markdown("---")

# Apply model filters to get qualifying prompts
filtered_prompts = set(state.models_evaluations[state.model_combinations[0]].index)  # Start with all prompts

for model_arch, model_size in state.model_combinations:
    model_key = f"{model_arch}-{model_size}"
    model_filter = st.session_state.model_filters.get(model_key, ModelFilterOption.ANY)

    if model_filter != ModelFilterOption.ANY:
        model_df = state.models_evaluations[(model_arch, model_size)]

        if model_filter == ModelFilterOption.CORRECT:
            filtered_prompts &= set(model_df[model_df[COLUMNS.MODEL_CORRECT]].index)
        else:  # INCORRECT
            filtered_prompts &= set(model_df[~model_df[COLUMNS.MODEL_CORRECT]].index)

# Create DataFrame with filtered prompts
prompts_df = pd.DataFrame({"original_idx": sorted(filtered_prompts)})

# Add correctness columns for each model
for model_arch, model_size in state.model_combinations:
    model_key = f"{model_arch}-{model_size}"
    model_df = state.models_evaluations[(model_arch, model_size)]
    prompts_df[f"{model_key}_correct"] = prompts_df[COLUMNS.ORIGINAL_IDX].isin(
        set(model_df[model_df[COLUMNS.MODEL_CORRECT]].index)
    )

# Display results count
st.write(
    HEATMAP_TEXTS.matching_counts(len(filtered_prompts), len(state.models_evaluations[state.model_combinations[0]]))
)

# Add pagination for prompts
prompts_pagination_config = create_pagination_config(
    total_items=len(prompts_df),
    default_page_size=5,
    key_prefix="prompts_",
)

# Apply pagination to prompts
paginated_prompts_df = apply_pagination(prompts_df, prompts_pagination_config)

# Display prompts
for _, row in paginated_prompts_df.iterrows():
    prompt_idx = row[COLUMNS.ORIGINAL_IDX]

    # Get prompt text for preview
    first_model_df = state.models_evaluations[state.model_combinations[0]]
    prompt_text = str(first_model_df.at[prompt_idx, COLUMNS.PROMPT])
    preview_length = 100
    prompt_preview = f"{prompt_text[:preview_length]}..." if len(prompt_text) > preview_length else prompt_text

    # Create a container for each prompt
    prompt_container = st.container()

    with prompt_container:
        col1, col2 = st.columns([0.9, 0.1])

        with col1:
            st.write(f"**Prompt {prompt_idx}**: {prompt_preview}")

        with col2:
            is_selected = st.checkbox(
                " ",
                value=prompt_idx in st.session_state.selected_prompts,
                key=f"select_prompt_{prompt_idx}",
                label_visibility="hidden",
            )
            if is_selected:
                st.session_state.selected_prompts.add(prompt_idx)
            else:
                st.session_state.selected_prompts.discard(prompt_idx)

        # Get prompt evaluations
        prompt_data, model_evals = get_merged_evaluations(prompt_idx)

        # Check which models have heatmaps available
        heatmaps_available = get_models_is_heatmap_available(prompt_idx)

        # Display model results
        for _, model_eval in model_evals.iterrows():
            model_arch = model_eval["model_arch"]
            model_size = model_eval["model_size"]
            model_key = f"{model_arch}-{model_size}"

            st.write(
                f"- **{model_key}**: {'✅' if row[f'{model_key}_correct'] else '❌'} "
                f"{'🔥' if heatmaps_available.get((model_arch, model_size), False) else ''}"
            )

# Add SLURM configuration in sidebar
with st.sidebar:
    with st.expander("Run Selected Prompts"):
        # Show count of selected prompts
        selected_count = len(st.session_state.selected_prompts)

        # SLURM configuration
        col1, col2 = st.columns(2)

        with col1:
            window_size = st.number_input("Window Size", value=DEFAULT_WINDOW_SIZE, min_value=1, max_value=100)

        with col2:
            gpu_options = [(gpu_type.value, gpu_type) for gpu_type in SLURM_GPU_TYPE]
            selected_gpu_value = st.selectbox(
                "GPU Type",
                options=[value for value, _ in gpu_options],
                index=[value for value, _ in gpu_options].index("l40s"),
            )
            selected_gpu = next(gpu_type for value, gpu_type in gpu_options if value == selected_gpu_value)

        # Run button
        if selected_count > 0 and st.button(f"🚀 Run {selected_count} Selected Prompts"):
            st.info(f"Preparing to run {selected_count} prompts...")

            success_count = 0
            failed_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Get all rows from filtered_df that match selected prompts
            selected_rows = prompts_df[prompts_df[COLUMNS.ORIGINAL_IDX].isin(st.session_state.selected_prompts)]

            # TODO: add heatmap creation
            # for i, (idx, row) in enumerate(selected_rows.iterrows()):
            #     try:
            #         prompt_idx = row[COLUMNS.ORIGINAL_IDX]

            #         # Get prompt evaluations
            #         prompt_data, model_evals = get_merged_evaluations(prompt_idx)

            #         # Create heatmap config
            #         config = HeatmapConfig(
            #             model_arch=model_arch,
            #             model_size=model_size,
            #             window_size=window_size,
            #             prompt_indices_rows=[],
            #             prompt_original_indices=[prompt_idx],
            #         )

            #         # Set running parameters
            #         config.set_running_params(
            #             with_slurm=True,
            #             slurm_gpu_type=selected_gpu,
            #         )

            #         # Submit job
            #         config.run()
            #         success_count += 1

            #     except Exception as e:
            #         st.error(f"Failed to submit job for prompt {prompt_idx}: {e}")
            #         failed_count += 1

            #     # Update progress
            #     progress = (i + 1) / len(selected_rows)
            #     progress_bar.progress(progress)
            #     status_text.text(f"Processed {i + 1} / {len(selected_rows)} prompts...")

            # Show final status
            if success_count > 0:
                st.success(f"Successfully submitted {success_count} jobs")
            if failed_count > 0:
                st.error(f"Failed to submit {failed_count} jobs")

            # Clear selected prompts
            st.session_state.selected_prompts.clear()
            st.rerun()

# Help text
st.markdown("""
### How to use:
1. Use the sidebar to set filters for each model:
   - ✅ **Correct**: Only include prompts where the model was correct
   - ❌ **Incorrect**: Only include prompts where the model was incorrect
   - **Any**: Include prompts regardless of model's correctness
2. View the filtered prompts that match all criteria
3. Select prompts using the checkboxes
4. Set the window size for the heatmap
5. Click "Run Selected Prompts" to generate the requirements
""")
