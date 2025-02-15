from enum import StrEnum
from typing import cast

import pandas as pd
import streamlit as st
from streamlit import cache_data as st_cache_data

from src.consts import COLUMNS, GRAPHS_ORDER, MODEL_ARCH, get_model_cat_size
from src.experiments.evaluate_model import EvaluateModelConfig
from src.experiments.heatmap import HeatmapConfig
from src.final_plots.app.utils import (
    apply_pagination,
    create_pagination_config,
)
from src.types import MODEL_SIZE_CAT, SLURM_GPU_TYPE

st.set_page_config(page_title="Heatmap Creation", page_icon="🔥", layout="wide")

st.title("Heatmap Creation 🔥")

FILTERING_MODLS_COLUMNS_N = 2

# Initialize session state for model filters if not exists
if "model_filters" not in st.session_state:
    st.session_state.model_filters = {}

# Initialize session state for selected prompts
if "selected_prompts" not in st.session_state:
    st.session_state.selected_prompts = set()


class ModelFilterOption(StrEnum):
    CORRECT = "correct"
    ANY = "any"
    INCORRECT = "incorrect"


MODEL_FILTER_OPTIONS: list[ModelFilterOption] = list(ModelFilterOption)
DEFAULT_VARIATION = "v3"
DEFAULT_WINDOW_SIZE = 9


@st_cache_data
def load_evaluation_data() -> dict[tuple[MODEL_ARCH, str], pd.DataFrame]:
    """Load evaluation data for all models with caching"""
    return {
        (model_arch, model_size): EvaluateModelConfig(
            model_arch=model_arch, model_size=model_size, variation=DEFAULT_VARIATION
        ).get_outputs()
        for model_arch, model_size, _ in GRAPHS_ORDER
    }


@st_cache_data
def get_model_combinations_prompts() -> pd.DataFrame:
    """Get all possible model combinations and their corresponding prompts.

    Returns:
        DataFrame with columns for each model combination and the count of prompts
        that match that combination pattern.
    """
    # Get all prompts
    all_prompts = set(models_evaluations[model_combinations[0]].index)

    # Create a DataFrame with correctness for each model
    correctness_df = pd.DataFrame(index=sorted(all_prompts))
    for model_arch, model_size in model_combinations:
        model_key = f"{model_arch}-{model_size}"
        model_df = models_evaluations[(model_arch, model_size)]
        correctness_df[model_key] = [
            model_df.at[idx, COLUMNS.MODEL_CORRECT] if idx in model_df.index else False for idx in correctness_df.index
        ]

    # Generate all possible combinations
    combinations = []

    # Convert to numpy for faster operations
    correctness_matrix = correctness_df.values
    model_names = correctness_df.columns.tolist()

    # For each possible combination of models
    for i in range(1, 2 ** len(model_combinations) + 1):
        # Convert number to binary to get combination
        binary = format(i - 1, f"0{len(model_combinations)}b")
        active_models = [model_names[j] for j, bit in enumerate(binary) if bit == "1"]

        if not active_models:
            continue

        # Find prompts that are correct for all active models
        mask = correctness_matrix[:, [j for j, bit in enumerate(binary) if bit == "1"]].all(axis=1)
        matching_prompts = correctness_df.index[mask].tolist()

        if matching_prompts:
            combinations.append(
                {
                    "active_models": active_models,
                    "binary_pattern": binary,
                    "prompt_count": len(matching_prompts),
                    "prompts": matching_prompts,
                }
            )

    # Convert to DataFrame and sort by count
    result_df = pd.DataFrame(combinations)
    result_df = result_df.sort_values("prompt_count", ascending=False)
    return result_df


# Load evaluation data
with st.spinner("Loading evaluation data...", show_time=True):
    models_evaluations = {
        k: v for k, v in load_evaluation_data().items() if get_model_cat_size(k[0], k[1]) != MODEL_SIZE_CAT.HUGE
    }

# Get unique model combinations
model_combinations = list(models_evaluations.keys())

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

# Column names in the evaluation DataFrame
EVAL_COLUMNS = [
    COLUMNS.PROMPT,
    COLUMNS.TARGET_TRUE,
    COLUMNS.TARGET_FALSE,
    COLUMNS.SUBJECT,
    COLUMNS.TARGET_FALSE_ID,
    COLUMNS.RELATION,
    COLUMNS.RELATION_PREFIX,
    COLUMNS.RELATION_SUFFIX,
    COLUMNS.RELATION_ID,
    COLUMNS.TARGET_TRUE_ID,
]

PROMPT_RELATED_COLUMNS = [
    COLUMNS.PROMPT,
    COLUMNS.TARGET_TRUE,
    COLUMNS.TARGET_FALSE,
    COLUMNS.SUBJECT,
    COLUMNS.TARGET_FALSE_ID,
    COLUMNS.RELATION,
]


@st_cache_data
def get_merged_evaluations(prompt_idx: int) -> tuple[pd.Series, pd.DataFrame]:
    """Get merged evaluations for a specific prompt.

    Args:
        prompt_idx: The prompt index to get evaluations for

    Returns:
        tuple of:
            - Series with prompt-specific data
            - DataFrame with model-specific evaluations merged
    """
    # Get the prompt data from first model (prompt data is the same for all models)
    first_model_df = models_evaluations[model_combinations[0]]
    prompt_data = cast(pd.Series, first_model_df.loc[prompt_idx])

    # Create list to hold each model's evaluation
    model_evals = []

    for model_arch, model_size in model_combinations:
        model_df = models_evaluations[(model_arch, model_size)]
        if prompt_idx not in model_df.index:
            continue

        row = model_df.loc[prompt_idx]

        # Filter out prompt-related columns (they're the same for all models)
        model_specific_data = {col: val for col, val in row.items() if col not in PROMPT_RELATED_COLUMNS}

        model_specific_data["model_arch"] = model_arch
        model_specific_data["model_size"] = model_size

        model_evals.append(model_specific_data)

    return prompt_data, pd.DataFrame(model_evals)


@st_cache_data
def get_models_is_heatmap_available(prompt_idx: int) -> dict[tuple[MODEL_ARCH, str], bool]:
    """Check if a model has a heatmap for a given prompt index."""
    return {
        (model_arch, model_size): HeatmapConfig(
            model_arch=model_arch,
            model_size=model_size,
            window_size=DEFAULT_WINDOW_SIZE,
            prompt_indices_rows=[],
            prompt_original_indices=[prompt_idx],
        )
        .output_heatmap_path(prompt_idx)
        .exists()
        for model_arch, model_size in model_combinations
    }


# Create model filter controls
with st.sidebar:
    with st.expander("Model Filters", expanded=True):
        # Create 3 columns
        cols = st.columns(FILTERING_MODLS_COLUMNS_N)

        # Distribute models across columns
        for i, (model_arch, model_size) in enumerate(model_combinations):
            with cols[i % FILTERING_MODLS_COLUMNS_N]:
                model_key = f"{model_arch}-{model_size}"
                if model_key not in st.session_state.model_filters:
                    st.session_state.model_filters[model_key] = ModelFilterOption.ANY

                st.write(f"**{model_key}**")

                selected_filter = st.select_slider(
                    f"Filter for {model_key}",
                    options=MODEL_FILTER_OPTIONS,
                    key=f"filter_{model_key}",
                    label_visibility="collapsed",
                    value=st.session_state.model_filters[model_key],
                )
                st.session_state.model_filters[model_key] = selected_filter

# Apply model filters to get qualifying prompts
filtered_prompts = set(models_evaluations[model_combinations[0]].index)  # Start with all prompts

for model_arch, model_size in model_combinations:
    model_key = f"{model_arch}-{model_size}"
    model_filter = st.session_state.model_filters[model_key]

    if model_filter != ModelFilterOption.ANY:
        model_df = models_evaluations[(model_arch, model_size)]

        if model_filter == ModelFilterOption.CORRECT:
            qualifying_prompts = set(model_df[model_df[COLUMNS.MODEL_CORRECT]].index)
        else:  # incorrect
            qualifying_prompts = set(model_df[~model_df[COLUMNS.MODEL_CORRECT]].index)

        filtered_prompts &= qualifying_prompts

# Create DataFrame with qualifying prompts
prompts_df = pd.DataFrame({COLUMNS.ORIGINAL_IDX: sorted(filtered_prompts)})

# Add model correctness columns for display
for model_arch, model_size in model_combinations:
    model_key = f"{model_arch}-{model_size}"
    model_df = models_evaluations[(model_arch, model_size)]
    prompts_df[f"{model_key}_correct"] = prompts_df[COLUMNS.ORIGINAL_IDX].isin(
        set(model_df[model_df[COLUMNS.MODEL_CORRECT]].index)
    )

# Display results count
st.write(
    f"Found {len(filtered_prompts)} / {len(models_evaluations[model_combinations[0]])}  prompts matching all criteria"
)

# Add pagination
pagination_config = create_pagination_config(
    total_items=len(prompts_df),
    default_page_size=10,
    key_prefix="prompts_",
)

# Apply pagination
paginated_df = apply_pagination(prompts_df, pagination_config)

# Display prompts with model correctness
for _, row in paginated_df.iterrows():
    prompt_idx = row[COLUMNS.ORIGINAL_IDX]

    # Create correctness indicators for summary
    correct_count = sum(1 for model_arch, model_size in model_combinations if row[f"{model_arch}-{model_size}_correct"])
    total_count = len(model_combinations)

    # Get prompt text for preview
    first_model_df = models_evaluations[model_combinations[0]]
    prompt_text = str(first_model_df.at[prompt_idx, COLUMNS.PROMPT])
    preview_length = 100
    prompt_preview = f"{prompt_text[:preview_length]}..." if len(prompt_text) > preview_length else prompt_text

    # Create a container for each prompt
    prompt_container = st.container()

    with prompt_container:
        # Header row with checkbox and summary
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
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

        with col2:
            # Create an expander with a nicely formatted header
            with st.expander(
                (
                    f"📝 Prompt #{prompt_idx} |"
                    f"✓ {sum(get_models_is_heatmap_available(prompt_idx).values())}/{total_count} models calculated\n\n"
                    f"**{prompt_preview}**"
                ),
                expanded=False,
            ):
                # Get merged evaluations (cached) only when details are shown
                prompt, model_evals_df = get_merged_evaluations(prompt_idx)

                # Show full prompt text in a more readable format
                st.markdown("### Full Prompt")
                st.markdown(f"_{prompt_text}_")
                st.markdown("---")

                # Show prompt details and model evaluations in tabs
                tabs = st.tabs(["Prompt Details", "Model Evaluations"])

                with tabs[0]:
                    # Format prompt details in a more readable way
                    details_df = pd.DataFrame({"Field": EVAL_COLUMNS, "Value": [prompt[col] for col in EVAL_COLUMNS]})
                    st.dataframe(
                        details_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Field": st.column_config.TextColumn("Field", width="medium"),
                            "Value": st.column_config.TextColumn("Value", width="large"),
                        },
                    )

                with tabs[1]:
                    # Show model evaluations as a dataframe
                    # Create index from model arch and size
                    model_evals_df["model_name"] = model_evals_df.apply(
                        lambda x: f"{x['model_arch']}-{x['model_size']}", axis=1
                    )
                    model_evals_df = model_evals_df.set_index("model_name").sort_index()

                    # Drop the now redundant columns
                    model_evals_df = model_evals_df.drop(columns=["model_arch", "model_size"])

                    st.dataframe(
                        model_evals_df,
                        use_container_width=True,
                        column_config={
                            "model_name": st.column_config.TextColumn("Model", width="medium"),
                            COLUMNS.MODEL_CORRECT: st.column_config.CheckboxColumn("Correct"),
                            "prediction": st.column_config.TextColumn("Prediction"),
                            COLUMNS.MODEL_TOP_OUTPUTS: st.column_config.ListColumn(),
                            "confidence": st.column_config.NumberColumn(
                                "Confidence",
                                format="%.2f%%",
                                width="small",
                            ),
                        },
                    )
                st.markdown("---")  # Add a separator between prompts

# Add button to create heatmap requirements
st.sidebar.header(f"Create Heatmap Requirements ({len(st.session_state.selected_prompts)} prompts)")
col1, col2 = st.sidebar.columns(2)
window_size = col1.number_input("Window Size", min_value=1, max_value=20, value=DEFAULT_WINDOW_SIZE)
variation = col2.text_input("Variation", value=DEFAULT_VARIATION)
with st.sidebar:
    gpu_options = [(gpu_type.value, gpu_type) for gpu_type in SLURM_GPU_TYPE]
    selected_gpu_value = st.selectbox(
        "GPU Type",
        options=[value for value, _ in gpu_options],
        index=[value for value, _ in gpu_options].index("l40s"),
    )
    selected_gpu = next(gpu_type for value, gpu_type in gpu_options if value == selected_gpu_value)

if st.sidebar.button(f"🔥 Create Heatmap Requirements ({len(st.session_state.selected_prompts)} prompts)"):
    st.sidebar.info("Creating heatmap requirements...")

    success_count = 0
    failed_count = 0

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    # TODO: Add heatmap creation
    # for model_arch, model_size in model_combinations:
    #     try:
    #         HeatmapConfig(
    #             model_arch=model_arch,
    #             model_size=model_size,
    #             window_size=window_size,
    #             variation=variation,
    #             prompt_indices_rows=[],
    #             prompt_original_indices=list(st.session_state.selected_prompts),
    #             with_slurm=True,
    #             slurm_gpu_type=selected_gpu,
    #         ).run()
    #         success_count += 1
    #     except Exception as e:
    #         st.sidebar.error(f"Failed to create requirement for prompt {prompt_idx}: {str(e)}")
    #         failed_count += 1

    #     # Update progress
    #     progress = (i + 1) / len(model_combinations)
    #     progress_bar.progress(progress)
    #     status_text.text(
    #         f"Processed: {i + 1}/{len(model_combinations)} | Success: {success_count} | Failed: {failed_count}"
    #     )

    if success_count > 0:
        st.sidebar.success(f"Successfully created {success_count} heatmap requirements")
    if failed_count > 0:
        st.sidebar.warning(f"Failed to create {failed_count} requirements")

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
5. Click "Create Heatmap Requirements" to generate the requirements
""")
