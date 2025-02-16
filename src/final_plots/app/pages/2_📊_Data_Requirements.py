# Purpose: Manage and display data requirements for experiments with filtering and execution capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Initialize session state and load data
# 3. Display and manage requirements with filters
# 4. Handle requirement selection and execution
# Outline Issues:
# - Consider adding batch operations for requirements
# - Add progress tracking for running requirements
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly


import streamlit as st

from src.final_plots.app.app_consts import (
    GLOBAL_APP_CONSTS,
    AppSessionKeys,
    DataReqCols,
    DataReqConsts,
    DataReqsSessionKeys,
)
from src.final_plots.app.components.inputs import select_variation
from src.final_plots.app.data_store import empty_selected_requirements, load_data
from src.final_plots.app.texts import DATA_REQUIREMENTS_TEXTS
from src.final_plots.app.utils import (
    apply_filters,
    apply_pagination,
    create_filters,
    create_pagination_config,
    get_data_req_from_df_row,
    show_filtered_count,
)
from src.final_plots.results_bank import ParamNames
from src.types import SLURM_GPU_TYPE

# region Page Configuration
st.set_page_config(page_title=DATA_REQUIREMENTS_TEXTS.title, page_icon=DATA_REQUIREMENTS_TEXTS.icon, layout="wide")
st.title(f"{DATA_REQUIREMENTS_TEXTS.title} {DATA_REQUIREMENTS_TEXTS.icon}")
# endregion

# region Data Loading and Preparation
# Load data
df = load_data()

# Create and apply filters
filters = create_filters(
    df,
    filter_columns=DataReqConsts.DATA_REQS_FILTER_COLUMNS,
    default_values=DataReqConsts.DATA_REQS_DEFAULT_FILTER_VALUES,
)
filtered_df = apply_filters(df, filters)

# Display results count
show_filtered_count(filtered_df, df, "requirements")
# endregion

# region Requirements Display and Management
# Add pagination
pagination_config = create_pagination_config(
    total_items=len(filtered_df),
    default_page_size=GLOBAL_APP_CONSTS.PaginationConfig.DATA_REQS["default_page_size"],
    key_prefix=GLOBAL_APP_CONSTS.PaginationConfig.DATA_REQS["key_prefix"],
    on_change=empty_selected_requirements,
)

# Apply pagination to filtered data
paginated_df = apply_pagination(filtered_df, pagination_config)

# Display requirements with expandable rows
for _, row in paginated_df.iterrows():
    col1, col2 = st.columns([1, 100], gap="small")

    with col1:
        key = row[DataReqCols.Key]
        is_selected = st.checkbox(
            " ",
            value=key in DataReqsSessionKeys.selected_requirements.get(),
            key=DataReqsSessionKeys.select_requirement(key).key,
            label_visibility="hidden",
        )
        if is_selected:
            DataReqsSessionKeys.selected_requirements.get().add(key)
        else:
            DataReqsSessionKeys.selected_requirements.get().discard(key)

    with col2:
        # Create an expander for the requirement
        with st.expander(f"Requirement {key}"):
            st.write(f"**Model:** {row[ParamNames.model_arch]} {row[ParamNames.model_size]}")
            st.write(f"**Window Size:** {row[ParamNames.window_size]}")
            if row[ParamNames.source]:
                st.write(f"**Source:** {row[ParamNames.source]}")
            if row[ParamNames.target]:
                st.write(f"**Target:** {row[ParamNames.target]}")
            if row[ParamNames.feature_category]:
                st.write(f"**Feature Category:** {row[ParamNames.feature_category]}")
            if row[ParamNames.prompt_idx]:
                st.write(f"**Prompt Index:** {row[ParamNames.prompt_idx]}")
# endregion

# region Requirement Execution
# Save button for overrides
if st.button(DATA_REQUIREMENTS_TEXTS.save_overrides):
    # Convert session state overrides to IDataFulfilled type
    pass
    # overrides: IDataFulfilled = {}
    # for key, path in DataReqsSessionKeys.overrides.get().items():
    #     data_req = get_data_req_from_df_row(filtered_df[filtered_df[DataReqCols.Key] == key].iloc[0])
    #     overrides[data_req] = path
    # save_data_fulfilled_overides(overrides)
    # st.success(DATA_REQUIREMENTS_TEXTS.overrides_saved)

# Add SLURM configuration in sidebar
with st.sidebar:
    with st.expander("Run Filtered Requirements"):
        # Show count of selected requirements
        selected_count = len(DataReqsSessionKeys.selected_requirements.get())

        # SLURM configuration
        col1, col2 = st.columns(2)

        with col1:
            select_variation()

        with col2:
            gpu_options = [(gpu_type.value, gpu_type) for gpu_type in SLURM_GPU_TYPE]
            selected_gpu_value = st.selectbox(
                "GPU Type",
                options=[value for value, _ in gpu_options],
                index=[value for value, _ in gpu_options].index("l40s"),
            )
            selected_gpu = next(gpu_type for value, gpu_type in gpu_options if value == selected_gpu_value)

        # Run button
        if selected_count > 0 and st.button(f"🚀 Run {selected_count} Selected Requirements"):
            st.info(f"Preparing to run {selected_count} requirements...")

            success_count = 0
            failed_count = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Get all rows from filtered_df that match selected requirements
            selected_rows = filtered_df[
                filtered_df[DataReqCols.Key].isin(DataReqsSessionKeys.selected_requirements.get())
            ]

            for i, (idx, row) in enumerate(selected_rows.iterrows()):
                try:
                    req = get_data_req_from_df_row(row)

                    # Get config and set running parameters
                    config = req.get_config(variation=AppSessionKeys.variation.get())
                    config.set_running_params(
                        with_slurm=True,
                        slurm_gpu_type=selected_gpu,
                    )

                    # Run the configuration
                    config.run()
                    success_count += 1

                except Exception as e:
                    st.error(f"Failed to run requirement: {str(e)}")
                    failed_count += 1

                # Update progress
                progress = (i + 1) / selected_count
                progress_bar.progress(progress)
                status_text.text(
                    f"Processed: {i + 1}/{selected_count} | Success: {success_count} | Failed: {failed_count}"
                )

            if success_count > 0:
                st.success(f"Successfully submitted {success_count} requirements to run")
            if failed_count > 0:
                st.warning(f"Failed to submit {failed_count} requirements")
# endregion
