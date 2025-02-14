from pathlib import Path

import pandas as pd
import streamlit as st

from src.consts import EXPERIMENT_NAMES
from src.final_plots.app.utils import (
    apply_filters,
    apply_pagination,
    cache_data,
    create_filters,
    create_pagination_config,
    format_path_for_display,
    show_filtered_count,
)
from src.final_plots.data_reqs import (
    DataReq,
    get_data_fullfment_options,
    load_data_fulfilled_overides,
    save_data_fulfilled_overides,
    update_data_reqs_with_latest_results,
)
from src.final_plots.results_bank import ParamNames
from src.types import SLURM_GPU_TYPE

st.set_page_config(page_title="Data Requirements", page_icon="📊", layout="wide")

st.title("Data Requirements and Overrides 📊")

# Initialize session state for overrides if not exists
if "overrides" not in st.session_state:
    st.session_state.overrides = load_data_fulfilled_overides()

# Initialize session state for selected requirements
if "selected_requirements" not in st.session_state:
    st.session_state.selected_requirements = set()


def empty_selected_requirements():
    st.session_state.selected_requirements = set()


# Add refresh button
with st.sidebar:
    if st.button("🔄 Update Latest Requirements"):
        update_data_reqs_with_latest_results()
        st.success("Requirements updated successfully!")
        st.rerun()


class ReqMetadataColumns:
    AvailableOptions = "Available Options"
    Options = "Options"
    CurrentOverride = "Current Override"
    Key = "Key"


@cache_data
def load_data() -> pd.DataFrame:
    """Load requirements and options data with caching"""
    options = get_data_fullfment_options()

    data = []
    for req, opts in options.items():
        override = st.session_state.overrides.get(str(req))

        row = {
            **{
                param: getattr(req, param, None)
                for param in ParamNames
                if param not in [ParamNames.path, ParamNames.variation]
            },
            ReqMetadataColumns.AvailableOptions: len(opts),
            ReqMetadataColumns.Options: opts,
            ReqMetadataColumns.CurrentOverride: override,
            ReqMetadataColumns.Key: str(req),
        }
        data.append(row)

    return pd.DataFrame(data)


# Load data
df = load_data()

# Create and apply filters
filter_columns = [
    ReqMetadataColumns.AvailableOptions,
    ParamNames.experiment_name,
    ParamNames.model_arch,
    ParamNames.model_size,
    ParamNames.window_size,
    ParamNames.is_all_correct,
    ParamNames.source,
    ParamNames.target,
    ParamNames.prompt_idx,
]
filters = create_filters(df, filter_columns=filter_columns)
filtered_df = apply_filters(df, filters)

# Display results count
show_filtered_count(filtered_df, df, "requirements")

# Add pagination
pagination_config = create_pagination_config(
    total_items=len(filtered_df),
    default_page_size=10,
    key_prefix="data_reqs_",
    on_change=lambda: empty_selected_requirements(),
)

# Apply pagination to filtered data
paginated_df = apply_pagination(filtered_df, pagination_config)

# Display requirements with expandable rows
for _, row in paginated_df.iterrows():
    col1, col2 = st.columns([1, 100], gap="small")

    with col1:
        key = row[ReqMetadataColumns.Key]
        is_selected = st.checkbox(
            " ",
            value=key in st.session_state.selected_requirements,
            key=f"select_{key}",
            label_visibility="hidden",
        )
        if is_selected:
            st.session_state.selected_requirements.add(key)
        else:
            st.session_state.selected_requirements.discard(key)

    with col2:
        label = (
            f"{row[ReqMetadataColumns.AvailableOptions]} Options | "
            f"**{row[ParamNames.experiment_name]}** | "
            f"**{row[ParamNames.model_arch]}-{row[ParamNames.model_size]}**"
            f" | ws=**{row[ParamNames.window_size]}**{' | **all_correct** ' if row[ParamNames.is_all_correct] else ''}"
        )
        if row[ParamNames.experiment_name] == EXPERIMENT_NAMES.INFO_FLOW:
            label += f" | source=**{row[ParamNames.source]}** target=**{row[ParamNames.target]}**" + (
                f" feature_category=**{row[ParamNames.feature_category]}**" if row[ParamNames.feature_category] else ""
            )
        elif row[ParamNames.experiment_name] == EXPERIMENT_NAMES.HEATMAP:
            label += f" | prompt_idx=**{row[ParamNames.prompt_idx]}**"

        with st.expander(label):
            st.write("### Requirement Details")
            details_col1, details_col2 = st.columns(2)

            with details_col1:
                st.write("**Parameters:**")
                for col in [
                    ParamNames.window_size,
                    ParamNames.is_all_correct,
                    ParamNames.source,
                    ParamNames.feature_category,
                    ParamNames.target,
                    ParamNames.prompt_idx,
                ]:
                    if pd.notna(row[col]):
                        st.write(f"- {col}: {row[col]}")

            with details_col2:
                st.write("**Available Options:**")
                if row[ReqMetadataColumns.AvailableOptions] > 0:
                    for opt in row[ReqMetadataColumns.Options]:
                        st.write(f"- `{format_path_for_display(opt)}`")
                else:
                    st.write("No options available")

            # Override management
            st.write("### Override Management")
            current_override = row[ReqMetadataColumns.CurrentOverride]

            if row[ReqMetadataColumns.AvailableOptions] > 0:
                options = ["None"] + [format_path_for_display(opt) for opt in row[ReqMetadataColumns.Options]]
                selected_option = st.selectbox(
                    "Select Override",
                    options,
                    index=0 if not current_override else options.index(format_path_for_display(current_override)),
                    key=f"override_{row[ReqMetadataColumns.Key]}",
                )

                if selected_option != "None":
                    st.session_state.overrides[row[ReqMetadataColumns.Key]] = Path(selected_option)
                elif row[ReqMetadataColumns.Key] in st.session_state.overrides:
                    del st.session_state.overrides[row[ReqMetadataColumns.Key]]

# Save button for overrides
if st.button("💾 Save Overrides"):
    save_data_fulfilled_overides(st.session_state.overrides)
    st.success("Overrides saved successfully!")

# Add SLURM configuration in sidebar
with st.sidebar:
    with st.expander("Run Filtered Requirements"):
        # Show count of selected requirements
        selected_count = len(st.session_state.selected_requirements)

        # SLURM configuration
        col1, col2 = st.columns(2)

        with col1:
            variation = st.text_input("Variation", value="v3")

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
                filtered_df[ReqMetadataColumns.Key].isin(st.session_state.selected_requirements)
            ]

            for i, (idx, row) in enumerate(selected_rows.iterrows()):
                try:
                    # Create DataReq object
                    req = DataReq(
                        **{
                            param: row[param]
                            for param in ParamNames
                            if param not in [ParamNames.path, ParamNames.variation]
                        },
                    )

                    # Get config and set running parameters
                    config = req.get_config()
                    config.set_running_params(
                        variation=variation,
                        is_slurm=True,
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
