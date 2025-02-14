from pathlib import Path

import pandas as pd
import streamlit as st

from src.consts import EXPERIMENT_NAMES
from src.final_plots.data_reqs import (
    get_data_fullfment_options,
    get_data_reqs,
    load_data_fulfilled_overides,
    save_data_fulfilled_overides,
    update_data_reqs_with_latest_results,
)
from src.final_plots.results_bank import ParamNames

st.set_page_config(page_title="Data Requirements", page_icon="📊", layout="wide")

st.title("Data Requirements and Overrides 📊")

# Initialize session state for overrides if not exists
if "overrides" not in st.session_state:
    st.session_state.overrides = load_data_fulfilled_overides()

# Add refresh button
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Update Latest Requirements"):
        update_data_reqs_with_latest_results()
        st.success("Requirements updated successfully!")
        st.rerun()


class ReqMetadataColumns:
    AvailableOptions = "Available Options"
    Options = "Options"
    CurrentOverride = "Current Override"
    Key = "Key"


# Get current data
@st.cache_data
def load_data():
    """Load requirements and options data with caching"""
    reqs = get_data_reqs()
    options = get_data_fullfment_options()

    data = []
    for req in reqs:
        opts = options.get(req, [])
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


if st.sidebar.button("🔄 Refresh Data"):
    # Clear all caches
    load_data.clear()  # type: ignore
    st.rerun()

# Load data
df = load_data()

# Add filters
st.sidebar.header("Filters")
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
filters = {}

for col in filter_columns:
    unique_values = sorted(df[col].dropna().unique())
    if len(unique_values) <= 1:
        continue
    filters[col] = st.sidebar.multiselect(f"Filter {col}", unique_values, default=[])

# Apply filters
filtered_df = df.copy()
for col, selected_values in filters.items():
    if selected_values:
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

# Display results count
st.write(f"Showing {len(filtered_df)} requirements out of {len(df)} total")

# Display requirements with expandable rows
for _, row in filtered_df.iterrows():
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
                    st.write(f"- `{opt}`")
            else:
                st.write("No options available")

        # Override management
        st.write("### Override Management")
        current_override = row[ReqMetadataColumns.CurrentOverride]

        if row[ReqMetadataColumns.AvailableOptions] > 0:
            options = ["None"] + [str(opt) for opt in row[ReqMetadataColumns.Options]]
            selected_option = st.selectbox(
                "Select Override",
                options,
                index=0 if not current_override else options.index(current_override),
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
