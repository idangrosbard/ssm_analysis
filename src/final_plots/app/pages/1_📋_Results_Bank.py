import pandas as pd
import streamlit as st

from src.final_plots.results_bank import (
    ParamNames,
    clear_results_bank_cache,
    get_results_bank,
)

st.set_page_config(page_title="Results Bank", page_icon="📋", layout="wide")

st.title("Results Bank 📋")


@st.cache_data
def load_results():
    """Load and process results with caching"""
    results = get_results_bank()
    results_data = []
    for result in results:
        result_dict = {param: getattr(result, param, None) for param in ParamNames}
        results_data.append(result_dict)

    return pd.DataFrame(results_data)


# Add refresh button in the sidebar
if st.sidebar.button("🔄 Refresh Data"):
    # Clear all caches
    clear_results_bank_cache()
    load_results.clear()  # type: ignore
    st.rerun()

# Get results using cached function
df = load_results()

# Add filters
st.sidebar.header("Filters")

# Create filters for each column (except Path)
filter_columns = [col for col in df.columns if col != ParamNames.path]
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
st.write(f"Showing {len(filtered_df)} results out of {len(df)} total")

# Display the table
st.dataframe(
    filtered_df,
    use_container_width=True,
    column_config={
        ParamNames.path: st.column_config.TextColumn(
            ParamNames.path, width="large", help="Full path to the result file"
        )
    },
)
