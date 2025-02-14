import pandas as pd
import streamlit as st

from src.final_plots.app.utils import (
    apply_filters,
    apply_pagination,
    cache_data,
    create_filters,
    create_pagination_config,
    format_path_for_display,
    show_filtered_count,
)
from src.final_plots.results_bank import (
    ParamNames,
    get_results_bank,
)

st.set_page_config(page_title="Results Bank", page_icon="📋", layout="wide")

st.title("Results Bank 📋")


@cache_data
def load_results() -> pd.DataFrame:
    """Load and process results with caching"""
    results = get_results_bank()
    results_data = []
    for result in results:
        result_dict = {param: getattr(result, param, None) for param in ParamNames}
        result_dict[ParamNames.path] = format_path_for_display(result_dict[ParamNames.path])
        results_data.append(result_dict)

    return pd.DataFrame(results_data)


# Get results using cached function
df = load_results()

# Create and apply filters
filters = create_filters(df, filter_columns=[col for col in df.columns if col != ParamNames.path])
filtered_df = apply_filters(df, filters)

# Display results count and create pagination
show_filtered_count(filtered_df, df)

# Add pagination
pagination_config = create_pagination_config(
    total_items=len(filtered_df), default_page_size=20, key_prefix="results_bank_"
)

# Apply pagination to filtered data
paginated_df = apply_pagination(filtered_df, pagination_config)

# Display the table
st.dataframe(
    paginated_df,
    use_container_width=True,
    column_config={
        ParamNames.path: st.column_config.TextColumn(
            ParamNames.path,
        )
    },
)
