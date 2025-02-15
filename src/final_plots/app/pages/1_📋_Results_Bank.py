import streamlit as st

from src.final_plots.app.data_store import load_results
from src.final_plots.app.utils import (
    apply_filters,
    apply_pagination,
    create_filters,
    create_pagination_config,
    show_filtered_count,
)
from src.final_plots.results_bank import (
    ParamNames,
)

st.set_page_config(page_title="Results Bank", page_icon="📋", layout="wide")

st.title("Results Bank 📋")

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
