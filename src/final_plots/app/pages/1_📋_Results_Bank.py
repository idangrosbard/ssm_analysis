# Purpose: Display and manage a bank of experiment results with filtering and pagination capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Load and prepare results data
# 3. Create and apply filters to results
# 4. Display filtered results with pagination
# Outline Issues:
# - Consider adding export functionality for filtered results
# - Add more detailed information for each result
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly

import streamlit as st

from src.final_plots.app.app_consts import GLOBAL_APP_CONSTS
from src.final_plots.app.data_store import load_experiment_results
from src.final_plots.app.texts import RESULTS_BANK_TEXTS
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

# region Page Configuration
st.set_page_config(page_title=RESULTS_BANK_TEXTS.title, page_icon=RESULTS_BANK_TEXTS.icon, layout="wide")
st.title(f"{RESULTS_BANK_TEXTS.title} {RESULTS_BANK_TEXTS.icon}")
# endregion

# region Data Loading and Preparation
# Get results using cached function
df = load_experiment_results()
# endregion

# region Filtering
# Create and apply filters
filters = create_filters(df, filter_columns=[col for col in df.columns if col != ParamNames.path])
filtered_df = apply_filters(df, filters)

# Display results count and create pagination
show_filtered_count(filtered_df, df)
# endregion

# region Results Display
# Add pagination
pagination_config = create_pagination_config(
    total_items=len(filtered_df),
    default_page_size=GLOBAL_APP_CONSTS.PaginationConfig.RESULTS_BANK["default_page_size"],
    key_prefix=GLOBAL_APP_CONSTS.PaginationConfig.RESULTS_BANK["key_prefix"],
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
# endregion
