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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from src.final_plots.app.data_store import load_experiment_results
from src.final_plots.app.texts import RESULTS_BANK_TEXTS
from src.final_plots.results_bank import ParamNames
from src.utils.streamlit_utils import StreamlitPage

st.set_page_config(page_title=RESULTS_BANK_TEXTS.title, page_icon=RESULTS_BANK_TEXTS.icon, layout="wide")
st.title(f"{RESULTS_BANK_TEXTS.title} {RESULTS_BANK_TEXTS.icon}")


class ResultsBankPage(StreamlitPage):
    def render(self):
        df = load_experiment_results()
        load_experiment_results.render()
        grid_builder = GridOptionsBuilder.from_dataframe(df)
        grid_builder.configure_pagination(enabled=True)
        # grid_builder.configure_selection(selection_mode="single", use_checkbox=True)
        grid_builder.configure_default_column(
            filter=True,
            floatingFilter=True,
        )
        grid_builder.configure_column(ParamNames.window_size, type=["textColumn"])
        grid_builder.configure_column(ParamNames.prompt_idx, type=["textColumn"])
        grid_builder.configure_side_bar()
        grid_options = grid_builder.build()

        # Display the table
        AgGrid(
            df,
            gridOptions=grid_options,
            height=1000,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key="results_bank",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
        )


if __name__ == "__main__":
    ResultsBankPage().render()
