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

from src.app.components.data_requirements import RequirementExecution, RequirementsDisplay
from src.app.data_store import load_fulfilled_reqs_df, load_latest_fulfilled_reqs
from src.app.texts import DATA_REQUIREMENTS_TEXTS
from src.core.names import ResultBankParamNames
from src.utils.streamlit.components.aagrid import SelectionMode
from src.utils.streamlit.helpers.component import StreamlitPage


class DataRequirementsPage(StreamlitPage):
    def render(self):
        # region Data Loading and Preparation
        # Load data
        df = load_fulfilled_reqs_df()
        load_fulfilled_reqs_df.render()

        # Filter the data
        # filtered_df = RequirementsFiltering(df).render()

        # Display requirements
        data_reqs_to_run = RequirementsDisplay(
            df,
            height=1000,
            selection_mode=SelectionMode.MULTIPLE,
            hide_columns=[ResultBankParamNames.is_all_correct],
        ).render()

        # Save button for overrides
        with st.sidebar.expander(DATA_REQUIREMENTS_TEXTS.reset_to_latest):
            load_latest_fulfilled_reqs.render()

        if data_reqs_to_run is not None:
            with st.sidebar:
                # Handle requirement execution
                RequirementExecution(data_reqs_to_run).render()


if __name__ == "__main__":
    st.set_page_config(page_title=DATA_REQUIREMENTS_TEXTS.title, page_icon=DATA_REQUIREMENTS_TEXTS.icon, layout="wide")
    st.title(f"{DATA_REQUIREMENTS_TEXTS.title} {DATA_REQUIREMENTS_TEXTS.icon}")

    DataRequirementsPage().render()
