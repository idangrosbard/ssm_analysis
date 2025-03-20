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

from src.app.components.result_bank import ShowResultsBank
from src.app.data_store import load_results_bank
from src.app.texts import RESULTS_BANK_TEXTS
from src.utils.streamlit.helpers.component import StreamlitPage


class ResultsBankPage(StreamlitPage):
    def render(self):
        results_bank = load_results_bank()
        load_results_bank.render()
        ShowResultsBank(results_bank).render()


if __name__ == "__main__":
    st.set_page_config(page_title=RESULTS_BANK_TEXTS.title, page_icon=RESULTS_BANK_TEXTS.icon, layout="wide")
    st.title(f"{RESULTS_BANK_TEXTS.title} {RESULTS_BANK_TEXTS.icon}")

    ResultsBankPage().render()
