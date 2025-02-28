"""
# Purpose: Display and analyze info flow requirements with interactive visualization capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Load and display latest fulfilled info flow requirements
# 3. Interactive selection of info flow data using RequirementsDisplay
# 4. Analysis of selected info flow data with animations using InfoFlowAnalysisComponent
# Outline Issues:
# - Add more interactive visualization options
# - Consider adding batch analysis capabilities
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly
"""

import streamlit as st

from src.consts import EXPERIMENT_NAMES
from src.experiments.info_flow import InfoFlowConfig
from src.final_plots.app.components.info_flow import InfoFlowAnalysisComponent
from src.final_plots.app.components.result_bank import SelectionMode, ShowResultsBank
from src.final_plots.app.texts import INFO_FLOW_ANALYSIS_TEXTS
from src.final_plots.app.utils import reverse_format_path_for_display
from src.final_plots.results_bank import ParamNames
from src.utils.streamlit_utils import StreamlitPage

st.set_page_config(page_title=INFO_FLOW_ANALYSIS_TEXTS.title, page_icon=INFO_FLOW_ANALYSIS_TEXTS.icon, layout="wide")
st.title(f"{INFO_FLOW_ANALYSIS_TEXTS.title} {INFO_FLOW_ANALYSIS_TEXTS.icon}")


class InfoFlowAnalysisPage(StreamlitPage):
    def render(self):
        selected_info_flow_results = (
            ShowResultsBank(
                filter_experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                filter_is_all_correct=False,
                selection_mode=SelectionMode.SINGLE,
                height=300,
            )
            .render()
            .selected_data
        )

        if selected_info_flow_results is None or selected_info_flow_results.empty:
            st.warning(INFO_FLOW_ANALYSIS_TEXTS.no_requirements)
            return

        # Display requirements table and get selection
        st.subheader("Latest Fulfilled Info Flow Requirements")
        info_flow_results = InfoFlowConfig.load_output(
            reverse_format_path_for_display(selected_info_flow_results.iloc[0][ParamNames.path])
        )
        InfoFlowAnalysisComponent(info_flow_results).render()


if __name__ == "__main__":
    InfoFlowAnalysisPage().render()
