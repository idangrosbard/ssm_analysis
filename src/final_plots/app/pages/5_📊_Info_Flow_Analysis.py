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

import random

import streamlit as st

from src.consts import COLUMNS, EXPERIMENT_NAMES, GRAPHS_ORDER
from src.experiments.info_flow import InfoFlowConfig
from src.final_plots.app.components.info_flow import InfoFlowAnalysisComponent
from src.final_plots.app.components.result_bank import SelectionMode, ShowResultsBank
from src.final_plots.app.texts import INFO_FLOW_ANALYSIS_TEXTS
from src.final_plots.app.utils import reverse_format_path_for_display
from src.final_plots.results_bank import ParamNames
from src.types import MODEL_SIZE_CAT, TInfoFlowOutput, TInfoFlowWindowValue
from src.utils.streamlit_utils import StreamlitPage
from src.utils.types_utils import select_indexes_from_list

st.set_page_config(page_title=INFO_FLOW_ANALYSIS_TEXTS.title, page_icon=INFO_FLOW_ANALYSIS_TEXTS.icon, layout="wide")
st.title(f"{INFO_FLOW_ANALYSIS_TEXTS.title} {INFO_FLOW_ANALYSIS_TEXTS.icon}")


def select_indexes_from_window_values(window_values: TInfoFlowWindowValue, indexes: list[int]) -> TInfoFlowWindowValue:
    return {
        COLUMNS.IF_HIT: select_indexes_from_list(window_values[COLUMNS.IF_HIT], indexes),
        COLUMNS.IF_TRUE_PROBS: select_indexes_from_list(window_values[COLUMNS.IF_TRUE_PROBS], indexes),
        COLUMNS.IF_DIFFS: select_indexes_from_list(window_values[COLUMNS.IF_DIFFS], indexes),
        COLUMNS.ORIGINAL_IDX: select_indexes_from_list(window_values[COLUMNS.ORIGINAL_IDX], indexes),
    }


def filter_info_flow_results(
    info_flow_results: TInfoFlowOutput, layers_range: tuple[int, int], sample_results_count: int, seed: int
) -> TInfoFlowOutput:
    random.seed(seed)
    index_chosen = random.sample(range(len(info_flow_results[0][COLUMNS.ORIGINAL_IDX])), sample_results_count)
    return {
        k: select_indexes_from_window_values(v, index_chosen)
        for k, v in info_flow_results.items()
        if layers_range[0] <= k <= layers_range[1]
    }


class SubsetInfoFlowResults:
    def __init__(self, info_flow_results: TInfoFlowOutput):
        self.info_flow_results = info_flow_results

    def render(self):
        info_flow_results = self.info_flow_results
        sample_results = st.checkbox(INFO_FLOW_ANALYSIS_TEXTS.sample_results, value=True)
        if sample_results:
            layers_count = len(self.info_flow_results)
            prompts_count = len(self.info_flow_results[0][COLUMNS.ORIGINAL_IDX])

            sample_results_count = st.slider(
                INFO_FLOW_ANALYSIS_TEXTS.sample_results_count,
                value=min(500, prompts_count),
                min_value=50,
                max_value=prompts_count,
                step=50,
            )
            layers_range = st.slider(
                INFO_FLOW_ANALYSIS_TEXTS.layers_range,
                value=(0, layers_count - 1),
                min_value=0,
                max_value=layers_count - 1,
                step=1,
            )
            seed = st.number_input(INFO_FLOW_ANALYSIS_TEXTS.seed, value=42, min_value=0, max_value=1000000, step=1)

            info_flow_results = filter_info_flow_results(info_flow_results, layers_range, sample_results_count, seed)

        return info_flow_results


class InfoFlowAnalysisPage(StreamlitPage):
    def render(self):
        selected_info_flow_results = (
            ShowResultsBank(
                filter_experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                filter_is_all_correct=False,
                selection_mode=SelectionMode.SINGLE,
                height=300,
                filters={
                    ParamNames.variation: ["v3"],
                    ParamNames.model_size: [
                        model_arch_and_size.size
                        for model_arch_and_size, size_cat in GRAPHS_ORDER.items()
                        if size_cat != MODEL_SIZE_CAT.SMALL
                    ],
                },
                hide_columns=[ParamNames.experiment_name, ParamNames.prompt_idx, ParamNames.is_all_correct],
                key="info_flow_results_bank",
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

        with st.sidebar:
            info_flow_results = SubsetInfoFlowResults(info_flow_results).render()
        InfoFlowAnalysisComponent(info_flow_results).render()


if __name__ == "__main__":
    InfoFlowAnalysisPage().render()
