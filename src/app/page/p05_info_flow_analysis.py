"""
# Purpose: Display and analyze info flow requirements with interactive visualization capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Load and display latest fulfilled info flow requirements
# 3. Interactive selection of multiple info flow data using RequirementsDisplay
# 4. Analysis of selected info flow data with visualizations using InfoFlowAnalysisComponent
# Outline Issues:
# - Add more interactive visualization options
# - Consider adding batch analysis capabilities
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly
"""

import streamlit as st

from src.app.components.info_flow import InfoFlowAnalysisComponent
from src.app.components.result_bank import SelectionMode, ShowResultsBank
from src.app.data_store import load_prompts, load_results_bank
from src.app.texts import INFO_FLOW_ANALYSIS_TEXTS
from src.core.consts import GRAPHS_ORDER
from src.core.names import COLS, InfoFlowMetricName, ResultBankParamNames
from src.core.types import (
    MODEL_SIZE_CAT,
    TInfoFlowWindowValue,
    TLayerIndex,
    TPromptOriginalIndex,
)
from src.data_ingestion.data_defs.data_defs import InfoFlowResults, Prompts
from src.utils.streamlit.helpers.component import StreamlitPage
from src.utils.types_utils import (
    get_list_indexes_of_set_values,
    select_indexes_from_list,
)


def select_indexes_from_window_values(
    window_values: TInfoFlowWindowValue, prompt_ids: list[TPromptOriginalIndex]
) -> TInfoFlowWindowValue:
    indexes = get_list_indexes_of_set_values(window_values[COLS.ORIGINAL_IDX], set(prompt_ids))
    return {
        InfoFlowMetricName.hit: select_indexes_from_list(window_values[InfoFlowMetricName.hit], indexes),
        InfoFlowMetricName.true_probs: select_indexes_from_list(window_values[InfoFlowMetricName.true_probs], indexes),
        InfoFlowMetricName.diffs: select_indexes_from_list(window_values[InfoFlowMetricName.diffs], indexes),
        COLS.ORIGINAL_IDX: prompt_ids,
    }


class SubsetInfoFlowResults:
    def __init__(
        self,
        info_flow_results: InfoFlowResults,
    ):
        self.info_flow_results = info_flow_results

    def render(
        self,
    ) -> tuple[Prompts, tuple[TLayerIndex, TLayerIndex]]:
        prompts = load_prompts().filter_by_prompt_ids(list(self.info_flow_results.get_common_indices()))

        if prompts.empty:
            return prompts, (0, 0)

        sample_results = st.checkbox(INFO_FLOW_ANALYSIS_TEXTS.sample_results, value=True)
        filter_relation_last_token = st.checkbox("Filter relation last token", value=False)

        # Find common indices across all info flow results
        max_layer = self.info_flow_results.max_layer()
        min_layer = self.info_flow_results.min_layer()

        if prompts.empty:
            st.warning(INFO_FLOW_ANALYSIS_TEXTS.no_common_indices)
            st.stop()

        # Filter by relation last token if needed
        if filter_relation_last_token:
            # Get model arch and size from first result
            prompts = prompts.filter_by_condition(lambda k, v: v.as_prompt().is_relation_last_token())

            if prompts.empty:
                st.warning("No prompts found with relation last token")
                st.stop()

        if sample_results:
            # Get the maximum number of layers across all info flow results

            # Calculate the number of common indices

            sample_results_count = st.slider(
                INFO_FLOW_ANALYSIS_TEXTS.sample_results_count,
                value=min(50, prompts.size),
                min_value=min(50, prompts.size),
                max_value=prompts.size,
                step=50,
            )

            layers_range = st.slider(
                INFO_FLOW_ANALYSIS_TEXTS.layers_range,
                value=(min_layer, max_layer),
                min_value=min_layer,
                max_value=max_layer,
                step=1,
            )

            seed = st.number_input(
                INFO_FLOW_ANALYSIS_TEXTS.seed,
                value=42,
                min_value=0,
                max_value=1000000,
                step=1,
            )

            # Sample from common indices
            sampled_indices = prompts.sample(sample_results_count, seed)
            return sampled_indices, layers_range

        # If not sampling, return the original info flow results
        return prompts, (min_layer, max_layer)


class InfoFlowAnalysisPage(StreamlitPage):
    def render(self):
        results_bank = load_results_bank.call_and_render().to_info_flow_results()

        result_bank = ShowResultsBank(
            results_bank,
            selection_mode=SelectionMode.MULTIPLE,  # Changed to MULTIPLE
            height=300,
            filters={
                # ResultBankParamNames.code_version: [GLOBAL_APP_CONSTS.DEFAULT_CODE_VERSION],
                ResultBankParamNames.model_size: [
                    model_arch_and_size.size
                    for model_arch_and_size, size_cat in GRAPHS_ORDER.items()
                    if size_cat.value > MODEL_SIZE_CAT.MEDIUM.value
                ],
                ResultBankParamNames.window_size: ["9", "15"],
            },
            hide_columns=[
                ResultBankParamNames.experiment_name,
            ],
            key="info_flow_results_bank",
        ).render()

        if result_bank.is_empty():
            st.warning(INFO_FLOW_ANALYSIS_TEXTS.no_requirements)
            return

        # Display requirements table and get selection
        st.subheader("Selected Info Flow Requirements")

        with st.sidebar:
            chosen_prompts, layers_range = SubsetInfoFlowResults(result_bank).render()

        result_bank = result_bank.subset_layers(layers_range).subset_prompts(chosen_prompts.original_idx)

        if chosen_prompts:
            # Combine model evaluations for all selected info flows

            # Create a combined model evaluations dataframe
            InfoFlowAnalysisComponent(
                result_bank,
            ).render()


if __name__ == "__main__":
    st.set_page_config(
        page_title=INFO_FLOW_ANALYSIS_TEXTS.title,
        page_icon=INFO_FLOW_ANALYSIS_TEXTS.icon,
        layout="wide",
    )
    st.title(f"{INFO_FLOW_ANALYSIS_TEXTS.title} {INFO_FLOW_ANALYSIS_TEXTS.icon}")

    InfoFlowAnalysisPage().render()
