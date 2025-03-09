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

import random
from typing import Any, Dict, List, cast

import streamlit as st
from st_aggrid import AgGridReturn

from src.consts import GRAPHS_ORDER
from src.experiments.info_flow import InfoFlowConfig
from src.final_plots.app.components.info_flow import InfoFlowAnalysisComponent
from src.final_plots.app.components.result_bank import SelectionMode, ShowResultsBank
from src.final_plots.app.data_store import load_model_evaluations
from src.final_plots.app.texts import INFO_FLOW_ANALYSIS_TEXTS
from src.final_plots.app.utils import reverse_format_path_for_display
from src.names import COLS, EXPERIMENT_NAMES, ResultBankParamNames
from src.types import (
    MODEL_ARCH_AND_SIZE,
    MODEL_SIZE_CAT,
    TInfoFlowOutput,
    TInfoFlowWindowValue,
    TLayerIndex,
    TPromptOriginalIndex,
)
from src.utils.streamlit_utils import StreamlitPage
from src.utils.types_utils import (
    first_dict_value,
    get_list_indexes_of_set_values,
    select_indexes_from_list,
)

st.set_page_config(
    page_title=INFO_FLOW_ANALYSIS_TEXTS.title,
    page_icon=INFO_FLOW_ANALYSIS_TEXTS.icon,
    layout="wide",
)
st.title(f"{INFO_FLOW_ANALYSIS_TEXTS.title} {INFO_FLOW_ANALYSIS_TEXTS.icon}")


def select_indexes_from_window_values(
    window_values: TInfoFlowWindowValue, prompt_ids: list[TPromptOriginalIndex]
) -> TInfoFlowWindowValue:
    indexes = get_list_indexes_of_set_values(window_values[COLS.ORIGINAL_IDX], set(prompt_ids))
    return {
        COLS.INFO_FLOW.HIT.value: select_indexes_from_list(window_values[COLS.INFO_FLOW.HIT.value], indexes),
        COLS.INFO_FLOW.TRUE_PROBS.value: select_indexes_from_list(
            window_values[COLS.INFO_FLOW.TRUE_PROBS.value], indexes
        ),
        COLS.INFO_FLOW.DIFFS.value: select_indexes_from_list(window_values[COLS.INFO_FLOW.DIFFS.value], indexes),
        COLS.ORIGINAL_IDX: prompt_ids,
    }


def find_common_indices(
    info_flow_results_list: List[TInfoFlowOutput],
) -> list[TPromptOriginalIndex]:
    """Find the intersection of original_idx across all info flow results."""
    if not info_flow_results_list:
        return []

    # Get the set of original indices from the first window of each info flow result
    all_indices_sets = []
    for info_flow_results in info_flow_results_list:
        if not info_flow_results:
            continue
        first_window = first_dict_value(info_flow_results)
        indices = set(first_window[COLS.ORIGINAL_IDX])
        all_indices_sets.append(indices)

    common_indices = all_indices_sets[0]
    for indices in all_indices_sets[1:]:
        common_indices &= indices

    return list(common_indices)


class SubsetInfoFlowResults:
    def __init__(self, info_flow_results_list: list[TInfoFlowOutput]):
        self.info_flow_results_list = info_flow_results_list

    def render(
        self,
    ) -> tuple[list[TPromptOriginalIndex], tuple[TLayerIndex, TLayerIndex]]:
        if not self.info_flow_results_list:
            return [], (0, 0)

        sample_results = st.checkbox(INFO_FLOW_ANALYSIS_TEXTS.sample_results, value=True)

        # Find common indices across all info flow results
        common_indices = find_common_indices([info_flow for info_flow in self.info_flow_results_list])
        max_layer = max(len(info_flow) for info_flow in self.info_flow_results_list) - 1
        min_layer = 0

        if len(common_indices) == 0:
            st.warning(INFO_FLOW_ANALYSIS_TEXTS.no_common_indices)
            st.stop()

        if sample_results:
            # Get the maximum number of layers across all info flow results

            # Calculate the number of common indices
            common_indices_count = len(common_indices)

            sample_results_count = st.slider(
                INFO_FLOW_ANALYSIS_TEXTS.sample_results_count,
                value=min(50, common_indices_count),
                min_value=min(50, common_indices_count),
                max_value=common_indices_count,
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
            random.seed(seed)
            sampled_indices = random.sample(range(len(common_indices)), sample_results_count)
            return select_indexes_from_list(common_indices, sampled_indices), layers_range

        # If not sampling, return the original info flow results
        return common_indices, (min_layer, max_layer)


class InfoFlowAnalysisPage(StreamlitPage):
    def render(self):
        result_bank: AgGridReturn = ShowResultsBank(
            filter_experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
            filter_is_all_correct=False,
            selection_mode=SelectionMode.MULTIPLE,  # Changed to MULTIPLE
            height=300,
            filters={
                ResultBankParamNames.variation: ["v3"],
                ResultBankParamNames.model_size: [
                    model_arch_and_size.size
                    for model_arch_and_size, size_cat in GRAPHS_ORDER.items()
                    if size_cat.value > MODEL_SIZE_CAT.MEDIUM.value
                ],
                ResultBankParamNames.window_size: ["9", "15"],
            },
            hide_columns=[
                ResultBankParamNames.experiment_name,
                ResultBankParamNames.prompt_idx,
                ResultBankParamNames.is_all_correct,
            ],
            key="info_flow_results_bank",
        ).render()

        selected_info_flow_results = result_bank.selected_rows

        if selected_info_flow_results is None or len(selected_info_flow_results) == 0:
            st.warning(INFO_FLOW_ANALYSIS_TEXTS.no_requirements)
            return

        # Display requirements table and get selection
        st.subheader("Selected Info Flow Requirements")

        # Process each selected info flow result
        chosen_info_flow_results_list = []
        metadata_list = []
        model_evaluations_list = []

        for _, selected_result in selected_info_flow_results.iterrows():
            # Cast selected_result to Dict[str, Any] to avoid type errors
            result_dict = cast(Dict[str, Any], dict(selected_result))
            path = result_dict.pop(ResultBankParamNames.path)
            for col in [
                ResultBankParamNames.is_all_correct,
                ResultBankParamNames.prompt_idx,
            ]:
                result_dict.pop(col)
            # Convert to proper types

            model_arch_and_size = MODEL_ARCH_AND_SIZE(
                result_dict[ResultBankParamNames.model_arch],
                result_dict[ResultBankParamNames.model_size],
            )

            # Get model evaluations if not already loaded
            model_evaluations_list.append(
                load_model_evaluations(result_dict[ResultBankParamNames.variation], model_arch_and_size)
            )

            # Load info flow results
            info_flow_results = InfoFlowConfig.load_output(reverse_format_path_for_display(path))

            chosen_info_flow_results_list.append(info_flow_results)
            metadata_list.append(result_dict)

        with st.sidebar:
            chosen_prompt_ids, layers_range = SubsetInfoFlowResults(chosen_info_flow_results_list).render()

        if chosen_prompt_ids:
            # Combine model evaluations for all selected info flows

            # Create a combined model evaluations dataframe
            InfoFlowAnalysisComponent(
                [
                    {
                        k: select_indexes_from_window_values(v, chosen_prompt_ids)
                        for k, v in info_flow.items()
                        if layers_range[0] <= k <= layers_range[1]
                    }
                    for info_flow in chosen_info_flow_results_list
                ],
                metadata_list=metadata_list,
                model_evaluations=[df.loc[chosen_prompt_ids] for df in model_evaluations_list],
            ).render()


if __name__ == "__main__":
    InfoFlowAnalysisPage().render()
