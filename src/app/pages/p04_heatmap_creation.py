# Purpose: Create and manage heatmaps for model analysis with filtering and batch processing capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Model combinations analysis
# 3. Prompt filtering and selection
# 4. Heatmap generation and execution
# Outline Issues:
# - Add comparison view for multiple heatmaps
# - Consider adding heatmap export functionality
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly


import streamlit as st
import streamlit_antd_components as sac

from src.app.app_consts import (
    GLOBAL_APP_CONSTS,
    AppSessionKeys,
)
from src.app.components.inputs import (
    select_models_and_sizes,
)
from src.app.components.multi_plots import HeatmapPlotGenerationComponent
from src.app.components.prompt_filter import ModelCombinations, PromptSelectionComponent
from src.app.components.data_requirements import HeatmapGenerationComponent
from src.app.data_store import (
    load_model_combinations_prompts,
    load_model_evaluations_dict,
)
from src.app.texts import COMMON_TEXTS, HEATMAP_TEXTS
from src.utils.streamlit.helpers.component import StreamlitPage

st.set_page_config(layout="wide", page_icon=HEATMAP_TEXTS.icon, page_title=HEATMAP_TEXTS.title)
st.header(HEATMAP_TEXTS.MODEL_COMBINATIONS_HEADER)


class HeatmapCreationPage(StreamlitPage):
    def render(self):
        # region Data Loading
        with st.sidebar:
            selected_models = select_models_and_sizes(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)

        with st.spinner(COMMON_TEXTS.LOADING("data"), show_time=True):
            # Get combinations data
            model_evaluations = load_model_evaluations_dict(AppSessionKeys.variation.value)
            representative_model_evaluations = next(iter(model_evaluations.values()))
            # Get combinations using selected models
            combinations_df = load_model_combinations_prompts(AppSessionKeys.variation.value, selected_models)
            combinations_df = sorted(combinations_df, key=lambda x: len(x.prompts), reverse=True)
        # endregion

        filtered_df, selected_combination_row = ModelCombinations(
            combinations_df, representative_model_evaluations
        ).render()

        if selected_combination_row is None:
            st.write(HEATMAP_TEXTS.NO_SELECTED_COMBINATION)
        else:
            tab = sac.tabs(
                [
                    sac.TabsItem(label=HEATMAP_TEXTS.TAB_SELECT_COMBINATION),
                    sac.TabsItem(label=HEATMAP_TEXTS.TAB_HEATMAP_PLOTS_GENERATION),
                    sac.TabsItem(label=HEATMAP_TEXTS.run_selected_prompts_button(len(filtered_df))),
                ]
            )
            combination_row = combinations_df[selected_combination_row]

            if tab == HEATMAP_TEXTS.TAB_SELECT_COMBINATION:
                PromptSelectionComponent(combination_row, representative_model_evaluations, combinations_df).render()
            elif tab == HEATMAP_TEXTS.TAB_HEATMAP_PLOTS_GENERATION:
                prompt_idx = combination_row.chosen_prompt
                if prompt_idx is not None:
                    HeatmapPlotGenerationComponent(prompt_idx).render()
            elif tab == HEATMAP_TEXTS.run_selected_prompts_button(len(filtered_df)):
                # Add SLURM configuration in sidebar
                HeatmapGenerationComponent(filtered_df).render()
            else:
                raise ValueError(f"Invalid tab: {tab}")


if __name__ == "__main__":
    HeatmapCreationPage().render()
