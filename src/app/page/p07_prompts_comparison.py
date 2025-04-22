import streamlit as st
import streamlit_antd_components as sac

from src.app.components.inputs import select_enum
from src.app.components.prompt_filter import FilterPromptsComponent
from src.app.components.result_bank import select_model_evaluations
from src.app.components.tokenization import TokenizationVisualizerComponent
from src.app.data_store import load_prompts, load_unique_tokenizers
from src.app.texts import PROMPTS_COMPARISON_TEXTS
from src.core.names import EvaluateModelMetricName
from src.utils.streamlit.helpers.component import StreamlitPage
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor, SessionKeysBase
from src.utils.types_utils import class_values


class _PromptsComparisonSessionKeys(SessionKeysBase["_PromptsComparisonSessionKeys"]):
    evaluate_model_metric_name = SessionKeyDescriptor[EvaluateModelMetricName](EvaluateModelMetricName.model_correct)


PromptsComparisonSessionKeys = _PromptsComparisonSessionKeys()


class PromptsComparisonPage(StreamlitPage):
    def render(self):
        # Select models to compare
        with st.expander("Select Models", expanded=True):
            results_bank = select_model_evaluations(key="prompts_comparison_select_model_evaluations")

        with st.expander("Prompts Filteration"):
            prompt_filteration = FilterPromptsComponent(
                key="prompts_comparison_filter_prompts",
                default_preset="selective",
            ).render()
            results_bank = results_bank.set_prompt_filteration(prompt_filteration)

        if results_bank.is_empty():
            st.warning("Please select at least one model to view tokenization.")
            return

        tab = sac.tabs([sac.TabsItem(label=tab_name) for tab_name in class_values(PROMPTS_COMPARISON_TEXTS.TABS)])

        if tab == PROMPTS_COMPARISON_TEXTS.TABS.SHOW_METRICS:
            select_enum(
                "Select Metric",
                EvaluateModelMetricName,
                PromptsComparisonSessionKeys.evaluate_model_metric_name,
            )

            with st.spinner("Loading data...", show_time=True):
                st.write(results_bank.get_hit_per_prompt(PromptsComparisonSessionKeys.evaluate_model_metric_name.value))
        elif tab == PROMPTS_COMPARISON_TEXTS.TABS.SHOW_TOKENIZATION:
            selected_models = results_bank.model_arch_and_sizes

            # Load prompts
            st.subheader("Select Prompt")
            prompts = load_prompts.call_and_render().filter_by_prompt_filteration(prompt_filteration)

            unique_tokenizers = load_unique_tokenizers.call_and_render(selected_models)

            # Show tokenization visualizer
            TokenizationVisualizerComponent(
                prompts,
                unique_tokenizers,
            ).render()
        else:
            raise ValueError(f"Invalid tab: {tab}")


if __name__ == "tokenization_compare":
    st.set_page_config(
        page_title=PROMPTS_COMPARISON_TEXTS.title,
        page_icon=PROMPTS_COMPARISON_TEXTS.icon,
        layout="wide",
    )

    # PromptsComparisonPage().render()
    PromptsComparisonPage().render()
