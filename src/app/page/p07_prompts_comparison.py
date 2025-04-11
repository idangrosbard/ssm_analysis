from typing import Any, Callable, Optional, cast

import streamlit as st
import streamlit_antd_components as sac

from src.analysis.prompt_filterations import (
    AllPromptFilteration,
    Correctness,
    SelectivePromptFilteration,
    get_shared_models_correctness_prompt_filteration,
)
from src.app.components.inputs import select_enum
from src.app.components.prompt_filter import PromptFilterationComponent
from src.app.components.result_bank import select_model_evaluations
from src.app.components.tokenization import TokenizationVisualizerComponent
from src.app.data_store import load_prompts, load_unique_tokenizers
from src.app.texts import PROMPTS_COMPARISON_TEXTS
from src.core.consts import ALL_IMPORTANT_MODELS, GRAPHS_ORDER
from src.core.names import EvaluateModelMetricName
from src.core.types import MODEL_ARCH_AND_SIZE, TPromptOriginalIndex
from src.experiments.infrastructure.base_runner import BasePromptFilteration
from src.utils.streamlit.helpers.component import StreamlitPage
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor, SessionKeysBase
from src.utils.types_utils import class_values

default_prompt_filteration: dict[str, Callable[[list[MODEL_ARCH_AND_SIZE]], BasePromptFilteration]] = {
    "all": lambda _: AllPromptFilteration(),
    "selected_correct": lambda selected_model_arch_and_sizes: get_shared_models_correctness_prompt_filteration(
        selected_model_arch_and_sizes,
        correctness=Correctness.correct,
    ),
    "all_correct": lambda _: get_shared_models_correctness_prompt_filteration(
        GRAPHS_ORDER.keys(),
        Correctness.correct,
    ),
    "all_important_correct": lambda _: get_shared_models_correctness_prompt_filteration(
        ALL_IMPORTANT_MODELS.keys(),
        Correctness.correct,
    ),
    "all_important_top_2_to_5_correct": lambda _: get_shared_models_correctness_prompt_filteration(
        ALL_IMPORTANT_MODELS.keys(),
        Correctness.top_2_to_5_correct,
    ),
    "selective": lambda _: SelectivePromptFilteration(
        prompt_ids=tuple(
            [
                TPromptOriginalIndex(i)
                for i in [
                    *[290, 4350, 6403, 14577],  # correct
                    *[6274, 9868, 18562, 12930],  # top 2 to 5 correct
                    *[4734, 4311, 13592, 18117],  # not correct
                ]
            ]
        )
    ),
}


class _PromptsComparisonSessionKeys(SessionKeysBase["_PromptsComparisonSessionKeys"]):
    evaluate_model_metric_name = SessionKeyDescriptor[EvaluateModelMetricName](EvaluateModelMetricName.model_correct)
    prompt_filteration = SessionKeyDescriptor[BasePromptFilteration](
        default_prompt_filteration["selective"](cast(Any, None))
    )
    selected_tree_item = SessionKeyDescriptor[Optional[str]](None)


PromptsComparisonSessionKeys = _PromptsComparisonSessionKeys()


class PromptsComparisonPage(StreamlitPage):
    def render(self):
        # Select models to compare
        with st.expander("Select Models"):
            results_bank = select_model_evaluations(key="prompts_comparison_select_model_evaluations")

        with st.expander("Prompts Filteration"):
            cols = st.columns(len(default_prompt_filteration))
            for i, prompt_filteration_name in enumerate(default_prompt_filteration):
                with cols[i]:
                    if st.button(f"set {prompt_filteration_name}"):
                        PromptsComparisonSessionKeys.prompt_filteration.value = default_prompt_filteration[
                            prompt_filteration_name
                        ](results_bank.model_arch_and_sizes)

            PromptFilterationComponent(
                PromptsComparisonSessionKeys.prompt_filteration,
                selected_tree_item_sk=PromptsComparisonSessionKeys.selected_tree_item,
            ).render()

            results_bank = results_bank.set_prompt_filteration(PromptsComparisonSessionKeys.prompt_filteration.value)

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
            prompts = load_prompts.call_and_render().filter_by_prompt_filteration(
                PromptsComparisonSessionKeys.prompt_filteration.value
            )

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
