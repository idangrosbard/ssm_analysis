from functools import lru_cache
from typing import Optional, Union, cast

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from annotated_text import annotated_text, annotation
from pandas import DataFrame
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode
from streamlit.delta_generator import DeltaGenerator

from src.analysis.experiment_results.model_prompt_combination import ModelCombination
from src.analysis.prompt_filterations import IntersectionPromptFilteration, UnionPromptFilteration
from src.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys
from src.app.app_utils import (
    filter_combinations,
    get_steamlit_dataframe_selected_row,
)
from src.app.data_store import get_merged_evaluations
from src.app.texts import HEATMAP_TEXTS
from src.core.names import COLS, HeatmapCols
from src.core.types import TPromptData, TPromptDataFlat, TPromptOriginalIndex
from src.data_ingestion.data_defs.data_defs import ModelCombinationsPrompts
from src.data_ingestion.datasets.download_dataset import (
    df_safe_operation,
    indexed_to_flat_prompt_data,
)
from src.data_ingestion.helpers.dataframe import (
    index_to_row_position,
    validate_one_selected_row_dataframe,
)
from src.data_ingestion.helpers.logits_utils import Prompt
from src.experiments.infrastructure.base_runner import BasePromptFilteration
from src.utils.streamlit.components.aagrid import (
    SelectionMode,
    base_grid_builder,
    set_pre_selected_rows,
)
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKey
from src.utils.types_utils import str_enum_values


def show_prompt(prompt: Prompt):
    annotated_text(
        [
            annotation(val.format(""), col)
            for col in [
                COLS.COUNTER_FACT.RELATION_PREFIX,
                COLS.COUNTER_FACT.SUBJECT,
                COLS.COUNTER_FACT.RELATION_SUFFIX,
                COLS.COUNTER_FACT.TARGET_TRUE,
            ]
            if pd.notna(val := prompt.get_column(col))
        ]
    )


class ShowModelCombinations(StreamlitComponent[tuple[pd.DataFrame, Optional[int]]]):
    def __init__(
        self,
        model_combinations_prompts: ModelCombinationsPrompts,
        representative_model_evaluations: pd.DataFrame,
        filters_container: Optional[DeltaGenerator] = None,
    ):
        self.model_combinations_prompts = model_combinations_prompts
        self.representative_model_evaluations = representative_model_evaluations
        self.filters_container = filters_container

    def render(self):
        display_df = self.model_combinations_prompts.to_display_df(GLOBAL_APP_CONSTS.MODELS_COMBINATIONS)
        assert display_df[HeatmapCols.PROMPT_COUNT].sum() == len(self.representative_model_evaluations), (
            "Display df prompt count mismatch, "
            f"{display_df[HeatmapCols.PROMPT_COUNT].sum()} != {len(self.representative_model_evaluations)}"
        )

        filter_container = self.filters_container
        if filter_container is None:
            filter_container = st.sidebar.expander(HEATMAP_TEXTS.MODEL_COMBINATIONS_FILTERING, expanded=True)
        with filter_container:
            filtered_df = filter_combinations(
                display_df,
                [model_name_and_size.model_name for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS],
            )

        selected_combination_row = get_steamlit_dataframe_selected_row(
            st.dataframe(
                filtered_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    HeatmapCols.PROMPT_COUNT: st.column_config.NumberColumn(pinned=True),
                    HeatmapCols.SELECTED_PROMPT: st.column_config.TextColumn(pinned=True),
                    **{
                        model_name_and_size.model_name: st.column_config.TextColumn()
                        for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
                    },
                },
            )
        )

        return filtered_df, selected_combination_row


class SelectPromptsComponent(StreamlitComponent[Optional[pd.DataFrame]]):
    def __init__(
        self,
        prompts_df: TPromptDataFlat,
        selection_mode: SelectionMode,
        key: str,
        pre_selected_rows: Optional[list[str]] = None,
    ):
        cols = [COLS.ORIGINAL_IDX, *str_enum_values(COLS.COUNTER_FACT)]
        remaining_cols = [col for col in prompts_df.columns if col not in cols]
        self.prompts_df = prompts_df[cols + remaining_cols]
        self.selection_mode = selection_mode
        self.key = key
        self.pre_selected_rows = pre_selected_rows

    def render_grid(self):
        df, grid_builder = base_grid_builder(self.prompts_df, self.selection_mode, [])
        grid_builder.configure_column(COLS.ORIGINAL_IDX, pinned=True)
        grid_builder.configure_column(COLS.COUNTER_FACT.PROMPT, pinned=True)
        grid_builder.configure_column(COLS.COUNTER_FACT.TARGET_TRUE, pinned=True)

        set_pre_selected_rows(grid_builder, self.pre_selected_rows)
        grid_options = grid_builder.build()
        grid_results = AgGrid(
            df,
            key=self.key,
            gridOptions=grid_options,
            height=300,
            # enable_enterprise_modules=False,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            # update_mode=GridUpdateMode.MANUAL,
            # fit_columns_on_grid_load=True,
            data_return_mode=DataReturnMode.AS_INPUT,
            floatingFilter=True,
            allow_unsafe_jscode=True,
        )
        return grid_results

    def render(self):
        grid_results = self.render_grid()
        return grid_results.selected_data

    def render_validate_single_selection(self):
        grid_results = self.render_grid()
        return validate_one_selected_row_dataframe(grid_results.selected_data)


class PromptSelectionForCombinationComponent(StreamlitComponent[TPromptOriginalIndex]):
    # TODO: remove this component
    def __init__(
        self,
        combination_row: ModelCombination,
        representative_model_evaluations: TPromptData,
        model_combinations_prompts: ModelCombinationsPrompts,
    ):
        self.combination_row = combination_row
        self.representative_model_evaluations = representative_model_evaluations
        self.model_combinations_prompts = model_combinations_prompts

    def render(self):
        possible_prompts = df_safe_operation(
            self.representative_model_evaluations,
            lambda df: df.loc[self.combination_row.prompts],
        )
        chosen_prompt_idx = self.combination_row.chosen_prompt
        selected_prompt_row = SelectPromptsComponent(
            indexed_to_flat_prompt_data(possible_prompts),
            SelectionMode.SINGLE,
            key=f"prompt_selection_component_{chosen_prompt_idx}",
            pre_selected_rows=(
                [] if chosen_prompt_idx is None else [str(index_to_row_position(possible_prompts, chosen_prompt_idx))]
            ),
        ).render_validate_single_selection()
        if selected_prompt_row is not None:
            selected_prompt_idx_new = int(selected_prompt_row[COLS.ORIGINAL_IDX])
            if selected_prompt_idx_new != chosen_prompt_idx:
                # Update the selected prompt index
                chosen_prompt_idx = TPromptOriginalIndex(selected_prompt_idx_new)

        if chosen_prompt_idx is not None:
            show_prompt(Prompt(possible_prompts.loc[chosen_prompt_idx]))
            model_evals: DataFrame = get_merged_evaluations(chosen_prompt_idx, AppSessionKeys.code_version.value)
            st.dataframe(
                (
                    model_evals.pipe(
                        lambda df: df[[col for col in df.columns if col not in str_enum_values(COLS.COUNTER_FACT)]]
                    )
                ),
                hide_index=True,
                column_config={
                    "model_arch": st.column_config.TextColumn(pinned=True),
                    "model_size": st.column_config.TextColumn(pinned=True),
                    COLS.EVALUATE_MODEL.MODEL_TOP_OUTPUTS: st.column_config.ListColumn(),
                },
            )
        return chosen_prompt_idx


class PromptFilterationComponent(StreamlitComponent):
    def __init__(
        self,
        prompt_filteration_sk: SessionKey[BasePromptFilteration],
        selected_tree_item_sk: SessionKey[Optional[str]],
    ):
        self.prompt_filteration_sk = prompt_filteration_sk
        self.selected_tree_item_sk = selected_tree_item_sk

    def _get_selected_tree_item(self) -> Optional[int]:
        selected_tree_item = self.selected_tree_item_sk.value
        if selected_tree_item is None:
            return None
        if isinstance(selected_tree_item, list):
            selected_tree_item = selected_tree_item[0]
        # assert isinstance(selected_tree_item, str)
        return int(selected_tree_item)

    @staticmethod
    @lru_cache(maxsize=5)
    def render_show_tree(prompt_filteration: BasePromptFilteration):
        # key_to_label: list[str] = []
        key_to_prompt_filteration: list[BasePromptFilteration] = []

        def register_label(label: str, prompt_filteration: BasePromptFilteration):
            # key_to_label.append(label)
            key_to_prompt_filteration.append(prompt_filteration)
            # return str(len(key_to_label) - 1)
            return label

        def recursive_build_items(prompt_filteration: BasePromptFilteration) -> Union[str, dict, sac.TreeItem]:
            children = None
            if isinstance(prompt_filteration, UnionPromptFilteration) or isinstance(
                prompt_filteration, IntersectionPromptFilteration
            ):
                children = [
                    recursive_build_items(sub_prompt_filteration)
                    for sub_prompt_filteration in prompt_filteration.prompt_filterations
                ]

            return sac.TreeItem(
                children=cast(list, children),
                label=register_label(prompt_filteration.display_name(), prompt_filteration),
                tag=f"Filters {len(prompt_filteration.get_prompt_ids())}",
            )

        items = [recursive_build_items(prompt_filteration)]

        return (
            items,
            # tuple(key_to_label),
            tuple(key_to_prompt_filteration),
        )

    def render(self):
        self.selected_tree_item_sk.init_default()

        (
            items,
            #  key_to_label,
            key_to_prompt_filteration,
        ) = self.render_show_tree(self.prompt_filteration_sk.value)

        selected_item = self._get_selected_tree_item()
        if selected_item is not None and selected_item < len(items):
            self.selected_tree_item_sk.reset_value()

        sac.tree(
            items=items,
            # format_func=lambda item: key_to_label[int(item)],
            label="Prompt Filteration",
            size="lg",
            open_all=True,
            checkbox_strict=True,
            return_index=True,
            key=self.selected_tree_item_sk.key_for_component,
        )

        selected_item = self._get_selected_tree_item()
        if selected_item is not None:
            st.write(str(key_to_prompt_filteration[int(selected_item)]))
