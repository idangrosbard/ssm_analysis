from typing import Optional

import pandas as pd
import streamlit as st
from annotated_text import annotated_text, annotation
from pandas import DataFrame
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

from src.analysis.experiment_results.data_requirements import ModelCombination, save_model_combinations_prompts
from src.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys
from src.app.app_utils import (
    filter_combinations,
    get_steamlit_dataframe_selected_row,
)
from src.app.data_store import get_merged_evaluations
from src.app.texts import HEATMAP_TEXTS
from src.core.names import COLS, HeatmapCols
from src.core.types import TPromptOriginalIndex
from src.data_ingestion.helpers.dataframe import (
    index_to_row_position,
    validate_one_selected_row_dataframe,
)
from src.data_ingestion.helpers.logits_utils import Prompt
from src.utils.streamlit.components.aagrid import (
    SelectionMode,
    base_grid_builder,
    set_pre_selected_rows,
)
from src.utils.streamlit.helpers.component import StreamlitComponent
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


class ModelCombinations(StreamlitComponent):
    def __init__(
        self,
        combinations_df: list[ModelCombination],
        representative_model_evaluations: pd.DataFrame,
    ):
        self.combinations_df = combinations_df
        self.representative_model_evaluations = representative_model_evaluations

    def render(self):
        table_data = []
        for row in self.combinations_df:
            # Create row with model correctness
            table_row = {}

            # Add prompt count and selected prompt first
            table_row[HeatmapCols.PROMPT_COUNT] = len(row.prompts)
            table_row[HeatmapCols.SELECTED_PROMPT] = row.chosen_prompt

            # Add model columns at the end
            for model_name_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
                model_name = model_name_and_size.model_name
                if model_name_and_size in row.correct_models:
                    table_row[model_name] = "✅"
                elif model_name_and_size in row.incorrect_models:
                    table_row[model_name] = "❌"
                else:
                    table_row[model_name] = "-"
            table_data.append(table_row)

        # endregion
        # region Create DataFrame for display
        display_df = pd.DataFrame(table_data)

        assert display_df[HeatmapCols.PROMPT_COUNT].sum() == len(self.representative_model_evaluations), (
            "Display df prompt count mismatch, "
            f"{display_df[HeatmapCols.PROMPT_COUNT].sum()} != {len(self.representative_model_evaluations)}"
        )

        with st.sidebar.expander(HEATMAP_TEXTS.MODEL_COMBINATIONS_FILTERING, expanded=True):
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


class ShowPromptsComponent(StreamlitComponent):
    def __init__(
        self,
        model_evals: pd.DataFrame,
        selection_mode: SelectionMode,
        key: str,
        pre_selected_rows: Optional[list[str]] = None,
    ):
        self.model_evals = model_evals.reset_index()[GLOBAL_APP_CONSTS.MODEL_EVALS_COLUMNS]
        self.selection_mode = selection_mode
        self.key = key
        self.pre_selected_rows = pre_selected_rows

    def render(self):
        df, grid_builder = base_grid_builder(self.model_evals, self.selection_mode, [])
        grid_builder.configure_first_column_as_index()
        grid_builder.configure_column(COLS.COUNTER_FACT.PROMPT, pinned=True)
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
            data_return_mode=DataReturnMode.AS_INPUT,
            floatingFilter=True,
            allow_unsafe_jscode=True,
        )
        return grid_results

    def render_validate_single_selection(self):
        grid_results = self.render()
        return validate_one_selected_row_dataframe(grid_results.selected_data)


class PromptSelectionComponent(StreamlitComponent):
    def __init__(
        self,
        combination_row: ModelCombination,
        representative_model_evaluations: pd.DataFrame,
        combinations_df: list[ModelCombination],
    ):
        self.combination_row = combination_row
        self.representative_model_evaluations = representative_model_evaluations
        self.combinations_df = combinations_df

    def render(self):
        possible_prompts = self.representative_model_evaluations.loc[self.combination_row.prompts]
        chosen_prompt_idx = self.combination_row.chosen_prompt
        selected_prompt_row = ShowPromptsComponent(
            possible_prompts,
            SelectionMode.SINGLE,
            key=f"prompt_selection_component_{chosen_prompt_idx}",
            pre_selected_rows=(
                [] if chosen_prompt_idx is None else [str(index_to_row_position(possible_prompts, chosen_prompt_idx))]
            ),
        ).render_validate_single_selection()
        if selected_prompt_row is not None:
            selected_prompt_idx_new = int(selected_prompt_row[COLS.ORIGINAL_IDX])
            if selected_prompt_idx_new != chosen_prompt_idx:
                if st.button(HEATMAP_TEXTS.BUT_SAVE_NEW_SELECTION(chosen_prompt_idx, selected_prompt_idx_new)):
                    # selected_prompt = possible_prompts.iloc[selected_row_idx]  # type: ignore
                    raise NotImplementedError("Saving is not implemented yet")
                    save_model_combinations_prompts(self.combinations_df)

                # Update the selected prompt index
                chosen_prompt_idx = TPromptOriginalIndex(selected_prompt_idx_new)

        if chosen_prompt_idx is not None:
            show_prompt(Prompt(possible_prompts.loc[chosen_prompt_idx]))
            model_evals: DataFrame = get_merged_evaluations(chosen_prompt_idx, AppSessionKeys.variation.value)
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
