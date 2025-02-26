from typing import cast

import pandas as pd
import streamlit as st
from pandas import DataFrame

from src.consts import COLUMNS
from src.final_plots.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys, HeatmapCols
from src.final_plots.app.components.prompt import show_prompt
from src.final_plots.app.data_store import get_merged_evaluations
from src.final_plots.app.texts import HEATMAP_TEXTS
from src.final_plots.app.utils import filter_combinations, get_steamlit_dataframe_selected_row
from src.final_plots.data_reqs import ModelCombination, save_model_combinations_prompts
from src.utils.logits import Prompt
from src.utils.streamlit_utils import StreamlitComponent


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
        selected_prompt_idx = self.combination_row.chosen_prompt
        selected_row_idx = get_steamlit_dataframe_selected_row(
            st.dataframe(
                possible_prompts[COLUMNS.PROMPT_DATA_COLS],
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    COLUMNS.PROMPT: st.column_config.TextColumn(
                        pinned=True,
                    )
                },
                use_container_width=True,
            )
        )
        if selected_row_idx is not None:
            selected_prompt_idx_new = int(cast(pd.Index, possible_prompts.iloc[selected_row_idx]).name)
            if selected_prompt_idx_new != selected_prompt_idx:
                if st.button(HEATMAP_TEXTS.BUT_SAVE_NEW_SELECTION(selected_prompt_idx, selected_prompt_idx_new)):
                    # selected_prompt = possible_prompts.iloc[selected_row_idx]  # type: ignore
                    raise NotImplementedError("Saving is not implemented yet")
                    save_model_combinations_prompts(self.combinations_df)

                # Update the selected prompt index
                selected_prompt_idx = selected_prompt_idx_new

        if selected_prompt_idx is not None:
            show_prompt(Prompt(possible_prompts.loc[selected_prompt_idx]))
            model_evals: DataFrame = get_merged_evaluations(selected_prompt_idx, AppSessionKeys.variation.value)

            st.dataframe(
                (model_evals.pipe(lambda df: df[[col for col in df.columns if col not in COLUMNS.PROMPT_DATA_COLS]])),
                hide_index=True,
                column_config={
                    "model_arch": st.column_config.TextColumn(pinned=True),
                    "model_size": st.column_config.TextColumn(pinned=True),
                    COLUMNS.MODEL_TOP_OUTPUTS: st.column_config.ListColumn(),
                },
            )
        return selected_prompt_idx
