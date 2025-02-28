from enum import StrEnum
from typing import Optional

from st_aggrid import AgGrid, AgGridReturn, DataReturnMode, GridOptionsBuilder, GridUpdateMode

from src.consts import EXPERIMENT_NAMES
from src.final_plots.app.data_store import load_experiment_results
from src.final_plots.results_bank import ParamNames
from src.utils.streamlit_utils import StreamlitComponent


class SelectionMode(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    DISABLED = "disabled"


class ShowResultsBank(StreamlitComponent):
    def __init__(
        self,
        filter_experiment_name: Optional[EXPERIMENT_NAMES] = None,
        filter_is_all_correct: Optional[bool] = None,
        selection_mode: SelectionMode = SelectionMode.DISABLED,
        height: int = 1000,
        key: str = "results_bank",
    ):
        super().__init__()
        self.filter_experiment_name = filter_experiment_name
        self.filter_is_all_correct = filter_is_all_correct
        self.selection_mode = selection_mode
        self.height = height
        self.key = key

    def _get_df(self):
        df = load_experiment_results()
        load_experiment_results.render()
        if self.filter_experiment_name is not None:
            df = df[df[ParamNames.experiment_name] == self.filter_experiment_name]
        if self.filter_is_all_correct is not None:
            df = df[df[ParamNames.is_all_correct] == self.filter_is_all_correct]
        return df

    def render(self) -> AgGridReturn:
        df = self._get_df()
        grid_builder = GridOptionsBuilder.from_dataframe(df)
        grid_builder.configure_pagination(enabled=True)
        grid_builder.configure_selection(selection_mode=self.selection_mode, use_checkbox=True)
        grid_builder.configure_default_column(
            filter=True,
            floatingFilter=True,
        )
        grid_builder.configure_column(ParamNames.window_size, type=["textColumn"])
        grid_builder.configure_column(ParamNames.prompt_idx, type=["textColumn"])
        grid_builder.configure_side_bar()
        grid_options = grid_builder.build()

        # Display the table
        return AgGrid(
            df,
            gridOptions=grid_options,
            height=self.height,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
        )
