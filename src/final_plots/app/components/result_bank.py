from enum import StrEnum
from typing import Optional

from st_aggrid import AgGrid, AgGridReturn, DataReturnMode, GridOptionsBuilder, GridUpdateMode

from src.consts import EXPERIMENT_NAMES
from src.final_plots.app.data_store import load_experiment_results
from src.final_plots.results_bank import ParamNames
from src.utils.streamlit.aagrid import set_aagrid_apply_default_filters
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
        filters: dict[str, list] = {},
        hide_columns: list[str] = [],
    ):
        super().__init__()
        self.filter_experiment_name = filter_experiment_name
        self.filter_is_all_correct = filter_is_all_correct
        self.selection_mode = selection_mode
        self.height = height
        self.key = key
        self.filters = filters
        self.hide_columns = hide_columns

    def _get_df(self):
        df = load_experiment_results()
        load_experiment_results.render()
        if self.filter_experiment_name is not None:
            df = df[df[ParamNames.experiment_name] == self.filter_experiment_name]
        if self.filter_is_all_correct is not None:
            df = df[df[ParamNames.is_all_correct] == self.filter_is_all_correct]
        # if the first column is hidden, we need to reorder the columns
        if self.selection_mode != SelectionMode.DISABLED and df.columns[0] in self.hide_columns:
            df = df[[*df.columns[1:], df.columns[0]]]
        return df

    def render(self) -> AgGridReturn:
        df = self._get_df()
        grid_builder = GridOptionsBuilder.from_dataframe((df))
        grid_builder.configure_pagination(enabled=True)

        if self.selection_mode != SelectionMode.DISABLED:
            grid_builder.configure_selection(
                selection_mode=self.selection_mode, use_checkbox=True, header_checkbox=True
            )

        grid_builder.configure_default_column(
            filter=True,
            floatingFilter=True,
        )
        set_aagrid_apply_default_filters(
            grid_builder,
            self.filters,
        )

        for col in self.hide_columns:
            grid_builder.configure_column(col, hide=True)
        for col in [ParamNames.window_size, ParamNames.prompt_idx]:
            if col not in self.hide_columns:
                grid_builder.configure_column(col, type=["textColumn"])

        grid_builder.configure_side_bar()
        grid_options = grid_builder.build()

        # Display the table
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=self.height,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key + "b",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
        )
        return grid_response
