from typing import Optional

from st_aggrid import AgGrid, AgGridReturn, DataReturnMode, GridUpdateMode

from src.core.names import EXPERIMENT_NAMES, ResultBankParamNames
from src.data_ingestion.data_defs import ResultBank
from src.data_ingestion.helpers.dataframe import validate_one_selected_row_dataframe
from src.utils.streamlit.components.aagrid import SelectionMode, base_grid_builder, set_aagrid_apply_default_filters
from src.utils.streamlit.helpers.component import StreamlitComponent


class ShowResultsBank(StreamlitComponent):
    def __init__(
        self,
        results_bank: ResultBank,
        filter_experiment_name: Optional[EXPERIMENT_NAMES] = None,
        selection_mode: SelectionMode = SelectionMode.DISABLED,
        height: int = 1000,
        key: str = "results_bank",
        filters: dict[str, list] = {},
        hide_columns: list[str] = [],
    ):
        super().__init__()
        self.results_bank = results_bank
        self.filter_experiment_name = filter_experiment_name
        self.selection_mode = selection_mode
        self.height = height
        self.key = key
        self.filters = filters
        self.hide_columns = hide_columns

    def _get_df(self):
        df = self.results_bank.to_experiment_results().to_df()
        if self.filter_experiment_name is not None:
            df = df[df[ResultBankParamNames.experiment_name] == self.filter_experiment_name]
        return df

    def render(self) -> AgGridReturn:
        df, grid_builder = base_grid_builder(self._get_df(), self.selection_mode, self.hide_columns)
        set_aagrid_apply_default_filters(
            grid_builder,
            self.filters,
        )
        for col in [ResultBankParamNames.window_size]:
            if col not in self.hide_columns:
                grid_builder.configure_column(col, type=["textColumn"])

        grid_options = grid_builder.build()

        # Display the table
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=self.height,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
        )
        return grid_response

    def render_validate_single_selection(self):
        grid_results = self.render()
        return validate_one_selected_row_dataframe(grid_results.selected_data)
