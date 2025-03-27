from typing import TypeVar

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

from src.core.names import ResultBankParamNames
from src.data_ingestion.data_defs import ResultBank
from src.utils.streamlit.components.aagrid import SelectionMode, base_grid_builder, set_aagrid_apply_default_filters
from src.utils.streamlit.helpers.component import StreamlitComponent

T_RESULT_BANK_TYPE = TypeVar("T_RESULT_BANK_TYPE", bound="ResultBank")


class ShowResultsBank(StreamlitComponent[T_RESULT_BANK_TYPE]):
    def __init__(
        self,
        results_bank: T_RESULT_BANK_TYPE,
        selection_mode: SelectionMode = SelectionMode.DISABLED,
        height: int = 1000,
        key: str = "results_bank",
        filters: dict[str, list] = {},
        hide_columns: list[str] = [],
    ):
        super().__init__()
        self.results_bank = results_bank
        self.selection_mode = selection_mode
        self.height = height
        self.key = key
        self.filters = filters
        self.hide_columns = [self.results_bank.KEY] + hide_columns

    def render(self) -> T_RESULT_BANK_TYPE:
        df, grid_builder = base_grid_builder(
            self.results_bank.to_experiment_results_df(), self.selection_mode, self.hide_columns
        )
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

        return self.results_bank.from_experiment_results_df(grid_response.selected_data)
