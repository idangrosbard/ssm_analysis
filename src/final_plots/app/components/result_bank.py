from typing import Optional

from st_aggrid import AgGrid, AgGridReturn, DataReturnMode, GridUpdateMode

from src.final_plots.app.data_store import load_experiment_results
from src.names import EXPERIMENT_NAMES, ResultBankParamNames
from src.utils.streamlit.aagrid import SelectionMode, base_grid_builder, set_aagrid_apply_default_filters
from src.utils.streamlit.dataframe import validate_one_selected_row_dataframe
from src.utils.streamlit_utils import StreamlitComponent


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
        df = load_experiment_results().to_df()
        load_experiment_results.render()
        if self.filter_experiment_name is not None:
            df = df[df[ResultBankParamNames.experiment_name] == self.filter_experiment_name]
        if self.filter_is_all_correct is not None:
            df = df[df[ResultBankParamNames.is_all_correct] == self.filter_is_all_correct]
        return df

    def render(self) -> AgGridReturn:
        df, grid_builder = base_grid_builder(self._get_df(), self.selection_mode, self.hide_columns)
        set_aagrid_apply_default_filters(
            grid_builder,
            self.filters,
        )
        for col in [ResultBankParamNames.window_size, ResultBankParamNames.prompt_idx]:
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
