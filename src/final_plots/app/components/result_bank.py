from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from src.final_plots.app.data_store import load_experiment_results
from src.final_plots.results_bank import ParamNames
from src.utils.streamlit_utils import StreamlitComponent


class ShowResultsBank(StreamlitComponent):
    def render(self) -> None:
        df = load_experiment_results()
        load_experiment_results.render()
        grid_builder = GridOptionsBuilder.from_dataframe(df)
        grid_builder.configure_pagination(enabled=True)
        # grid_builder.configure_selection(selection_mode="single", use_checkbox=True)
        grid_builder.configure_default_column(
            filter=True,
            floatingFilter=True,
        )
        grid_builder.configure_column(ParamNames.window_size, type=["textColumn"])
        grid_builder.configure_column(ParamNames.prompt_idx, type=["textColumn"])
        grid_builder.configure_side_bar()
        grid_options = grid_builder.build()

        # Display the table
        AgGrid(
            df,
            gridOptions=grid_options,
            height=1000,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key="results_bank",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
        )
