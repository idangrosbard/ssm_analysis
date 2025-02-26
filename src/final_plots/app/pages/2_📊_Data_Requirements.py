# Purpose: Manage and display data requirements for experiments with filtering and execution capabilities
# High Level Outline:
# 1. Page setup and configuration
# 2. Initialize session state and load data
# 3. Display and manage requirements with filters
# 4. Handle requirement selection and execution
# Outline Issues:
# - Consider adding batch operations for requirements
# - Add progress tracking for running requirements
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

import src.final_plots.app  # noqa: F401
from src.final_plots.app.app_consts import AppSessionKeys, DataReqConsts
from src.final_plots.app.components.inputs import select_gpu_type, select_variation
from src.final_plots.app.data_store import (
    load_fulfilled_reqs_df,
    load_latest_fulfilled_reqs,
)
from src.final_plots.app.texts import DATA_REQUIREMENTS_TEXTS
from src.final_plots.app.utils import get_data_req_from_df_row
from src.final_plots.data_reqs import _save_data_fulfilled
from src.utils.streamlit_utils import StreamlitComponent, StreamlitPage

# region Page Configuration
st.set_page_config(page_title=DATA_REQUIREMENTS_TEXTS.title, page_icon=DATA_REQUIREMENTS_TEXTS.icon, layout="wide")
st.title(f"{DATA_REQUIREMENTS_TEXTS.title} {DATA_REQUIREMENTS_TEXTS.icon}")
# endregion


class RequirementsDisplay(StreamlitComponent):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render(self) -> pd.DataFrame | None:
        data_reqs_df = self.df[DataReqConsts.DATA_REQS_FILTER_COLUMNS]

        grid_builder = GridOptionsBuilder.from_dataframe(data_reqs_df)
        grid_builder.configure_pagination(enabled=True)
        grid_builder.configure_selection(selection_mode="multiple", use_checkbox=True, header_checkbox=True)
        grid_builder.configure_default_column(
            filter=True,
            floatingFilter=True,
        )
        for col in DataReqConsts.DATA_REQS_FILTER_COLUMNS:
            grid_builder.configure_column(col, type=["textColumn"])
        grid_builder.configure_side_bar()
        grid_options = grid_builder.build()

        # Display the table
        grid_response = AgGrid(
            data_reqs_df,
            gridOptions=grid_options,
            height=1000,
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key="data_requirements",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
        )
        if grid_response["selected_data"] is None or len(grid_response["selected_data"]) == 0:
            st.warning("No requirements selected")
            return None
        filtered_reqs = self.df.loc[pd.to_numeric(grid_response["selected_data"].index)]
        st.write(filtered_reqs)
        st.write(f"Selected {len(filtered_reqs)} requirements")
        return filtered_reqs


class RequirementExecution(StreamlitComponent):
    def __init__(self, data_reqs_to_run: pd.DataFrame):
        self.data_reqs_to_run = data_reqs_to_run

    def render(self):
        # Add SLURM configuration in sidebar
        selected_count = len(self.data_reqs_to_run)
        with st.sidebar:
            with st.expander(f"Run {selected_count} Filtered Requirements"):
                # Show count of selected requirements

                # SLURM configuration
                col1, col2 = st.columns(2)

                with col1:
                    select_variation()

                with col2:
                    select_gpu_type()

                # Run button
                if len(self.data_reqs_to_run) > 0 and st.button(f"🚀 Run {selected_count} Selected Requirements"):
                    st.info(f"Preparing to run {selected_count} requirements...")

                    success_count = 0
                    failed_count = 0

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Get all rows from filtered_df that match selected requirements

                    for i, (idx, row) in enumerate(self.data_reqs_to_run.iterrows()):
                        try:
                            req = get_data_req_from_df_row(row)

                            # Get config and set running parameters
                            config = req.get_config(variation=AppSessionKeys.variation.value)
                            config.set_running_params(
                                with_slurm=True,
                                slurm_gpu_type=AppSessionKeys.get_selected_gpu(req.model_arch_and_size),
                            )

                            # Run the configuration
                            config.run()
                            success_count += 1

                        except Exception as e:
                            st.error(f"Failed to run requirement: {str(e)}")
                            failed_count += 1

                        # Update progress
                        progress = (i + 1) / selected_count
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Processed: {i + 1}/{selected_count} | Success: {success_count} | Failed: {failed_count}"
                        )

                    if success_count > 0:
                        st.success(f"Successfully submitted {success_count} requirements to run")
                    if failed_count > 0:
                        st.warning(f"Failed to submit {failed_count} requirements")


class DataRequirementsPage(StreamlitPage):
    def render(self):
        # region Data Loading and Preparation
        # Load data
        df = load_fulfilled_reqs_df()
        load_fulfilled_reqs_df.render()

        # Filter the data
        # filtered_df = RequirementsFiltering(df).render()

        # Display requirements
        data_reqs_to_run = RequirementsDisplay(df).render()

        # Save button for overrides
        with st.sidebar.expander(DATA_REQUIREMENTS_TEXTS.reset_to_latest):
            load_latest_fulfilled_reqs.render()
            if st.button(DATA_REQUIREMENTS_TEXTS.reset_to_latest):
                _save_data_fulfilled(load_latest_fulfilled_reqs())
                st.success("Requirements updated successfully!")

        if data_reqs_to_run is not None:
            # Handle requirement execution
            RequirementExecution(data_reqs_to_run).render()


if __name__ == "__main__":
    DataRequirementsPage().render()
