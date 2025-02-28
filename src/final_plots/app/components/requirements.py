import json
from typing import cast

import pandas as pd
import rich
import rich.errors
import rich.traceback
import streamlit as st
from rich.console import Console
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode

from src.final_plots.app.app_consts import GLOBAL_APP_CONSTS, AppSessionKeys, DataReqCols, DataReqConsts, HeatmapCols
from src.final_plots.app.components.inputs import select_gpu_type, select_variation, select_window_size
from src.final_plots.app.data_store import get_models_remaining_prompts
from src.final_plots.app.texts import HEATMAP_TEXTS
from src.final_plots.app.utils import get_data_req_from_df_row
from src.types import MODEL_ARCH_AND_SIZE, TPromptOriginalIndex
from src.utils.streamlit.aagrid import set_aagrid_apply_default_filters
from src.utils.streamlit_utils import StreamlitComponent

console = Console()


class RequirementsDisplay(StreamlitComponent):
    def __init__(self, df: pd.DataFrame, height: int | None = None, key: str = "data_requirements"):
        self.df = df
        self.height = height
        self.key = key

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
        set_aagrid_apply_default_filters(
            grid_builder,
            {DataReqCols.AvailableOptions: ["0"]},
        )
        # Display the table
        grid_response = AgGrid(
            data_reqs_df,
            gridOptions=grid_options,
            height=cast(int, self.height),  # allow None
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
        )
        if grid_response["selected_data"] is None or len(grid_response["selected_data"]) == 0:
            st.warning("No requirements selected")
            return None
        filtered_reqs = self.df.loc[pd.to_numeric(grid_response["selected_data"].index)]
        st.write(filtered_reqs.drop(columns=[DataReqCols.Options]))
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
                with_slurm = True
                if selected_count == 1:
                    with_slurm = st.checkbox("Run with SLURM", value=False)
                # Show count of selected requirements

                # SLURM configuration
                col1, col2 = st.columns(2)

                with col1:
                    select_variation()

                with col2:
                    if with_slurm:
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
                        req = get_data_req_from_df_row(row)
                        try:
                            # Get config and set running parameters
                            config = req.get_config(variation=AppSessionKeys.variation.value)
                            if with_slurm:
                                config.set_running_params(
                                    with_slurm=True,
                                    slurm_gpu_type=AppSessionKeys.get_selected_gpu(req.model_arch_and_size),
                                )

                            # Run the configuration
                            config.run()
                            success_count += 1

                        except Exception as e:
                            st.error(f"Failed to run requirement: {str(json.dumps(req._asdict(), indent=4))}")
                            st.exception(e)
                            failed_count += 1
                            console.print(
                                rich.traceback.Traceback.from_exception(
                                    exc_type=type(e), exc_value=e, traceback=e.__traceback__
                                )
                            )

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


class HeatmapGenerationComponent(StreamlitComponent):
    def __init__(self, filtered_df: pd.DataFrame):
        self.filtered_df = filtered_df

    def render(self):
        # Show count of selected prompts
        # SLURM configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            select_window_size()
        with col2:
            select_variation()
        with col3:
            select_gpu_type()

        test_existing_prompts = st.checkbox("Test existing prompts", value=False)

        if test_existing_prompts:
            if st.button("reset remaining prompts"):
                get_models_remaining_prompts.clear()  # type: ignore
            prompt_original_indices = [
                TPromptOriginalIndex(int(x)) for x in self.filtered_df[HeatmapCols.SELECTED_PROMPT]
            ]
            with st.spinner("Calculating remaining prompts to run...", show_time=True):
                models_remaining_prompts = get_models_remaining_prompts(
                    GLOBAL_APP_CONSTS.MODELS_COMBINATIONS,
                    AppSessionKeys.window_size.value,
                    AppSessionKeys.variation.value,
                    prompt_original_indices,
                )

            table_data = []
            for model_arch_and_size, heatmap_config in models_remaining_prompts.items():
                model_name = model_arch_and_size.model_name
                table_data.append(
                    {
                        "Model": model_name,
                        "Prompt Count": len(heatmap_config.prompt_original_indices),
                        "Running": heatmap_config.is_running(),
                        "GPU": AppSessionKeys.get_selected_gpu(model_arch_and_size),
                    }
                )
            st.table(table_data)

            skip_running = st.checkbox("Skip running", value=True)
            # Run button
            if models_remaining_prompts and st.button(
                HEATMAP_TEXTS.run_selected_prompts(len(models_remaining_prompts))
            ):
                success_count = 0
                failed_count = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                # Get all rows from filtered_df that match selected prompts
                for i, heatmap_config in enumerate(models_remaining_prompts.values()):
                    try:
                        if skip_running and heatmap_config.is_running():
                            st.warning(
                                HEATMAP_TEXTS.skipping_running(heatmap_config.model_arch, heatmap_config.model_size)
                            )
                            continue

                        # Set running parameters
                        heatmap_config.set_running_params(
                            with_slurm=True,
                            slurm_gpu_type=AppSessionKeys.get_selected_gpu(
                                MODEL_ARCH_AND_SIZE(heatmap_config.model_arch, heatmap_config.model_size)
                            ),
                        )

                        # Submit job
                        heatmap_config.run()
                        success_count += 1

                    except Exception as e:
                        st.error(HEATMAP_TEXTS.submit_failed(heatmap_config.prompt_original_indices, e))
                        failed_count += 1

                    # Update progress
                    progress = (i + 1) / len(models_remaining_prompts)
                    progress_bar.progress(progress)
                    status_text.text(HEATMAP_TEXTS.processing_status(i + 1, len(models_remaining_prompts)))

                # Show final status
                if success_count > 0:
                    st.success(HEATMAP_TEXTS.success_status(success_count))
                if failed_count > 0:
                    st.warning(HEATMAP_TEXTS.error_status(failed_count))
