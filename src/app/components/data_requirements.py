import json
from typing import cast

import pandas as pd
import rich
import rich.errors
import rich.traceback
import streamlit as st
from rich.console import Console
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

from src.app.app_consts import (
    GLOBAL_APP_CONSTS,
    AppSessionKeys,
    DataReqConsts,
)
from src.app.components.inputs import select_gpu_type, select_window_size
from src.app.texts import HEATMAP_TEXTS
from src.core.names import DATASETS, HeatmapCols, SlurmStatus, SummarizedDataFulfilledReqsCols
from src.core.types import MODEL_ARCH_AND_SIZE, TPromptOriginalIndex, TVariationName, TWindowSize
from src.data_ingestion.data_defs import DataReqs, SummarizedDataFulfilledReqs
from src.experiments.infrastructure.base_config import CommonParams, SelectivePromptFilteration
from src.experiments.runners.heatmap import HeatmapConfig, HeatmapParams
from src.utils.streamlit.components.aagrid import SelectionMode, base_grid_builder, set_aagrid_apply_default_filters
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.types_utils import select_indexes_from_list

console = Console()


class RequirementsDisplay(StreamlitComponent):
    def __init__(
        self,
        summarized_data_fulfilled_reqs: SummarizedDataFulfilledReqs,
        selection_mode: SelectionMode,
        height: int | None = None,
        key: str = "data_requirements",
        hide_columns: list[str] = [],
    ):
        self.summarized_data_fulfilled_reqs = summarized_data_fulfilled_reqs
        self.height = height
        self.key = key
        self.selection_mode = selection_mode
        self.hide_columns = hide_columns

    def render(self) -> DataReqs | None:
        original_df = self.summarized_data_fulfilled_reqs.to_df()
        data_reqs_df = original_df[DataReqConsts.DATA_REQS_FILTER_COLUMNS]

        df, grid_builder = base_grid_builder(data_reqs_df, self.selection_mode, hide_columns=self.hide_columns)
        for col in DataReqConsts.DATA_REQS_FILTER_COLUMNS:
            grid_builder.configure_column(col, type=["textColumn"])
        grid_options = grid_builder.build()
        set_aagrid_apply_default_filters(
            grid_builder,
            {SummarizedDataFulfilledReqsCols.AvailableOptions: ["0"]},
        )
        # Display the table
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=cast(int, self.height),  # allow None
            fit_columns_on_grid_load=True,
            floatingFilter=True,
            key=self.key,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
        )
        if self.selection_mode == SelectionMode.DISABLED:
            return None

        if grid_response["selected_data"] is None or len(grid_response["selected_data"]) == 0:
            st.warning("No requirements selected")
            return None
        st.write(f"Selected {len(grid_response['selected_data'])} requirements")
        return DataReqs(
            set(
                select_indexes_from_list(
                    self.summarized_data_fulfilled_reqs.to_data_reqs().to_rows(),
                    [int(i) for i in grid_response["selected_data"].index],
                )
            )
        )


class RequirementExecution(StreamlitComponent):
    def __init__(self, data_reqs_to_run: DataReqs):
        self.data_reqs_to_run = data_reqs_to_run

    def render(self):
        # Add SLURM configuration in sidebar
        selected_count = len(self.data_reqs_to_run.to_rows())

        with st.expander(f"Run {selected_count} Filtered Requirements"):
            with_slurm = True
            if selected_count == 1:
                with_slurm = st.checkbox("Run with SLURM", value=False)
            # Show count of selected requirements

            # SLURM configuration
            col1, col2 = st.columns(2)

            with col1:
                AppSessionKeys.variation.create_input_widget()

            with col2:
                if with_slurm:
                    select_gpu_type()

            # Run button
            if selected_count > 0 and st.button(f"🚀 Run {selected_count} Selected Requirements"):
                st.info(f"Preparing to run {selected_count} requirements...")

                success_count = 0
                failed_count = 0

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Get all rows from filtered_df that match selected requirements
                last_error = None
                for i, req in enumerate(self.data_reqs_to_run.to_rows()):
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
                        last_error = e
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
                    raise Exception(f"Failed to submit {failed_count} requirements") from last_error


def get_models_remaining_prompts(
    model_combinations: list[MODEL_ARCH_AND_SIZE],
    window_size: TWindowSize,
    variation: TVariationName,
    prompt_original_indices: list[TPromptOriginalIndex],
) -> dict[MODEL_ARCH_AND_SIZE, HeatmapConfig]:
    """Get the remaining prompts for each model."""
    res = {}
    for model_arch, model_size in model_combinations:
        config = HeatmapConfig(
            variation=variation,
            common_params=CommonParams(
                model_arch=model_arch,
                model_size=model_size,
            ),
            prompt_filteration=SelectivePromptFilteration(
                dataset_name=DATASETS.COUNTER_FACT,
                prompt_ids=prompt_original_indices,
            ),
            runner_params=HeatmapParams(
                window_size=window_size,
            ),
        )
        if config.get_remaining_prompt_original_indices():
            res[MODEL_ARCH_AND_SIZE(model_arch, model_size)] = config
    return res


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
            AppSessionKeys.variation.create_input_widget()
        with col3:
            select_gpu_type()

        test_existing_prompts = st.checkbox("Test existing prompts", value=False)

        if test_existing_prompts:
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
                        "Prompt Count": len(heatmap_config.get_remaining_prompt_original_indices()),
                        "Status": heatmap_config.get_slurm_status(),
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
                last_error = None
                for i, heatmap_config in enumerate(models_remaining_prompts.values()):
                    try:
                        if skip_running and heatmap_config.get_slurm_status() in [
                            SlurmStatus.RUNNING,
                            SlurmStatus.PENDING,
                        ]:
                            st.warning(
                                HEATMAP_TEXTS.skipping_running(
                                    heatmap_config.common_params.model_arch, heatmap_config.common_params.model_size
                                )
                            )
                            continue

                        # Set running parameters
                        heatmap_config.set_running_params(
                            with_slurm=True,
                            slurm_gpu_type=AppSessionKeys.get_selected_gpu(
                                heatmap_config.common_params.model_arch_and_size
                            ),
                        )

                        # Submit job
                        heatmap_config.run()
                        success_count += 1

                    except Exception as e:
                        st.exception(e)
                        last_error = e
                        st.error(HEATMAP_TEXTS.submit_failed(heatmap_config.get_remaining_prompt_original_indices(), e))
                        failed_count += 1

                    # Update progress
                    progress = (i + 1) / len(models_remaining_prompts)
                    progress_bar.progress(progress)
                    status_text.text(HEATMAP_TEXTS.processing_status(i + 1, len(models_remaining_prompts)))

                # Show final status
                if success_count > 0:
                    st.success(HEATMAP_TEXTS.success_status(success_count))
                if failed_count > 0:
                    raise Exception(HEATMAP_TEXTS.error_status(failed_count)) from last_error
