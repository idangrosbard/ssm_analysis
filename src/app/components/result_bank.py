from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode
from streamlit_modal import Modal

from src.core.names import ResultBankParamNames
from src.data_ingestion.data_defs.data_defs import ResultBank
from src.experiments.infrastructure.base_runner import BaseRunner
from src.utils.streamlit.components.aagrid import SelectionMode, base_grid_builder, set_aagrid_apply_default_filters
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKey

T_RESULT_BANK_TYPE = TypeVar("T_RESULT_BANK_TYPE", bound=ResultBank)


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


modal = Modal(key="job_output_modal", title="Job Output", max_width=1000)


class ShowRunnerStatus(StreamlitComponent):
    def __init__(self, runner: BaseRunner):
        super().__init__()
        self.runner = runner

    def render(self):
        st.code(self.runner.variation_paths.variation_base_path, wrap_lines=True)
        st.expander(expanded=False, label="Params").write(asdict(self.runner))
        # Computation status

        all_slurm_jobs = self.runner.slurm_job_folder.get_all_slurm_jobs()
        all_slurm_jobs = sorted(all_slurm_jobs, key=lambda x: -int(x.job_id))
        sk_file_path = SessionKey[Path](f"job_output_path_{self.runner.experiment_name}", None, allow_none=True)

        if modal.is_open():
            with modal.container():
                lines = sk_file_path.value.read_text().split("\n")
                if len(lines) > 300:
                    first_100 = lines[:100]
                    last_200 = lines[-200:]
                    skipped_lines = len(lines) - 300
                    lines = [
                        *first_100,
                        f"...Skipped {skipped_lines} lines",
                        "...",
                        *last_200,
                    ]

                lines = lines[::-1]
                st.code(sk_file_path.value, wrap_lines=True)
                st.code("\n".join(lines))

        with st.expander(expanded=False, label=f"{len(all_slurm_jobs)} Slurm Runs"):
            for slurm_job in all_slurm_jobs:
                st.markdown(f"**Job ID:** {slurm_job.job_id} - **Status:** {slurm_job.get_slurm_status()}")

                cols = st.columns(2)
                with cols[0]:
                    if st.button("Show Job Output", key=f"show_job_output_{slurm_job.job_id}"):
                        sk_file_path.value = slurm_job.slurm_job_output_path
                        modal.open()

                with cols[1]:
                    if st.button("Show Job Error", key=f"show_job_error_{slurm_job.job_id}"):
                        sk_file_path.value = slurm_job.slurm_job_error_path
                        modal.open()

        # Show uncomputed dependencies if any
        if not self.runner.dependencies_are_computed():
            with st.expander("**Uncomputed Dependencies** ❌"):
                uncomputed = self.runner.uncomputed_dependencies()

                def render_dependencies(deps, level=0):
                    for key, dep in deps.items():
                        indent = "&nbsp;" * (4 * level)
                        if isinstance(dep, BaseRunner):
                            st.markdown(
                                f"{indent}• {key}: {dep.experiment_name} ({dep.variant_params.model_arch_and_size})",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f"{indent}• {key}:", unsafe_allow_html=True)
                            render_dependencies(dep, level + 1)

                render_dependencies(uncomputed)

                if st.button("Compute Dependencies"):
                    with st.spinner("Computing..."):
                        try:
                            self.runner.compute_dependencies(1)
                            st.success("Computation completed or submitted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during computation: {e}")

        # # Add a button to compute if not computed yet
        # if not self.runner.is_computed():
        #     cols = st.columns([1, 2])
        #     with cols[0]:
        #         st.markdown("**Computed:** ❌")
        #     with cols[1]:
        #         if st.button("Compute Runner"):
        #             with st.spinner("Computing..."):
        #                 try:
        #                     self.runner.run(with_dependencies=False)
        #                     st.success("Computation completed or submitted!")
        #                     st.rerun()
        #                 except Exception as e:
        #                     st.error(f"Error during computation: {e}")
