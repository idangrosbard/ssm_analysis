import shutil

import streamlit as st

from src.analysis.experiment_results.helpers import result_record_to_data_req
from src.core.types import TVariationName
from src.data_ingestion.data_defs import ResultBank
from src.experiments.runners.evaluate_model import EvaluateModelConfig
from src.experiments.runners.heatmap import HeatmapConfig
from src.experiments.runners.info_flow import InfoFlowConfig
from src.utils.file_system import fast_relative_to
from src.utils.streamlit.helpers.component import StreamlitComponent


class MigrateResults(StreamlitComponent):
    def __init__(self, results_bank: ResultBank, is_test_results: bool):
        self.results_bank = results_bank
        self.is_test_results = is_test_results

    def render(self):
        from src.core.consts import PATHS
        from tests.src.experiments.test_full_pipeline import TEST_BASE_PATH

        dry_run = st.checkbox("Dry Run", value=True, disabled=True)
        rows = self.results_bank.to_rows()

        st.write(len(rows))
        rows = [row for row in rows if row.variation == TVariationName("v3")]

        st.write(len(rows))

        # st.stop()
        rows_count = len(rows)
        progress_bar = st.progress(0, text="Starting...")
        prev_path = PATHS.PROJECT_DIR
        try:
            if self.is_test_results:
                PATHS.PROJECT_DIR = TEST_BASE_PATH

            for i, result_record in enumerate(rows):
                data_req = result_record_to_data_req(result_record)
                config = data_req.get_config(TVariationName("v1"))
                source_path = result_record.path
                if isinstance(config, EvaluateModelConfig):
                    assert isinstance(config, EvaluateModelConfig)
                    new_path = config.output_result_path
                elif isinstance(config, InfoFlowConfig):
                    assert isinstance(config, InfoFlowConfig)
                    new_path = config.output_block_target_source_path()
                elif isinstance(config, HeatmapConfig):
                    assert isinstance(config, HeatmapConfig)
                    assert len(config.prompt_ids) == 1
                    new_path = config.output_heatmap_path(config.prompt_ids[0])
                else:
                    raise ValueError(f"Unknown config type: {type(config)}")

                new_path = PATHS.PROJECT_DIR / "outputs.new" / fast_relative_to(new_path, PATHS.OUTPUT_DIR)

                if not dry_run:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(source_path, new_path)

                progress = min((i + 1) / rows_count, 1.0)
                progress_bar.progress(
                    progress,
                    text="\n \n".join(
                        [
                            f"Processed {i + 1} items{' [DRY RUN]' if dry_run else ''}",
                            f"{fast_relative_to(source_path, PATHS.PROJECT_DIR)} ->",
                            f"{fast_relative_to(new_path, PATHS.PROJECT_DIR)}",
                        ]
                    ),
                )

        finally:
            PATHS.PROJECT_DIR = prev_path
