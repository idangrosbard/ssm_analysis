import contextlib
import shutil

import h5py
import pandas as pd
import streamlit as st

from src.analysis.experiment_results.helpers import result_record_to_data_req
from src.core.names import EXPERIMENT_NAMES
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

        dry_run = st.checkbox("Dry Run", value=True, disabled=False)
        rows = self.results_bank.to_rows()

        st.write(len(rows))
        rows = [row for row in rows if row.experiment_name == EXPERIMENT_NAMES.HEATMAP]

        st.write(len(rows))

        # st.stop()
        rows_count = len(rows)
        rows_progress_bar = st.progress(0, text="Starting...")
        prev_path = PATHS.PROJECT_DIR
        try:
            if self.is_test_results:
                PATHS.PROJECT_DIR = TEST_BASE_PATH

            for i, result_record in enumerate(rows):
                data_req = result_record_to_data_req(result_record)
                config = data_req.get_config(TVariationName(result_record.variation))
                source_path = result_record.path
                if isinstance(config, EvaluateModelConfig):
                    assert isinstance(config, EvaluateModelConfig)
                    new_path = config.output_result_path
                elif isinstance(config, InfoFlowConfig):
                    assert isinstance(config, InfoFlowConfig)
                    new_path = config.output_block_target_source_path()
                elif isinstance(config, HeatmapConfig):
                    assert isinstance(config, HeatmapConfig)
                    new_path = config.output_hdf5_path.path
                else:
                    raise ValueError(f"Unknown config type: {type(config)}")

                # new_path = PATHS.PROJECT_DIR / "outputs.new" / fast_relative_to(new_path, PATHS.OUTPUT_DIR)
                assert new_path.suffix == ".h5"

                ctx = h5py.File(new_path, "a") if not dry_run else contextlib.nullcontext()

                with ctx as hf:
                    assert data_req.prompt_idx is not None
                    total = len(data_req.prompt_idx)
                    item_progress_bar = st.progress(0, text="Copying...")
                    for j, prompt_idx in enumerate(data_req.prompt_idx):
                        source_file = source_path / f"idx={prompt_idx}.csv"
                        data = pd.read_csv(source_file)

                        if not dry_run:
                            assert hf is not None
                            hf.create_dataset(str(prompt_idx), data=data)
                            backup_path = (
                                PATHS.PROJECT_DIR
                                / "outputs.backup"
                                / fast_relative_to(source_path, PATHS.OUTPUT_DIR)
                                / f"idx={prompt_idx}.csv"
                            )
                            backup_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(source_file, backup_path)
                            source_file.unlink()

                        progress = min((j + 1) / total, 1.0)
                        item_progress_bar.progress(
                            progress,
                            text="\n \n".join(
                                [
                                    f"Processed {j + 1} items{' [DRY RUN]' if dry_run else ''}",
                                    f"{fast_relative_to(source_file, PATHS.PROJECT_DIR)} ->",
                                    f"{fast_relative_to(new_path, PATHS.PROJECT_DIR)}",
                                ]
                            ),
                        )
                rows_progress_bar.progress(
                    min((i + 1) / rows_count, 1.0),
                    text="\n \n".join(
                        [
                            f"Processed {i + 1} rows {' [DRY RUN]' if dry_run else ''}",
                            f"{fast_relative_to(source_path, PATHS.PROJECT_DIR)} ->",
                            f"{fast_relative_to(new_path, PATHS.PROJECT_DIR)}",
                        ]
                    ),
                )
        finally:
            PATHS.PROJECT_DIR = prev_path
