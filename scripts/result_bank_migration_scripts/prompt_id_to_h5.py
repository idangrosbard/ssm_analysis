import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import streamlit as st
from tqdm import tqdm

from src.analysis.experiment_results.helpers import result_record_to_data_req
from src.core.consts import PATHS
from src.core.names import EXPERIMENT_NAMES
from src.core.types import TCodeVersionName, TInfoFlowOutput, TLayerIndex, TPromptOriginalIndex
from src.data_ingestion.data_defs import ResultBank
from src.experiments.runners.evaluate_model import EvaluateModelConfig
from src.experiments.runners.heatmap import HeatmapConfig
from src.experiments.runners.info_flow import (
    InfoFlowConfig,
    InfoFlowFileContent,
    InfoFlowMetadata,
    InfoFlowPromptLayerValue,
)
from src.utils.infra.slurm import submit_cpu_job
from src.utils.streamlit.helpers.component import StreamlitComponent


def read_prev_info_flow_file(path: Path) -> TInfoFlowOutput:
    def convert_json_output_to_output(
        json_output,
    ) -> TInfoFlowOutput:
        return {int(k): v for k, v in json_output.items()}

    return convert_json_output_to_output(json.load(path.open("r")))


def prev_format_to_new_format(prev_data: TInfoFlowOutput) -> InfoFlowFileContent:
    data = cast(
        dict[TPromptOriginalIndex, dict[TLayerIndex, InfoFlowPromptLayerValue]], defaultdict(lambda: defaultdict(dict))
    )
    layers_amount = 0
    for layer_id, layer_data in prev_data.items():
        layers_amount = max(layers_amount, layer_id)
        for i, prompt_id in enumerate(layer_data["original_idx"]):
            data[TPromptOriginalIndex(prompt_id)][TLayerIndex(layer_id)] = InfoFlowPromptLayerValue(
                hit=layer_data["hit"][i],
                true_probs=layer_data["true_probs"][i],
                diffs=layer_data["diffs"][i],
            )

    return InfoFlowFileContent(
        metadata=InfoFlowMetadata(layers_amount=layers_amount + 1, banned_prompts={}),
        data=data,
    )


def migrate_info_flow(tasks: list[tuple[Path, InfoFlowConfig]]):
    for source_path, new_config in tqdm(tasks):
        print(f"Migrating {source_path} to {new_config.output_file.path}")
        prev_data = read_prev_info_flow_file(source_path)
        new_data = prev_format_to_new_format(prev_data)
        new_config.create_experiment_run_path()
        new_config.output_file.save(new_data)


class MigrateResults(StreamlitComponent):
    def __init__(self, results_bank: ResultBank, is_test_results: bool):
        self.results_bank = results_bank
        self.is_test_results = is_test_results

    def render(self):
        from tests.src.experiments.test_full_pipeline import TEST_BASE_PATH

        run = st.button("Run")
        rows = self.results_bank.to_rows()

        st.write(len(rows))
        rows = [row for row in rows if row.experiment_name == EXPERIMENT_NAMES.INFO_FLOW]

        st.write(len(rows))

        prev_path = PATHS.PROJECT_DIR
        tasks_to_run: list[tuple[Path, InfoFlowConfig]] = []

        try:
            if self.is_test_results:
                PATHS.PROJECT_DIR = TEST_BASE_PATH

            for result_record in rows:
                data_req = result_record_to_data_req(result_record)
                new_config = data_req.get_config(TCodeVersionName("v2"))
                source_path = result_record.path
                if isinstance(new_config, EvaluateModelConfig):
                    assert isinstance(new_config, EvaluateModelConfig)
                    new_path = new_config.output_result_path
                elif isinstance(new_config, InfoFlowConfig):
                    assert isinstance(new_config, InfoFlowConfig)
                    new_path = new_config.output_file.path
                elif isinstance(new_config, HeatmapConfig):
                    assert isinstance(new_config, HeatmapConfig)
                    new_path = new_config.output_hdf5_path.path
                else:
                    raise ValueError(f"Unknown config type: {type(new_config)}")

                assert isinstance(new_config, InfoFlowConfig)

                if new_path.exists():
                    continue

                tasks_to_run.append((source_path, new_config))

            n_parallel_jobs = 14
            st.write(len(tasks_to_run))

            if run:
                chunk_size = len(tasks_to_run) // n_parallel_jobs
                for i_job in range(n_parallel_jobs):
                    job_name = f"chunk_{i_job} of {n_parallel_jobs}"
                    job = submit_cpu_job(
                        migrate_info_flow,
                        tasks_to_run[chunk_size * i_job : chunk_size * (i_job + 1)],
                        job_name=job_name,
                        log_folder=str(PATHS.SLURM_DIR / "migrate_info_flow" / job_name / "%j"),
                    )
                    print(f"{job}: {job_name}")

        finally:
            PATHS.PROJECT_DIR = prev_path
