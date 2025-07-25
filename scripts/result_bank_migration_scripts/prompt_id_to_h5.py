import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import streamlit as st
from tqdm import tqdm

from src.core.consts import PATHS
from src.core.names import ExperimentName
from src.core.types import TCodeVersionName, TInfoFlowOutput, TLayerIndex, TPromptOriginalIndex
from src.data_ingestion.data_defs.data_defs import ResultBank
from src.experiments.infrastructure.base_runner import MetadataParams
from src.experiments.runners.evaluate_model import EvaluateModelRunner
from src.experiments.runners.heatmap import HeatmapRunner
from src.experiments.runners.info_flow import (
    InfoFlowFileContent,
    InfoFlowMetadata,
    InfoFlowPromptLayerValue,
    InfoFlowRunner,
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


def migrate_info_flow(tasks: list[tuple[Path, InfoFlowRunner]]):
    for source_path, new_config in tqdm(tasks):
        print(f"Migrating {source_path} to {new_config.output_file.path}")
        prev_data = read_prev_info_flow_file(source_path)
        new_data = prev_format_to_new_format(prev_data)
        new_config.create_experiment_dir()
        new_config.output_file.save(new_data)


class MigrateResults(StreamlitComponent):
    def __init__(self, results_bank: ResultBank, is_test_results: bool):
        self.results_bank = results_bank
        self.is_test_results = is_test_results

    def render(self):
        from tests.src.experiments.baseline_builder import TEST_BASE_PATH

        run = st.button("Run")
        st.write(len(self.results_bank))
        rows = [row for row in self.results_bank if row.experiment_name == ExperimentName.info_flow]

        st.write(len(rows))

        prev_path = PATHS.PROJECT_DIR
        tasks_to_run: list[tuple[Path, InfoFlowRunner]] = []

        try:
            if self.is_test_results:
                PATHS.PROJECT_DIR = TEST_BASE_PATH

            for result_record in rows:
                new_config = result_record.init_from_runner(
                    result_record,
                    result_record.variant_params,
                    metadata_params=MetadataParams(code_version=TCodeVersionName("v2")),
                )
                source_path = result_record.path  # type: ignore
                if isinstance(new_config, EvaluateModelRunner):
                    assert isinstance(new_config, EvaluateModelRunner)
                    new_path = new_config.output_result_path
                elif isinstance(new_config, InfoFlowRunner):
                    assert isinstance(new_config, InfoFlowRunner)
                    new_path = new_config.output_file.path
                elif isinstance(new_config, HeatmapRunner):
                    assert isinstance(new_config, HeatmapRunner)
                    new_path = new_config.output_hdf5_path.path
                else:
                    raise ValueError(f"Unknown config type: {type(new_config)}")

                assert isinstance(new_config, InfoFlowRunner)

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
