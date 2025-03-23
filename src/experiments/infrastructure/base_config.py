import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, Mapping, Optional, TypeVar, Union, final

from submitit.slurm.slurm import SlurmJob

from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS, PathsConfig, RunnerPaths
from src.core.names import EXPERIMENT_NAMES, SlurmStatus
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    TBatchSize,
    TModelID,
    TModelSize,
    TPromptOriginalIndex,
    TTokenizer,
    TVariationName,
    TWindowSize,
)
from src.data_ingestion.datasets.download_dataset import DATASETS, get_prompt_ids
from src.experiments.infrastructure.model_interface import ModelInterface, get_model_interface
from src.experiments.infrastructure.setup_models import get_tokenizer
from src.utils.infra.experiment_helper import create_run_id
from src.utils.infra.output_path import OutputKey, combine_output_keys
from src.utils.infra.slurm import SLURM_GPU_TYPE, submit_job
from src.utils.types_utils import create_mutable_field


class BASE_OUTPUT_KEYS:
    MODEL_ID = OutputKey[TModelID]("model_id", key_display_name="")
    MODEL_ARCH = OutputKey[MODEL_ARCH]("model_arch", key_display_name="arch=")
    MODEL_SIZE = OutputKey[TModelSize]("model_size", key_display_name="size=")
    VARIATION = OutputKey[TVariationName]("variation", key_display_name="v=")
    EXPERIMENT_NAME = OutputKey[EXPERIMENT_NAMES]("experiment_name", key_display_name="")
    DATASET_NAME = OutputKey[DATASETS]("dataset_name", key_display_name="ds=")
    WINDOW_SIZE = OutputKey[TWindowSize]("window_size", key_display_name="ws=")


_TRunnerOutputs = TypeVar("_TRunnerOutputs", bound=Any)
_TRunnerParams = TypeVar("_TRunnerParams", bound=Any)


@dataclass
class RunParams:
    _batch_size: TBatchSize = TBatchSize(1)  # Adjust based on GPU memory
    with_slurm: bool = False
    # slurm_gpu_type: SLURM_GPU_TYPE = SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN
    slurm_gpu_type: SLURM_GPU_TYPE = SLURM_GPU_TYPE.L40S
    slurm_gpus_per_node: int = 1
    overwrite_existing_outputs: bool = False


TDependencies = Mapping[str, Union["BaseRunner", "TDependencies"]]


@dataclass
class BasePromptFilteration(ABC):
    """Filteration of prompts to run the experiment on."""

    dataset_name: DATASETS

    @abstractmethod
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return get_prompt_ids(self.dataset_name)

    @abstractmethod
    def get_dependencies(self) -> TDependencies:
        pass


@dataclass
class CommonParams:
    model_arch: MODEL_ARCH
    model_size: TModelSize
    dataset_name: DATASETS = DATASETS.COUNTER_FACT

    @property
    def model_arch_and_size(self) -> MODEL_ARCH_AND_SIZE:
        return MODEL_ARCH_AND_SIZE(self.model_arch, self.model_size)

    def get_model_interface(self) -> ModelInterface:
        return get_model_interface(self.model_arch_and_size)

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]

    @property
    def get_tokenizer(self) -> TTokenizer:
        return get_tokenizer(self.model_arch, self.model_size)


@dataclass
class BaseRunner(ABC, Generic[_TRunnerParams, _TRunnerOutputs]):
    """Base configuration class with common parameters across all scripts."""

    variation: TVariationName
    common_params: CommonParams
    prompt_filteration: BasePromptFilteration
    runner_params: _TRunnerParams
    run_params: RunParams = create_mutable_field(lambda: RunParams())

    @property
    def global_path_config(self) -> PathsConfig:
        return PATHS

    @property
    @abstractmethod
    def experiment_name(self) -> EXPERIMENT_NAMES:
        pass

    @property
    def batch_size(self) -> TBatchSize:
        assert self.run_params._batch_size == 1, "Batch size must be 1, unless we debug the issue"
        return TBatchSize(1) if (self.common_params.model_arch == MODEL_ARCH.MAMBA2) else self.run_params._batch_size

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.common_params.model_arch][self.common_params.model_size]

    @property
    @abstractmethod
    def experiment_output_keys(self) -> list[OutputKey]:
        return []

    @property
    def variation_relative_path(self) -> Path:
        path = Path(".")

        path /= combine_output_keys(
            self,
            [
                BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
                BASE_OUTPUT_KEYS.VARIATION,
            ],
            sep="/",
        )

        path /= combine_output_keys(
            self.common_params,
            [
                BASE_OUTPUT_KEYS.MODEL_ARCH,
                BASE_OUTPUT_KEYS.MODEL_SIZE,
                BASE_OUTPUT_KEYS.DATASET_NAME,
            ],
            sep="/",
        )

        path /= combine_output_keys(
            self.runner_params,
            self.experiment_output_keys,
            sep="/",
        )
        return path

    @final
    @property
    def variation_paths(self) -> RunnerPaths:
        return RunnerPaths(self.global_path_config.OUTPUT_DIR / self.variation_relative_path)

    @property
    def job_name(self) -> str:
        sep = "_"
        return sep.join(
            [
                combine_output_keys(
                    self,
                    [
                        BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
                        BASE_OUTPUT_KEYS.VARIATION,
                    ],
                    sep=sep,
                ),
                combine_output_keys(
                    self.common_params,
                    [
                        BASE_OUTPUT_KEYS.MODEL_ARCH,
                        BASE_OUTPUT_KEYS.MODEL_SIZE,
                        BASE_OUTPUT_KEYS.DATASET_NAME,
                    ],
                    sep=sep,
                ),
                combine_output_keys(
                    self.runner_params,
                    self.experiment_output_keys,
                    sep=sep,
                ),
            ]
        )

    def set_running_params(
        self,
        with_slurm: bool,
        slurm_gpu_type: SLURM_GPU_TYPE,
        slurm_gpus_per_node: Optional[int] = None,
    ):
        self.run_params.with_slurm = with_slurm
        self.run_params.slurm_gpu_type = slurm_gpu_type
        if slurm_gpus_per_node is not None:
            self.run_params.slurm_gpus_per_node = slurm_gpus_per_node

    @abstractmethod
    def is_computed(self) -> bool:
        pass

    @abstractmethod
    def get_runner_dependencies(self) -> TDependencies:
        pass

    def uncomputed_dependencies(self) -> TDependencies:
        def rec_uncomputed_dependencies(dependencies: TDependencies) -> TDependencies:
            res = {}
            for k, v in dependencies.items():
                if isinstance(v, BaseRunner):
                    if not v.is_computed():
                        res[k] = v
                else:
                    res[k] = rec_uncomputed_dependencies(v)
            return res

        return rec_uncomputed_dependencies(self.get_runner_dependencies())

    def dependencies_are_computed(self) -> bool:
        return len(self.uncomputed_dependencies()) == 0

    @property
    def prompt_ids(self) -> list[TPromptOriginalIndex]:
        return self.prompt_filteration.get_prompt_ids()

    def create_experiment_run_path(self) -> None:
        self.variation_paths.running_history_path.mkdir(parents=True, exist_ok=True)
        self.variation_paths.plots_path.mkdir(parents=True, exist_ok=True)
        self.variation_paths.outputs_path.mkdir(parents=True, exist_ok=True)

        run_id = create_run_id(None)

        params = asdict(self)
        params["run_id"] = run_id
        try:
            params["git_commit_hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            pass

        json.dump(params, self.variation_paths.running_history_json_path(run_id).open("w"), indent=4)

    @abstractmethod
    def get_outputs(self) -> _TRunnerOutputs:
        pass

    @abstractmethod
    def compute(self) -> None:
        pass

    def get_latest_slurm_job(self) -> Optional[SlurmJob]:
        slurm_logs_path = self.variation_paths.slurm_logs_path
        if not slurm_logs_path.exists():
            return None

        job_paths = list(slurm_logs_path.glob("*"))
        if not job_paths:
            return None
        job_path = max(job_paths, key=lambda x: int(x.stem))
        submission_file_path = list(job_path.glob("*_submission.sh"))
        if len(submission_file_path) != 1:
            return None
        return SlurmJob(submission_file_path[0], job_id=job_path.stem)

    def get_slurm_status(self) -> SlurmStatus:
        latest_job = self.get_latest_slurm_job()
        if latest_job is None:
            return SlurmStatus.NOT_SUBMITTED
        return SlurmStatus[latest_job.state]

    def run(self) -> None:
        if not self.run_params.with_slurm:
            self.compute()
            return
        else:
            job = submit_job(
                self.compute,
                log_folder=str(self.global_path_config.get_slurm_job_log_folder(self.job_name, "%j")),
                job_name=self.job_name,
                # timeout_min=1200,
                gpu_type=self.run_params.slurm_gpu_type,
                slurm_gpus_per_node=self.run_params.slurm_gpus_per_node,
            )
            self.variation_paths.slurm_logs_path.mkdir(parents=True, exist_ok=True)
            # create symlink to slurm logs
            (self.variation_paths.slurm_log_folder(job_id=job.job_id)).symlink_to(
                self.global_path_config.get_slurm_job_log_folder(self.job_name, job.job_id)
            )

            self.global_path_config.get_slurm_job_submission_file_path(self.job_name, job.job_id).symlink_to(
                self.variation_paths.variation_base_path
            )

            print(f"{job}: {self.job_name}")

    def compute_with_dependencies(self) -> None:
        def rec_compute_with_dependencies(dependencies: TDependencies) -> None:
            for dependency in dependencies.values():
                if isinstance(dependency, BaseRunner):
                    dependency.compute_with_dependencies()
                else:
                    rec_compute_with_dependencies(dependency)

        rec_compute_with_dependencies(self.get_runner_dependencies())
        self.compute()

    @classmethod
    def init_from_config(
        cls,
        config: "BaseRunner",
        runner_params: _TRunnerParams,
        variation: Optional[TVariationName] = None,
        common_params: Optional[CommonParams] = None,
        prompt_filteration: Optional[BasePromptFilteration] = None,
        run_params: Optional[RunParams] = None,
    ):
        return cls(
            variation=variation or config.variation,
            common_params=common_params or config.common_params,
            prompt_filteration=prompt_filteration or config.prompt_filteration,
            runner_params=runner_params,
            run_params=run_params or config.run_params,
        )
