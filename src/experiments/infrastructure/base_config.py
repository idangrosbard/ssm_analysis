from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Generic, Mapping, Optional, Type, TypeVar, Union, assert_never, cast, final

from submitit.slurm.slurm import SlurmJob

from src.core.consts import (
    BASE_OUTPUT_KEYS,
    MODEL_SIZES_PER_ARCH_TO_MODEL_ID,
    PATHS,
    PathsConfig,
    RunnerPaths,
)
from src.core.names import BASE_CONFIG_HP_COLS, EXPERIMENT_NAMES, RunningHistoryCols, SlurmStatus
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    TBatchSize,
    TCodeVersionName,
    TModelID,
    TModelSize,
    TPromptOriginalIndex,
    TTokenizer,
)
from src.data_ingestion.datasets.download_dataset import DATASETS
from src.experiments.infrastructure.model_interface import ModelInterface, get_model_interface
from src.experiments.infrastructure.setup_models import get_tokenizer
from src.utils.infra.experiment_helper import create_run_id
from src.utils.infra.git import get_git_commit_hash
from src.utils.infra.output_path import OutputKey, combine_output_keys
from src.utils.infra.slurm import SLURM_GPU_TYPE, submit_job
from src.utils.types_utils import json_dumps_dataclass, str_enum_values

TDependencies = Mapping[str, Union["BaseRunner", "TDependencies"]]


@dataclass(frozen=True)
class BasePromptFilteration(ABC):
    """Filteration of prompts to run the experiment on."""

    @abstractmethod
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        pass

    @abstractmethod
    def get_dependencies(self) -> TDependencies:
        pass


@dataclass(frozen=True)
class BaseVariantParams(ABC):
    model_arch: MODEL_ARCH
    model_size: TModelSize
    experiment_name: EXPERIMENT_NAMES = field(init=False)

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

    def modify(
        self,
        **kwargs,
    ):
        return replace(self, **kwargs)


@dataclass
class InputParams:
    filteration: BasePromptFilteration
    dataset_name: DATASETS = DATASETS.COUNTER_FACT


@dataclass
class MetadataParams:
    code_version: TCodeVersionName
    requested_batch_size: TBatchSize = TBatchSize(1)  # Adjust based on GPU memory
    with_slurm: bool = False
    # slurm_gpu_type: SLURM_GPU_TYPE = SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN
    slurm_gpu_type: SLURM_GPU_TYPE = SLURM_GPU_TYPE.L40S
    slurm_gpus_per_node: int = 1
    overwrite_existing_outputs: bool = False
    override_base_project_dir: Optional[str] = None


_TVariantParams = TypeVar("_TVariantParams", bound=BaseVariantParams)


@dataclass
class BaseRunner(ABC, Generic[_TVariantParams]):
    """Base configuration class with common parameters across all scripts."""

    variant_params: _TVariantParams
    input_params: InputParams
    metadata_params: MetadataParams

    @property
    def experiment_name(self) -> EXPERIMENT_NAMES:
        return self.variant_params.experiment_name

    @staticmethod
    @abstractmethod
    def _get_variant_params() -> Type[_TVariantParams]:
        pass

    @classmethod
    def init_from_runner(
        cls,
        runner: "BaseRunner",
        variant_params: _TVariantParams,
        input_params: Optional[InputParams] = None,
        metadata_params: Optional[MetadataParams] = None,
    ):
        return cls(
            variant_params=variant_params,
            input_params=input_params or runner.input_params,
            metadata_params=metadata_params or runner.metadata_params,
        )

    def modify(
        self,
        variant_params: Optional[_TVariantParams] = None,
        input_params: Optional[InputParams] = None,
        metadata_params: Optional[MetadataParams] = None,
    ):
        return self.__class__(
            variant_params=variant_params or self.variant_params,
            input_params=input_params or self.input_params,
            metadata_params=metadata_params or self.metadata_params,
        )

    @property
    def global_path_config(self) -> PathsConfig:
        return PathsConfig(PROJECT_DIR=Path(self.metadata_params.override_base_project_dir or PATHS.PROJECT_DIR))

    @property
    def effective_batch_size(self) -> TBatchSize:
        assert self.metadata_params.requested_batch_size == 1, "Batch size must be 1, unless we debug the issue"
        return (
            TBatchSize(1)
            if (self.variant_params.model_arch == MODEL_ARCH.MAMBA2)
            else self.metadata_params.requested_batch_size
        )

    @classmethod
    @abstractmethod
    def get_variant_output_keys(cls) -> list[OutputKey]:
        return [
            BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
            BASE_OUTPUT_KEYS.CODE_VERSION,
            BASE_OUTPUT_KEYS.MODEL_ARCH,
            BASE_OUTPUT_KEYS.MODEL_SIZE,
            BASE_OUTPUT_KEYS.DATASET_NAME,
        ]

    @property
    def shared_param_namespace(self):
        class CombineParams:
            @classmethod
            def __getattr__(cls, item: str) -> Any:
                if item in str_enum_values(BASE_CONFIG_HP_COLS):
                    item = cast(BASE_CONFIG_HP_COLS, item)
                    match item:
                        case BASE_CONFIG_HP_COLS.experiment_name:
                            return self.experiment_name
                        case BASE_CONFIG_HP_COLS.code_version:
                            return self.metadata_params.code_version
                        case BASE_CONFIG_HP_COLS.dataset_name:
                            return self.input_params.dataset_name
                        case BASE_CONFIG_HP_COLS.prompt_filteration:
                            return self.input_params.filteration
                        case BASE_CONFIG_HP_COLS.model_arch:
                            return self.variant_params.model_arch
                        case BASE_CONFIG_HP_COLS.model_size:
                            return self.variant_params.model_size
                        case _:
                            assert_never(item)
                else:
                    return getattr(self.variant_params, item)

        return CombineParams()

    def combine_output_keys(self, sep: str) -> str:
        return combine_output_keys(
            self.shared_param_namespace,
            self.get_variant_output_keys(),
            sep=sep,
        )

    @property
    def variation_relative_path(self) -> Path:
        return Path(".") / self.combine_output_keys(sep="/")

    @final
    @property
    def variation_paths(self) -> RunnerPaths:
        return RunnerPaths(self.global_path_config.OUTPUT_DIR / self.variation_relative_path)

    @property
    def job_name(self) -> str:
        return self.combine_output_keys(sep="_")

    def set_running_params(
        self,
        with_slurm: bool,
        slurm_gpu_type: SLURM_GPU_TYPE,
        slurm_gpus_per_node: Optional[int] = None,
    ):
        self.metadata_params.with_slurm = with_slurm
        self.metadata_params.slurm_gpu_type = slurm_gpu_type
        if slurm_gpus_per_node is not None:
            self.metadata_params.slurm_gpus_per_node = slurm_gpus_per_node

    def should_skip_task(self) -> bool:
        return False

    @abstractmethod
    def get_runner_dependencies(self) -> TDependencies:
        pass

    @abstractmethod
    def _compute_impl(self) -> None:
        pass

    @abstractmethod
    def is_computed(self) -> bool:
        pass

    @abstractmethod
    def get_outputs(self) -> Any:
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

    def create_experiment_dir(self) -> None:
        self.variation_paths.running_history_path.mkdir(parents=True, exist_ok=True)
        self.variation_paths.plots_path.mkdir(parents=True, exist_ok=True)
        self.variation_paths.outputs_path.mkdir(parents=True, exist_ok=True)

        run_id = create_run_id(None)

        params = asdict(self)
        params[RunningHistoryCols.run_id] = run_id
        params[RunningHistoryCols.git_commit_hash] = get_git_commit_hash()

        self.variation_paths.running_history_json_path(run_id).write_text(json_dumps_dataclass(params, indent=4))

    def compute_dependencies(self, rec_depth: int = -1) -> None:
        """
        Compute the dependencies of the current runner.
        if rec_depth is N, only the N-th level dependencies will be computed.
        If rec_depth is 0 nothing will be computed.
        if rec_depth is -1, all the way down dependencies will be computed.
        """
        if rec_depth == 0:
            return

        def rec_compute_with_dependencies(dependencies: TDependencies) -> None:
            for dependency in dependencies.values():
                if isinstance(dependency, BaseRunner):
                    dependency.compute_dependencies(rec_depth - 1)
                    dependency.run(with_dependencies=True)
                else:
                    rec_compute_with_dependencies(dependency)

        rec_compute_with_dependencies(self.get_runner_dependencies())

    def run(self, with_dependencies: bool) -> None:
        if self.should_skip_task():
            return
        if self.is_computed():
            return
        if not self.dependencies_are_computed():
            if with_dependencies:
                self.compute_dependencies(1)
            else:
                raise ValueError("Dependencies are not computed")

        if not self.metadata_params.with_slurm:
            self._compute_impl()
            return
        else:
            if self.should_skip_task():
                return
            job = submit_job(
                self._compute_impl,
                log_folder=str(self.global_path_config.get_slurm_job_log_folder(self.job_name, "%j")),
                job_name=self.job_name,
                # timeout_min=1200,
                gpu_type=self.metadata_params.slurm_gpu_type,
                slurm_gpus_per_node=self.metadata_params.slurm_gpus_per_node,
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

    def get_slurm_status(self) -> SlurmStatus | str:
        latest_job = self.get_latest_slurm_job()
        if latest_job is None:
            return SlurmStatus.NOT_SUBMITTED
        try:
            return SlurmStatus[latest_job.state]
        except KeyError:
            return latest_job.state
