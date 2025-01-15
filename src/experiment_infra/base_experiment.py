from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from src.consts import PATHS
from src.datasets.download_dataset import get_hit_dataset
from src.experiment_infra.config import BaseConfig
from src.models.model_interface import ModelInterface, get_model_interface
from src.types import TPromptData
from src.utils.slurm import submit_job

TConfig = TypeVar("TConfig", bound=BaseConfig)
TSingleData = TypeVar("TSingleData")
TSingleResult = TypeVar("TSingleResult")
TCombinedResult = TypeVar("TCombinedResult")


class BaseExperiment(ABC, Generic[TConfig, TSingleData, TSingleResult, TCombinedResult]):
    """Base class for all experiments"""

    def __init__(self, config: TConfig):
        self.config = config

        # Lazy loaded properties
        self._model_interface: Optional[ModelInterface] = None
        self._dataset: Optional[TPromptData] = None

    def _setup_paths(self):
        """Setup experiment directories"""
        self.config.get_output_path().mkdir(parents=True, exist_ok=True)

    @property
    def model_interface(self) -> ModelInterface:
        """Lazy load model interface"""
        if self._model_interface is None:
            self._model_interface = get_model_interface(self.config.model_arch, self.config.model_size)
        return self._model_interface

    @property
    def dataset(self) -> TPromptData:
        """Lazy load dataset"""
        if self._dataset is None:
            self._dataset = get_hit_dataset(
                model_id=f"{self.config.model_arch}_{self.config.model_size}", dataset_args=self.config.dataset_args
            )
        return self._dataset

    @abstractmethod
    def run_single_evaluation(self, data: TSingleData) -> TSingleResult:
        """Run single evaluation step"""
        pass

    @abstractmethod
    def evaluation_data(self) -> list[TSingleData]:
        """Get evaluation data"""
        pass

    @abstractmethod
    def save_results(self, results: TCombinedResult):
        """Save experiment results"""
        pass

    @abstractmethod
    def save_single_results(self, results: list[TSingleResult]):
        """Save single experiment results"""
        pass

    @abstractmethod
    def get_results(self) -> Optional[TCombinedResult]:
        """Load cached results if they exist"""
        pass

    @abstractmethod
    def combine_results(self, results: list[TSingleResult]) -> TCombinedResult:
        """Combine results"""
        pass

    def run_local(self) -> None:
        """Run experiment locally"""
        print(f"Running {self.config.experiment_name} locally")

        # Check cache first
        if self.get_results():
            print("Using cached results")

        results = []
        for data in self.evaluation_data():
            results.append(self.run_single_evaluation(data))
            self.save_single_results(results)
        self.save_results(self.combine_results(results))

    def run_remote(
        self,
        gpu_type: str,
        slurm_gpus_per_node: int,
    ) -> None:
        """Run experiment on Slurm"""

        job = submit_job(
            self.run_local,
            self.config,
            log_folder=str(PATHS.SLURM_DIR / self.config.experiment_name / "%j"),
            job_name=self.config.experiment_name,
            gpu_type=gpu_type,
            slurm_gpus_per_node=slurm_gpus_per_node,
        )
        print(f"{job}: {self.config.experiment_name}")
