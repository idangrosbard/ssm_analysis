"""
BaseExperiment: A Framework for Structured Experiment Design in the context of this project.

The context of the experiment is the analysis of prompt-level information flow through model layers.
The base experiment is designed to be a generalized structured framework for running experiments in this context.


Considerations:
- The base experiment is designed to be easily runnable in a distributed environment (e.g. Slurm)
    - Allocating of GPU can take a long time.
    - Model loading can take a long time.
    - We want to save intermediate results to avoid re-running the same
        experiment, and allow continuation of experiments.
- Some models support running prompts in batches
    - We want to be agnostic to this
- We want it to be easily testable
- We want to provide flexibility for different types of experiments
- We want to support plotting and visualization based on the intermediate results
    - We might want to update the plotting and visualization code based on the intermediate results


Core concepts:
- Model Interface: A class that provides a unified interface for interacting with different models.
- Dataset: Provide prompt-level data for the experiment.
- Experiment Config: Allows for easy configuration of the experiment.
- Single evaluation: Where we call the model interface to get the results of a single evaluation
- Inner loop: Where we call the model interface to get the results of a single sub-task
- Sub-task: A single unit of work that we want to evaluate
- Combined result: The result of the experiment, which is the combination of all sub-task results
- Resumability:
    - We want to identify sub-tasks that have already been evaluated


Experiment Flow:
- Find sub-tasks that have not been evaluated
- if sub-task has not been evaluated, prepare the data and load the model
- Iterate over sub-tasks
    - Run inner loop
    - Combine inner loop results to get sub-task result
- Save sub-task results
- Combine sub-task results
- Save combined results
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Generator, Generic, Optional, TypeVar

from src.consts import PATHS
from src.datasets.download_dataset import get_hit_dataset
from src.experiment_infra.base_config import BaseConfig
from src.models.model_interface import ModelInterface, get_model_interface
from src.types import TPromptData
from src.utils.slurm import submit_job

TConfig = TypeVar("TConfig", bound=BaseConfig)
TInnerLoopData = TypeVar("TInnerLoopData")
TInnerLoopResult = TypeVar("TInnerLoopResult")
TSubTasksData = TypeVar("TSubTasksData")
TSubTasksResult = TypeVar("TSubTasksResult")
TCombinedResult = TypeVar("TCombinedResult")


class BaseExperiment(
    ABC,
    Generic[
        TConfig,
        TInnerLoopData,
        TInnerLoopResult,
        TSubTasksData,
        TSubTasksResult,
        TCombinedResult,
    ],
):
    """Base class for all experiments"""

    def __init__(self, config: TConfig):
        self.config = config

        # Lazy loaded properties
        self._model_interface: Optional[ModelInterface] = None
        self._dataset: Optional[TPromptData] = None

    def _setup_paths(self):
        """Setup experiment directories"""
        self.config.output_path.mkdir(parents=True, exist_ok=True)

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
            self._dataset = get_hit_dataset(model_id=self.config.model_id, dataset_args=self.config.dataset_args)
        return self._dataset

    @abstractmethod
    def sub_tasks(self) -> Generator[TSubTasksData, None, None]:
        """Get evaluation data"""
        pass

    @abstractmethod
    def inner_loop(self, data: TSubTasksData) -> Generator[TInnerLoopData, None, None]:
        """Get evaluation data"""
        pass

    @abstractmethod
    def run_single_inner_evaluation(self, data: tuple[TSubTasksData, TInnerLoopData]) -> TInnerLoopResult:
        """Run single evaluation step"""
        pass

    @abstractmethod
    def combine_inner_results(self, results: list[tuple[TInnerLoopData, TInnerLoopResult]]) -> TSubTasksResult:
        """Combine inner results"""
        pass

    @abstractmethod
    def save_sub_task_results(self, results: list[tuple[TSubTasksData, TSubTasksResult]]):
        """Save single experiment results"""
        pass

    @abstractmethod
    def load_sub_task_result(self, data: TSubTasksData) -> Optional[TSubTasksResult]:
        """Load a single result if it exists"""
        pass

    @abstractmethod
    def combine_sub_task_results(self, results: list[tuple[TSubTasksData, TSubTasksResult]]) -> TCombinedResult:
        """Combine results"""
        pass

    @abstractmethod
    def save_results(self, results: TCombinedResult):
        """Save experiment results"""
        pass

    def load_results(self) -> Optional[TCombinedResult]:
        """Load cached results if they exist. By default, tries to load all individual results."""
        results = []
        for data_point in self.sub_tasks():
            result = self.load_sub_task_result(data_point)
            if result is None:
                return None
            results.append((data_point, result))
        return self.combine_sub_task_results(results)

    def run_local(self) -> None:
        """Run experiment locally"""
        print(f"Config:\n{json.dumps(asdict(self.config), indent=2)}")

        self._setup_paths()

        # Check cache first
        if self.load_results():
            print("Results already cached")
            return

        sub_task_results = []
        for sub_task in self.sub_tasks():
            existing_result = self.load_sub_task_result(sub_task)
            if existing_result is not None:
                print(f"Skipping existing sub-task result for {sub_task}")
                continue
            inner_results = []
            for inner_data_point in self.inner_loop(sub_task):
                inner_results.append((inner_data_point, self.run_single_inner_evaluation((sub_task, inner_data_point))))
            sub_task_result = self.combine_inner_results(inner_results)
            sub_task_results.append((sub_task, sub_task_result))
            self.save_sub_task_results(sub_task_results)
        self.save_results(self.combine_sub_task_results(sub_task_results))

    @classmethod
    def run_remote(
        cls,
        config: TConfig,
        gpu_type: str,
        slurm_gpus_per_node: int,
    ) -> None:
        """Run experiment on Slurm"""

        job = submit_job(
            cls.run_local,
            config,
            log_folder=str(PATHS.SLURM_DIR / config.experiment_name / "%j"),
            job_name=config.experiment_name,
            gpu_type=gpu_type,
            slurm_gpus_per_node=slurm_gpus_per_node,
        )
        print(f"{job}: {config.experiment_name}")
