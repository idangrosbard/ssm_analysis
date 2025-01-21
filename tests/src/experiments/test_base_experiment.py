from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from src.experiment_infra.base_config import BaseConfig
from src.experiment_infra.base_experiment import BaseExperiment


@dataclass
class MockConfig(BaseConfig):
    """Mock configuration for testing."""

    experiment_name: str = "mock_experiment"
    max_number: int = 10
    window_size: int = 2

    @property
    def output_path(self) -> Path:
        return Path("/tmp/mock_experiment")


class MockExperiment(
    BaseExperiment[
        MockConfig,  # TConfig
        list[int],  # TInnerLoopData - window of numbers
        int,  # TInnerLoopResult - sum of numbers in window
        int,  # TSubTasksData - number to process
        np.ndarray,  # TSubTasksResult - array of window sums
        dict[int, np.ndarray],  # TCombinedResult - number -> array of sums
    ]
):
    """A mock experiment that processes windows of numbers."""

    def sub_tasks(self) -> Generator[int, None, None]:
        """Generate numbers to process."""
        for i in range(self.config.max_number):
            yield i

    def inner_loop(self, data: int) -> Generator[list[int], None, None]:
        """Generate windows of numbers."""
        for i in range(0, self.config.window_size):
            yield list(range(i, i + self.config.window_size))

    def run_single_inner_evaluation(self, data: tuple[int, list[int]]):
        """Sum numbers in the window."""
        number, window = data
        return sum(window)

    def combine_inner_results(self, results):
        """Combine window sums into an array."""
        return np.array([result for _, result in results])

    def combine_sub_task_results(self, results):
        """Combine all results into a dictionary."""
        return {number: sums for number, sums in results}

    def save_results(self, results):
        """Mock save results."""
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        for number, sums in results.items():
            np.save(self.config.output_path / f"{number}.npy", sums)

    def save_sub_task_results(self, results):
        """Mock save sub task results."""
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        for number, sums in results:
            np.save(self.config.output_path / f"{number}.npy", sums)

    def load_sub_task_result(self, data: int):
        """Mock load sub task result."""
        path = self.config.output_path / f"{data}.npy"
        if path.exists():
            return np.load(path)
        return None


def test_mock_experiment():
    """Test the mock experiment."""
    config = MockConfig(max_number=3, window_size=2)
    experiment = MockExperiment(config)

    # Test sub_tasks
    sub_tasks = list(experiment.sub_tasks())
    assert sub_tasks == [0, 1, 2]

    # Test inner_loop
    windows = list(experiment.inner_loop(0))
    assert windows == [[0, 1], [1, 2]]

    # Test single evaluation
    result = experiment.run_single_inner_evaluation((0, [0, 1]))
    assert result == 1

    # Test combine inner results
    inner_results = [(w, sum(w)) for w in [[0, 1], [1, 2]]]
    combined = experiment.combine_inner_results(inner_results)
    assert np.array_equal(combined, np.array([1, 3]))

    # Test full run
    experiment.run_local()

    # Test results were saved
    loaded = experiment.load_results()
    assert loaded is not None
    assert 0 in loaded
    assert 1 in loaded
    assert 2 in loaded
    assert np.array_equal(loaded[0], np.array([1, 3]))  # For number 0, windows [0,1] and [1,2]
