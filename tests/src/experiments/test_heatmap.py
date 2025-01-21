from typing import Dict, Iterable, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments.heatmap import HeatmapConfig, HeatmapExperiment
from src.models.model_interface import ModelInterface
from src.types import TPromptData


def create_mock_dataset() -> TPromptData:
    """Create a mock dataset with test prompts."""
    df = pd.DataFrame(
        {
            "prompt": ["test prompt"] * 10,
            "subject": ["test subject"] * 10,
            "target_true": ["test target"] * 10,
            "true_prob": [0.5] * 10,
        }
    )
    return TPromptData(df)


class MockModelInterface(ModelInterface):
    def __init__(self):
        self.device = "cpu"
        self.tokenizer = MagicMock()
        self.model = MagicMock()
        self.model.backbone.layers = [MagicMock() for _ in range(10)]

    def setup(self, layers: Optional[Iterable[int]] = None):
        pass

    def generate_logits(
        self,
        input_ids: torch.Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        # Return mock probabilities - just return the layer number as the probability
        batch_size = input_ids.shape[0]
        return torch.tensor([[0.1]]).expand(batch_size, -1)


@pytest.fixture
def mock_experiment():
    config = HeatmapConfig(
        experiment_name="test_heatmap",
        window_size=2,
        prompt_indices=[0, 1],
    )

    experiment = HeatmapExperiment(config)

    # Mock the model interface and dataset
    experiment._model_interface = MockModelInterface()
    experiment._dataset = create_mock_dataset()

    # Mock tokenizer behavior
    experiment._model_interface.tokenizer.return_value = MagicMock(input_ids=torch.tensor([[0, 1, 2, 3]]))

    return experiment


def test_heatmap_experiment_sub_tasks(mock_experiment):
    """Test that sub_tasks yields correct prompt indices."""
    sub_tasks = list(mock_experiment.sub_tasks())
    assert sub_tasks == [0, 1]


def test_heatmap_experiment_inner_loop(mock_experiment):
    """Test that inner_loop yields correct windows."""
    windows = list(mock_experiment.inner_loop(0))
    assert len(windows) == 9  # For 10 layers and window_size=2
    assert windows[0] == [0, 1]
    assert windows[-1] == [8, 9]


def test_heatmap_experiment_single_evaluation(mock_experiment):
    """Test single evaluation step."""
    result = mock_experiment.run_single_inner_evaluation((0, [0, 1]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)  # Length of input sequence


def test_heatmap_experiment_combine_inner_results(mock_experiment):
    """Test combining inner results."""
    mock_results = [
        ([0, 1], np.array([0.1, 0.1, 0.1, 0.1])),
        ([1, 2], np.array([0.2, 0.2, 0.2, 0.2])),
    ]
    result = mock_experiment.combine_inner_results(mock_results)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)  # (sequence_length, num_windows)


def test_heatmap_experiment_full_run(mock_experiment):
    """Test full experiment run."""
    # Run experiment
    mock_experiment.run_local()

    # Load results
    results = mock_experiment.load_results()

    # Check results
    assert isinstance(results, dict)
    assert 0 in results
    assert 1 in results
    assert isinstance(results[0], np.ndarray)
    assert results[0].shape[0] == 4  # sequence length
