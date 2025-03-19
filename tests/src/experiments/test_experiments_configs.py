from src.experiments.runners.evaluate_model import EvaluateModelConfig
from src.experiments.runners.heatmap import HeatmapConfig
from src.experiments.runners.info_flow import InfoFlowConfig


def test_experiments_configs():
    evaluate_model_config = EvaluateModelConfig()
    heatmap_config = HeatmapConfig()
    info_flow_config = InfoFlowConfig()

    assert evaluate_model_config
    assert heatmap_config
    assert info_flow_config
