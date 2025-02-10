from src.experiments.data_construction import DataConstructionConfig
from src.experiments.evaluate_model import EvaluateModelConfig
from src.experiments.heatmap import HeatmapConfig
from src.experiments.info_flow import InfoFlowConfig


def test_data_construction_config():
    data_construction_config = DataConstructionConfig()
    evaluate_model_config = EvaluateModelConfig()
    heatmap_config = HeatmapConfig()
    info_flow_config = InfoFlowConfig()

    assert data_construction_config
    assert evaluate_model_config
    assert heatmap_config
    assert info_flow_config
