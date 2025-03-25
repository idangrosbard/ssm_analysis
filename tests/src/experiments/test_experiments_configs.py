from src.analysis.prompt_filterations import AllPromptFilteration
from src.core.names import DATASETS
from src.core.types import MODEL_ARCH, FeatureCategory, TCodeVersionName, TModelSize, TokenType, TWindowSize
from src.experiments.infrastructure.base_config import (
    CommonParams,
)
from src.experiments.runners.evaluate_model import EvaluateModelConfig
from src.experiments.runners.heatmap import HeatmapConfig, HeatmapParams
from src.experiments.runners.info_flow import InfoFlowConfig, InfoFlowParams


def test_experiments_configs():
    code_version = TCodeVersionName("test")
    common_params = CommonParams(
        model_arch=MODEL_ARCH.MAMBA1,
        model_size=TModelSize("130M"),
    )
    prompt_filteration = AllPromptFilteration(dataset_name=DATASETS.COUNTER_FACT)
    window_size = TWindowSize(10)
    evaluate_model_config = EvaluateModelConfig(
        code_version=code_version,
        common_params=common_params,
        prompt_filteration=prompt_filteration,
    )
    heatmap_config = HeatmapConfig(
        code_version=code_version,
        common_params=common_params,
        prompt_filteration=prompt_filteration,
        runner_params=HeatmapParams(
            window_size=window_size,
        ),
    )
    info_flow_config = InfoFlowConfig(
        code_version=code_version,
        common_params=common_params,
        prompt_filteration=prompt_filteration,
        runner_params=InfoFlowParams(
            window_size=window_size,
            source=TokenType.last,
            target=TokenType.last,
            feature_category=FeatureCategory.ALL,
        ),
    )
    assert evaluate_model_config
    assert heatmap_config
    assert info_flow_config
