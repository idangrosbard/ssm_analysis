from dataclasses import asdict

from src.analysis.prompt_filterations import AllPromptFilteration
from src.core.names import DATASETS
from src.core.types import MODEL_ARCH, FeatureCategory, TModelSize, TokenType, TWindowSize
from src.experiments.infrastructure.base_config import (
    BaseVariantParams,
    InputParams,
)
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams
from src.experiments.runners.heatmap import HeatmapConfig, HeatmapParams
from src.experiments.runners.info_flow import InfoFlowConfig, InfoFlowParams


def test_experiments_configs():
    variant_params = BaseVariantParams(
        model_arch=MODEL_ARCH.MAMBA1,
        model_size=TModelSize("130M"),
    )
    prompt_filteration = AllPromptFilteration(dataset_name=DATASETS.COUNTER_FACT)
    window_size = TWindowSize(10)
    evaluate_model_config = EvaluateModelConfig(
        variant_params=EvaluateModelParams(
            **asdict(variant_params),
        ),
        input_params=InputParams(filteration=prompt_filteration),
    )
    heatmap_config = HeatmapConfig(
        variant_params=HeatmapParams(
            **asdict(variant_params),
            window_size=window_size,
        ),
        input_params=InputParams(filteration=prompt_filteration),
    )
    info_flow_config = InfoFlowConfig(
        variant_params=InfoFlowParams(
            **asdict(variant_params),
            window_size=window_size,
            source=TokenType.last,
            target=TokenType.last,
            feature_category=FeatureCategory.ALL,
        ),
        input_params=InputParams(filteration=prompt_filteration),
    )
    assert evaluate_model_config
    assert heatmap_config
    assert info_flow_config
