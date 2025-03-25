from typing import NamedTuple, Optional, Union

from src.core.consts import MODEL_ARCH, TokenType
from src.core.names import EXPERIMENT_NAMES, DataReqCols
from src.core.types import (
    MODEL_ARCH_AND_SIZE,
    FeatureCategory,
    TCodeVersionName,
    TModelSize,
    TWindowSize,
)
from src.experiments.infrastructure.base_config import BasePromptFilteration, InputParams, MetadataParams
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams
from src.experiments.runners.heatmap import HeatmapConfig, HeatmapParams
from src.experiments.runners.info_flow import InfoFlowConfig, InfoFlowParams


class DataReq(NamedTuple):
    experiment_name: EXPERIMENT_NAMES
    model_arch: MODEL_ARCH
    model_size: TModelSize
    prompt_filteration: BasePromptFilteration
    window_size: Optional[TWindowSize]
    source: Optional[TokenType]
    feature_category: Optional[FeatureCategory]
    target: Optional[TokenType]

    def validate(self):
        experiment_name = EXPERIMENT_NAMES.get_experiment_name_by_str(self.experiment_name)
        for col in DataReqCols.get_cols_by_experiment_name(experiment_name):
            assert getattr(self, col) is not None, f"{experiment_name} requires '{col}'"
        return self

    @classmethod
    def create_and_validate(cls, **kwargs):
        return cls(**kwargs).validate()

    @property
    def model_arch_and_size(self) -> MODEL_ARCH_AND_SIZE:
        return MODEL_ARCH_AND_SIZE(self.model_arch, self.model_size)

    def get_config(self, code_version: TCodeVersionName) -> Union[InfoFlowConfig, HeatmapConfig, EvaluateModelConfig]:
        match self.experiment_name:
            case EXPERIMENT_NAMES.INFO_FLOW:
                assert self.source is not None
                assert self.feature_category is not None
                assert self.target is not None
                assert self.window_size is not None
                config = InfoFlowConfig(
                    variant_params=InfoFlowParams(
                        model_arch=self.model_arch,
                        model_size=self.model_size,
                        window_size=self.window_size,
                        source=self.source,
                        feature_category=self.feature_category,
                        target=self.target,
                    ),
                    input_params=InputParams(
                        filteration=self.prompt_filteration,
                    ),
                    metadata_params=MetadataParams(
                        code_version=code_version,
                    ),
                )
            case EXPERIMENT_NAMES.HEATMAP:
                assert self.window_size is not None
                config = HeatmapConfig(
                    variant_params=HeatmapParams(
                        model_arch=self.model_arch,
                        model_size=self.model_size,
                        window_size=self.window_size,
                    ),
                    input_params=InputParams(
                        filteration=self.prompt_filteration,
                    ),
                    metadata_params=MetadataParams(
                        code_version=code_version,
                    ),
                )
            case EXPERIMENT_NAMES.EVALUATE_MODEL:
                config = EvaluateModelConfig(
                    variant_params=EvaluateModelParams(
                        model_arch=self.model_arch,
                        model_size=self.model_size,
                    ),
                    input_params=InputParams(
                        filteration=self.prompt_filteration,
                    ),
                    metadata_params=MetadataParams(
                        code_version=code_version,
                    ),
                )
            case _:
                raise ValueError(f"Unknown experiment name: {self.experiment_name}")

        return config
