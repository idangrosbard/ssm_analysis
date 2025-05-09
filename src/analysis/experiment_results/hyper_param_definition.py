from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Generic, Literal, Sequence, TypeVar, Union, assert_never, cast

from pydantic import BaseModel

from src.analysis.prompt_filterations import (
    AnyExistingCompletePromptFilteration,
    Correctness,
    ModelCorrectPromptFilteration,
)
from src.core.consts import DEFAULT_MODEL_CORRECT_DATASET_NAME, DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION, GRAPHS_ORDER
from src.core.names import (
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
    ExperimentHyperParams,
    InfoFlowVariantParam,
    WindowedVariantParam,
)
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    FeatureCategory,
    TModelSize,
    TokenType,
    TPromptOriginalIndex,
    TWindowSize,
)
from src.data_ingestion.data_defs.data_defs import ResultBank
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration, LogicalPromptFilteration
from src.experiments.runners.heatmap import HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowRunner
from src.utils.types_utils import str_enum_values

_T = TypeVar("_T")


class HyperParamDefinition(ABC, Generic[_T]):
    @abstractmethod
    def get_result_bank_options(self, result_bank: ResultBank) -> Sequence[_T]:
        pass

    @abstractmethod
    def get_static_options(self) -> Sequence[_T]:
        pass

    def get_options(self, result_bank: ResultBank) -> Sequence[_T]:
        return self.get_static_options()

    @abstractmethod
    def get_display_name(self, option: _T) -> str:
        pass

    def default_fix_value(self) -> _T:
        raise NotImplementedError(f"Default fix value not implemented for {self.__class__.__name__}")

    @abstractmethod
    def derived_variants_params(
        self,
    ) -> Sequence[
        Union[
            VARIANT_PARAM_NAME,
            Literal[ExperimentHyperParams.prompt_idx],
            Literal[ExperimentHyperParams.filteration_factory],
        ]
    ]:
        pass


class ModelArchAndSizeHPD(HyperParamDefinition[MODEL_ARCH_AND_SIZE]):
    def get_result_bank_options(self, result_bank: ResultBank) -> list[MODEL_ARCH_AND_SIZE]:
        return list(
            [
                MODEL_ARCH_AND_SIZE(result.variant_params.model_arch, result.variant_params.model_size)
                for result in result_bank
            ]
        )

    def get_static_options(self) -> Sequence[MODEL_ARCH_AND_SIZE]:
        return list(GRAPHS_ORDER.keys())

    def get_display_name(self, option: MODEL_ARCH_AND_SIZE) -> str:
        return option.model_name

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [BaseVariantParamName.model_arch, BaseVariantParamName.model_size]


class ModelArchHPD(HyperParamDefinition[MODEL_ARCH]):
    def get_result_bank_options(self, result_bank: ResultBank):
        return list(set([result.variant_params.model_arch for result in result_bank]))

    def get_static_options(self):
        return str_enum_values(MODEL_ARCH)

    def get_display_name(self, option):
        return option

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [BaseVariantParamName.model_arch]


class ModelSizeHPD(HyperParamDefinition[TModelSize]):
    def get_result_bank_options(self, result_bank: ResultBank):
        return list(set([result.variant_params.model_size for result in result_bank]))

    def get_static_options(self):
        return list({size: size for _, size in GRAPHS_ORDER.keys()}.keys())

    def get_display_name(self, option):
        return option

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [BaseVariantParamName.model_size]


class SourceHPD(HyperParamDefinition[TokenType]):
    def get_result_bank_options(self, result_bank):
        sources = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner):
                sources.add(result.variant_params.source)
        return list(sources)

    def get_static_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [InfoFlowVariantParam.source]


class TargetHPD(HyperParamDefinition[TokenType]):
    def get_result_bank_options(self, result_bank):
        targets = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner):
                targets.add(result.variant_params.target)
        return list(targets)

    def get_static_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def default_fix_value(self) -> TokenType:
        return TokenType.last

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [InfoFlowVariantParam.target]


class FeatureCategoryHPD(HyperParamDefinition[FeatureCategory]):
    def get_result_bank_options(self, result_bank):
        features = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner):
                features.add(result.variant_params.feature_category)
        return list(features)

    def get_static_options(self):
        return str_enum_values(FeatureCategory)

    def get_display_name(self, option: FeatureCategory) -> str:
        return str(option)

    def default_fix_value(self) -> FeatureCategory:
        return FeatureCategory.ALL

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [InfoFlowVariantParam.feature_category]


class WindowSizeHPD(HyperParamDefinition[TWindowSize]):
    def get_result_bank_options(self, result_bank):
        window_sizes = set()
        for result in result_bank:
            if isinstance(result, InfoFlowRunner) or isinstance(result, HeatmapRunner):
                window_sizes.add(result.variant_params.window_size)
        return list(window_sizes)

    def get_static_options(self):
        return list([TWindowSize(i) for i in range(1, 20)])

    def get_display_name(self, option: TWindowSize) -> str:
        return f"{option}"

    def default_fix_value(self) -> TWindowSize:
        return TWindowSize(9)

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [WindowedVariantParam.window_size]


class PromptIdxHPD(HyperParamDefinition[TPromptOriginalIndex]):
    def get_result_bank_options(self, result_bank: ResultBank) -> Sequence[TPromptOriginalIndex]:
        prompts: set[TPromptOriginalIndex] = set()
        for result in result_bank:
            if isinstance(result, HeatmapRunner):
                prompts.update(set(result.output_hdf5_path.get_existing_prompt_idx()))
        return sorted(prompts)

    def get_static_options(self):
        raise NotImplementedError("PromptIdxVariationOption does not have static options")

    def get_options(self, result_bank: ResultBank) -> Sequence[TPromptOriginalIndex]:
        return self.get_result_bank_options(result_bank)

    def get_display_name(self, option: TPromptOriginalIndex) -> str:
        return f"{option}"

    def derived_variants_params(self):
        return cast(Sequence[Literal[ExperimentHyperParams.prompt_idx]], [ExperimentHyperParams.prompt_idx])


class EnumSelectFilterationContext(StrEnum):
    current_model_all = "current_model_all"
    current_model_conditional_any_existing = "current_model_conditional_any_existing"
    current_model_any_existing = "current_model_any_existing"
    context_models_intersect = "context_models_intersect"


class PromptFilterationFactory(BaseModel):
    filteration_context: EnumSelectFilterationContext
    correctness: Correctness

    def get_filteration(self, context_model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]) -> BasePromptFilteration:
        match self.filteration_context:
            case (
                EnumSelectFilterationContext.current_model_all
                | EnumSelectFilterationContext.current_model_conditional_any_existing
            ):
                filteration = ModelCorrectPromptFilteration(
                    dataset_name=DEFAULT_MODEL_CORRECT_DATASET_NAME,
                    model_arch_and_size=None,
                    correctness=self.correctness,
                    code_version=DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
                )
                if self.filteration_context == EnumSelectFilterationContext.current_model_conditional_any_existing:
                    filteration = filteration & AnyExistingCompletePromptFilteration()
                return filteration
            case EnumSelectFilterationContext.current_model_any_existing:
                return AnyExistingCompletePromptFilteration()
            case EnumSelectFilterationContext.context_models_intersect:
                return LogicalPromptFilteration.create_and(
                    [
                        ModelCorrectPromptFilteration(
                            dataset_name=DEFAULT_MODEL_CORRECT_DATASET_NAME,
                            model_arch_and_size=model_arch_and_size,
                            correctness=self.correctness,
                            code_version=DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
                        )
                        for model_arch_and_size in context_model_arch_and_sizes
                    ]
                )
            case _:
                assert_never(self.filteration_context)


class FilterationHPD(HyperParamDefinition[PromptFilterationFactory]):
    def get_result_bank_options(self, result_bank: ResultBank):
        raise NotImplementedError("FilterationHPD does not have result bank options")

    def get_static_options(self):
        raise NotImplementedError("FilterationHPD does not have result bank options")

    def get_options(self, result_bank: ResultBank):
        return self.get_static_options()

    def get_display_name(self, option: PromptFilterationFactory) -> str:
        return f"{option.filteration_context} {option.correctness}"

    def derived_variants_params(self):
        return cast(
            Sequence[Literal[ExperimentHyperParams.filteration_factory]], [ExperimentHyperParams.filteration_factory]
        )


def get_hyper_param_definition(option: ExperimentHyperParams) -> HyperParamDefinition:
    match option:
        case ExperimentHyperParams.model_arch_and_size:
            return ModelArchAndSizeHPD()
        case ExperimentHyperParams.model_arch:
            return ModelArchHPD()
        case ExperimentHyperParams.model_size:
            return ModelSizeHPD()
        case ExperimentHyperParams.source:
            return SourceHPD()
        case ExperimentHyperParams.target:
            return TargetHPD()
        case ExperimentHyperParams.feature_category:
            return FeatureCategoryHPD()
        case ExperimentHyperParams.window_size:
            return WindowSizeHPD()
        case ExperimentHyperParams.prompt_idx:
            return PromptIdxHPD()
        case ExperimentHyperParams.filteration_factory:
            return FilterationHPD()
        case _:
            raise ValueError(f"Unsupported variation option: {option}")


PossibleHPDTypes = Union[
    MODEL_ARCH_AND_SIZE,
    MODEL_ARCH,
    TModelSize,
    TokenType,
    FeatureCategory,
    TWindowSize,
    TPromptOriginalIndex,
    PromptFilterationFactory,
]
