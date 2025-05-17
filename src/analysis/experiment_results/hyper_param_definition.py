from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Generic, Sequence, TypeVar, Union

from src.analysis.experiment_results.prompt_filteration_factory import (
    AllImportantModelsFilterationFactory,
    ContextModelsFilterationFactory,
    CurrentModelFilterationFactory,
    ExistingPromptsFilterationFactory,
    PresetFilterationFactory,
    PromptFilterationFactory,
    PromptFilterationFactoryUnion,
)
from src.analysis.prompt_filterations import (
    AllPromptFilteration,
    Correctness,
)
from src.core.consts import (
    GRAPHS_ORDER,
)
from src.core.names import (
    VARIANT_PARAM_NAME,
    BaseVariantParamName,
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
from src.experiments.infrastructure.base_prompt_filteration import BasePromptFilteration, SelectivePromptFilteration
from src.utils.types_utils import str_enum_values

_T = TypeVar("_T")


PossibleHPDTypes = Union[
    MODEL_ARCH_AND_SIZE,
    MODEL_ARCH,
    TModelSize,
    TokenType,
    FeatureCategory,
    TWindowSize,
    TPromptOriginalIndex,
    PromptFilterationFactoryUnion,
]

PossibleDerivedHPDTypes = Union[MODEL_ARCH, TModelSize, TokenType, FeatureCategory, TWindowSize, BasePromptFilteration]


class VirtualExperimentHyperParams(StrEnum):
    model_arch_and_size = "model_arch_and_size"
    filteration_factory = "filteration_factory"
    prompt_idx = "prompt_idx"


TExperimentHyperParams = Union[
    VARIANT_PARAM_NAME,
    VirtualExperimentHyperParams,
]


class HyperParamDefinition(ABC, Generic[_T]):
    @abstractmethod
    def get_options(self) -> Sequence[_T]:
        pass

    @abstractmethod
    def get_display_name(self, option: _T) -> str:
        pass

    def default_fix_value(self) -> _T:
        raise NotImplementedError(f"Default fix value not implemented for {self.__class__.__name__}")

    @abstractmethod
    def derived_variants_params(
        self,
    ) -> Sequence[VARIANT_PARAM_NAME]:
        pass

    @abstractmethod
    def get_derived_hpds(self, option: _T) -> dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]:
        pass


_T_BaseVariant = TypeVar("_T_BaseVariant", bound=PossibleDerivedHPDTypes)


class BaseVariantHPD(HyperParamDefinition[_T_BaseVariant]):
    def get_display_name(self, option: _T_BaseVariant) -> str:
        return str(option)

    @abstractmethod
    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        pass

    def get_derived_hpds(self, option: _T_BaseVariant) -> dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]:
        return {
            self.derived_variants_params()[0]: option,
        }


# region Variant HPDs


class ModelArchHPD(BaseVariantHPD[MODEL_ARCH]):
    def get_options(self):
        return str_enum_values(MODEL_ARCH)

    def get_display_name(self, option):
        return option

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [BaseVariantParamName.model_arch]


class ModelSizeHPD(BaseVariantHPD[TModelSize]):
    def get_options(self):
        return list({size: size for _, size in GRAPHS_ORDER.keys()}.keys())

    def get_display_name(self, option):
        return option

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [BaseVariantParamName.model_size]


class SourceHPD(BaseVariantHPD[TokenType]):
    def get_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [InfoFlowVariantParam.source]


class TargetHPD(BaseVariantHPD[TokenType]):
    def get_options(self):
        return str_enum_values(TokenType)

    def get_display_name(self, option: TokenType) -> str:
        return option

    def default_fix_value(self) -> TokenType:
        return TokenType.last

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [InfoFlowVariantParam.target]


class FeatureCategoryHPD(BaseVariantHPD[FeatureCategory]):
    def get_options(self):
        return str_enum_values(FeatureCategory)

    def get_display_name(self, option: FeatureCategory) -> str:
        return str(option)

    def default_fix_value(self) -> FeatureCategory:
        return FeatureCategory.ALL

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [InfoFlowVariantParam.feature_category]


class WindowSizeHPD(BaseVariantHPD[TWindowSize]):
    def get_options(self):
        return list([TWindowSize(i) for i in range(1, 20)])

    def get_display_name(self, option: TWindowSize) -> str:
        return f"{option}"

    def default_fix_value(self) -> TWindowSize:
        return TWindowSize(9)

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [WindowedVariantParam.window_size]


# endregion

# region Virtual HPDs


class ModelArchAndSizeHPD(HyperParamDefinition[MODEL_ARCH_AND_SIZE]):
    def get_options(self) -> Sequence[MODEL_ARCH_AND_SIZE]:
        return list(GRAPHS_ORDER.keys())

    def get_display_name(self, option: MODEL_ARCH_AND_SIZE) -> str:
        return option.model_name

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return [BaseVariantParamName.model_arch, BaseVariantParamName.model_size]

    def get_derived_hpds(self, option: MODEL_ARCH_AND_SIZE) -> dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]:
        return {
            BaseVariantParamName.model_arch: option.arch,
            BaseVariantParamName.model_size: option.size,
        }


class PromptFilterationHPD(HyperParamDefinition[_T]):
    def get_derived_hpds(self, option):
        raise NotImplementedError("Only get_derived_hpd_with_context should be called")

    @abstractmethod
    def get_derived_hpd_with_context(
        self, option: _T, context_model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
    ) -> tuple[BasePromptFilteration, dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]: ...


class PromptIdxHPD(PromptFilterationHPD[TPromptOriginalIndex]):
    def get_options(self):
        return AllPromptFilteration().get_prompt_ids()

    def get_display_name(self, option: TPromptOriginalIndex) -> str:
        return f"{option}"

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return []

    def get_derived_hpd_with_context(
        self, option: TPromptOriginalIndex, context_model_arch_and_sizes
    ) -> tuple[BasePromptFilteration, dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        return SelectivePromptFilteration(prompt_ids=(option,)), {}


class FilterationFactoryHPD(PromptFilterationHPD[PromptFilterationFactory]):
    def get_options(self) -> Sequence[PromptFilterationFactory]:
        from src.data_ingestion.data_defs.data_defs import PromptFilterationsPresets

        options = []

        # Add preset options
        presets = PromptFilterationsPresets.load()
        for preset_id in presets:
            options.append(PresetFilterationFactory(preset_id=preset_id))

        # Add model correctness options
        for correctness in Correctness:
            # Current model
            options.append(CurrentModelFilterationFactory(correctness=correctness, combine_with_existing=False))
            options.append(CurrentModelFilterationFactory(correctness=correctness, combine_with_existing=True))

            # Context models
            options.append(ContextModelsFilterationFactory(correctness=correctness, combine_with_existing=False))
            options.append(ContextModelsFilterationFactory(correctness=correctness, combine_with_existing=True))

            # All important models
            options.append(AllImportantModelsFilterationFactory(correctness=correctness, combine_with_existing=False))
            options.append(AllImportantModelsFilterationFactory(correctness=correctness, combine_with_existing=True))

        # Add existing prompts option
        options.append(ExistingPromptsFilterationFactory())

        return options

    def get_display_name(self, option: PromptFilterationFactory) -> str:
        return option.display_name

    def derived_variants_params(self) -> Sequence[VARIANT_PARAM_NAME]:
        return []

    def get_derived_hpd_with_context(
        self, option: PromptFilterationFactory, context_model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
    ) -> tuple[BasePromptFilteration, dict[VARIANT_PARAM_NAME, PossibleDerivedHPDTypes]]:
        return option.get_filteration(context_model_arch_and_sizes), {}


# endregion

# region Experiment Hyper Params


def get_hyper_param_definition(option: TExperimentHyperParams) -> HyperParamDefinition:
    match option:
        case VirtualExperimentHyperParams.model_arch_and_size:
            return ModelArchAndSizeHPD()
        case BaseVariantParamName.model_arch:
            return ModelArchHPD()
        case BaseVariantParamName.model_size:
            return ModelSizeHPD()
        case InfoFlowVariantParam.source:
            return SourceHPD()
        case InfoFlowVariantParam.target:
            return TargetHPD()
        case InfoFlowVariantParam.feature_category:
            return FeatureCategoryHPD()
        case WindowedVariantParam.window_size:
            return WindowSizeHPD()
        case VirtualExperimentHyperParams.prompt_idx:
            return PromptIdxHPD()
        case VirtualExperimentHyperParams.filteration_factory:
            return FilterationFactoryHPD()
        case _:
            raise ValueError(f"Unsupported variation option: {option}")
