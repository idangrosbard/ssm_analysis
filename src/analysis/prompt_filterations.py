from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

from src.core.names import COLS, DATASETS
from src.core.types import (
    ALL_SPLITS_LITERAL,
    MODEL_ARCH,
    TCodeVersionName,
    TModelSize,
    TPromptOriginalIndex,
    TSplitChoise,
)
from src.data_ingestion.datasets.download_dataset import get_prompt_ids
from src.experiments.infrastructure.base_config import BasePromptFilteration, InputParams, MetadataParams, TDependencies
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams


@dataclass(frozen=True)
class AllPromptFilteration(BasePromptFilteration):
    dataset_name: DATASETS = DATASETS.COUNTER_FACT
    split: TSplitChoise = ALL_SPLITS_LITERAL

    @lru_cache(maxsize=1)
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:  # type: ignore
        return get_prompt_ids(self.dataset_name, self.split)

    def get_dependencies(self) -> TDependencies:
        return {}


@dataclass(frozen=True)
class AnyExistingPromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        raise ValueError("Should not be called")

    def get_dependencies(self) -> TDependencies:
        raise ValueError("Should not be called")


@dataclass(frozen=True)
class AnyExistingCompletePromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        raise ValueError("Should not be called")

    def get_dependencies(self) -> TDependencies:
        raise ValueError("Should not be called")


@dataclass(frozen=True)
class SelectivePromptFilteration(BasePromptFilteration):
    prompt_ids: tuple[TPromptOriginalIndex, ...]

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return list(self.prompt_ids)

    def get_dependencies(self) -> TDependencies:
        return {}


@dataclass(frozen=True)
class IntersectionPromptFilteration(BasePromptFilteration):
    prompt_filterations: set[BasePromptFilteration]

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        prompt_ids = super().get_prompt_ids()

        for prompt_filteration in self.prompt_filterations:
            prompt_ids = [prompt_id for prompt_id in prompt_ids if prompt_id in prompt_filteration.get_prompt_ids()]
        return prompt_ids

    def get_dependencies(self) -> TDependencies:
        dependencies = {}
        for prompt_filteration in self.prompt_filterations:
            dependencies.update(prompt_filteration.get_dependencies())
        return dependencies


@dataclass(frozen=True)
class UnionPromptFilteration(BasePromptFilteration):
    prompt_filterations: set[BasePromptFilteration] = field(default_factory=set)

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        prompt_ids = set()
        for prompt_filteration in self.prompt_filterations:
            prompt_ids.update(prompt_filteration.get_prompt_ids())
        return list(prompt_ids)

    def get_dependencies(self) -> TDependencies:
        dependencies = {}
        for prompt_filteration in self.prompt_filterations:
            dependencies.update(prompt_filteration.get_dependencies())
        return dependencies

    def add_prompt_filteration(self, prompt_filteration: BasePromptFilteration):
        if isinstance(prompt_filteration, UnionPromptFilteration):
            return UnionPromptFilteration(self.prompt_filterations.union(prompt_filteration.prompt_filterations))
        elif isinstance(prompt_filteration, AnyExistingCompletePromptFilteration):
            return self
        return UnionPromptFilteration(self.prompt_filterations.union({prompt_filteration}))


class Correctness(StrEnum):
    correct = "correct"
    top_5_correct = "top_5_correct"


@dataclass(frozen=True)
class ModelCorrectPromptFilteration(BasePromptFilteration):
    dataset_name: DATASETS
    model_arch: MODEL_ARCH
    model_size: TModelSize
    correctness: Correctness
    code_version: TCodeVersionName

    @lru_cache(maxsize=1)
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:  # type: ignore
        df = self.get_dependencies()["evaluate_model"].get_outputs()
        match self.correctness:
            case Correctness.correct:
                df = df[df[COLS.EVALUATE_MODEL.MODEL_CORRECT]]
            case Correctness.top_5_correct:
                df = df[df[COLS.EVALUATE_MODEL.MODEL_CORRECT]]
            case _:
                raise NotImplementedError(f"Correctness {self.correctness} not implemented")

        return df[COLS.ORIGINAL_IDX].tolist()

    def get_dependencies(self):
        return {
            "evaluate_model": EvaluateModelConfig(
                variant_params=EvaluateModelParams(
                    model_arch=self.model_arch,
                    model_size=self.model_size,
                ),
                input_params=InputParams(
                    filteration=AllPromptFilteration(dataset_name=self.dataset_name),
                ),
                metadata_params=MetadataParams(
                    code_version=self.code_version,
                ),
            ),
        }
