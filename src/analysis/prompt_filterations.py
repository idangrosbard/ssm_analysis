from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import Iterable

from src.core.consts import DEFAULT_MODEL_CORRECT_DATASET_NAME, DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION
from src.core.names import COLS, DatasetName
from src.core.types import (
    ALL_SPLITS_LITERAL,
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    TCodeVersionName,
    TModelSize,
    TPromptOriginalIndex,
    TSplitChoise,
)
from src.data_ingestion.datasets.download_dataset import get_prompt_ids
from src.experiments.infrastructure.base_runner import BasePromptFilteration, InputParams, MetadataParams, TDependencies
from src.experiments.runners.evaluate_model import EvaluateModelParams, EvaluateModelRunner


@dataclass(frozen=True)
class AllPromptFilteration(BasePromptFilteration):
    dataset_name: DatasetName = DatasetName.counter_fact
    split: TSplitChoise = ALL_SPLITS_LITERAL

    @lru_cache(maxsize=1)
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:  # type: ignore
        return get_prompt_ids(self.dataset_name, self.split)

    def get_dependencies(self) -> TDependencies:
        return {}

    def display_name(self) -> str:
        return f"{self.dataset_name} {self.split}"


@dataclass(frozen=True)
class AnyExistingPromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        raise ValueError("Should not be called")

    def get_dependencies(self) -> TDependencies:
        raise ValueError("Should not be called")

    def display_name(self) -> str:
        return "Any Existing"


@dataclass(frozen=True)
class AnyExistingCompletePromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        raise ValueError("Should not be called")

    def get_dependencies(self) -> TDependencies:
        raise ValueError("Should not be called")

    def display_name(self) -> str:
        return "Any Existing (Complete Only)"


@dataclass(frozen=True)
class SelectivePromptFilteration(BasePromptFilteration):
    prompt_ids: tuple[TPromptOriginalIndex, ...]

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return list(self.prompt_ids)

    def get_dependencies(self) -> TDependencies:
        return {}

    def display_name(self) -> str:
        amount = len(self.prompt_ids)
        if amount > 5:
            return f"Selective ({amount})"
        else:
            return f"Selective ({', '.join(str(prompt_id) for prompt_id in self.prompt_ids)})"


@dataclass(frozen=True)
class IntersectionPromptFilteration(BasePromptFilteration):
    prompt_filterations: tuple[BasePromptFilteration, ...]
    base_prompt_filteration: BasePromptFilteration

    @lru_cache(maxsize=1)
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:  # type: ignore
        prompt_ids = set(self.base_prompt_filteration.get_prompt_ids())

        for prompt_filteration in self.prompt_filterations:
            prompt_ids = prompt_ids.intersection(set(prompt_filteration.get_prompt_ids()))
        return list(prompt_ids)

    def get_dependencies(self) -> TDependencies:
        dependencies = {}
        for prompt_filteration in self.prompt_filterations:
            deps = prompt_filteration.get_dependencies()
            if deps:
                dependencies[prompt_filteration] = deps
        return dependencies

    def display_name(self) -> str:
        return "Intersection"


@dataclass(frozen=True)
class UnionPromptFilteration(BasePromptFilteration):
    prompt_filterations: tuple[BasePromptFilteration, ...] = field(default_factory=tuple)

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        prompt_ids = set()
        for prompt_filteration in self.prompt_filterations:
            prompt_ids.update(prompt_filteration.get_prompt_ids())
        return list(prompt_ids)

    def get_dependencies(self) -> TDependencies:
        dependencies = {}
        for prompt_filteration in self.prompt_filterations:
            deps = prompt_filteration.get_dependencies()
            if deps:
                dependencies[prompt_filteration] = deps
        return dependencies

    def add_prompt_filteration(self, prompt_filteration: BasePromptFilteration):
        if isinstance(prompt_filteration, UnionPromptFilteration):
            return UnionPromptFilteration(
                tuple(set(self.prompt_filterations).union(prompt_filteration.prompt_filterations))
            )
        elif isinstance(prompt_filteration, AnyExistingCompletePromptFilteration):
            return self
        return UnionPromptFilteration(tuple(set(self.prompt_filterations).union({prompt_filteration})))

    def display_name(self) -> str:
        return "Union"


class Correctness(StrEnum):
    correct = "correct"
    top_5_correct = "top_5_correct"
    top_2_to_5_correct = "top_2_to_5_correct"


@dataclass(frozen=True)
class ModelCorrectPromptFilteration(BasePromptFilteration):
    dataset_name: DatasetName
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
                df = df[df[COLS.EVALUATE_MODEL.TARGET_RANK] <= 5]
            case Correctness.top_2_to_5_correct:
                df = df[(2 <= df[COLS.EVALUATE_MODEL.TARGET_RANK]) & (df[COLS.EVALUATE_MODEL.TARGET_RANK] <= 5)]
            case _:
                raise NotImplementedError(f"Correctness {self.correctness} not implemented")

        return df[COLS.ORIGINAL_IDX].tolist()

    def get_dependencies(self):
        return {
            "evaluate_model": EvaluateModelRunner(
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

    def is_computed(self) -> bool:
        return self.get_dependencies()["evaluate_model"].is_computed()

    def display_name(self) -> str:
        return f"Model {self.correctness.value} on {self.model_arch} {self.model_size}"


def get_shared_models_correctness_prompt_filteration(
    model_arch_and_sizes: Iterable[MODEL_ARCH_AND_SIZE],
    correctness: Correctness,
    code_version: TCodeVersionName = DEFAULT_MODEL_CORRECT_MODEL_CODE_VERSION,
    dataset_name: DatasetName = DEFAULT_MODEL_CORRECT_DATASET_NAME,
):
    return IntersectionPromptFilteration(
        tuple(
            ModelCorrectPromptFilteration(
                dataset_name=dataset_name,
                model_arch=model_arch_and_size.arch,
                model_size=model_arch_and_size.size,
                correctness=correctness,
                code_version=code_version,
            )
            for model_arch_and_size in model_arch_and_sizes
        ),
        base_prompt_filteration=AllPromptFilteration(dataset_name=dataset_name),
    )
