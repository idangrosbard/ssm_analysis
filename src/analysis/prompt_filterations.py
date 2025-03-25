from dataclasses import dataclass
from enum import StrEnum

from src.core.names import COLS
from src.core.types import MODEL_ARCH, TModelSize, TPromptOriginalIndex, TVersionName
from src.experiments.infrastructure.base_config import BasePromptFilteration, CommonParams, TDependencies
from src.experiments.runners.evaluate_model import EvaluateModelConfig


@dataclass
class AllPromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return super().get_prompt_ids()

    def get_dependencies(self) -> TDependencies:
        return {}


@dataclass
class AnyExistingPromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        raise ValueError("Should not be called")

    def get_dependencies(self) -> TDependencies:
        raise ValueError("Should not be called")


@dataclass
class AnyExistingCompletePromptFilteration(BasePromptFilteration):
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        raise ValueError("Should not be called")

    def get_dependencies(self) -> TDependencies:
        raise ValueError("Should not be called")


@dataclass
class SelectivePromptFilteration(BasePromptFilteration):
    prompt_ids: tuple[TPromptOriginalIndex, ...]

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return list(self.prompt_ids)

    def get_dependencies(self) -> TDependencies:
        return {}


@dataclass
class MultiplePromptFilteration(BasePromptFilteration):
    prompt_filterations: list[BasePromptFilteration]

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


class Correctness(StrEnum):
    correct = "correct"
    top_5_correct = "top_5_correct"


@dataclass
class ModelCorrectPromptFilteration(BasePromptFilteration):
    model_arch: MODEL_ARCH
    model_size: TModelSize
    correctness: Correctness
    version: TVersionName

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
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
                common_params=CommonParams(
                    model_arch=self.model_arch,
                    model_size=self.model_size,
                    dataset_name=self.dataset_name,
                ),
                prompt_filteration=AllPromptFilteration(dataset_name=self.dataset_name),
                version=self.version,
            ),
        }
