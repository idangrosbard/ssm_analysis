from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional, Type

from git import Union

from src.consts import EXPERIMENT_NAMES, MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS, reverse_model_id
from src.experiment_infra.base_config import BASE_OUTPUT_KEYS, DATASETS, MODEL_ARCH
from src.experiment_infra.output_path import OutputKey, OutputPath
from src.types import FeatureCategory, TModelID, TokenType


class ParamNames(StrEnum):
    experiment_name = "experiment_name"
    variation = "variation"
    model_arch = "model_arch"
    model_size = "model_size"
    window_size = "window_size"
    is_all_correct = "is_all_correct"
    source = "source"
    feature_category = "feature_category"
    target = "target"
    prompt_idx = "prompt_idx"
    path = "path"


class IntermediateParamNames:
    _model_id_source = "_model_id_source"
    _model_id_name = "_model_id_name"
    _experiment_name_and_variation = "_experiment_name_and_variation"
    _dataset_and_filteration = "_dataset_and_filteration"
    _source_and_feature_category = "_source_and_feature_category"
    _block_target = "_block_target"


class RESULTS_BASE_PATH(StrEnum):
    Prev = "prev"
    New = "new"

    @property
    def path(self) -> Path:
        if self == RESULTS_BASE_PATH.Prev:
            return PATHS.PROJECT_DIR / "output.old"
        else:
            return PATHS.OUTPUT_DIR

    @classmethod
    def from_path(cls, path: Path) -> "RESULTS_BASE_PATH":
        for p in cls:
            if path.is_relative_to(p.path):
                return p
        raise ValueError(f"Path {path} is not a valid results base path")

    def pattern_output_path(self, middle_experiment_keys: list[OutputKey]) -> OutputPath:
        dataset_output_key = OutputKey(key_name="dataset_and_filteration", key_display_name="ds=")
        if self == RESULTS_BASE_PATH.Prev:
            return OutputPath(
                self.path,
                [
                    OutputKey(key_name=IntermediateParamNames._model_id_source, key_display_name=""),
                    OutputKey(key_name=IntermediateParamNames._model_id_name, key_display_name=""),
                    OutputKey(key_name=IntermediateParamNames._experiment_name_and_variation, key_display_name=""),
                    dataset_output_key,
                    *middle_experiment_keys,
                ],
            )
        else:
            return OutputPath(
                self.path,
                [
                    BASE_OUTPUT_KEYS.EXPERIMENT_NAME,
                    BASE_OUTPUT_KEYS.VARIATION,
                    BASE_OUTPUT_KEYS.MODEL_ARCH,
                    BASE_OUTPUT_KEYS.MODEL_SIZE,
                    dataset_output_key,
                    *middle_experiment_keys,
                    OutputKey(key_name="_", key_display_name="outputs"),
                ],
            )

    def process_values(self, values: dict[str, str]) -> Optional[dict[str, str]]:
        experiment_name: str = values.pop(ParamNames.experiment_name)
        values.pop("_", None)

        if self == RESULTS_BASE_PATH.Prev:
            model_id_source = values.pop(IntermediateParamNames._model_id_source)
            model_id_name = values.pop(IntermediateParamNames._model_id_name)
            model_arch, model_size = reverse_model_id(f"{model_id_source}/{model_id_name}")
            values[ParamNames.model_arch] = model_arch.value
            values[ParamNames.model_size] = model_size

            experiment_name_and_variation = values.pop(IntermediateParamNames._experiment_name_and_variation)
            if not experiment_name_and_variation.startswith(experiment_name):
                return None
            values[ParamNames.variation] = experiment_name_and_variation[len(experiment_name) :]

        return values

    @property
    def heatmap_suffix(self) -> str:
        if self == RESULTS_BASE_PATH.Prev:
            return ".npy"
        else:
            return ".csv"

    @property
    def info_flow_suffix(self) -> str:
        return ".csv"


@dataclass
class ResultRecord(ABC):
    experiment_name: EXPERIMENT_NAMES = field(init=False)
    path: Path
    variation: str
    model_arch: MODEL_ARCH
    model_size: str
    dataset_and_filteration: str
    window_size: int
    results_base_path: RESULTS_BASE_PATH

    def __post_init__(self):
        self.model_arch = MODEL_ARCH(self.model_arch)
        self.window_size = int(self.window_size)

    @property
    def dataset(self) -> DATASETS:
        assert self.dataset_and_filteration.startswith(DATASETS.COUNTER_FACT.value)
        return DATASETS.COUNTER_FACT

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]

    @classmethod
    def get_results_base_path(cls, path: Path) -> RESULTS_BASE_PATH:
        return RESULTS_BASE_PATH.from_path(path)

    @classmethod
    @abstractmethod
    def get_results_output_path(cls, path: Path) -> OutputPath:
        pass

    @property
    def is_all_correct(self) -> bool:
        if self.results_base_path == RESULTS_BASE_PATH.Prev:
            return True
        filteration = self.dataset_and_filteration[len(self.dataset) :]
        if filteration:
            assert filteration == "_all_correct"
            return True
        return False

    @classmethod
    def from_path(cls, path: Path) -> Union["HeatmapRecord", None]:
        result_output_path = cls.get_results_output_path(path)
        try:
            values = result_output_path.extract_values_from_path(path)
            return cls(
                path=path,
                **values,  # type: ignore
            )
        except ValueError:
            return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultRecord):
            return False
        return self.path == other.path

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ResultRecord):
            raise ValueError(f"Cannot compare {type(self)} with {type(other)}")
        if self.results_base_path != other.results_base_path:
            return self.results_base_path == RESULTS_BASE_PATH.Prev
        elif self.variation != other.variation:
            return self.variation < other.variation
        else:
            raise ValueError(f"Cannot compare {self} with {other}")


@dataclass
class HeatmapRecord(ResultRecord):
    experiment_name = EXPERIMENT_NAMES.HEATMAP
    prompt_idx: int

    def __post_init__(self):
        super().__post_init__()
        assert self.prompt_idx is not None
        self.prompt_idx = int(self.prompt_idx)

    @classmethod
    def get_results_output_path(cls, path: Path) -> OutputPath:
        results_base_path = cls.get_results_base_path(path)
        return results_base_path.pattern_output_path(
            [
                BASE_OUTPUT_KEYS.WINDOW_SIZE,
            ]
        ).add(
            [
                OutputKey(
                    key_name=ParamNames.prompt_idx, key_display_name="idx=", suffix=results_base_path.heatmap_suffix
                ),
            ]
        )


@dataclass
class InfoFlowRecord(ResultRecord):
    experiment_name = EXPERIMENT_NAMES.INFO_FLOW
    _target: str = ""
    _source_and_feature_category: Optional[str] = None
    _block_target: Optional[str] = None

    @property
    def target(self) -> TokenType:
        if self.results_base_path == RESULTS_BASE_PATH.Prev:
            assert self._block_target is not None
            assert self._target == ""
            return TokenType(self._block_target.split("_target_")[1])
        else:
            return TokenType(self._target)

    @property
    def source(self) -> TokenType:
        if self.results_base_path == RESULTS_BASE_PATH.Prev:
            assert self._block_target is not None
            return TokenType(self._block_target.split("_target_")[0])
        else:
            assert self._source_and_feature_category is not None
            sep = "_feature_category="
            if sep in self._source_and_feature_category:
                return TokenType(self._source_and_feature_category.split(sep)[0])
            else:
                return TokenType(self._source_and_feature_category)

    @property
    def feature_category(self) -> Optional[FeatureCategory]:
        if self.results_base_path == RESULTS_BASE_PATH.Prev:
            return None
        else:
            assert self._source_and_feature_category is not None
            sep = "_feature_category="
            if sep in self._source_and_feature_category:
                return FeatureCategory(self._source_and_feature_category.split(sep)[1])
            else:
                return None

    @classmethod
    def get_results_output_path(cls, path: Path) -> OutputPath:
        results_base_path = cls.get_results_base_path(path)
        output_path = results_base_path.pattern_output_path(
            [
                BASE_OUTPUT_KEYS.WINDOW_SIZE,
            ]
        )
        if results_base_path == RESULTS_BASE_PATH.Prev:
            output_path = output_path.add(
                [
                    OutputKey(key_name=IntermediateParamNames._block_target, key_display_name="block_"),
                    OutputKey("_", key_display_name="outputs.json"),
                ]
            )
        else:
            output_path = output_path.add(
                [
                    OutputKey(key_name=f"_{ParamNames.target}", key_display_name="target="),
                    OutputKey(
                        key_name=IntermediateParamNames._source_and_feature_category,
                        key_display_name="source=",
                        suffix=results_base_path.info_flow_suffix,
                    ),
                ]
            )

        return output_path


CACHE_RESULTS_BANK: Optional[list[ResultRecord]] = None


def clear_results_bank_cache():
    global CACHE_RESULTS_BANK
    CACHE_RESULTS_BANK = None


def get_experiment_results_bank(
    results_base_paths: Optional[list[RESULTS_BASE_PATH]] = None,
    experiments: Optional[list[Type[ResultRecord]]] = None,
    update: bool = False,
) -> list[ResultRecord]:
    global CACHE_RESULTS_BANK
    with_cache = results_base_paths is None and experiments is None
    if with_cache and (CACHE_RESULTS_BANK is not None) and not update:
        return CACHE_RESULTS_BANK

    if results_base_paths is None:
        results_base_paths = [RESULTS_BASE_PATH.Prev, RESULTS_BASE_PATH.New]

    if experiments is None:
        experiments = [HeatmapRecord, InfoFlowRecord]

    print("Results bank cache is being updated")
    results: list[ResultRecord] = []
    for results_base_path in results_base_paths:
        for experiment in experiments:
            output_path = experiment.get_results_output_path(results_base_path.path)
            in_pattern, out_of_pattern = output_path.process_path()
            for path, values in in_pattern:
                values[ParamNames.experiment_name] = experiment.experiment_name
                processed_values = results_base_path.process_values(values=values)
                if not processed_values:
                    continue

                results.append(
                    experiment(
                        path=path,
                        results_base_path=results_base_path,
                        **processed_values,  # type: ignore
                    )
                )

    if with_cache:
        CACHE_RESULTS_BANK = results
    return results
