from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional, Sequence, Type, assert_never

from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, PATHS, reverse_model_id
from src.data_defs import ResultBank
from src.experiment_infra.base_config import BASE_OUTPUT_KEYS, DATASETS, MODEL_ARCH
from src.experiment_infra.output_path import OutputKey, OutputPath
from src.names import EXPERIMENT_NAMES, ResultBankParamNames
from src.types import (
    FeatureCategory,
    TModelID,
    TModelSize,
    TokenType,
    TPromptOriginalIndex,
    TVariationName,
    TWindowSize,
)


class IntermediateParamNames:
    _model_id_source = "_model_id_source"
    _model_id_name = "_model_id_name"
    _experiment_name_and_variation = "_experiment_name_and_variation"
    _dataset_and_filteration = "_dataset_and_filteration"
    _source_and_feature_category = "_source_and_feature_category"
    _block_target = "_block_target"


class RESULTS_BASE_PATH(StrEnum):
    v1 = "v1"
    CURRENT = "new"

    @property
    def path(self) -> Path:
        match self:
            case RESULTS_BASE_PATH.CURRENT:
                return PATHS.OUTPUT_DIR
            case _:
                return PATHS.OUTPUT_DIR.parent / f"{PATHS.OUTPUT_DIR.name}.{self}"

    @classmethod
    def from_path(cls, path: Path) -> "RESULTS_BASE_PATH":
        for p in cls:
            if path.is_relative_to(p.path):
                return p
        raise ValueError(f"Path {path} is not a valid results base path")

    def pattern_output_path(self, middle_experiment_keys: list[OutputKey]) -> OutputPath:
        dataset_output_key = OutputKey(key_name="dataset_and_filteration", key_display_name="ds=")
        match self:
            case RESULTS_BASE_PATH.v1:
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
            case RESULTS_BASE_PATH.CURRENT:
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
            case _:
                assert_never(self)

    def process_values(self, values: dict[str, str]) -> Optional[dict[str, str]]:
        experiment_name: str = values.pop(ResultBankParamNames.experiment_name)
        values.pop("_", None)

        if self == RESULTS_BASE_PATH.v1:
            model_id_source = values.pop(IntermediateParamNames._model_id_source)
            model_id_name = values.pop(IntermediateParamNames._model_id_name)
            model_arch, model_size = reverse_model_id(TModelID(f"{model_id_source}/{model_id_name}"))
            values[ResultBankParamNames.model_arch] = model_arch.value
            values[ResultBankParamNames.model_size] = model_size

            experiment_name_and_variation = values.pop(IntermediateParamNames._experiment_name_and_variation)
            if not experiment_name_and_variation.startswith(experiment_name):
                return None
            values[ResultBankParamNames.variation] = experiment_name_and_variation[len(experiment_name) :]

        return values

    @property
    def heatmap_suffix(self) -> str:
        match self:
            case RESULTS_BASE_PATH.v1:
                return ".npy"
            case RESULTS_BASE_PATH.CURRENT:
                return ".csv"
            case _:
                assert_never(self)

    @property
    def info_flow_suffix(self) -> str:
        return ".csv"


@dataclass
class ResultRecord(ABC):
    experiment_name: EXPERIMENT_NAMES = field(init=False)
    path: Path
    variation: TVariationName
    model_arch: MODEL_ARCH
    model_size: TModelSize
    dataset_and_filteration: str
    window_size: TWindowSize
    results_base_path: RESULTS_BASE_PATH

    def __post_init__(self):
        self.model_arch = MODEL_ARCH(self.model_arch)
        self.window_size = TWindowSize(int(self.window_size))

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
        if self.results_base_path == RESULTS_BASE_PATH.v1:
            return True
        filteration = self.dataset_and_filteration[len(self.dataset) :]
        if filteration:
            assert filteration == "_all_correct"
            return True
        return False

    @classmethod
    def from_path(cls, path: Path) -> Optional["ResultRecord"]:
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
            return self.results_base_path == RESULTS_BASE_PATH.v1
        elif self.variation != other.variation:
            return self.variation < other.variation
        else:
            raise ValueError(f"Cannot compare {self} with {other}")

    @classmethod
    def init_from_processed_values(
        cls, path: Path, results_base_path: RESULTS_BASE_PATH, values: dict[str, str]
    ) -> Optional["ResultRecord"]:
        return cls(
            path=path,
            results_base_path=results_base_path,
            **values,  # type: ignore
        )


@dataclass
class HeatmapRecord(ResultRecord):
    experiment_name = EXPERIMENT_NAMES.HEATMAP
    prompt_idx: TPromptOriginalIndex

    def __post_init__(self):
        super().__post_init__()
        assert self.prompt_idx is not None
        self.prompt_idx = TPromptOriginalIndex(int(self.prompt_idx))

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
                    key_name=ResultBankParamNames.prompt_idx,
                    key_display_name="idx=",
                    suffix=results_base_path.heatmap_suffix,
                ),
            ]
        )


@dataclass
class InfoFlowRecord(ResultRecord):
    experiment_name = EXPERIMENT_NAMES.INFO_FLOW
    _block_target: Optional[str] = None  # v1
    _target: Optional[str] = None  # v2
    _source_and_feature_category: Optional[str] = None  # v2

    @property
    def target(self) -> TokenType:
        if self.results_base_path == RESULTS_BASE_PATH.v1:
            assert self._block_target is not None
            return TokenType(self._block_target.split("_target_")[1])
        else:
            assert self._target is not None
            return TokenType(self._target)

    @property
    def source(self) -> TokenType:
        if self.results_base_path == RESULTS_BASE_PATH.v1:
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
    def _feature_category_str(self) -> str:
        if self.results_base_path == RESULTS_BASE_PATH.v1:
            return FeatureCategory.ALL
        else:
            assert self._source_and_feature_category is not None
            sep = "_feature_category="
            if sep in self._source_and_feature_category:
                return self._source_and_feature_category.split(sep)[1]
            else:
                return FeatureCategory.ALL

    @property
    def feature_category(self) -> FeatureCategory:
        return FeatureCategory(self._feature_category_str)

    @classmethod
    def get_results_output_path(cls, path: Path) -> OutputPath:
        results_base_path = cls.get_results_base_path(path)
        output_path = results_base_path.pattern_output_path(
            [
                BASE_OUTPUT_KEYS.WINDOW_SIZE,
            ]
        )
        if results_base_path == RESULTS_BASE_PATH.v1:
            output_path = output_path.add(
                [
                    OutputKey(key_name=IntermediateParamNames._block_target, key_display_name="block_"),
                    OutputKey("_", key_display_name="outputs.json"),
                ]
            )
        else:
            output_path = output_path.add(
                [
                    OutputKey(key_name=f"_{ResultBankParamNames.target}", key_display_name="target="),
                    OutputKey(
                        key_name=IntermediateParamNames._source_and_feature_category,
                        key_display_name="source=",
                        suffix=results_base_path.info_flow_suffix,
                    ),
                ]
            )

        return output_path

    @classmethod
    def init_from_processed_values(
        cls, path: Path, results_base_path: RESULTS_BASE_PATH, values: dict[str, str]
    ) -> Optional["ResultRecord"]:
        result_record = super().init_from_processed_values(path, results_base_path, values)
        assert isinstance(result_record, InfoFlowRecord)
        if results_base_path == RESULTS_BASE_PATH.CURRENT:
            if result_record._feature_category_str == "NONE":
                return None
        return result_record


def get_experiment_results_bank(
    results_base_paths: Sequence[RESULTS_BASE_PATH] = (
        # RESULTS_BASE_PATH.Prev,
        RESULTS_BASE_PATH.CURRENT,
    ),
    experiment_records: Sequence[Type[ResultRecord]] = (HeatmapRecord, InfoFlowRecord),
) -> ResultBank:
    results: list[ResultRecord] = []
    for results_base_path in results_base_paths:
        for experiment_record in experiment_records:
            output_path = experiment_record.get_results_output_path(results_base_path.path).enforce_value(
                ResultBankParamNames.experiment_name, experiment_record.experiment_name
            )
            in_pattern, _ = output_path.process_path()
            for path, values in in_pattern:
                values[ResultBankParamNames.experiment_name] = experiment_record.experiment_name
                processed_values = results_base_path.process_values(values=values)
                if not processed_values:
                    continue
                result_record = experiment_record.init_from_processed_values(
                    path=path,
                    results_base_path=results_base_path,
                    values=processed_values,
                )
                if result_record is not None:
                    results.append(result_record)
    return ResultBank(results)
