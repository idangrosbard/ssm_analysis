import json
from dataclasses import asdict
from typing import cast

import pandas as pd

from src.analysis.prompt_filterations import AllPromptFilteration
from src.core.names import COLS, DATASETS, EXPERIMENT_NAMES, DataReqCols
from src.core.types import MODEL_ARCH_AND_SIZE, TCodeVersionName
from src.data_ingestion.data_defs import ResultBank
from src.experiments.infrastructure.base_config import (
    BasePromptFilteration,
    BaseRunner,
    BaseVariantParams,
    InputParams,
    MetadataParams,
)
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams
from src.experiments.runners.heatmap import HeatmapParams, HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowParams, InfoFlowRunner


def get_model_evaluations(
    code_version: TCodeVersionName, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
) -> dict[MODEL_ARCH_AND_SIZE, pd.DataFrame]:
    return {
        model_arch_and_size: EvaluateModelConfig(
            variant_params=EvaluateModelParams(
                model_arch=model_arch_and_size[0],
                model_size=model_arch_and_size[1],
            ),
            input_params=InputParams(
                filteration=AllPromptFilteration(dataset_name=DATASETS.COUNTER_FACT),
            ),
            metadata_params=MetadataParams(
                code_version=code_version,
            ),
        )
        .get_outputs()
        .set_index(COLS.ORIGINAL_IDX)
        for model_arch_and_size in model_arch_and_sizes
    }


def init_variant_params_from_values(dict_values: dict) -> BaseVariantParams:
    experiment_name = cast(EXPERIMENT_NAMES, dict_values.pop(DataReqCols.experiment_name))
    match experiment_name:
        case EXPERIMENT_NAMES.EVALUATE_MODEL:
            return EvaluateModelParams(**dict_values)
        case EXPERIMENT_NAMES.INFO_FLOW:
            return InfoFlowParams(**dict_values)
        case EXPERIMENT_NAMES.HEATMAP:
            return HeatmapParams(**dict_values)
        case _:
            raise ValueError(f"Unsupported experiment name: {experiment_name}")


def init_runner_from_params(
    variant_params: BaseVariantParams,
    input_params: InputParams,
    metadata_params: MetadataParams,
) -> BaseRunner:
    match variant_params:
        case EvaluateModelParams():
            return EvaluateModelConfig(
                variant_params=variant_params,
                input_params=input_params,
                metadata_params=metadata_params,
            )
        case InfoFlowParams():
            return InfoFlowRunner(
                variant_params=variant_params,
                input_params=input_params,
                metadata_params=metadata_params,
            )
        case HeatmapParams():
            return HeatmapRunner(
                variant_params=variant_params,
                input_params=input_params,
                metadata_params=metadata_params,
            )
        case _:
            raise ValueError(f"Unsupported variant params: {variant_params}")


def serialize_result_bank(result_bank: ResultBank) -> str:
    def rec_serialize_dependencies(item):
        if isinstance(item, BaseRunner):
            return [
                rec_serialize_dependencies(asdict(item.variant_params)),
                rec_serialize_dependencies(asdict(item.input_params)),
                rec_serialize_dependencies(item.get_outputs()),
            ]
        if isinstance(item, dict):
            res = {}
            for k, v in item.items():
                if isinstance(k, tuple):
                    k = str(k)
                res[k] = rec_serialize_dependencies(v)
            return res
        elif isinstance(item, list):
            return [rec_serialize_dependencies(v) for v in item]
        elif isinstance(item, BasePromptFilteration):
            return {item.__class__.__name__: rec_serialize_dependencies(asdict(item))}
        elif isinstance(item, pd.DataFrame):
            return item.to_dict()
        else:
            return item

    def sort_key(item):
        assert len(item) == 3
        item = item[0]
        return tuple([item[col] for col in DataReqCols.get_cols_by_experiment_name(item[DataReqCols.experiment_name])])

    return json.dumps(sorted(rec_serialize_dependencies([item for item in result_bank]), key=sort_key), indent=4)
