import json
from dataclasses import asdict
from typing import Optional

import pandas as pd

from src.analysis.experiment_results.data_requirements import DataReq
from src.analysis.experiment_results.results_bank import (
    EvaluateModelRecord,
    HeatmapRecord,
    InfoFlowRecord,
    ResultRecord,
)
from src.analysis.prompt_filterations import AllPromptFilteration, AnyExistingPromptFilteration
from src.core.names import COLS, DATASETS, DataReqCols
from src.core.types import MODEL_ARCH_AND_SIZE, TCodeVersionName
from src.data_ingestion.data_defs import DataReqs, FulfilledReqs, ResultBank
from src.experiments.infrastructure.base_config import BasePromptFilteration, BaseRunner, CommonParams
from src.experiments.runners.evaluate_model import EvaluateModelConfig
from src.utils.types_utils import str_enum_values


def get_model_evaluations(
    code_version: TCodeVersionName, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
) -> dict[MODEL_ARCH_AND_SIZE, pd.DataFrame]:
    return {
        model_arch_and_size: EvaluateModelConfig(
            code_version=code_version,
            common_params=CommonParams(
                model_arch=model_arch_and_size[0],
                model_size=model_arch_and_size[1],
            ),
            prompt_filteration=AllPromptFilteration(dataset_name=DATASETS.COUNTER_FACT),
        )
        .get_outputs()
        .set_index(COLS.ORIGINAL_IDX)
        for model_arch_and_size in model_arch_and_sizes
    }


IDataFulfilled = dict[DataReq, Optional[ResultRecord]]


def choose_latest_data_fulfilled(
    data_reqs_options: FulfilledReqs,
) -> IDataFulfilled:
    return {
        data_req: max(options, key=lambda x: x.path) if options else None
        for data_req, options in data_reqs_options._raw.items()
    }


def get_data_fullfment_options(data_reqs: DataReqs, result_bank: ResultBank) -> FulfilledReqs:
    data_reqs_options: FulfilledReqs = FulfilledReqs({data_req: [] for data_req in data_reqs.to_rows()})
    for result in result_bank.to_rows():
        data_req = result_record_to_data_req(result)
        if data_req in data_reqs_options._raw:
            data_reqs_options._raw[data_req].append(result)
    return data_reqs_options


def result_record_to_data_req(
    result_record: ResultRecord, prompt_filteration: Optional[BasePromptFilteration] = None
) -> DataReq:
    if prompt_filteration is None:
        prompt_filteration = AnyExistingPromptFilteration(DATASETS.COUNTER_FACT)
    if isinstance(result_record, InfoFlowRecord):
        target = result_record.target
        feature_category = result_record.feature_category
        source = result_record.source
        window_size = result_record.window_size
    elif isinstance(result_record, HeatmapRecord):
        target = None
        feature_category = None
        source = None
        window_size = result_record.window_size
    elif isinstance(result_record, EvaluateModelRecord):
        target = None
        feature_category = None
        source = None
        window_size = None
    else:
        raise ValueError(f"Unknown result record type: {type(result_record)}")

    return DataReq(
        experiment_name=result_record.experiment_name,
        model_arch=result_record.model_arch,
        model_size=result_record.model_size,
        window_size=window_size,
        source=source,
        feature_category=feature_category,
        target=target,
        prompt_filteration=prompt_filteration,
    ).validate()


def result_record_to_config(result_record: ResultRecord, prompt_filteration: BasePromptFilteration) -> BaseRunner:
    data_req = result_record_to_data_req(result_record)
    return data_req.get_config(result_record.code_version)


def serialize_result_bank(result_bank: ResultBank) -> str:
    def rec_serialize_dependencies(item):
        if isinstance(item, ResultRecord):
            data_req = result_record_to_data_req(item)
            config = data_req.get_config(item.code_version)
            return [rec_serialize_dependencies(data_req._asdict()), rec_serialize_dependencies(config.get_outputs())]
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
        assert len(item) == 2
        item = item[0]
        return tuple([item[col] for col in str_enum_values(DataReqCols) if col != DataReqCols.prompt_filteration])

    return json.dumps(sorted(rec_serialize_dependencies(result_bank.to_rows()), key=sort_key), indent=4)
