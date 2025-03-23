from typing import Optional

import pandas as pd

from src.analysis.experiment_results.data_requirements import DataReq
from src.analysis.experiment_results.results_bank import (
    EvaluateModelRecord,
    HeatmapRecord,
    InfoFlowRecord,
    ResultRecord,
)
from src.core.names import COLS, DATASETS
from src.core.types import MODEL_ARCH_AND_SIZE, TVariationName
from src.data_ingestion.data_defs import DataReqs, FulfilledReqs, ResultBank
from src.experiments.infrastructure.base_config import CommonParams
from src.analysis.prompt_filterations import AllPromptFilteration
from src.experiments.runners.evaluate_model import EvaluateModelConfig


def get_model_evaluations(
    variation: TVariationName, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
) -> dict[MODEL_ARCH_AND_SIZE, pd.DataFrame]:
    return {
        model_arch_and_size: EvaluateModelConfig(
            variation=variation,
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


def result_record_to_data_req(result_record: ResultRecord) -> DataReq:
    if isinstance(result_record, InfoFlowRecord):
        target = result_record.target
        feature_category = result_record.feature_category
        source = result_record.source
        prompt_idx = None
        window_size = result_record.window_size
    elif isinstance(result_record, HeatmapRecord):
        target = None
        feature_category = None
        source = None
        prompt_idx = result_record.prompt_idx
        window_size = result_record.window_size
    elif isinstance(result_record, EvaluateModelRecord):
        target = None
        feature_category = None
        source = None
        prompt_idx = None
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
        prompt_idx=prompt_idx,
    ).validate()
