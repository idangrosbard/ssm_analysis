import functools
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Optional, Type, TypeVar, cast

import pandas as pd

from src.analysis.prompt_filterations import SelectivePromptFilteration, UnionPromptFilteration
from src.core.consts import PATHS
from src.core.names import (
    EXPERIMENT_NAMES,
    HeatmapCols,
    ModelCombinationCols,
    ResultBankParamNames,
    SummarizedDataFulfilledReqsCols,
)
from src.core.types import MODEL_ARCH_AND_SIZE, TPlotID, TPromptOriginalIndex, TTokenizer
from src.data_ingestion.helpers.logits_utils import Prompt
from src.experiments.infrastructure.base_config import BasePromptFilteration, BaseRunner, BaseVariantParams, InputParams
from src.experiments.runners.info_flow import InfoFlowRunner, TWindowLayerStartIndex
from src.utils.infra.data_object import DataObject
from src.utils.types_utils import (
    get_dict_keys_by_condition,
    select_indexes_from_list,
    subset_dict_by_keys,
)

if TYPE_CHECKING:
    from src.analysis.experiment_results.model_prompt_combination import ModelCombination
    from src.analysis.experiment_results.plot_plan import PlotPlan


class DataReqiermentCollection:
    def __init__(self):
        self.data_reqs: dict[BaseVariantParams, UnionPromptFilteration] = defaultdict(UnionPromptFilteration)

    def add_data_req(self, data_req: BaseVariantParams, prompt_filteration: BasePromptFilteration):
        self.data_reqs[data_req] = self.data_reqs[data_req].add_prompt_filteration(prompt_filteration)


class DataReqs(DataObject):
    def __init__(self, data_reqs: dict[BaseVariantParams, UnionPromptFilteration]):
        self._raw = data_reqs

    def to_rows(self):
        return list(self._raw.items())

    def to_fulfilled_reqs(self, result_bank: "ResultBank") -> "FulfilledReqs":
        data_reqs_options = {data_req: [] for data_req, _ in self.to_rows()}

        for runner in result_bank.to_rows():
            if runner.variant_params in self._raw:
                prompt_filterations = self._raw[runner.variant_params]

                if runner.init_from_runner(
                    runner,
                    variant_params=runner.variant_params,
                    input_params=InputParams(filteration=prompt_filterations),
                ).is_computed():
                    data_reqs_options[runner.variant_params].append(runner)

        return FulfilledReqs(
            {data_req: (self._raw[data_req], options) for data_req, options in data_reqs_options.items()}
        )

    @classmethod
    def from_data_reqs_collection(cls, data_reqs: DataReqiermentCollection) -> "DataReqs":
        return cls(data_reqs.data_reqs)

    def size(self) -> int:
        return len(self._raw)


class FulfilledReqs(DataObject):
    def __init__(
        self,
        fulfilled_reqs: dict[BaseVariantParams, tuple[UnionPromptFilteration, list[BaseRunner]]],
    ):
        self._raw = fulfilled_reqs

    def summarize(self) -> "SummarizedDataFulfilledReqs":
        return SummarizedDataFulfilledReqs(self)

    def choose_latest_fulfilled(self) -> "FulfilledReqs":
        return FulfilledReqs(
            {
                data_req: (filteration, [max(options, key=lambda x: x.metadata_params.code_version)])
                for data_req, (filteration, options) in self._raw.items()
            }
        )

    def get_config(self):
        return {req: runners[0] for req, runners in self.to_rows() if runners}

    def to_rows(self):
        return list(self._raw.items())


class SummarizedDataFulfilledReqs(DataObject):
    def __init__(self, fulfilled_reqs: FulfilledReqs):
        self._fulfilled_reqs = fulfilled_reqs
        self._raw = []

        for req, (filteration, opts) in fulfilled_reqs._raw.items():
            row = {
                **{
                    param: getattr(req, param, None)
                    for param in ResultBankParamNames
                    if param not in [ResultBankParamNames.path, ResultBankParamNames.code_version]
                },
                SummarizedDataFulfilledReqsCols.AvailableOptions: len(opts),
                SummarizedDataFulfilledReqsCols.Options: opts,
                SummarizedDataFulfilledReqsCols.Key: str(req),
                SummarizedDataFulfilledReqsCols.filters_requested: len(str(filteration)),
            }
            self._raw.append(row)

    def to_rows(self) -> list[dict]:
        return self._raw

    def to_data_reqs(self) -> DataReqs:
        data_reqs = DataReqiermentCollection()
        for req, (filteration, _) in self._fulfilled_reqs._raw.items():
            data_reqs.add_data_req(req, filteration)
        return DataReqs(data_reqs.data_reqs)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._raw)


class PlotPlans(DataObject):
    def __init__(self, plot_plans: dict[str, "PlotPlan"]):
        self._raw: dict[TPlotID, PlotPlan] = {}

    @staticmethod
    def get_plot_plan_dir(plot_id: TPlotID) -> Path:
        return PATHS.FINAL_PLOTS_DIR / plot_id

    @staticmethod
    def get_json_path() -> Path:
        return PATHS.FINAL_PLOTS_DIR / "plot_plans.json"

    @classmethod
    def get_cache_dir(cls, plot_id: TPlotID) -> Path:
        return cls.get_plot_plan_dir(plot_id) / "cache"

    def add_plan(self, plan: "PlotPlan") -> None:
        if plan.plot_id in self._raw:
            raise ValueError(f"Plot plan with title {plan.plot_id} already exists")
        self._raw[plan.plot_id] = plan
        self.order_plans()

    def order_plans(self) -> None:
        self._raw = dict(sorted(self._raw.items(), key=lambda x: x[1].order))

    def remove_plan(self, plot_id: TPlotID) -> None:
        if plot_id not in self._raw:
            raise ValueError(f"Plot plan with title {plot_id} does not exist")
        self._raw.pop(plot_id)

    def is_plan_exists(self, plot_id: TPlotID) -> bool:
        return plot_id in self._raw

    def get_plan(self, plot_id: TPlotID) -> "PlotPlan":
        return self._raw[plot_id]

    def save(self) -> None:
        self.order_plans()
        self.get_json_path().parent.mkdir(parents=True, exist_ok=True)

        self.get_json_path().write_text(json.dumps([plot_plan.to_dict() for plot_plan in self.to_rows()], indent=4))

    def is_empty(self) -> bool:
        return not self._raw

    @classmethod
    def load(cls) -> "PlotPlans":
        from src.analysis.experiment_results.plot_plan import PlotPlan

        plot_plans = cls({})
        if cls.get_json_path().exists():
            for plot_plan in json.loads(cls.get_json_path().read_text()):
                plot_plans._raw[TPlotID(plot_plan["plot_id"])] = PlotPlan.from_dict(plot_plan)
        return plot_plans

    def to_rows(self) -> list["PlotPlan"]:
        return list(self._raw.values())


class ModelCombinationsPrompts(DataObject):
    def __init__(self, model_combinations: list["ModelCombination"]):
        self._raw = model_combinations

    def cols_enum(self) -> Type[ModelCombinationCols]:
        return ModelCombinationCols

    def to_rows(self) -> list["ModelCombination"]:
        return self._raw

    def sort_by_prompt_count(self) -> "ModelCombinationsPrompts":
        return ModelCombinationsPrompts(sorted(self._raw, key=lambda x: len(x.prompts), reverse=True))

    def change_chosen_prompt_by_seed(self, seed: int) -> "ModelCombinationsPrompts":
        return ModelCombinationsPrompts([combination.choose_prompt_by_seed(seed) for combination in self._raw])

    def to_display_df(self, models_combinations: list[MODEL_ARCH_AND_SIZE]) -> pd.DataFrame:
        table_data = []
        for row in self.to_rows():
            # Create row with model correctness
            table_row = {}

            # Add prompt count and selected prompt first
            table_row[HeatmapCols.PROMPT_COUNT] = len(row.prompts)
            table_row[HeatmapCols.SELECTED_PROMPT] = row.chosen_prompt

            # Add model columns at the end
            for model_name_and_size in models_combinations:
                model_name = model_name_and_size.model_name
                if model_name_and_size in row.correct_models:
                    table_row[model_name] = "✅"
                elif model_name_and_size in row.incorrect_models:
                    table_row[model_name] = "❌"
                else:
                    table_row[model_name] = "-"
            table_data.append(table_row)
        return pd.DataFrame(table_data)


T_RUNNER_TYPE = TypeVar("T_RUNNER_TYPE", bound=BaseRunner)


class ResultBank(DataObject, Generic[T_RUNNER_TYPE]):
    KEY = "key"

    def __init__(self, result_bank: list[T_RUNNER_TYPE]):
        self._raw = result_bank

    def to_rows(self) -> list[T_RUNNER_TYPE]:
        return self._raw

    def to_experiment_results_df(self) -> pd.DataFrame:
        results_data = []
        for i, result in enumerate(self.to_rows()):
            result_dict: dict = {param: getattr(result.variant_params, param, None) for param in ResultBankParamNames}
            result_dict[ResultBankParamNames.path] = str(result.variation_relative_path)
            result_dict[ResultBankParamNames.code_version] = result.metadata_params.code_version
            result_dict[ResultBank.KEY] = i
            results_data.append(result_dict)
        return pd.DataFrame(results_data)

    def from_experiment_results_df(self, experiment_results_df: Optional[pd.DataFrame]):
        if experiment_results_df is None:
            return self.__class__([])
        return self.__class__(select_indexes_from_list(self.to_rows(), experiment_results_df[ResultBank.KEY].tolist()))

    def to_info_flow_results(self) -> "InfoFlowResults":
        results = [
            result for result in self.to_rows() if result.variant_params.experiment_name == EXPERIMENT_NAMES.INFO_FLOW
        ]
        return InfoFlowResults(cast(list[InfoFlowRunner], results))

    def is_empty(self) -> bool:
        return len(self.to_rows()) == 0

    def get_common_and_different_params(self) -> tuple[dict, list[dict]]:
        if self.is_empty():
            return {}, []

        common_params = {}
        different_params_list = []

        all_keys = set()
        for result in self.to_rows():
            all_keys.update(asdict(result.variant_params).keys())

        for key in all_keys:
            values = [getattr(result.variant_params, key, None) for result in self.to_rows()]
            unique_values = set(values)

            if len(unique_values) == 1:
                common_params[key] = next(iter(unique_values))
            else:
                for i, result in enumerate(self.to_rows()):
                    if i >= len(different_params_list):
                        different_params_list.append({})
                    variant_params = asdict(result.variant_params)
                    if key in variant_params:
                        different_params_list[i][key] = variant_params[key]

        return common_params, different_params_list


class InfoFlowResults(ResultBank[InfoFlowRunner]):
    def to_rows(self) -> list[InfoFlowRunner]:
        return self._raw

    def get_common_indices(self) -> set[TPromptOriginalIndex]:
        existing_ids_list = [info_flow.output_file.get_computed_prompt_idx() for info_flow in self.to_rows()]
        return functools.reduce(lambda x, y: x.intersection(y), existing_ids_list)

    def max_layer(self) -> int:
        return max(info_flow.output_file.get_statistics().layers_amount for info_flow in self.to_rows()) - 1

    def min_layer(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.to_rows())

    def subset_layers(self, layer_idx_subset: TWindowLayerStartIndex) -> "InfoFlowResults":
        return InfoFlowResults(
            [
                info_flow.modify(variant_params=info_flow.variant_params.modify(subset_layers=layer_idx_subset))
                for info_flow in self.to_rows()
            ]
        )

    def subset_prompts(self, prompt_ids: list[TPromptOriginalIndex]) -> "InfoFlowResults":
        return InfoFlowResults(
            [
                info_flow.modify(input_params=InputParams(filteration=SelectivePromptFilteration(tuple(prompt_ids))))
                for info_flow in self.to_rows()
            ]
        )


class PromptNew(DataObject):
    def __init__(self, prompt: dict):
        self._prompt = prompt

    def as_prompt(self) -> Prompt:
        return Prompt(self._prompt)  # type: ignore


class Prompts(DataObject):
    def __init__(self, df: dict[TPromptOriginalIndex, PromptNew], tokenizer: TTokenizer):
        self._raw = df
        self._tokenizer = tokenizer

    def filter_by_prompt_ids(self, prompt_ids: list[TPromptOriginalIndex]):
        return Prompts(subset_dict_by_keys(self._raw, prompt_ids), self._tokenizer)

    def filter_by_condition(self, condition: Callable[[TPromptOriginalIndex, PromptNew], bool]):
        return self.filter_by_prompt_ids(get_dict_keys_by_condition(self._raw, condition))

    @property
    def empty(self) -> bool:
        return len(self._raw) == 0

    @property
    def size(self) -> int:
        return len(self._raw)

    @property
    def original_idx(self) -> list[TPromptOriginalIndex]:
        return list(self._raw.keys())

    def sample(self, sample_size: int, seed: int) -> "Prompts":
        random.seed(seed)
        sampled_indices = random.choices(list(self._raw.keys()), k=sample_size)
        return self.filter_by_prompt_ids(sampled_indices)
