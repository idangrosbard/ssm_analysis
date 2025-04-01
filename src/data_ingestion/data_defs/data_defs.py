import functools
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

import pandas as pd

from src.analysis.prompt_filterations import SelectivePromptFilteration, UnionPromptFilteration
from src.core.consts import PATHS
from src.core.names import (
    ExperimentName,
    HeatmapCols,
    ModelCombinationCols,
    ResultBankParamNames,
    SummarizedDataFulfilledReqsCols,
)
from src.core.types import MODEL_ARCH_AND_SIZE, TPlotID, TPromptOriginalIndex, TTokenizer
from src.data_ingestion.helpers.logits_utils import Prompt
from src.experiments.infrastructure.base_runner import (
    BasePromptFilteration,
    BaseRunner,
    BaseVariantParams,
    InputParams,
)
from src.experiments.runners.info_flow import InfoFlowRunner, TWindowLayerStartIndex
from src.utils.infra.data_object import DataObject, IndexableDataObject, IterableDataObject
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
        self._data_reqs: dict[BaseVariantParams, UnionPromptFilteration] = defaultdict(UnionPromptFilteration)

    def add_data_req(self, data_req: BaseVariantParams, prompt_filteration: BasePromptFilteration):
        self._data_reqs[data_req] = self._data_reqs[data_req].add_prompt_filteration(prompt_filteration)
        for dependency in prompt_filteration.uncomputed_dependencies():
            self.add_data_req(dependency.variant_params, dependency.input_params.filteration)

    def to_dict(self) -> dict[BaseVariantParams, BasePromptFilteration]:
        return dict(self._data_reqs)


class DataReqs(IndexableDataObject[BaseVariantParams, BasePromptFilteration]):
    def __init__(self, data_reqs: dict[BaseVariantParams, BasePromptFilteration]):
        super().__init__(data_reqs)

    def to_fulfilled_reqs(self, result_bank: "ResultBank[BaseRunner]") -> "FulfilledReqs":
        data_reqs_options: dict[BaseVariantParams, list[BaseRunner]] = {
            data_req: [] for data_req, _ in self._items.items()
        }

        for runner in result_bank:
            if runner.variant_params in self._items:
                prompt_filterations = self._items[runner.variant_params]

                if prompt_filterations.dependencies_are_computed():
                    if runner.init_from_runner(
                        runner,
                        variant_params=runner.variant_params,
                        input_params=InputParams(filteration=prompt_filterations),
                    ).is_computed():
                        data_reqs_options[runner.variant_params].append(runner)

        return FulfilledReqs(
            {data_req: (self._items[data_req], options) for data_req, options in data_reqs_options.items()}
        )

    @classmethod
    def from_data_reqs_collection(cls, data_reqs: DataReqiermentCollection) -> "DataReqs":
        return cls(data_reqs.to_dict())


class FulfilledReqs(IndexableDataObject[BaseVariantParams, tuple[BasePromptFilteration, List[BaseRunner]]]):
    def __init__(
        self,
        fulfilled_reqs: Dict[BaseVariantParams, tuple[BasePromptFilteration, List[BaseRunner]]],
    ):
        super().__init__(fulfilled_reqs)

    def summarize(self) -> "SummarizedDataFulfilledReqs":
        return SummarizedDataFulfilledReqs(self)

    def choose_latest_fulfilled(self) -> "FulfilledReqs":
        def get_latest_results() -> Dict[BaseVariantParams, tuple[BasePromptFilteration, List[BaseRunner]]]:
            return {
                data_req: (filteration, [max(options, key=lambda x: x.metadata_params.code_version)])
                for data_req, (filteration, options) in self._items.items()
            }

        return FulfilledReqs(get_latest_results())

    def get_config(self) -> Dict[BaseVariantParams, BaseRunner]:
        return {req: runners[0] for req, (_, runners) in self._items.items() if runners}


class SummarizedDataFulfilledReqs(IterableDataObject[dict[str, Any]]):
    def __init__(self, fulfilled_reqs: FulfilledReqs):
        self._fulfilled_reqs = fulfilled_reqs

        raw = []

        for req, (filteration, opts) in fulfilled_reqs._items.items():
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
            raw.append(row)
        super().__init__(raw)

    def to_data_reqs(self) -> DataReqs:
        data_reqs = DataReqiermentCollection()
        for req, (filteration, _) in self._fulfilled_reqs._items.items():
            data_reqs.add_data_req(req, filteration)
        return DataReqs(data_reqs.to_dict())

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._items)


class PlotPlans(IndexableDataObject[TPlotID, "PlotPlan"]):
    def __init__(self, plot_plans: Dict[TPlotID, "PlotPlan"]):
        super().__init__(plot_plans)

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
        if plan.plot_id in self._items:
            raise ValueError(f"Plot plan with title {plan.plot_id} already exists")
        self._items[plan.plot_id] = plan
        self.order_plans()

    def order_plans(self) -> None:
        self._items = dict(sorted(self._items.items(), key=lambda x: x[1].order))

    def remove_plan(self, plot_id: TPlotID) -> None:
        if plot_id not in self._items:
            raise ValueError(f"Plot plan with title {plot_id} does not exist")
        self._items.pop(plot_id)

    def is_plan_exists(self, plot_id: TPlotID) -> bool:
        return plot_id in self._items

    def get_plan(self, plot_id: TPlotID) -> "PlotPlan":
        return self._items[plot_id]

    def save(self) -> None:
        self.order_plans()
        self.get_json_path().parent.mkdir(parents=True, exist_ok=True)

        self.get_json_path().write_text(json.dumps([plot_plan.to_dict() for plot_plan in self.values()], indent=4))

    def is_empty(self) -> bool:
        return not self._items

    @classmethod
    def load(cls) -> "PlotPlans":
        from src.analysis.experiment_results.plot_plan import PlotPlan

        plot_plans = cls({})
        if cls.get_json_path().exists():
            for plot_plan in json.loads(cls.get_json_path().read_text()):
                plot_plans._items[TPlotID(plot_plan["plot_id"])] = PlotPlan.from_dict(plot_plan)
        return plot_plans


class ModelCombinationsPrompts(IterableDataObject["ModelCombination"]):
    def __init__(self, model_combinations: list["ModelCombination"]):
        super().__init__(model_combinations)

    def cols_enum(self) -> Type[ModelCombinationCols]:
        return ModelCombinationCols

    def sort_by_prompt_count(self) -> "ModelCombinationsPrompts":
        return ModelCombinationsPrompts(sorted(self._items, key=lambda x: len(x.prompts), reverse=True))

    def change_chosen_prompt_by_seed(self, seed: int) -> "ModelCombinationsPrompts":
        return ModelCombinationsPrompts([combination.choose_prompt_by_seed(seed) for combination in self._items])

    def to_display_df(self, models_combinations: list[MODEL_ARCH_AND_SIZE]) -> pd.DataFrame:
        table_data: List[Dict[str, Union[int, str]]] = []
        for row in self._items:
            # Create row with model correctness
            table_row: Dict[str, Union[int, str]] = {}

            # Add prompt count and selected prompt first
            table_row[HeatmapCols.PROMPT_COUNT] = len(row.prompts)
            if row.chosen_prompt is not None:
                table_row[HeatmapCols.SELECTED_PROMPT] = str(row.chosen_prompt)
            else:
                table_row[HeatmapCols.SELECTED_PROMPT] = ""

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


class ResultBank(IterableDataObject[T_RUNNER_TYPE]):
    KEY = "key"

    def to_experiment_results_df(self) -> pd.DataFrame:
        results_data = []
        for i, result in enumerate(self._items):
            result_dict: dict = {param: getattr(result.variant_params, param, None) for param in ResultBankParamNames}
            result_dict[ResultBankParamNames.path] = str(result.variation_relative_path)
            result_dict[ResultBankParamNames.code_version] = result.metadata_params.code_version
            result_dict[ResultBank.KEY] = i
            results_data.append(result_dict)
        return pd.DataFrame(results_data)

    def from_experiment_results_df(self, experiment_results_df: Optional[pd.DataFrame]):
        if experiment_results_df is None:
            return self.__class__([])
        return self.__class__(select_indexes_from_list(self._items, experiment_results_df[ResultBank.KEY].tolist()))

    def to_info_flow_results(self) -> "InfoFlowResults":
        results = [result for result in self if result.variant_params.experiment_name == ExperimentName.info_flow]
        return InfoFlowResults(cast(list[InfoFlowRunner], results))

    def is_empty(self) -> bool:
        return len(self) == 0

    def get_common_and_different_params(self) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.is_empty():
            return {}, []

        common_params: Dict[str, Any] = {}
        different_params_list: List[Dict[str, Any]] = []

        all_keys: Set[str] = set()
        for result in self:
            all_keys.update(asdict(result.variant_params).keys())

        for key in all_keys:
            values = [getattr(result.variant_params, key, None) for result in self]
            unique_values = set(values)

            if len(unique_values) == 1:
                common_params[key] = next(iter(unique_values))
            else:
                for i, result in enumerate(self):
                    if i >= len(different_params_list):
                        different_params_list.append({})
                    variant_params = asdict(result.variant_params)
                    if key in variant_params:
                        different_params_list[i][key] = variant_params[key]

        return common_params, different_params_list


class InfoFlowResults(ResultBank[InfoFlowRunner]):
    def get_common_indices(self) -> set[TPromptOriginalIndex]:
        existing_ids_list = [info_flow.output_file.get_computed_prompt_idx() for info_flow in self]
        return functools.reduce(lambda x, y: x.intersection(y), existing_ids_list)

    def max_layer(self) -> int:
        return max(info_flow.output_file.get_statistics().layers_amount for info_flow in self) - 1

    def min_layer(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self)

    def subset_layers(self, layer_idx_subset: TWindowLayerStartIndex) -> "InfoFlowResults":
        return InfoFlowResults(
            [
                info_flow.modify(variant_params=info_flow.variant_params.modify(subset_layers=layer_idx_subset))
                for info_flow in self
            ]
        )

    def subset_prompts(self, prompt_ids: list[TPromptOriginalIndex]) -> "InfoFlowResults":
        return InfoFlowResults(
            [
                info_flow.modify(input_params=InputParams(filteration=SelectivePromptFilteration(tuple(prompt_ids))))
                for info_flow in self
            ]
        )


class PromptNew(DataObject):
    def __init__(self, prompt: dict):
        self._prompt = prompt

    def as_prompt(self) -> Prompt:
        return Prompt(self._prompt)  # type: ignore


class Prompts(IndexableDataObject[TPromptOriginalIndex, PromptNew]):
    def __init__(self, df: Dict[TPromptOriginalIndex, PromptNew], tokenizer: TTokenizer):
        super().__init__(df)
        self._tokenizer = tokenizer

    def filter_by_prompt_ids(self, prompt_ids: list[TPromptOriginalIndex]):
        return Prompts(subset_dict_by_keys(self._items, prompt_ids), self._tokenizer)

    def filter_by_condition(self, condition: Callable[[TPromptOriginalIndex, PromptNew], bool]):
        return self.filter_by_prompt_ids(get_dict_keys_by_condition(self._items, condition))

    @property
    def empty(self) -> bool:
        return len(self._items) == 0

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def original_idx(self) -> list[TPromptOriginalIndex]:
        return list(self._items.keys())

    def sample(self, sample_size: int, seed: int) -> "Prompts":
        random.seed(seed)
        sampled_indices = random.choices(list(self._items.keys()), k=sample_size)
        return self.filter_by_prompt_ids(sampled_indices)
