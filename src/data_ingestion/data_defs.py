import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Type

import pandas as pd

from src.analysis.prompt_filterations import UnionPromptFilteration
from src.core.consts import PATHS
from src.core.names import HeatmapCols, ModelCombinationCols, ResultBankParamNames, SummarizedDataFulfilledReqsCols
from src.core.types import MODEL_ARCH_AND_SIZE, TPlotID
from src.experiments.infrastructure.base_config import BasePromptFilteration, BaseRunner, BaseVariantParams, InputParams
from src.utils.infra.data_object import DataObject

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
                # if isinstance(runner, InfoFlowConfig):
                #     if (
                #         len(
                #             set(prompt_filterations.get_prompt_ids())
                #             - set(runner.output_file.get_computed_prompt_idx(include_banned=True))
                #         )
                #         == 0
                #     ):
                #         data_reqs_options[runner.variant_params].append(runner)

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


class ExperimentDisplayResults(DataObject):
    def __init__(self, results_data: list[BaseRunner]):
        self._raw = results_data

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._raw)


class ResultBank(DataObject):
    def __init__(self, result_bank: list[BaseRunner]):
        self._raw = result_bank

    def to_rows(self) -> list[BaseRunner]:
        return self._raw

    def to_experiment_results(self) -> ExperimentDisplayResults:
        from src.app.app_utils import format_path_for_display

        results_data = []
        for result in self.to_rows():
            result_dict = {param: getattr(result, param, None) for param in ResultBankParamNames}
            result_dict[ResultBankParamNames.path] = format_path_for_display(result_dict[ResultBankParamNames.path])
            results_data.append(result_dict)
        return ExperimentDisplayResults(results_data)


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
