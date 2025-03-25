import json
from pathlib import Path
from typing import TYPE_CHECKING, Type

import pandas as pd

from src.core.consts import PATHS
from src.core.names import HeatmapCols, ModelCombinationCols, ResultBankParamNames
from src.core.types import MODEL_ARCH_AND_SIZE, TPlotID
from src.utils.infra.data_object import DataObject
from src.utils.types_utils import str_enum_values

if TYPE_CHECKING:
    from src.analysis.experiment_results.data_requirements import DataReq
    from src.analysis.experiment_results.model_prompt_combination import ModelCombination
    from src.analysis.experiment_results.plot_plan import PlotPlan
    from src.analysis.experiment_results.results_bank import ResultRecord


class DataReqs(DataObject):
    def __init__(self, data_reqs: set["DataReq"]):
        self._raw = data_reqs

    def to_rows(self) -> list["DataReq"]:
        return list(self._raw)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DataReqs":
        from src.analysis.experiment_results.data_requirements import DataReq

        return cls(
            set(
                {
                    DataReq.create_and_validate(
                        **{
                            col: row[col]
                            for col in str_enum_values(ResultBankParamNames)
                            if col not in [ResultBankParamNames.path, ResultBankParamNames.version]
                        }
                    )
                    for row in df.to_dict(orient="records")
                }
            )
        )

    def to_fulfilled_reqs(self, result_bank: "ResultBank") -> "FulfilledReqs":
        from src.analysis.experiment_results.helpers import get_data_fullfment_options

        return get_data_fullfment_options(self, result_bank)


class FulfilledReqs(DataObject):
    def __init__(self, fulfilled_reqs: dict["DataReq", list["ResultRecord"]]):
        self._raw = fulfilled_reqs

    def summarize(self) -> "SummarizedDataFulfilledReqs":
        return SummarizedDataFulfilledReqs(self)

    def choose_latest_fulfilled(self, result_bank: "ResultBank") -> "FulfilledReqs":
        from src.analysis.experiment_results.helpers import choose_latest_data_fulfilled

        return FulfilledReqs(
            {req: [] if path is None else [path] for req, path in choose_latest_data_fulfilled(self).items()}
        )

    def get_config(self):
        return {req: req.get_config(result_records[0].version) for req, result_records in self.to_rows()}

    def to_rows(self) -> list[tuple["DataReq", list["ResultRecord"]]]:
        return list(self._raw.items())


class ExperimentDisplayResults(DataObject):
    def __init__(self, results_data: list["ResultRecord"]):
        self._raw = results_data

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._raw)


class ResultBank(DataObject):
    def __init__(self, result_bank: list["ResultRecord"]):
        self._raw = result_bank

    def to_rows(self) -> list["ResultRecord"]:
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
        from src.core.names import SummarizedDataFulfilledReqsCols

        self._raw = []
        for req, opts in fulfilled_reqs._raw.items():
            row = {
                **{
                    param: getattr(req, param, None)
                    for param in ResultBankParamNames
                    if param not in [ResultBankParamNames.path, ResultBankParamNames.version]
                },
                SummarizedDataFulfilledReqsCols.AvailableOptions: len(opts),
                SummarizedDataFulfilledReqsCols.Options: opts,
                SummarizedDataFulfilledReqsCols.Key: str(req),
            }
            self._raw.append(row)

    def to_rows(self) -> list[dict]:
        return self._raw

    def to_data_reqs(self) -> DataReqs:
        from src.analysis.experiment_results.data_requirements import DataReq

        return DataReqs(
            set(
                DataReq.create_and_validate(
                    **{
                        param: row[param]
                        for param in ResultBankParamNames
                        if param not in [ResultBankParamNames.path, ResultBankParamNames.version]
                    }
                )
                for row in self._raw
            )
        )

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
