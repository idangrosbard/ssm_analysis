import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.final_plots.app.app_consts import SummarizedDataFulfilledReqsCols
from src.names import ResultBankParamNames
from src.utils.data_object import DataObject
from src.utils.types_utils import str_enum_values

if TYPE_CHECKING:
    from src.final_plots.data_reqs import DataReq
    from src.final_plots.plot_plan import PlotPlan
    from src.final_plots.results_bank import ResultRecord


class DataReqs(DataObject):
    def __init__(self, data_reqs: set["DataReq"]):
        self._raw = data_reqs

    def to_rows(self) -> list["DataReq"]:
        return list(self._raw)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DataReqs":
        from src.final_plots.data_reqs import DataReq

        return cls(
            set(
                {
                    DataReq(
                        **{
                            col: row[col]
                            for col in str_enum_values(ResultBankParamNames)
                            if col not in [ResultBankParamNames.path, ResultBankParamNames.variation]
                        }
                    )
                    for row in df.to_dict(orient="records")
                }
            )
        )

    def to_fulfilled_reqs(self, result_bank: "ResultBank") -> "FulfilledReqs":
        from src.final_plots.data_reqs import get_data_fullfment_options

        return get_data_fullfment_options(self, result_bank)


class FulfilledReqs(DataObject):
    def __init__(self, fulfilled_reqs: dict["DataReq", list[Path]]):
        self._raw = fulfilled_reqs

    def summarize(self, overrides: Optional[dict["DataReq", Optional[Path]]]) -> "SummarizedDataFulfilledReqs":
        return SummarizedDataFulfilledReqs(self, overrides)

    def choose_latest_fulfilled(self, result_bank: "ResultBank") -> "FulfilledReqs":
        from src.final_plots.data_reqs import choose_latest_data_fulfilled

        return FulfilledReqs(
            {req: [] if path is None else [path] for req, path in choose_latest_data_fulfilled(self).items()}
        )

    def to_rows(self) -> list["DataReq"]:
        return list(self._raw.keys())


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
        from src.final_plots.app.utils import format_path_for_display

        results_data = []
        for result in self.to_rows():
            result_dict = {param: getattr(result, param, None) for param in ResultBankParamNames}
            result_dict[ResultBankParamNames.path] = format_path_for_display(result_dict[ResultBankParamNames.path])
            results_data.append(result_dict)
        return ExperimentDisplayResults(results_data)


class SummarizedDataFulfilledReqs(DataObject):
    def __init__(self, fulfilled_reqs: FulfilledReqs, overrides: Optional[dict["DataReq", Optional[Path]]]):
        self._raw = []
        for req, opts in fulfilled_reqs._raw.items():
            override = overrides.get(req, None) if overrides else None
            assert override is None or override in opts
            row = {
                **{
                    param: getattr(req, param, None)
                    for param in ResultBankParamNames
                    if param not in [ResultBankParamNames.path, ResultBankParamNames.variation]
                },
                SummarizedDataFulfilledReqsCols.AvailableOptions: len(opts),
                SummarizedDataFulfilledReqsCols.Options: opts,
                SummarizedDataFulfilledReqsCols.CurrentOverride: override,
                SummarizedDataFulfilledReqsCols.Key: str(req),
            }
            self._raw.append(row)

    def to_rows(self) -> list[dict]:
        return self._raw

    def to_data_reqs(self) -> DataReqs:
        from src.final_plots.data_reqs import DataReq

        return DataReqs(
            set(
                DataReq(
                    **{
                        param: row[param]
                        for param in ResultBankParamNames
                        if param not in [ResultBankParamNames.path, ResultBankParamNames.variation]
                    }
                )
                for row in self._raw
            )
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._raw)


class PlotPlans(DataObject):
    def __init__(self, plot_plans: dict[str, "PlotPlan"]):
        self._raw = plot_plans

    def add_plan(self, plan: "PlotPlan") -> None:
        if plan.title in self._raw:
            raise ValueError(f"Plot plan with title {plan.title} already exists")
        self._raw[plan.title] = plan
        self.order_plans()

    def order_plans(self) -> None:
        self._raw = dict(sorted(self._raw.items(), key=lambda x: x[1].order))

    def remove_plan(self, plan_title: str) -> None:
        if plan_title not in self._raw:
            raise ValueError(f"Plot plan with title {plan_title} does not exist")
        self._raw.pop(plan_title)

    def get_plan(self, plan_title: str) -> Optional["PlotPlan"]:
        return self._raw.get(plan_title)

    def save(self, path: Path) -> None:
        self.order_plans()
        with open(path, "w") as f:
            json.dump([plan.to_dict() for plan in self._raw.values()], f, indent=2)

    def is_empty(self) -> bool:
        return not self._raw

    @classmethod
    def load(cls, path: Path) -> "PlotPlans":
        from src.final_plots.plot_plan import PlotPlan

        with open(path, "r") as f:
            data = json.load(f)

        return cls(plot_plans={plan["title"]: PlotPlan.from_dict(plan) for plan in data})

    def to_rows(self) -> list["PlotPlan"]:
        return list(self._raw.values())
