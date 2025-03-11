from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.final_plots.app.app_consts import SummarizedDataFulfilledReqsCols
from src.names import ResultBankParamNames
from src.utils.data_object import DataObject
from src.utils.types_utils import str_enum_values

if TYPE_CHECKING:
    from src.final_plots.data_reqs import DataReq
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


class FulfilledReqs(DataObject):
    def __init__(self, fulfilled_reqs: dict["DataReq", list[Path]]):
        self._raw = fulfilled_reqs


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
    def __init__(self, fulfilled_reqs: FulfilledReqs, overrides: dict["DataReq", Optional[Path]]):
        self._raw = []
        for req, opts in fulfilled_reqs._raw.items():
            override = overrides.get(req)
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

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._raw)
