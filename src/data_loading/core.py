from functools import lru_cache

from src.analysis.experiment_results.results_bank import get_experiment_results_bank
from src.core.names import COLS, DatasetName
from src.core.types import TPromptOriginalIndex
from src.data_ingestion.data_defs.data_defs import PromptNew, Prompts, ResultBank
from src.data_ingestion.datasets.download_dataset import get_row_data


@lru_cache(maxsize=None)
def load_results_bank() -> ResultBank:
    return get_experiment_results_bank()


@lru_cache(maxsize=None)
def load_prompts(
    dataset: DatasetName = DatasetName.counter_fact,
) -> Prompts:
    df = get_row_data(dataset)

    return Prompts(
        {TPromptOriginalIndex(int(row[COLS.ORIGINAL_IDX])): PromptNew(dict(row)) for _, row in df.iterrows()},
    )
