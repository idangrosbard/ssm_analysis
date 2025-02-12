from typing import Optional

import pandas as pd

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from datasets import (
    load_dataset as huggingface_load_dataset,
)
from scripts.counterfact.splitting import split_dataset
from src.consts import COLUMNS, COUNTER_FACT_2_KNOWN1000_COL_CONV, DATASETS_IDS, PATHS
from src.types import DATASETS, FILTERATIONS, SPLIT, DatasetArgs, TSplit


def load_splitted_counter_fact(
    split: TSplit = (SPLIT.TRAIN1,),
    add_split_name_column: bool = False,
    filteration: Optional[FILTERATIONS] = None,
    align_to_known: bool = False,
) -> Dataset:
    splitted_path = PATHS.COUNTER_FACT_DIR / "splitted"

    if not splitted_path.exists():
        print("Creating splitted dataset")
        dataset_name = DATASETS_IDS[DATASETS.COUNTER_FACT]
        num_splits = 5
        split_ratio = 0.1
        seed = 42

        dataset = huggingface_load_dataset(dataset_name)["train"]  # type: ignore

        splitted_dataset = split_dataset(dataset, num_splits, split_ratio, seed)
        splitted_dataset.save_to_disk(str(splitted_path))

    data: DatasetDict = load_from_disk(str(splitted_path))  # type: ignore

    if split == "all":
        split = list(data.keys())
    if isinstance(split, str):
        split = [SPLIT(split)]

    datasets = [data[split_name] for split_name in split]

    if add_split_name_column:
        for i, (split_name, dataset) in enumerate(zip(split, datasets)):
            dataset = dataset.add_column("split_name", [split_name] * len(dataset))  # type: ignore
            datasets[i] = dataset

    dataset = concatenate_datasets(datasets)

    if align_to_known:
        for counter_fact_col, known1000_col in COUNTER_FACT_2_KNOWN1000_COL_CONV.items():
            dataset = dataset.rename_column(counter_fact_col, known1000_col)
        dataset = dataset.remove_columns([COLUMNS.TARGET_FALSE, COLUMNS.TARGET_FALSE_ID])

    if filteration is not None:
        original_idx = pd.read_csv(PATHS.COUNTER_FACT_FILTERATIONS_DIR / f"{filteration}.csv")[
            COLUMNS.ORIGINAL_IDX
        ].to_list()
        dataset = dataset.filter(lambda x: x[COLUMNS.ORIGINAL_IDX] in original_idx)

    return dataset


def load_dataset(dataset_args: DatasetArgs) -> Dataset:
    if dataset_args.name == DATASETS.COUNTER_FACT:
        return load_splitted_counter_fact(dataset_args.splits)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_args.name}")
