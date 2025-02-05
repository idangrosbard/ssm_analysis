from typing import Optional

import pandas as pd

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from datasets import load_dataset as huggingface_load_dataset
from scripts.counterfact.splitting import split_dataset
from src.consts import COLUMNS, DATASETS_IDS, FILTERATIONS, PATHS
from src.types import DATASETS, SPLIT, DatasetArgs, TPromptData, TSplit


def load_splitted_counter_fact(
    split: TSplit = (SPLIT.TRAIN1,),
    add_split_name_column: bool = False,
    filteration: Optional[FILTERATIONS] = None,
    align_to_known: bool = True,
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
        # rename 'target_true' -> 'attribute',
        dataset = dataset.rename_column("target_true", "attribute")
        # remove: 'target_false', 'target_false_id', 'target_true_id'
        dataset = dataset.remove_columns(["target_false", "target_false_id", "target_true_id"])

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


def get_hit_dataset(model_id: str, dataset_args: DatasetArgs) -> TPromptData:
    original_res, attn_res = [
        pd.read_parquet(
            PATHS.OUTPUT_DIR
            / model_id
            / "data_construction"
            / f"ds={dataset_args.dataset_name}"
            / f"entire_results_{'attention' if attention else 'original'}.parquet"
        )
        for attention in [True, False]
    ]

    mask = (original_res["hit"] == attn_res["hit"]) & (attn_res["hit"])
    return attn_res[mask]  # type: ignore
