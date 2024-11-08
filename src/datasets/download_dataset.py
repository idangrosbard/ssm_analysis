from pathlib import Path

import pandas as pd
import wget

from scripts.counterfact.splitting import split_dataset
from src.consts import DATASETS_IDS, PATHS
import tempfile
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from src.types import DATASETS, SPLIT, DatasetArgs, TSplit
from datasets import load_dataset, DatasetDict


def load_knowns() -> Dataset:
    if not PATHS.RAW_KNOWN_1000_DIR.exists():
        with tempfile.TemporaryDirectory() as tmpdirname:
            wget.download(
                "https://rome.baulab.info/data/dsets/known_1000.json", out=tmpdirname
            )
            knowns_df = (
                pd.read_json(Path(tmpdirname) / "known_1000.json")
                .set_index("known_id")
                # add space before values in the 'attribute' col to match counterfact
                .assign(**{'attribute':lambda x: " " + x["attribute"]})
            )
            
            dataset = Dataset.from_pandas(knowns_df)
            dataset.save_to_disk(PATHS.RAW_KNOWN_1000_DIR)

    return load_from_disk(str(PATHS.RAW_KNOWN_1000_DIR))  # type: ignore


def load_splitted_knowns(split: TSplit = (SPLIT.TRAIN1,)) -> Dataset:
    splitted_path = PATHS.PROCESSED_KNOWN_DIR / "splitted"

    if not splitted_path.exists():
        dataset = load_knowns()

        num_splits = 3
        split_ratio = 0.2
        seed = 42

        splitted_dataset = split_dataset(dataset, num_splits, split_ratio, seed)
        splitted_dataset.save_to_disk(str(splitted_path))

    data = load_from_disk(str(splitted_path))  # type: ignore

    if split == "all":
        split = list(data.keys())
    if isinstance(split, str):
        split = [split]

    return concatenate_datasets([data[split] for split in split])


def load_knowns_pd() -> pd.DataFrame:
    return pd.DataFrame(load_knowns())


def load_splitted_counter_fact(split: TSplit = (SPLIT.TRAIN1,)) -> Dataset:
    splitted_path = PATHS.COUNTER_FACT_DIR / "splitted"

    if not splitted_path.exists():
        print("Creating splitted dataset")
        dataset_name = DATASETS_IDS[DATASETS.COUNTER_FACT]
        num_splits = 5
        split_ratio = 0.1
        seed = 42

        dataset = load_dataset(dataset_name)["train"]

        # Align counterfact to known
        # rename 'target_true' -> 'attribute',
        dataset = dataset.rename_column("target_true", "attribute")
        # remove: 'target_false', 'target_false_id', 'target_true_id'
        dataset = dataset.remove_columns(
            ["target_false", "target_false_id", "target_true_id"]
        )

        splitted_dataset = split_dataset(dataset, num_splits, split_ratio, seed)
        splitted_dataset.save_to_disk(str(splitted_path))

    data = load_from_disk(str(splitted_path))  # type: ignore

    if split == "all":
        split = list(data.keys())
    if isinstance(split, str):
        split = [split]

    return concatenate_datasets([data[split] for split in split])


def load_dataset(dataset_args: DatasetArgs) -> Dataset:
    if dataset_args.name == DATASETS.KNOWN_1000:
        return load_splitted_knowns(dataset_args.splits)
    elif dataset_args.name == DATASETS.COUNTER_FACT:
        return load_splitted_counter_fact(dataset_args.splits)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_args.name}")
