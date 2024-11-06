from datasets import load_dataset, DatasetDict

from src.consts import DATASETS_IDS, PATHS
from src.types import DATASETS
import random


def split_dataset(dataset_name, num_splits, split_ratio, seed):
    """
    Split the dataset into multiple train splits and a test set.

    Args:
        dataset_name (str): The name or path of the dataset to load.
        num_splits (int): Number of train splits.
        split_ratio (float): Proportion of data for each train split.
        seed (int): Seed for shuffling to ensure reproducibility.

    Returns:
        DatasetDict: A dictionary containing the train splits and test set.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)["train"]
    # rename 'target_true' -> 'attribute',
    dataset = dataset.rename_column('target_true', 'attribute')
    
    # remove: 'target_false', 'target_false_id', 'target_true_id'
    dataset = dataset.remove_columns(['target_false', 'target_false_id', 'target_true_id'])
    
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=seed)
    # Keep original indices while shuffling
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    dataset = dataset.select(indices)
    # Add original indices as a feature
    dataset = dataset.map(lambda example, idx: {'original_idx': indices[idx]}, with_indices=True)
    # Calculate sizes
    num_examples = len(dataset)
    split_size = int(split_ratio * num_examples)

    # Create splits
    splits = {}
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        splits[f"train{i+1}"] = dataset.select(range(start_idx, end_idx))

    # Remaining data for test split
    remaining_start_idx = num_splits * split_size
    splits["test"] = dataset.select(range(remaining_start_idx, num_examples))

    return DatasetDict(splits)


def save_splitted_dataset(splitted_dataset, save_path):
    """
    Save the split dataset to disk.

    Args:
        splitted_dataset (DatasetDict): The dataset with splits to save.
        save_path (str): The path where to save the dataset.
    """
    splitted_dataset.save_to_disk(save_path)
