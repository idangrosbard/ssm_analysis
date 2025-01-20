from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal, NewType

import pandas as pd

from src.utils.types_utils import STREnum


class SPLIT(STREnum):
    TRAIN1 = "train1"
    TRAIN2 = "train2"
    TRAIN3 = "train3"
    TRAIN4 = "train4"
    TRAIN5 = "train5"
    TEST = "test"


class MODEL_ARCH(STREnum):
    MAMBA1 = "mamba"
    MAMBA2 = "mamba2"
    MINIMAL_MAMBA1 = "minimal_mamba1"
    MINIMAL_MAMBA2 = "minimal_mamba2"
    MINIMAL_MAMBA2_new = "minimal_mamba2_new"
    LLAMA2 = "llama2"
    LLAMA3_2 = "llama3.2"

    @property
    def model_title(self) -> str:
        match self:
            case MODEL_ARCH.MAMBA1 | MODEL_ARCH.MINIMAL_MAMBA1:
                return "Mamba1"
            case MODEL_ARCH.MAMBA2 | MODEL_ARCH.MINIMAL_MAMBA2 | MODEL_ARCH.MINIMAL_MAMBA2_new:
                return "Mamba2"
        return self.value


class MODEL_SIZE_CAT(STREnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class DATASETS(STREnum):
    KNOWN_1000 = "known_1000"
    COUNTER_FACT = "counter_fact"


TModelID = NewType("TModelID", str)
TDatasetID = NewType("TDatasetID", str)
TSplit = SPLIT | Iterable[SPLIT] | Literal["all"]

TNum2Mask = NewType("TNum2Mask", dict[int, list[tuple[int, int]]])
TWindow = NewType("TWindow", list[int])
TPromptData = NewType("TPromptData", pd.DataFrame)


class TokenType(STREnum):
    first = "first"
    last = "last"
    subject = "subject"
    relation = "relation"
    context = "context"


@dataclass
class DatasetArgs:
    name: DATASETS
    splits: TSplit = "all"

    def __post_init__(self):
        if self.splits != "all" and isinstance(self.splits, str):
            self.splits = [SPLIT(self.splits)]

    @property
    def dataset_name(self) -> str:
        split_name = ""
        if self.splits != "all":
            split_name = f"_{self.splits}"
        return self.name + split_name


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    TTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast
