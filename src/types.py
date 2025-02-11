from dataclasses import dataclass
from enum import Enum
from typing import Literal, NewType, Optional, Sequence, Union, assert_never

import pandas as pd
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.utils.types_utils import STREnum


class SPLIT(STREnum):
    TRAIN1 = "train1"
    TRAIN2 = "train2"
    TRAIN3 = "train3"
    TRAIN4 = "train4"
    TRAIN5 = "train5"
    TEST = "test"


class MODEL_ARCH(STREnum):
    MAMBA1 = "mamba1"
    MAMBA2 = "mamba2"
    GPT2 = "gpt2"
    LLAMA2 = "llama2"
    LLAMA3_2 = "llama3.2"

    @property
    def model_title(self) -> str:
        match self:
            case MODEL_ARCH.MAMBA1:
                return "Mamba1"
            case MODEL_ARCH.MAMBA2:
                return "Mamba2"
            case MODEL_ARCH.LLAMA2:
                return "Llama2"
            case MODEL_ARCH.LLAMA3_2:
                return "Llama3.2"
            case MODEL_ARCH.GPT2:
                return "GPT2"
        assert_never(self.value)


class MODEL_SIZE_CAT(STREnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class DATASETS(STREnum):
    COUNTER_FACT = "counter_fact"


TModelID = NewType("TModelID", str)
TDatasetID = NewType("TDatasetID", str)
TSplit = Union[SPLIT, Sequence[SPLIT], Literal["all"]]

TNum2Mask = NewType("TNum2Mask", dict[int, list[tuple[int, int]]])
TWindow = NewType("TWindow", list[int])
TPromptData = NewType("TPromptData", pd.DataFrame)


class TokenType(STREnum):
    first = "first"
    last = "last"
    subject = "subject"
    relation = "relation"
    context = "context"


class FILTERATIONS(STREnum):
    all_correct = "all_correct"
    all_any_correct = "all_any_correct"


@dataclass
class DatasetArgs:
    name: DATASETS
    splits: TSplit = "all"
    filteration: Optional[FILTERATIONS] = FILTERATIONS.all_correct

    def __post_init__(self):
        if self.splits != "all" and isinstance(self.splits, str):
            self.splits = [SPLIT(self.splits)]

    @property
    def display_name(self) -> str:
        split_name = ""
        if self.splits != "all":
            split_name = f"_{self.splits}"
        if self.filteration is not None:
            split_name = f"_{self.filteration}"
        return self.name + split_name


TTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

TSSMState = Float[Tensor, "batch hidden_size ssm_dim"]
TSSMInput = Float[Tensor, "batch hidden_size seq_len"]
TSSM_A = Float[Tensor, "batch hidden_size seq_len ssm_dim"]
TSSM_B = Float[Tensor, "batch hidden_size seq_len ssm_dim"]
TSSM_Bu = Float[Tensor, "batch hidden_size seq_len ssm_dim"]
TSSM_C = Float[Tensor, "batch seq_len ssm_dim"]


class KnockoutMode(Enum):
    ZERO_ATTENTION = 0
    ZERO_DELTA = 1
    IGNORE_CONTEXT = 2
    ONLY_CONTEXT = 3
    IGNORE_LAYER = 4
    IGNORE_SSM = 5
    INCREASE_DELTA = 6
