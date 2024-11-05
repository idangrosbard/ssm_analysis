from re import L
from typing import Any, Iterable, Literal
from typing import Dict
from typing import NamedTuple
from typing import NewType
from typing import Optional
from typing import TypedDict

from attr import dataclass
from regex import D

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
    MINIMAL_MAMBA1 = "minimal_mamba1"
    MINIMAL_MAMBA2 = "minimal_mamba2"
    LLAMA2 = "llama2"
    LLAMA3_2 = "llama3.2"


class DATASETS(STREnum):
    KNOWN_1000 = "known_1000"
    COUNTER_FACT = "counter_fact"


TModelID = NewType("TModelID", str)
TDatasetID = NewType("TDatasetID", str)
TSplit = SPLIT | Iterable[SPLIT] | None | Literal['all']

@dataclass
class DatasetArgs:
    name: DATASETS
    splits: TSplit = 'all'
