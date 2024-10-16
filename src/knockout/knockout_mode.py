from enum import Enum


class KnockoutMode(Enum):
    ZERO_ATTENTION = 0
    ZERO_DELTA = 1
    IGNORE_CONTEXT = 2