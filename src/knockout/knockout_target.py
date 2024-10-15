from enum import Enum


class KnockoutTarget(Enum):
    ENTIRE_SUBJ = 0
    SUBJ_LAST = 1
    FIRST = 2
    LAST = 3
    RANDOM = 4
    RANDOM_SPAN = 5
    ALL_CONTEXT = 6
    