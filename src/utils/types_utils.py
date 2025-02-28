from enum import StrEnum
from typing import Type, TypeVar


def class_values(cls: Type) -> list[str]:
    if issubclass(cls, StrEnum):
        return [member.value for member in cls]  # Handle StrEnum
    return [value for key, value in vars(cls).items() if not key.startswith("__")]


_T = TypeVar("_T")


def select_indexes_from_list(lst: list[_T], indexes: list[int]) -> list[_T]:
    return [lst[i] for i in indexes]
