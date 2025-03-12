import contextlib
from enum import StrEnum
from typing import Any, ContextManager, Type, TypeVar, cast


def class_values(cls: Type) -> list[str]:
    if issubclass(cls, StrEnum):
        return [member.value for member in cls]  # Handle StrEnum
    return [value for key, value in vars(cls).items() if not key.startswith("__")]


_T_STR_ENUM = TypeVar("_T_STR_ENUM", bound=StrEnum)


def str_enum_values(cls: Type[_T_STR_ENUM]) -> list[_T_STR_ENUM]:
    return cast(list[_T_STR_ENUM], class_values(cls))


_T = TypeVar("_T")


def select_indexes_from_list(lst: list[_T], indexes: list[int]) -> list[_T]:
    return [lst[i] for i in indexes]


def get_list_indexes_of_set_values(lst: list[_T], values: set[_T]) -> list[int]:
    return [i for i, v in enumerate(lst) if v in values]


def first_dict_value(d: dict[Any, _T]) -> _T:
    return next(iter(d.values()))


def first_dict_key(d: dict[Any, _T]) -> Any:
    return next(iter(d.keys()))


def conditional_context_manager(use_ctx: bool, ctx: ContextManager[None]) -> ContextManager[None]:
    """
    Returns the given context manager if use_ctx is True, otherwise returns a dummy context.

    :param use_ctx: Boolean flag to determine whether to use the actual context manager.
    :param ctx: The actual context manager to use if use_ctx is True.
    :return: The appropriate context manager (either ctx or nullcontext).
    """
    return ctx if use_ctx else contextlib.nullcontext()
