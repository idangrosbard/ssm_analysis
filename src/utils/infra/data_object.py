from abc import ABC
from typing import Generic, TypeVar


class DataObject(ABC):
    pass


T_VALUE = TypeVar("T_VALUE")


class IterableDataObject(DataObject, Generic[T_VALUE]):
    def __init__(self, items: list[T_VALUE]):
        self._items = items  # Make it immutable

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index: int) -> T_VALUE:
        return self._items[index]


T_KEY = TypeVar("T_KEY")


class IndexableDataObject(DataObject, Generic[T_KEY, T_VALUE]):
    """Base class for objects that wrap dict-like data"""

    def __init__(self, items: dict[T_KEY, T_VALUE]):
        self._items = items

    def __getitem__(self, index: T_KEY) -> T_VALUE:
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def items(self):
        return self._items.items()

    def to_rows(self):
        return list(self._items.items())

    def __len__(self):
        return len(self._items)

    def is_empty(self):
        return len(self._items) == 0

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()
