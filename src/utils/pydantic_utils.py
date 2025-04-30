from typing import Any, Literal


def create_literal_value(values: list[str]) -> Any:
    return Literal[*values]  # type: ignore
