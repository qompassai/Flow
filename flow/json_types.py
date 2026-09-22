"""Checked container narrowing at untrusted JSON boundaries."""

from collections.abc import Mapping, Sequence
from typing import TypeGuard, cast


def is_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    # The container is known; neither its keys nor its values are trusted yet.
    container = cast(Mapping[object, object], value)
    return all(isinstance(key, str) for key in container)


def is_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def object_map(value: object) -> Mapping[str, object]:
    if not is_object(value):
        raise ValueError("Expected a JSON object with string keys")
    return value


def object_list(value: object) -> Sequence[object]:
    if not is_array(value):
        raise ValueError("Expected a JSON array")
    return value
