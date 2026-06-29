"""Shared helpers and minimal fixtures for serialization contract tests."""

from __future__ import annotations

import json
from typing import Any

import pydantic

import dspy
from dr_dspy.serialization import SerializationError, to_jsonable
from dspy.utils.dummies import DummyLM

_JSON_TYPES = (type(None), bool, int, float, str, list, dict)


def assert_json_dumps(value: Any) -> None:
    json.dumps(value, ensure_ascii=False)


def assert_only_json_types(value: Any) -> None:
    if isinstance(value, _JSON_TYPES):
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    msg = f"non-string dict key: {key!r}"
                    raise AssertionError(msg)
                assert_only_json_types(item)
        elif isinstance(value, list):
            for item in value:
                assert_only_json_types(item)
        return
    if isinstance(value, tuple):
        for item in value:
            assert_only_json_types(item)
        return
    msg = f"non-JSON type: {type(value).__name__}"
    raise AssertionError(msg)


def assert_to_jsonable(value: Any, *, max_bytes: int | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if max_bytes is not None:
        kwargs["max_bytes"] = max_bytes
    result = to_jsonable(value, **kwargs)
    assert_json_dumps(result)
    assert_only_json_types(result)
    return result


def assert_diagnostics(
    exc: SerializationError,
    required_keys: set[str],
    **expected_fields: Any,
) -> dict[str, Any]:
    diagnostics = exc.diagnostics()
    assert set(diagnostics) >= required_keys
    for key, expected in expected_fields.items():
        assert diagnostics[key] == expected, (
            f"diagnostics[{key!r}]: {diagnostics[key]!r} != {expected!r}"
        )
    return diagnostics


def nested_list(depth: int, leaf: str = "x") -> list[Any]:
    value: Any = leaf
    for _ in range(depth):
        value = [value]
    return value


def large_payload(char_count: int) -> dict[str, str]:
    return {"blob": "a" * char_count}


class QASig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def minimal_example() -> dspy.Example:
    return dspy.Example(question="q", answer="a")


def stub_lm(**kwargs: Any) -> dspy.BaseLM:
    lm = DummyLM([{}])
    lm.kwargs = dict(kwargs)
    return lm


class OkModel(pydantic.BaseModel):
    name: str
    count: int


class SerializedNameModel(pydantic.BaseModel):
    name: str

    @pydantic.field_serializer("name")
    def serialize_name(self, value: str) -> str:
        return value.upper()


def ok_pydantic_model() -> OkModel:
    return OkModel(name="n", count=1)


class BadModel(pydantic.BaseModel):
    x: object


def bad_pydantic_model() -> BadModel:
    return BadModel(x=object())


class SimpleObject:
    def __init__(self) -> None:
        self.a = 1
        self.label = "test"
