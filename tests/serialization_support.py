"""DSPy fixtures and helpers for whetstone's serialization handler tests.

Generic serialization contract tests moved to the dr-serialize repo with
the engine; only the DSPy-specific pieces stay here.
"""

from __future__ import annotations

import json
from typing import Any

import pydantic

import dspy
from dspy.utils.dummies import DummyLM
from whetstone.dspy_serialization import dspy_serializer

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


def assert_to_jsonable(value: Any) -> Any:
    result = dspy_serializer().to_jsonable(value)
    assert_json_dumps(result)
    assert_only_json_types(result)
    return result


class QASig(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def minimal_example() -> dspy.Example:
    return dspy.Example(question="q", answer="a")


def stub_lm(**kwargs: Any) -> dspy.BaseLM:
    lm = DummyLM([{}])
    lm.kwargs = dict(kwargs)
    return lm


class BadModel(pydantic.BaseModel):
    x: object


def bad_pydantic_model() -> BadModel:
    return BadModel(x=object())
