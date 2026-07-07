"""Contract tests for whetstone's DSPy serialization handlers.

The generic engine tests live in the dr-serialize repo; these cover the
app-side handlers registered at whetstone package import.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from dr_serialize import (
    postgres_jsonb_limits,
    registered_handlers,
    to_jsonable,
)

import dspy
import whetstone.dspy_serialization as dspy_serialization
from tests.serialization_support import (
    QASig,
    assert_to_jsonable,
    minimal_example,
    stub_lm,
)
from whetstone.dspy_serialization import (
    ExampleSerializationError,
    SignatureSummaryError,
)

DEFAULT_LIMITS = postgres_jsonb_limits()


def assert_diagnostics_shape(exc: Any) -> None:
    diagnostics = exc.diagnostics()
    assert set(diagnostics) >= {
        "path",
        "detail",
        "value_preview",
        "underlying",
    }


def test_handlers_registered_at_package_import() -> None:
    handlers = registered_handlers()
    assert dspy_serialization.jsonable_dspy_example in handlers
    assert dspy_serialization.jsonable_dspy_signature_type in handlers
    assert dspy_serialization.jsonable_dspy_lm in handlers


def test_repeated_registration_is_idempotent() -> None:
    before = registered_handlers()
    dspy_serialization.register_dspy_handlers()
    assert registered_handlers() == before


def test_dspy_example() -> None:
    result = assert_to_jsonable(minimal_example())
    assert result == {"question": "q", "answer": "a"}


def test_dspy_signature_type() -> None:
    result = assert_to_jsonable(QASig)
    assert set(result) == {"signature", "instructions", "fields"}
    assert isinstance(result["fields"], list)
    assert all(isinstance(field, tuple) for field in result["fields"])


def test_plain_type_falls_through_to_generic_handler() -> None:
    result = assert_to_jsonable(int)
    assert isinstance(result, str)
    assert result.startswith("<class ")
    assert "int" in result


def test_dspy_base_lm() -> None:
    lm = stub_lm(api_key="secret", temperature=0.7)
    result = assert_to_jsonable(lm)
    assert result["_kind"] == "BaseLM"
    assert result["class"] == "dspy.utils.dummies.DummyLM"
    assert result["model"] == "dummy"
    assert result["kwargs"] == {
        "api_key": "<redacted>",
        "temperature": 0.7,
    }


def test_example_serialization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = minimal_example()

    def boom(self: dspy.Example) -> dict[str, Any]:
        raise RuntimeError("toDict failed")

    monkeypatch.setattr(dspy.Example, "toDict", boom)
    with pytest.raises(ExampleSerializationError) as exc_info:
        to_jsonable(example, limits=DEFAULT_LIMITS)
    assert_diagnostics_shape(exc_info.value)


def test_signature_summary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadFields:
        def items(self) -> Any:
            raise RuntimeError("fields broke")

    class FakeSig:
        fields = BadFields()
        signature = "question -> answer"
        instructions = "test instructions"

    original = dspy_serialization._signature_summary

    def intercept(
        sig_cls: type[dspy.Signature],
        path: tuple[str | int, ...],
    ) -> Any:
        if sig_cls is QASig:
            fake_sig = cast(type[dspy.Signature], cast(Any, FakeSig))
            return original(fake_sig, path)
        return original(sig_cls, path)

    monkeypatch.setattr(
        dspy_serialization, "_signature_summary", intercept
    )
    with pytest.raises(SignatureSummaryError) as exc_info:
        to_jsonable(QASig, limits=DEFAULT_LIMITS)
    assert_diagnostics_shape(exc_info.value)
