"""Contract tests for dr_dspy.serialization.

Deliberately not covered here:
- Full round-trip / lossless serialization
- Exhaustive Python type zoo (datetime, Decimal, UUID)
- lm_logging stderr behavior
- Exact preview truncation byte lengths
- Private _jsonable_* handler unit tests
"""

from __future__ import annotations

from typing import Any, cast

import pytest

import dr_dspy.serialization as serialization
import dspy
from dr_dspy.serialization import (
    MAX_JSONABLE_DEPTH,
    POSTGRES_JSONB_MAX_BYTES,
    ExampleSerializationError,
    JsonEncodeError,
    MaxDepthExceededError,
    ModelDumpError,
    ObjectVarsSerializationError,
    PayloadTooLargeError,
    SerializationError,
    SignatureSummaryError,
    sanitize_lm_kwargs,
    to_jsonable,
)
from tests.serialization_support import (
    BadModel,
    QASig,
    SerializedNameModel,
    SimpleObject,
    assert_diagnostics,
    assert_to_jsonable,
    bad_pydantic_model,
    large_payload,
    minimal_example,
    nested_list,
    ok_pydantic_model,
    stub_lm,
)


class TestSanitizeLmKwargs:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (None, {}),
            ({}, {}),
            (
                {"api_key": "secret", "temperature": 0.7},
                {"api_key": "<redacted>", "temperature": 0.7},
            ),
            (
                {"API_BASE": "https://x", "max_tokens": 100},
                {"API_BASE": "<redacted>", "max_tokens": 100},
            ),
            (
                {
                    "authorization": "Bearer x",
                    "model_list": ["a"],
                    "base_url": "https://y",
                    "other": "keep",
                },
                {
                    "authorization": "<redacted>",
                    "model_list": "<redacted>",
                    "base_url": "<redacted>",
                    "other": "keep",
                },
            ),
        ],
        ids=[
            "none",
            "empty",
            "api_key_mixed",
            "case_insensitive_key",
            "all_sensitive_keys",
        ],
    )
    def test_sanitize(
        self,
        kwargs: dict[str, Any] | None,
        expected: dict[str, Any],
    ) -> None:
        assert sanitize_lm_kwargs(kwargs) == expected


class TestToJsonableInvariants:
    @pytest.mark.parametrize(
        ("input_value", "check"),
        [
            (None, lambda r: r is None),
            (True, lambda r: r is True),
            (42, lambda r: r == 42),
            (1.5, lambda r: r == 1.5),
            ("hi", lambda r: r == "hi"),
            ({"a": 1, "b": {"c": 2}}, lambda r: r == {"a": 1, "b": {"c": 2}}),
            ([1, [2, 3]], lambda r: r == [1, [2, 3]]),
            ((1, 2), lambda r: r == [1, 2]),
            ({1, 2}, lambda r: sorted(r) == [1, 2]),
            (frozenset({3}), lambda r: r == [3]),
            ({1: "one"}, lambda r: r == {"1": "one"}),
        ],
        ids=[
            "none",
            "bool",
            "int",
            "float",
            "str",
            "nested_dict",
            "nested_list",
            "tuple_to_list",
            "set_to_list",
            "frozenset_to_list",
            "int_dict_key_to_str",
        ],
    )
    def test_happy_path(
        self,
        input_value: Any,
        check: Any,
    ) -> None:
        result = assert_to_jsonable(input_value)
        assert check(result)

    def test_lm_logging_shaped_payload(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        assert assert_to_jsonable(payload) == payload


class TestDomainTransforms:
    def test_dspy_example(self) -> None:
        result = assert_to_jsonable(minimal_example())
        assert result == {"question": "q", "answer": "a"}

    def test_dspy_signature_type(self) -> None:
        result = assert_to_jsonable(QASig)
        assert set(result) == {"signature", "instructions", "fields"}
        assert isinstance(result["fields"], list)
        assert all(isinstance(field, tuple) for field in result["fields"])

    @pytest.mark.parametrize(
        ("type_value", "expected_substring"),
        [
            (int, "int"),
            (SimpleObject, "SimpleObject"),
        ],
        ids=["builtin_type", "local_class"],
    )
    def test_plain_type(
        self,
        type_value: type,
        expected_substring: str,
    ) -> None:
        result = assert_to_jsonable(type_value)
        assert isinstance(result, str)
        assert result.startswith("<class ")
        assert expected_substring in result

    def test_dspy_base_lm(self) -> None:
        lm = stub_lm(api_key="secret", temperature=0.7)
        result = assert_to_jsonable(lm)
        assert result["_kind"] == "BaseLM"
        assert result["class"] == "dspy.utils.dummies.DummyLM"
        assert result["model"] == "dummy"
        assert result["kwargs"] == {
            "api_key": "<redacted>",
            "temperature": 0.7,
        }

    def test_pydantic_model(self) -> None:
        model = ok_pydantic_model()
        assert assert_to_jsonable(model) == model.model_dump(mode="json")

    def test_pydantic_precedence_over_object_vars(self) -> None:
        model = SerializedNameModel(name="n")
        result = assert_to_jsonable(model)
        assert result == model.model_dump(mode="json")
        assert result["name"] == "N"
        assert vars(model)["name"] == "n"

    def test_bytes(self) -> None:
        assert assert_to_jsonable(b"hello") == "<bytes len=5>"

    def test_generator(self) -> None:
        def gen() -> Any:
            yield 1

        result = assert_to_jsonable(gen())
        assert result == "<generator>"

    def test_coroutine(self) -> None:
        async def coro() -> None:
            return None

        with pytest.warns(
            RuntimeWarning,
            match="coroutine .* was never awaited",
        ):
            result = assert_to_jsonable(coro())
        assert result == "<coroutine>"

    def test_simple_object_vars(self) -> None:
        result = assert_to_jsonable(SimpleObject())
        assert result == {"a": 1, "label": "test"}


class TestGuardrails:
    def test_max_depth_exceeded(self) -> None:
        with pytest.raises(MaxDepthExceededError) as exc_info:
            to_jsonable(nested_list(101))
        exc = exc_info.value
        assert exc.depth == 101
        assert exc.max_depth == MAX_JSONABLE_DEPTH
        assert_diagnostics(
            exc,
            {"path", "detail", "depth", "max_depth", "value_preview"},
            depth=101,
            max_depth=MAX_JSONABLE_DEPTH,
        )

    def test_max_depth_nested_path(self) -> None:
        payload = {"outer": {"inner": nested_list(101)}}
        with pytest.raises(MaxDepthExceededError) as exc_info:
            to_jsonable(payload)
        path = exc_info.value.path
        assert path[0] == "outer"
        assert path[1] == "inner"

    def test_payload_too_large(self) -> None:
        with pytest.raises(PayloadTooLargeError) as exc_info:
            to_jsonable(large_payload(500), max_bytes=100)
        exc = exc_info.value
        assert exc.size_bytes > exc.max_bytes
        assert exc.max_bytes == 100
        assert exc.postgres_max_bytes == POSTGRES_JSONB_MAX_BYTES
        assert "blob" in exc.top_level_sizes
        assert exc.preview_head
        assert_diagnostics(
            exc,
            {
                "path",
                "detail",
                "size_bytes",
                "max_bytes",
                "postgres_max_bytes",
                "top_level_sizes",
                "preview_head",
                "preview_tail",
            },
            max_bytes=100,
            postgres_max_bytes=POSTGRES_JSONB_MAX_BYTES,
        )

    def test_json_encode_error(self) -> None:
        with pytest.raises(JsonEncodeError) as exc_info:
            to_jsonable({"bad": object()})
        exc = exc_info.value
        assert exc.type_name == "object"
        assert exc.path == ("bad",)
        assert_diagnostics(
            exc,
            {"path", "detail", "type_name", "value_preview", "underlying"},
            path=["bad"],
            type_name="object",
        )


class TestStructuredErrors:
    def test_example_serialization_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        example = minimal_example()

        def boom(self: dspy.Example) -> dict[str, Any]:
            raise RuntimeError("toDict failed")

        monkeypatch.setattr(dspy.Example, "toDict", boom)
        with pytest.raises(ExampleSerializationError) as exc_info:
            to_jsonable(example)
        assert_diagnostics(
            exc_info.value,
            {"path", "detail", "value_preview", "underlying"},
        )

    def test_model_dump_error(self) -> None:
        with pytest.raises(ModelDumpError) as exc_info:
            to_jsonable(bad_pydantic_model())
        assert_diagnostics(
            exc_info.value,
            {"path", "detail", "value_preview", "underlying"},
        )

    def test_object_vars_serialization_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = SimpleObject()
        original = serialization._to_jsonable_inner

        def patched(
            x: Any,
            depth: int = 0,
            path: tuple[str | int, ...] = (),
        ) -> Any:
            if path == ("label",):
                raise RuntimeError("vars walk failed")
            return original(x, depth, path)

        monkeypatch.setattr(serialization, "_to_jsonable_inner", patched)
        with pytest.raises(ObjectVarsSerializationError) as exc_info:
            to_jsonable(target)
        assert_diagnostics(
            exc_info.value,
            {"path", "detail", "value_preview", "underlying"},
        )

    def test_signature_summary_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class BadFields:
            def items(self) -> Any:
                raise RuntimeError("fields broke")

        class FakeSig:
            fields = BadFields()
            signature = "question -> answer"
            instructions = "test instructions"

        original = serialization._signature_summary

        def intercept(
            sig_cls: type[dspy.Signature],
            path: tuple[str | int, ...],
        ) -> Any:
            if sig_cls is QASig:
                fake_sig = cast(type[dspy.Signature], cast(Any, FakeSig))
                return original(fake_sig, path)
            return original(sig_cls, path)

        monkeypatch.setattr(serialization, "_signature_summary", intercept)
        with pytest.raises(SignatureSummaryError) as exc_info:
            to_jsonable(QASig)
        assert_diagnostics(
            exc_info.value,
            {"path", "detail", "value_preview", "underlying"},
        )

    def test_all_serialization_errors_implement_diagnostics(self) -> None:
        """Smoke: concrete subclasses return diagnostics without raising."""
        triggers: list[
            tuple[type[SerializationError], SerializationError]
        ] = []

        with pytest.raises(MaxDepthExceededError) as exc_info:
            to_jsonable(nested_list(101))
        triggers.append((MaxDepthExceededError, exc_info.value))

        with pytest.raises(JsonEncodeError) as exc_info:
            to_jsonable({"bad": object()})
        triggers.append((JsonEncodeError, exc_info.value))

        with pytest.raises(PayloadTooLargeError) as exc_info:
            to_jsonable(large_payload(500), max_bytes=100)
        triggers.append((PayloadTooLargeError, exc_info.value))

        with pytest.raises(ModelDumpError) as exc_info:
            to_jsonable(BadModel(x=object()))
        triggers.append((ModelDumpError, exc_info.value))

        for exc_type, exc in triggers:
            diag = exc.diagnostics()
            assert isinstance(diag, dict)
            assert "path" in diag
            assert issubclass(type(exc), exc_type)
