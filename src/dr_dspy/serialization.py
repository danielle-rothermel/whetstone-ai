"""DSPy-aware serialization helpers for experiment telemetry."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import pydantic

import dspy

# PostgreSQL jsonb/text per-value maximum (~1 GiB; see PG MaxAllocSize).
POSTGRES_JSONB_MAX_BYTES = 1 << 30  # 1_073_741_824

# Empirical headroom for compact JSON text → jsonb binary expansion.
# PG does not define this; 25% is a conservative upper bound for structured
# telemetry JSON (see depesz.com JSON vs JSONB sizing benchmarks).
POSTGRES_JSONB_TEXT_TO_BINARY_OVERHEAD_RATIO = 0.25

# Practical JSON nesting guard. PG has no fixed cap, but 100 matches common
# JSON storage limits and catches runaway recursion before insert.
POSTGRES_JSON_MAX_DEPTH = 100

PAYLOAD_MAX_BYTES = POSTGRES_JSONB_MAX_BYTES - int(
    POSTGRES_JSONB_MAX_BYTES * POSTGRES_JSONB_TEXT_TO_BINARY_OVERHEAD_RATIO
)  # 805_306_368 bytes (~768 MiB)

MAX_JSONABLE_DEPTH = POSTGRES_JSON_MAX_DEPTH
REPR_TRUNCATE = 4096
SANITIZE_KEYS = frozenset(
    {"api_key", "api_base", "base_url", "model_list", "authorization"}
)

type JsonableHandle = tuple[bool, Any]


class SerializationError(Exception):
    """Base for telemetry serialization failures."""


class MaxDepthExceededError(SerializationError):
    def __init__(
        self,
        *,
        depth: int,
        max_depth: int,
        value_preview: str,
    ) -> None:
        self.depth = depth
        self.max_depth = max_depth
        self.value_preview = value_preview
        super().__init__(
            f"serialization exceeded max depth {max_depth} at depth "
            f"{depth}: {value_preview}"
        )


class JsonEncodeError(SerializationError):
    def __init__(
        self,
        *,
        value_preview: str,
        underlying: TypeError,
    ) -> None:
        self.value_preview = value_preview
        self.underlying = underlying
        super().__init__(
            f"value is not JSON-serializable: {value_preview}"
        )


class PayloadTooLargeError(SerializationError):
    def __init__(
        self,
        *,
        size_bytes: int,
        max_bytes: int,
        postgres_max_bytes: int,
        preview: str,
    ) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        self.postgres_max_bytes = postgres_max_bytes
        self.preview = preview
        super().__init__(
            f"serialized payload size {size_bytes} bytes exceeds limit "
            f"{max_bytes} bytes (postgres jsonb ceiling "
            f"{postgres_max_bytes} bytes): {preview}"
        )


def sanitize_lm_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip credential-like keys from an LM kwargs dict before logging."""
    if not kwargs:
        return {}
    return {
        k: ("<redacted>" if k.lower() in SANITIZE_KEYS else v)
        for k, v in kwargs.items()
    }


def _truncate_repr(x: Any) -> str:
    """Truncated repr for error messages only."""
    return repr(x)[:REPR_TRUNCATE]


def _signature_summary(sig_cls: type[dspy.Signature]) -> dict[str, Any]:
    """Summarize a Signature class for logging."""
    try:
        fields_summary = [
            (
                name,
                str(field.annotation),
                (field.json_schema_extra or {}).get("__dspy_field_type")
                if isinstance(field.json_schema_extra, dict)
                else None,
            )
            for name, field in sig_cls.fields.items()
        ]
    except Exception:
        fields_summary = []
    return {
        "signature": getattr(sig_cls, "signature", repr(sig_cls)),
        "instructions": getattr(sig_cls, "instructions", ""),
        "fields": fields_summary,
    }


def _check_max_depth(x: Any, depth: int) -> None:
    if depth > MAX_JSONABLE_DEPTH:
        raise MaxDepthExceededError(
            depth=depth,
            max_depth=MAX_JSONABLE_DEPTH,
            value_preview=_truncate_repr(x),
        )


def _jsonable_scalar(x: Any, depth: int) -> JsonableHandle:
    del depth
    if x is None or isinstance(x, (bool, int, float, str)):
        return True, x
    return False, None


def _jsonable_sequence(x: Any, depth: int) -> JsonableHandle:
    if isinstance(x, (list, tuple, set, frozenset)):
        return True, [_to_jsonable_inner(v, depth + 1) for v in x]
    return False, None


def _jsonable_mapping(x: Any, depth: int) -> JsonableHandle:
    if isinstance(x, dict):
        return True, {
            str(k): _to_jsonable_inner(v, depth + 1) for k, v in x.items()
        }
    return False, None


def _jsonable_bytes(x: Any, depth: int) -> JsonableHandle:
    del depth
    if isinstance(x, bytes):
        return True, f"<bytes len={len(x)}>"
    return False, None


def _jsonable_dspy_example(x: Any, depth: int) -> JsonableHandle:
    if isinstance(x, dspy.Example):
        try:
            return True, _to_jsonable_inner(x.toDict(), depth + 1)
        except Exception:
            return True, _truncate_repr(x)
    return False, None


def _jsonable_type(x: Any, depth: int) -> JsonableHandle:
    del depth
    if not isinstance(x, type):
        return False, None
    try:
        if issubclass(x, dspy.Signature):
            return True, _signature_summary(x)
    except TypeError:
        pass
    return True, f"<class {x.__module__}.{x.__name__}>"


def _jsonable_dspy_lm(x: Any, depth: int) -> JsonableHandle:
    del depth
    if isinstance(x, dspy.BaseLM):
        return True, {
            "_kind": "BaseLM",
            "class": f"{type(x).__module__}.{type(x).__name__}",
            "model": getattr(x, "model", None),
            "kwargs": sanitize_lm_kwargs(getattr(x, "kwargs", {})),
        }
    return False, None


def _jsonable_pydantic_model(x: Any, depth: int) -> JsonableHandle:
    del depth
    if isinstance(x, pydantic.BaseModel):
        try:
            return True, x.model_dump(mode="json")
        except Exception:
            return True, _truncate_repr(x)
    return False, None


def _jsonable_async_or_generator(x: Any, depth: int) -> JsonableHandle:
    del depth
    if (
        inspect.iscoroutine(x)
        or inspect.isasyncgen(x)
        or inspect.isgenerator(x)
    ):
        return True, f"<{type(x).__name__}>"
    return False, None


def _jsonable_object_vars(x: Any, depth: int) -> JsonableHandle:
    if hasattr(x, "__dict__") and not callable(x):
        try:
            return True, {
                k: _to_jsonable_inner(v, depth + 1)
                for k, v in vars(x).items()
            }
        except Exception:
            return True, _truncate_repr(x)
    return False, None


_HANDLERS: tuple[Callable[[Any, int], JsonableHandle], ...] = (
    _jsonable_scalar,
    _jsonable_sequence,
    _jsonable_mapping,
    _jsonable_bytes,
    _jsonable_dspy_example,
    _jsonable_type,
    _jsonable_dspy_lm,
    _jsonable_pydantic_model,
    _jsonable_async_or_generator,
    _jsonable_object_vars,
)


def _to_jsonable_inner(x: Any, depth: int = 0) -> Any:
    """Recursive, depth-bounded worker for to_jsonable."""
    _check_max_depth(x, depth)
    for handler in _HANDLERS:
        handled, value = handler(x, depth)
        if handled:
            return value
    return x


def to_jsonable(x: Any, *, max_bytes: int = PAYLOAD_MAX_BYTES) -> Any:
    value = _to_jsonable_inner(x)
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except TypeError as e:
        raise JsonEncodeError(
            value_preview=_truncate_repr(x),
            underlying=e,
        ) from e
    size_bytes = len(encoded.encode("utf-8"))
    if size_bytes > max_bytes:
        raise PayloadTooLargeError(
            size_bytes=size_bytes,
            max_bytes=max_bytes,
            postgres_max_bytes=POSTGRES_JSONB_MAX_BYTES,
            preview=encoded[:REPR_TRUNCATE],
        )
    return value
