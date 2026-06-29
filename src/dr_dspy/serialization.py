"""DSPy-aware serialization helpers for experiment telemetry.

Deferred: masking-site re-raises; pytest suite; failure propagation in
lm_logging/DB writes; persist exc.diagnostics() to metadata; classify as
RecordingFailureError in failures/.
"""

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
MESSAGE_PREVIEW = 512
DEBUG_DETAIL_LIMIT = 256 * 1024
ENCODED_PREVIEW_SLICE = 8192
SANITIZE_KEYS = frozenset(
    {"api_key", "api_base", "base_url", "model_list", "authorization"}
)

type JsonPath = tuple[str | int, ...]
type JsonableHandle = tuple[bool, Any]

_JSON_LEAF_TYPES = (type(None), bool, int, float, str)
_JSON_CONTAINER_TYPES = (*_JSON_LEAF_TYPES, dict, list)


class SerializationError(Exception):
    """Base for telemetry serialization failures."""

    path: JsonPath
    detail: str

    def diagnostics(self) -> dict[str, Any]:
        raise NotImplementedError


class MaxDepthExceededError(SerializationError):
    def __init__(
        self,
        *,
        depth: int,
        max_depth: int,
        path: JsonPath,
        value_preview: str,
        detail: str,
    ) -> None:
        self.depth = depth
        self.max_depth = max_depth
        self.path = path
        self.value_preview = value_preview
        self.detail = detail
        super().__init__(
            f"max depth {max_depth} exceeded at depth {depth} "
            f"path {path!r}"
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "detail": self.detail,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "value_preview": self.value_preview,
        }


class JsonEncodeError(SerializationError):
    def __init__(
        self,
        *,
        path: JsonPath,
        type_name: str,
        detail: str,
        underlying: TypeError,
        value_preview: str,
    ) -> None:
        self.path = path
        self.type_name = type_name
        self.detail = detail
        self.underlying = underlying
        self.value_preview = value_preview
        super().__init__(
            f"not JSON-serializable at path {path!r} type {type_name}"
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "detail": self.detail,
            "type_name": self.type_name,
            "value_preview": self.value_preview,
            "underlying": repr(self.underlying),
        }


class PayloadTooLargeError(SerializationError):
    def __init__(
        self,
        *,
        size_bytes: int,
        max_bytes: int,
        postgres_max_bytes: int,
        path: JsonPath,
        top_level_sizes: dict[str, int],
        preview_head: str,
        preview_tail: str,
        detail: str,
    ) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        self.postgres_max_bytes = postgres_max_bytes
        self.path = path
        self.top_level_sizes = top_level_sizes
        self.preview_head = preview_head
        self.preview_tail = preview_tail
        self.detail = detail
        sizes_summary = _format_top_level_sizes(top_level_sizes)
        sizes_part = f" top keys: {sizes_summary}" if sizes_summary else ""
        super().__init__(
            f"payload {size_bytes} bytes exceeds limit {max_bytes} "
            f"at path {path!r}{sizes_part}"
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "detail": self.detail,
            "size_bytes": self.size_bytes,
            "max_bytes": self.max_bytes,
            "postgres_max_bytes": self.postgres_max_bytes,
            "top_level_sizes": self.top_level_sizes,
            "preview_head": self.preview_head,
            "preview_tail": self.preview_tail,
        }


def sanitize_lm_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip credential-like keys from an LM kwargs dict before logging."""
    if not kwargs:
        return {}
    return {
        k: ("<redacted>" if k.lower() in SANITIZE_KEYS else v)
        for k, v in kwargs.items()
    }


def _preview_repr(x: Any) -> str:
    return repr(x)[:MESSAGE_PREVIEW]


def _detail_repr(x: Any) -> str:
    return repr(x)[:DEBUG_DETAIL_LIMIT]


def _format_top_level_sizes(
    sizes: dict[str, int],
    *,
    limit: int = 10,
) -> str:
    items = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return ", ".join(f"{key}={size}" for key, size in items)


def _encoded_preview_slices(encoded: str) -> tuple[str, str, str]:
    head = encoded[:ENCODED_PREVIEW_SLICE]
    if len(encoded) > ENCODED_PREVIEW_SLICE:
        tail = encoded[-ENCODED_PREVIEW_SLICE:]
    else:
        tail = ""
    detail = f"head:\n{head}"
    if tail:
        detail = f"{detail}\n\ntail:\n{tail}"
    if len(detail) > DEBUG_DETAIL_LIMIT:
        detail = detail[:DEBUG_DETAIL_LIMIT]
    return head, tail, detail


def _top_level_key_sizes(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): len(json.dumps(item, ensure_ascii=False).encode())
        for key, item in value.items()
    }


def _find_non_jsonable_path(
    value: Any,
    path: JsonPath = (),
) -> tuple[JsonPath, Any]:
    if isinstance(value, _JSON_LEAF_TYPES):
        return path, value
    if isinstance(value, dict):
        for key, item in value.items():
            sub_path = (*path, str(key))
            if not isinstance(item, _JSON_CONTAINER_TYPES):
                return sub_path, item
            found_path, leaf = _find_non_jsonable_path(item, sub_path)
            if not isinstance(leaf, _JSON_LEAF_TYPES):
                return found_path, leaf
        return path, value
    if isinstance(value, list):
        for index, item in enumerate(value):
            sub_path = (*path, index)
            if not isinstance(item, _JSON_CONTAINER_TYPES):
                return sub_path, item
            found_path, leaf = _find_non_jsonable_path(item, sub_path)
            if not isinstance(leaf, _JSON_LEAF_TYPES):
                return found_path, leaf
        return path, value
    return path, value


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


def _check_max_depth(x: Any, depth: int, path: JsonPath) -> None:
    if depth > MAX_JSONABLE_DEPTH:
        raise MaxDepthExceededError(
            depth=depth,
            max_depth=MAX_JSONABLE_DEPTH,
            path=path,
            value_preview=_preview_repr(x),
            detail=_detail_repr(x),
        )


def _jsonable_scalar(x: Any, depth: int, path: JsonPath) -> JsonableHandle:
    del depth, path
    if x is None or isinstance(x, (bool, int, float, str)):
        return True, x
    return False, None


def _jsonable_sequence(x: Any, depth: int, path: JsonPath) -> JsonableHandle:
    if isinstance(x, (list, tuple, set, frozenset)):
        return True, [
            _to_jsonable_inner(item, depth + 1, (*path, index))
            for index, item in enumerate(x)
        ]
    return False, None


def _jsonable_mapping(x: Any, depth: int, path: JsonPath) -> JsonableHandle:
    if isinstance(x, dict):
        return True, {
            str(key): _to_jsonable_inner(item, depth + 1, (*path, str(key)))
            for key, item in x.items()
        }
    return False, None


def _jsonable_bytes(x: Any, depth: int, path: JsonPath) -> JsonableHandle:
    del depth, path
    if isinstance(x, bytes):
        return True, f"<bytes len={len(x)}>"
    return False, None


def _jsonable_dspy_example(
    x: Any, depth: int, path: JsonPath
) -> JsonableHandle:
    if isinstance(x, dspy.Example):
        try:
            return True, _to_jsonable_inner(x.toDict(), depth + 1, path)
        except Exception:
            return True, _preview_repr(x)
    return False, None


def _jsonable_type(x: Any, depth: int, path: JsonPath) -> JsonableHandle:
    del depth, path
    if not isinstance(x, type):
        return False, None
    try:
        if issubclass(x, dspy.Signature):
            return True, _signature_summary(x)
    except TypeError:
        pass
    return True, f"<class {x.__module__}.{x.__name__}>"


def _jsonable_dspy_lm(x: Any, depth: int, path: JsonPath) -> JsonableHandle:
    del depth, path
    if isinstance(x, dspy.BaseLM):
        return True, {
            "_kind": "BaseLM",
            "class": f"{type(x).__module__}.{type(x).__name__}",
            "model": getattr(x, "model", None),
            "kwargs": sanitize_lm_kwargs(getattr(x, "kwargs", {})),
        }
    return False, None


def _jsonable_pydantic_model(
    x: Any, depth: int, path: JsonPath
) -> JsonableHandle:
    del depth, path
    if isinstance(x, pydantic.BaseModel):
        try:
            return True, x.model_dump(mode="json")
        except Exception:
            return True, _preview_repr(x)
    return False, None


def _jsonable_async_or_generator(
    x: Any, depth: int, path: JsonPath
) -> JsonableHandle:
    del depth, path
    if (
        inspect.iscoroutine(x)
        or inspect.isasyncgen(x)
        or inspect.isgenerator(x)
    ):
        return True, f"<{type(x).__name__}>"
    return False, None


def _jsonable_object_vars(
    x: Any, depth: int, path: JsonPath
) -> JsonableHandle:
    if hasattr(x, "__dict__") and not callable(x):
        try:
            return True, {
                key: _to_jsonable_inner(value, depth + 1, (*path, key))
                for key, value in vars(x).items()
            }
        except Exception:
            return True, _preview_repr(x)
    return False, None


_HANDLERS: tuple[Callable[[Any, int, JsonPath], JsonableHandle], ...] = (
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


def _to_jsonable_inner(
    x: Any,
    depth: int = 0,
    path: JsonPath = (),
) -> Any:
    """Recursive, depth-bounded worker for to_jsonable."""
    _check_max_depth(x, depth, path)
    for handler in _HANDLERS:
        handled, value = handler(x, depth, path)
        if handled:
            return value
    return x


def to_jsonable(x: Any, *, max_bytes: int = PAYLOAD_MAX_BYTES) -> Any:
    value = _to_jsonable_inner(x)
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except TypeError as error:
        failure_path, leaf = _find_non_jsonable_path(value)
        type_name = type(leaf).__name__
        raise JsonEncodeError(
            path=failure_path,
            type_name=type_name,
            detail=_detail_repr(leaf),
            underlying=error,
            value_preview=_preview_repr(x),
        ) from error
    size_bytes = len(encoded.encode("utf-8"))
    if size_bytes > max_bytes:
        preview_head, preview_tail, detail = _encoded_preview_slices(encoded)
        raise PayloadTooLargeError(
            size_bytes=size_bytes,
            max_bytes=max_bytes,
            postgres_max_bytes=POSTGRES_JSONB_MAX_BYTES,
            path=(),
            top_level_sizes=_top_level_key_sizes(value),
            preview_head=preview_head,
            preview_tail=preview_tail,
            detail=detail,
        )
    return value
