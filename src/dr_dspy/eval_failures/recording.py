from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from dr_dspy.eval_failures.exceptions import (
    EvalFailureError,
    RecordingFailureError,
)
from dr_dspy.serialization import (
    PAYLOAD_MAX_BYTES,
    SerializationError,
    to_jsonable,
)


def ensure_recordable(
    value: Any,
    *,
    max_bytes: int = PAYLOAD_MAX_BYTES,
) -> Any:
    """Shared path for all storable JSON/JSONB values."""
    try:
        return to_jsonable(value, max_bytes=max_bytes)
    except SerializationError as exc:
        raise RecordingFailureError(str(exc), underlying=exc) from exc


def recordable_jsonb(
    value: Any,
    *,
    max_bytes: int = PAYLOAD_MAX_BYTES,
) -> Jsonb:
    return Jsonb(ensure_recordable(value, max_bytes=max_bytes))


def failure_metadata_from_exception(error: BaseException) -> dict[str, Any]:
    """Extract SerializationError diagnostics or EvalFailureError metadata."""
    current: BaseException | None = error
    seen: set[int] = set()
    eval_failure_metadata: dict[str, Any] | None = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SerializationError):
            return current.diagnostics()
        if (
            isinstance(current, EvalFailureError)
            and eval_failure_metadata is None
        ):
            if current.metadata:
                eval_failure_metadata = dict(current.metadata)
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if current.__context__ is not None:
            current = current.__context__
            continue
        underlying = getattr(current, "underlying", None)
        if isinstance(underlying, BaseException):
            current = underlying
            continue
        break
    return eval_failure_metadata or {}
