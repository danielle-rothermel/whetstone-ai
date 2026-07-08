"""Recordability boundary for storable failure and telemetry payloads.

``recordable_jsonb`` is the one Postgres JSONB adapter exception in
``eval_failures``. Core classification and failure models should remain free of
database, DBOS, and v0 workflow imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg.types.json import Jsonb

    from whetstone.records import FailureMetadataPayload

from dr_serialize import (
    POSTGRES_JSONB_PAYLOAD_MAX_BYTES,
    SerializationError,
)

from whetstone.dspy_serialization import dspy_serializer
from whetstone.eval_failures.exceptions import (
    EvalFailureError,
    RecordingFailureError,
)


def ensure_recordable(
    value: Any,
    *,
    max_bytes: int = POSTGRES_JSONB_PAYLOAD_MAX_BYTES,
) -> Any:
    """Shared path for all storable JSON/JSONB values."""
    try:
        return dspy_serializer(max_bytes).to_jsonable(value)
    except SerializationError as exc:
        raise RecordingFailureError(str(exc), underlying=exc) from exc


def recordable_text(value: Any) -> str:
    """Convert a payload to canonical text for metrics recording."""
    if isinstance(value, str):
        return value
    from dr_serialize import canonical_json

    return canonical_json(ensure_recordable(value))


def recordable_jsonb(
    value: Any,
    *,
    max_bytes: int = POSTGRES_JSONB_PAYLOAD_MAX_BYTES,
) -> Jsonb:
    from psycopg.types.json import Jsonb

    return Jsonb(ensure_recordable(value, max_bytes=max_bytes))


def failure_metadata_from_exception(
    error: BaseException,
) -> FailureMetadataPayload:
    """Build a storable failure metadata record from any exception."""
    from whetstone.eval_failures.policy import summarize_exception
    from whetstone.records import FailureMetadataPayload as Payload

    summary = summarize_exception(error)
    return Payload(
        failure_class=summary.failure_class,
        error_type=summary.failure_exception_type,
        message=summary.message,
        metadata=summary.failure_metadata,
    )


def failure_metadata_dict_from_exception(
    error: BaseException,
) -> dict[str, Any]:
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
