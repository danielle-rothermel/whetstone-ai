"""Recordability boundary for storable failure and telemetry payloads.

Core classification and failure models remain free of database and DBOS
workflow imports; these helpers convert arbitrary values into JSON-safe
payloads and extract diagnostics from exception chains.
"""

from __future__ import annotations

from typing import Any

from dr_serialize import (
    POSTGRES_JSONB_PAYLOAD_MAX_BYTES,
    SerializationError,
    Serializer,
    postgres_jsonb_limits,
)

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
        return Serializer(
            limits=postgres_jsonb_limits(max_bytes)
        ).to_jsonable(value)
    except SerializationError as exc:
        raise RecordingFailureError(str(exc), underlying=exc) from exc


def recordable_text(value: Any) -> str:
    """Convert a payload to canonical text for metrics recording."""
    if isinstance(value, str):
        return value
    from dr_serialize import canonical_json

    return canonical_json(value)


def failure_metadata_dict_from_exception(
    error: BaseException,
) -> dict[str, Any]:
    """Extract SerializationError diagnostics or EvalFailureError metadata."""
    # Exceptions can carry several links at once (a raise inside an
    # except block sets __context__ even when an ``underlying`` was
    # attached), so walk all of them depth-first, __cause__ subtree
    # before __context__ before ``underlying``.
    stack: list[BaseException] = [error]
    seen: set[int] = set()
    eval_failure_metadata: dict[str, Any] | None = None
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, SerializationError):
            return current.diagnostics()
        if (
            isinstance(current, EvalFailureError)
            and eval_failure_metadata is None
            and current.metadata
        ):
            eval_failure_metadata = dict(current.metadata)
        underlying = getattr(current, "underlying", None)
        for link in (underlying, current.__context__, current.__cause__):
            if isinstance(link, BaseException):
                stack.append(link)
    return eval_failure_metadata or {}
