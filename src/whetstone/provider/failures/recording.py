from __future__ import annotations

from typing import Any

from dr_serialize import (
    POSTGRES_JSONB_PAYLOAD_MAX_BYTES,
    SerializationError,
    Serializer,
    postgres_jsonb_limits,
)

from whetstone.provider.failures.exceptions import (
    EvalFailureError,
    RecordingFailureError,
)


def ensure_recordable(
    value: Any,
    *,
    max_bytes: int = POSTGRES_JSONB_PAYLOAD_MAX_BYTES,
) -> Any:
    try:
        return Serializer(limits=postgres_jsonb_limits(max_bytes)).to_jsonable(
            value
        )
    except SerializationError as exc:
        raise RecordingFailureError(str(exc), underlying=exc) from exc


def recordable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    from dr_serialize import canonical_json

    return canonical_json(value)


def failure_metadata_dict_from_exception(
    error: BaseException,
) -> dict[str, Any]:

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
