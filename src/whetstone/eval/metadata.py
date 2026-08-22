from __future__ import annotations

from whetstone.core.identity import ImmutableJsonObject

PURPOSE_METADATA_KEY = "purpose"
TASK_IDS_METADATA_KEY = "task_ids"

__all__ = [
    "PURPOSE_METADATA_KEY",
    "TASK_IDS_METADATA_KEY",
    "eval_purpose",
    "eval_task_ids",
    "metadata_with_purpose",
]


def eval_purpose(metadata: ImmutableJsonObject) -> str | None:
    value = metadata.get(PURPOSE_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{PURPOSE_METADATA_KEY!r} metadata must be a string")
    if not value.strip():
        raise ValueError(f"{PURPOSE_METADATA_KEY!r} metadata must be non-empty")
    return value


def eval_task_ids(metadata: ImmutableJsonObject) -> tuple[str, ...] | None:
    """Per-intent task set, when the intent is not the full engine sample.

    Absent means "use the bound engine's tasks" (COPRO). Present means the
    caller already scoped the work — fan-out and INLINE evaluate must honor
    that set rather than expand to every engine task.
    """
    value = metadata.get(TASK_IDS_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{TASK_IDS_METADATA_KEY!r} metadata must be a list")
    task_ids = tuple(item for item in value)
    if any(not isinstance(item, str) or not item for item in task_ids):
        raise ValueError(
            f"{TASK_IDS_METADATA_KEY!r} metadata must be non-empty strings"
        )
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{TASK_IDS_METADATA_KEY!r} metadata must be unique")
    return task_ids


def metadata_with_purpose(purpose: str, **extra: str) -> ImmutableJsonObject:
    if not purpose.strip():
        raise ValueError("purpose must be non-empty")
    payload: dict[str, str] = {PURPOSE_METADATA_KEY: purpose, **extra}
    return ImmutableJsonObject(payload)
