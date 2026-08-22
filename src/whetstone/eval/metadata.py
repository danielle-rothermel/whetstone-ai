from __future__ import annotations

from whetstone.core.identity import ImmutableJsonObject

PURPOSE_METADATA_KEY = "purpose"

__all__ = [
    "PURPOSE_METADATA_KEY",
    "eval_purpose",
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


def metadata_with_purpose(purpose: str, **extra: str) -> ImmutableJsonObject:
    if not purpose.strip():
        raise ValueError("purpose must be non-empty")
    payload: dict[str, str] = {PURPOSE_METADATA_KEY: purpose, **extra}
    return ImmutableJsonObject(payload)
