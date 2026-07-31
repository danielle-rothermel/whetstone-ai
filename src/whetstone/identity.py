"""Canonical validation for Whetstone identity-bearing values."""

from __future__ import annotations

_LOWERCASE_HEX = frozenset("0123456789abcdef")


def require_full_hash(value: object, *, field: str) -> str:
    """Require a canonical lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise ValueError(
            f"{field} must be a full 64-char lowercase SHA-256 hash, "
            f"got {value!r}"
        )
    return value


__all__ = ["require_full_hash"]
