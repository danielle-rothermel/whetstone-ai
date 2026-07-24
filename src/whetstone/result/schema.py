"""Shared record-schema constants and validators for the result package.

A Rollout Result is stored by typed Object Reference plus Content Hash. The
schema name below is a dr-store *record* schema (for content addressing), not
a dr-serialize Identity Document schema: a Rollout Result has a Content Hash,
never an Identity Hash.
"""

from __future__ import annotations

# dr-store record schema for the terminal Rollout Result Object.
ROLLOUT_RESULT_SCHEMA = "whetstone.rollout_result"

_HEX = frozenset("0123456789abcdef")


def require_full_hash(value: str, *, field: str) -> str:
    """Require a full 64-char lowercase SHA-256 hex digest."""
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(
            f"{field} must be a full 64-char lowercase SHA-256 hash, "
            f"got {value!r}"
        )
    return value


__all__ = ["ROLLOUT_RESULT_SCHEMA", "require_full_hash"]
