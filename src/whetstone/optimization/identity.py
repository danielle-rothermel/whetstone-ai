"""Shared identity, reference, and validation helpers for the harness.

The durable Optimization Step protocol distinguishes two hashes throughout,
exactly as the vocabulary does:

* an **Identity Hash** — the full 64-character lowercase SHA-256 digest of
  Canonical Identity JSON derived from a versioned Identity Document. Config
  and Definition identities (Optimization Run/Step/Config, Tool Config, Tool
  Definition, Eval Config, candidate, Reward Policy) are addressed by Identity
  Hash.

* a **Content Hash** — the full 64-character lowercase SHA-256 digest of a
  complete canonical persisted record. Stored objects (Step Request, Step
  Result, checkpointed proposal output, state/history snapshots, Tool Results,
  evaluation evidence) are addressed by a typed
  :class:`~dr_store.ObjectReference` plus Content Hash.

A :class:`TypedRef` pairs a stored object's typed Object Reference with its
Content Hash so a Step Request or Step Result can name a prior object without
carrying its body. Every ``TypedRef.content_hash`` equals its
``reference.content_hash``; the two are never allowed to disagree.
"""

from __future__ import annotations

from typing import Any

from dr_serialize import (
    StrictJsonError,
    build_identity_document,
    identity_document_hash,
    validate_strict_json,
)
from dr_store import ObjectReference
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

__all__ = [
    "TypedRef",
    "compute_identity_hash",
    "reject_non_json",
    "require_full_hash",
    "typed_ref_for_record",
]


def reject_non_json(value: Any, *, field: str) -> Any:
    """Require a value to be strict finite JSON.

    This is the seam that rejects a mutable process object — a live client, an
    open connection, an executable closure, or a Runtime Tool Handle — from a
    free-form Step Request field (``pools``, ``hyperparameters``, ``args``,
    ``payload``). A non-JSON or non-finite value fails loudly at construction
    rather than at a later canonicalization the caller cannot see.
    """
    try:
        validate_strict_json(value)
    except StrictJsonError as exc:
        raise ValueError(
            f"{field} must be strict finite JSON (no runtime handles, "
            f"clients, connections, or closures): {exc}"
        ) from exc
    return value

_HEX = frozenset("0123456789abcdef")


def require_full_hash(value: str, *, field: str) -> str:
    """Require a full 64-char lowercase SHA-256 hex digest."""
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(
            f"{field} must be a full 64-char lowercase SHA-256 hash, "
            f"got {value!r}"
        )
    return value


def compute_identity_hash(
    *, schema: str, schema_version: int, payload: Any
) -> str:
    """Compute the Identity Hash of a versioned identity payload.

    A thin, uniform wrapper over dr-serialize's
    :func:`build_identity_document` + :func:`identity_hash` so every harness
    identity is derived through the one canonical lane.
    """
    document = build_identity_document(
        schema=schema, schema_version=schema_version, payload=payload
    )
    return identity_document_hash(document)


class TypedRef(BaseModel):
    """A typed Object Reference plus its Content Hash.

    This is the immutable-reference primitive the design requires everywhere a
    Step Request or Step Result names a prior stored object: the ``schema``
    plus ``content_hash`` are the typed :class:`~dr_store.ObjectReference`, and
    ``content_hash`` is repeated as a first-class field so the pairing is
    explicit and self-describing in canonical JSON. The two can never diverge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: StrictStr
    content_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> TypedRef:
        if not self.schema_name:
            raise ValueError("schema_name must be non-empty")
        require_full_hash(self.content_hash, field="content_hash")
        return self

    @property
    def reference(self) -> ObjectReference:
        """The typed dr-store Object Reference this pair denotes."""
        return ObjectReference(
            schema=self.schema_name, content_hash=self.content_hash
        )


def typed_ref_for_record(schema: str, record: Any) -> TypedRef:
    """Build the :class:`TypedRef` a record resolves under (Content Hash)."""
    reference = ObjectReference.for_record(schema, record)
    return TypedRef(
        schema_name=reference.schema, content_hash=reference.content_hash
    )
