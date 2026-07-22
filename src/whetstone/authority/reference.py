"""Typed content-addressed reference used inside authority records.

Official records name prior stored objects (Rollout Results, aggregates,
selection evidence, Materialization Records) by their typed
:class:`~dr_store.ObjectReference` plus Content Hash. :class:`TypedContentRef`
is the immutable, self-describing pairing embedded in those records so a record
cites a stored object without carrying its body, and so ``schema`` +
``content_hash`` can never diverge.

This mirrors the harness ``TypedRef`` but lives in the authority package so the
authority record schemas depend only on the authority surface, not on the
optimizer harness.
"""

from __future__ import annotations

from dr_store import ObjectReference
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

__all__ = ["TypedContentRef"]

_HEX = frozenset("0123456789abcdef")


class TypedContentRef(BaseModel):
    """A typed Object Reference (``schema``) plus its Content Hash.

    Frozen and hashable so it can key convergence checks and be compared for
    equality when two mapping entries must agree on a shared aggregate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: StrictStr
    content_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> TypedContentRef:
        if not self.schema_name:
            raise ValueError("schema_name must be non-empty")
        value = self.content_hash
        if len(value) != 64 or any(char not in _HEX for char in value):
            raise ValueError(
                "content_hash must be a full 64-char lowercase SHA-256 hash, "
                f"got {value!r}"
            )
        return self

    @property
    def reference(self) -> ObjectReference:
        """The typed dr-store Object Reference this pair denotes."""
        return ObjectReference(
            schema=self.schema_name, content_hash=self.content_hash
        )

    @classmethod
    def from_reference(cls, reference: ObjectReference) -> TypedContentRef:
        """Build a :class:`TypedContentRef` from a dr-store reference."""
        return cls(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )
