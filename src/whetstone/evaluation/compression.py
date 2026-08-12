"""Typed compression references and ratio calculation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Self

import zstandard
from pydantic import field_serializer, field_validator

from whetstone.evaluation.config import (
    SCHEMA_COMPRESSION_REFERENCE_ARTIFACT,
    SCHEMA_COMPRESSION_REFERENCE_KEY,
    _FrozenModel,
    identity_hash_for,
)

#: The pinned zstd level for description-length measurement. Fixed by the
#: experiment; a level change is a deliberate breaking measurement change.
ZSTD_LEVEL = 19


class CompressionReferenceKey(_FrozenModel):
    namespace: str
    name: str

    @field_validator("namespace", "name")
    @classmethod
    def reject_empty(cls, value: str) -> str:
        if not value:
            raise ValueError(
                "compression reference key parts must be non-empty"
            )
        return value

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_COMPRESSION_REFERENCE_KEY,
            payload={"namespace": self.namespace, "name": self.name},
        )


class CompressionReferenceArtifact(_FrozenModel):
    content: bytes

    @field_serializer("content")
    def serialize_content(self, content: bytes) -> str:
        return base64.b64encode(content).decode("ascii")

    @field_validator("content", mode="before")
    @classmethod
    def accept_base64(cls, value: object) -> object:
        if isinstance(value, str):
            return base64.b64decode(value.encode("ascii"), validate=True)
        return value

    @property
    def byte_length(self) -> int:
        return len(self.content)

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_COMPRESSION_REFERENCE_ARTIFACT,
            payload={
                "byte_length": self.byte_length,
                "content_b64": base64.b64encode(self.content).decode("ascii"),
            },
        )


class ReferenceResolutionError(KeyError):
    """A compression-reference key has no bound artifact."""


@dataclass(frozen=True, slots=True)
class CompressionReferenceResolver:
    bindings: tuple[
        tuple[CompressionReferenceKey, CompressionReferenceArtifact], ...
    ] = ()

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[CompressionReferenceKey, CompressionReferenceArtifact],
    ) -> Self:
        return cls(bindings=tuple(mapping.items()))

    def resolve(
        self, key: CompressionReferenceKey
    ) -> CompressionReferenceArtifact:
        for bound_key, artifact in self.bindings:
            if bound_key == key:
                return artifact
        raise ReferenceResolutionError(
            f"no compression reference bound for {key.namespace}/{key.name}"
        )


ZERO_DENOMINATOR: None = None


def compression_ratio(
    *,
    numerator_bytes: int,
    reference: CompressionReferenceArtifact,
) -> float | None:
    if numerator_bytes < 0:
        raise ValueError("numerator_bytes must be non-negative")
    if reference.byte_length == 0:
        return ZERO_DENOMINATOR
    return numerator_bytes / reference.byte_length


def zstd_compressed_utf8_byte_length(
    text: str,
    *,
    level: int = ZSTD_LEVEL,
) -> int:
    """Return the zstd-compressed byte length of UTF-8-encoded text.

    Pure and deterministic given ``text`` and ``level``.
    """

    payload = text.encode("utf-8")
    compressed = zstandard.ZstdCompressor(level=level).compress(payload)
    return len(compressed)


__all__ = [
    "ZERO_DENOMINATOR",
    "ZSTD_LEVEL",
    "CompressionReferenceArtifact",
    "CompressionReferenceKey",
    "CompressionReferenceResolver",
    "ReferenceResolutionError",
    "compression_ratio",
    "zstd_compressed_utf8_byte_length",
]
