from __future__ import annotations

from whetstone.evaluation import (
    Applicability,
    CompressionReferenceArtifact,
    MetricFact,
    OperatorLineage,
    Score,
    compression_ratio,
    zstd_compressed_utf8_byte_length,
)

_BYTES_UNIT = "bytes"
_RATIO_UNIT = "ratio"


def utf8_description_length_fact(
    name: str,
    text: str,
    *,
    lineage: OperatorLineage,
) -> MetricFact:
    """Return one lineage-bearing description-length fact for UTF-8 text."""

    return MetricFact(
        name=name,
        value=zstd_compressed_utf8_byte_length(text),
        unit=_BYTES_UNIT,
        applicability=Applicability.APPLICABLE,
        lineage=lineage,
    )


def compression_ratio_from_bytes(
    *,
    numerator_bytes: int,
    reference: CompressionReferenceArtifact,
) -> float | None:
    """Return the compression ratio, or ``None`` for a zero denominator."""

    return compression_ratio(
        numerator_bytes=numerator_bytes,
        reference=reference,
    )


def compression_ratio_score_from_bytes(
    *,
    name: str,
    numerator_bytes: int,
    reference: CompressionReferenceArtifact,
    evaluation_procedure_config_hash: str,
    derived_from: tuple[str, ...],
) -> Score | None:
    """Derive a compression-ratio score, or ``None`` if denominator is zero."""

    ratio = compression_ratio_from_bytes(
        numerator_bytes=numerator_bytes,
        reference=reference,
    )
    if ratio is None:
        return None
    return Score(
        name=name,
        value=ratio,
        unit=_RATIO_UNIT,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        derived_from=derived_from,
    )


__all__ = [
    "compression_ratio_from_bytes",
    "compression_ratio_score_from_bytes",
    "utf8_description_length_fact",
]
