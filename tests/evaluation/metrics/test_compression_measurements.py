from __future__ import annotations

from whetstone.evaluation import (
    Applicability,
    CompressionReferenceArtifact,
)
from whetstone.evaluation.compression import zstd_compressed_utf8_byte_length
from whetstone.evaluation.metrics.compression_measurements import (
    compression_ratio_from_bytes,
    compression_ratio_score_from_bytes,
    utf8_description_length_fact,
)

from ..code.support import FULL_HASH, operator_lineage

ENCODER_TEXT = "def f(x):\n    return x + 1\n" * 8
DESCRIPTION_LENGTH_NAME = "compressed_description_length"
COMPRESSION_RATIO_NAME = "compression_ratio"


def test_description_length_fact_uses_generic_zstd_measurement() -> None:
    fact = utf8_description_length_fact(
        DESCRIPTION_LENGTH_NAME,
        ENCODER_TEXT,
        lineage=operator_lineage(),
    )
    assert fact.name == DESCRIPTION_LENGTH_NAME
    assert fact.value == zstd_compressed_utf8_byte_length(ENCODER_TEXT)
    assert fact.unit == "bytes"
    assert fact.applicability is Applicability.APPLICABLE
    assert fact.lineage.operator == "compressed_length"


def test_compression_ratio_over_nonzero_reference() -> None:
    reference = CompressionReferenceArtifact(content=b"abcdefghij")
    ratio = compression_ratio_from_bytes(
        numerator_bytes=5, reference=reference
    )
    assert ratio == 0.5


def test_compression_ratio_score_has_lineage() -> None:
    reference = CompressionReferenceArtifact(content=b"abcdefghij")
    score = compression_ratio_score_from_bytes(
        name=COMPRESSION_RATIO_NAME,
        numerator_bytes=5,
        reference=reference,
        evaluation_procedure_config_hash=FULL_HASH,
        derived_from=(DESCRIPTION_LENGTH_NAME,),
    )
    assert score is not None
    assert score.name == COMPRESSION_RATIO_NAME
    assert score.value == 0.5
    assert score.unit == "ratio"
    assert score.derived_from == (DESCRIPTION_LENGTH_NAME,)


def test_compression_ratio_zero_denominator_is_none_never_coerced() -> None:
    empty_reference = CompressionReferenceArtifact(content=b"")
    assert (
        compression_ratio_from_bytes(
            numerator_bytes=5, reference=empty_reference
        )
        is None
    )
    score = compression_ratio_score_from_bytes(
        name=COMPRESSION_RATIO_NAME,
        numerator_bytes=5,
        reference=empty_reference,
        evaluation_procedure_config_hash=FULL_HASH,
        derived_from=(DESCRIPTION_LENGTH_NAME,),
    )
    assert score is None
