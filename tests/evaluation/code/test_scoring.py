from __future__ import annotations

from whetstone.evaluation import (
    Applicability,
    CompressionReferenceArtifact,
)
from whetstone.evaluation.code import (
    COMPRESSED_DESCRIPTION_LENGTH_NAME,
    COMPRESSION_RATIO_NAME,
    compressed_description_length_bytes,
    compressed_description_length_fact,
    compression_ratio_score,
    compression_ratio_value,
)

from .support import FULL_HASH, operator_lineage

ENCODER_TEXT = "def f(x):\n    return x + 1\n" * 8


def test_cdl_wrapper_delegates_to_generic_zstd_measurement() -> None:
    assert isinstance(compressed_description_length_bytes(ENCODER_TEXT), int)


def test_cdl_fact_carries_unit_and_lineage() -> None:
    fact = compressed_description_length_fact(
        ENCODER_TEXT, lineage=operator_lineage()
    )
    assert fact.name == COMPRESSED_DESCRIPTION_LENGTH_NAME
    assert fact.unit == "bytes"
    assert fact.applicability is Applicability.APPLICABLE
    assert isinstance(fact.value, int)
    assert fact.lineage.operator == "compressed_length"


def test_compression_ratio_over_nonzero_reference() -> None:
    reference = CompressionReferenceArtifact(content=b"abcdefghij")
    ratio = compression_ratio_value(
        compressed_description_length=5, reference=reference
    )
    assert ratio == 0.5


def test_compression_ratio_score_has_lineage() -> None:
    reference = CompressionReferenceArtifact(content=b"abcdefghij")
    score = compression_ratio_score(
        compressed_description_length=5,
        reference=reference,
        evaluation_procedure_config_hash=FULL_HASH,
    )
    assert score is not None
    assert score.name == COMPRESSION_RATIO_NAME
    assert score.value == 0.5
    assert score.unit == "ratio"
    assert score.derived_from == (COMPRESSED_DESCRIPTION_LENGTH_NAME,)


def test_compression_ratio_zero_denominator_is_none_never_coerced() -> None:
    empty_reference = CompressionReferenceArtifact(content=b"")
    assert (
        compression_ratio_value(
            compressed_description_length=5, reference=empty_reference
        )
        is None
    )
    score = compression_ratio_score(
        compressed_description_length=5,
        reference=empty_reference,
        evaluation_procedure_config_hash=FULL_HASH,
    )
    assert score is None
