"""Binary Test Pass Score, Compressed Description Length, Compression Ratio."""

from __future__ import annotations

import pytest
import zstandard
from dr_code.eval import (
    AbsenceMode,
    Applicability,
    CodeCandidateSet,
    CompressionReferenceArtifact,
    Score,
)

from whetstone.code_eval import (
    COMPRESSED_DESCRIPTION_LENGTH_NAME,
    COMPRESSION_RATIO_NAME,
    ZSTD_LEVEL,
    CandidateVerdict,
    CorrectnessOutcome,
    InfrastructureUnknownScoreError,
    binary_test_pass_score,
    compressed_description_length_bytes,
    compressed_description_length_fact,
    compression_ratio_score,
    compression_ratio_value,
    correctness_absent,
    evaluate_candidate_correctness,
)

from .support import FULL_HASH, operator_lineage

ENCODER_TEXT = "def f(x):\n    return x + 1\n" * 8


def _passed_result():
    cs = CodeCandidateSet.from_sources(
        ("def f():\n    return 1\n",), origin="t"
    )
    return evaluate_candidate_correctness(
        cs, runner=lambda _a: CandidateVerdict.PASSED
    )


def _failed_result():
    cs = CodeCandidateSet.from_sources(
        ("def f():\n    return 1\n",), origin="t"
    )
    return evaluate_candidate_correctness(
        cs, runner=lambda _a: CandidateVerdict.FAILED
    )


def _infra_result():
    cs = CodeCandidateSet.from_sources(
        ("def f():\n    return 1\n",), origin="t"
    )
    return evaluate_candidate_correctness(
        cs, runner=lambda _a: CandidateVerdict.INFRASTRUCTURE_UNKNOWN
    )


# --- Binary Test Pass Score ------------------------------------------------


def test_binary_score_one_on_any_pass() -> None:
    score = binary_test_pass_score(
        _passed_result(), evaluation_procedure_config_hash=FULL_HASH
    )
    assert isinstance(score, Score)
    assert score.value == 1


def test_binary_score_zero_on_definitive_fail() -> None:
    score = binary_test_pass_score(
        _failed_result(), evaluation_procedure_config_hash=FULL_HASH
    )
    assert score.value == 0


def test_binary_score_zero_on_absence_causes() -> None:
    # Compile failure / empty set / absence causes are scorable definitives.
    for result in (
        correctness_absent(AbsenceMode.NO_INPUT),
        correctness_absent(AbsenceMode.EMPTY_CANDIDATE_SET),
    ):
        score = binary_test_pass_score(
            result, evaluation_procedure_config_hash=FULL_HASH
        )
        assert score.value == 0


def test_binary_score_infrastructure_unknown_raises_not_zero() -> None:
    # Infrastructure-unknown has NO score: deriving one raises so the rollout
    # fails rather than producing a spurious 0.
    result = _infra_result()
    assert result.outcome is CorrectnessOutcome.INFRASTRUCTURE_FAILURE
    with pytest.raises(InfrastructureUnknownScoreError):
        binary_test_pass_score(
            result, evaluation_procedure_config_hash=FULL_HASH
        )


# --- Compressed Description Length -----------------------------------------


def test_cdl_is_zstd19_utf8_byte_count() -> None:
    expected = len(
        zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
            ENCODER_TEXT.encode("utf-8")
        )
    )
    assert compressed_description_length_bytes(ENCODER_TEXT) == expected
    assert ZSTD_LEVEL == 19


def test_cdl_is_nonnegative_integer() -> None:
    value = compressed_description_length_bytes("")
    assert isinstance(value, int)
    assert value >= 0


def test_cdl_uses_exact_utf8_bytes() -> None:
    text = "print('π —')"
    expected = len(
        zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
            text.encode("utf-8")
        )
    )
    assert compressed_description_length_bytes(text) == expected


def test_cdl_fact_carries_unit_and_lineage() -> None:
    fact = compressed_description_length_fact(
        ENCODER_TEXT, lineage=operator_lineage()
    )
    assert fact.name == COMPRESSED_DESCRIPTION_LENGTH_NAME
    assert fact.unit == "bytes"
    assert fact.applicability is Applicability.APPLICABLE
    assert isinstance(fact.value, int)
    assert fact.lineage.operator == "compressed_length"


# --- Compression Ratio -----------------------------------------------------


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
    # Value path: explicit None, never 0.0 / 1.0.
    assert (
        compression_ratio_value(
            compressed_description_length=5, reference=empty_reference
        )
        is None
    )
    # Score path: explicit None (invalid/N-A behavior), never a coerced Score.
    score = compression_ratio_score(
        compressed_description_length=5,
        reference=empty_reference,
        evaluation_procedure_config_hash=FULL_HASH,
    )
    assert score is None
