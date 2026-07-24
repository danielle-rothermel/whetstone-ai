"""Whetstone code-eval Score / Metric Fact derivations.

Three experiment-specific derivations over dr-code's kernel primitives.
Whetstone owns these; dr-code owns the generic ``compression_ratio``,
``MetricFact``, and ``Score`` types they build on.

* **Binary Test Pass Score** — a Whetstone :class:`~dr_code.eval.Score` equal
  to ``1`` when Candidate Correctness ``PASSED`` and ``0`` when it is a
  ``DEFINITIVE_TEST_FAILURE`` (or any other definitive non-passing scorable
  outcome). An infrastructure-unknown correctness result has **no** Binary
  Test Pass Score — deriving one raises, so the rollout fails instead of
  producing a spurious ``0``.
* **Compressed Description Length** — a Whetstone
  :class:`~dr_code.eval.MetricFact` equal to the nonnegative integer byte
  count produced by **zstd level 19** over the exact encoder Generation
  encoded as **UTF-8**.
* **Compression Ratio** — a Whetstone :class:`~dr_code.eval.Score` equal to
  the
  Compressed Description Length divided by the selected nonzero Compression
  Reference Artifact byte length. A zero denominator produces the explicit
  invalid / not-applicable behavior the Procedure/Aggregation Configs declare
  (surfaced via dr-code's ``compression_ratio`` returning the
  ``ZERO_DENOMINATOR`` sentinel); it is never silently coerced.
"""

from __future__ import annotations

import zstandard
from dr_code.eval import (
    Applicability,
    CompressionReferenceArtifact,
    MetricFact,
    OperatorLineage,
    Score,
    compression_ratio,
)

from whetstone.code_eval.correctness import (
    CorrectnessOutcome,
    CorrectnessResult,
)

#: The pinned zstd level for Compressed Description Length. Fixed by the
#: experiment; a level change is a deliberate breaking measurement change.
ZSTD_LEVEL = 19

BINARY_TEST_PASS_SCORE_NAME = "binary_test_pass"
COMPRESSED_DESCRIPTION_LENGTH_NAME = "compressed_description_length"
COMPRESSION_RATIO_NAME = "compression_ratio"

_BINARY_UNIT = "pass"
_BYTES_UNIT = "bytes"
_RATIO_UNIT = "ratio"


class InfrastructureUnknownScoreError(ValueError):
    """Raised when a Binary Test Pass Score is requested for an
    infrastructure-unknown correctness result.

    Infrastructure uncertainty must fail the rollout, never become score 0.
    """


def binary_test_pass_score(
    result: CorrectnessResult,
    *,
    evaluation_procedure_config_hash: str,
    derived_from: tuple[str, ...] = (),
) -> Score:
    """Derive the Binary Test Pass Score from a Candidate Correctness result.

    ``1`` when correctness ``PASSED``; ``0`` when it is any definitive
    non-passing scorable outcome (definitive test failure, compile failure,
    empty set, or an absence cause). Raises
    :class:`InfrastructureUnknownScoreError` for an infrastructure-unknown
    result — that case has no score and must terminate the rollout as failed.
    """

    if result.is_infrastructure_failure:
        raise InfrastructureUnknownScoreError(
            "infrastructure-unknown correctness has no Binary Test Pass "
            "Score; the rollout must fail rather than score 0"
        )
    value = 1 if result.outcome is CorrectnessOutcome.PASSED else 0
    return Score(
        name=BINARY_TEST_PASS_SCORE_NAME,
        value=value,
        unit=_BINARY_UNIT,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        derived_from=derived_from,
    )


def compressed_description_length_bytes(encoder_generation: str) -> int:
    """The Compressed Description Length in bytes.

    The nonnegative integer byte count produced by zstd level 19 over the
    exact encoder Generation encoded as UTF-8. Pure and deterministic.
    """

    payload = encoder_generation.encode("utf-8")
    compressed = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(payload)
    return len(compressed)


def compressed_description_length_fact(
    encoder_generation: str,
    *,
    lineage: OperatorLineage,
) -> MetricFact:
    """The Compressed Description Length as a lineage-bearing Metric Fact.

    An integer ``bytes`` fact carrying the resolved operator/step lineage.
    """

    return MetricFact(
        name=COMPRESSED_DESCRIPTION_LENGTH_NAME,
        value=compressed_description_length_bytes(encoder_generation),
        unit=_BYTES_UNIT,
        applicability=Applicability.APPLICABLE,
        lineage=lineage,
    )


def compression_ratio_value(
    *,
    compressed_description_length: int,
    reference: CompressionReferenceArtifact,
) -> float | None:
    """The Compression Ratio value, or ``None`` for a zero denominator.

    Delegates the denominator handling to dr-code's generic
    ``compression_ratio``: a zero-length reference returns the explicit
    ``ZERO_DENOMINATOR`` sentinel (``None``), never coerced to ``0.0``/``1.0``.
    """

    return compression_ratio(
        numerator_bytes=compressed_description_length,
        reference=reference,
    )


def compression_ratio_score(
    *,
    compressed_description_length: int,
    reference: CompressionReferenceArtifact,
    evaluation_procedure_config_hash: str,
    derived_from: tuple[str, ...] = (
        COMPRESSED_DESCRIPTION_LENGTH_NAME,
    ),
) -> Score | None:
    """Derive the Compression Ratio Score, or ``None`` for a zero denominator.

    ``None`` (the explicit zero-denominator outcome) is returned when the
    reference has zero bytes; the caller surfaces it as the invalid /
    not-applicable behavior declared by its Procedure/Aggregation Config. It
    is never coerced to a numeric Score.
    """

    ratio = compression_ratio_value(
        compressed_description_length=compressed_description_length,
        reference=reference,
    )
    if ratio is None:
        return None
    return Score(
        name=COMPRESSION_RATIO_NAME,
        value=ratio,
        unit=_RATIO_UNIT,
        evaluation_procedure_config_hash=evaluation_procedure_config_hash,
        derived_from=derived_from,
    )


__all__ = [
    "BINARY_TEST_PASS_SCORE_NAME",
    "COMPRESSED_DESCRIPTION_LENGTH_NAME",
    "COMPRESSION_RATIO_NAME",
    "ZSTD_LEVEL",
    "InfrastructureUnknownScoreError",
    "binary_test_pass_score",
    "compressed_description_length_bytes",
    "compressed_description_length_fact",
    "compression_ratio_score",
    "compression_ratio_value",
]
