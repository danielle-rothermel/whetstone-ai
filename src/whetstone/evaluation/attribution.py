"""F4 attribution: typed outcomes into exactly one accounting cell.

This module owns the pinned attribution table and the named per-flow
derivations that place every planned evaluation row in exactly one cell of
``{scored, missing, failed, invalid}``. It replaces the previously
independent flag ladders in the generation coordinators, the read-side
evidence re-derivation, and the re-implemented exclusivity invariants.

The standing rule from the row-accounting contract holds everywhere:
numeric zero remains a scored value — a measured ``0.0`` is present, never
missing, failed, or invalid.

Outcome kinds not yet produced by the current flows (cancellation,
empty candidate set, token-limit truncation, vanished published evidence,
crash, abandonment) are pinned now so later executor and provider wiring
maps into the same table.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import UNIQUE, StrEnum, verify
from types import MappingProxyType
from typing import TYPE_CHECKING

from whetstone.evaluation.traces import ExecutedRowState
from whetstone.provider.classification import SemanticFailureClass

if TYPE_CHECKING:
    from whetstone.evaluation.aggregate import RowValue

__all__ = [
    "ROW_ATTRIBUTION_TABLE",
    "AccountingCell",
    "AttributedOutcome",
    "attribute_compression_row",
    "attribute_generated_row",
    "attribute_outcome",
    "attribute_published_row",
    "require_exclusive_row_state",
    "require_exhaustive_row_accounting",
]


@verify(UNIQUE)
class AccountingCell(StrEnum):
    """The four row-accounting cells of the evaluation matrix."""

    SCORED = "scored"
    MISSING = "missing"
    FAILED = "failed"
    INVALID = "invalid"


@verify(UNIQUE)
class AttributedOutcome(StrEnum):
    """Closed set of named unscored-outcome kinds the table attributes."""

    #: The planned slot's work was cancelled before completion.
    CANCELLATION = "cancellation"
    #: No recorded evidence exists for the planned slot.
    ZERO_EVIDENCE = "zero-evidence"
    #: Provider, execution, or storage infrastructure failed the row.
    INFRASTRUCTURE = "infrastructure"
    #: The provider cleanly rejected the semantic request.
    SEMANTIC_REJECTION = "semantic-rejection"
    #: The row executed but produced empty output (e.g. blank text, a
    #: zero-denominator Compression Ratio).
    EMPTY_OUTPUT = "empty-output"
    #: The row executed but yielded an empty candidate set.
    EMPTY_CANDIDATE_SET = "empty-candidate-set"
    #: The generation was truncated by a token limit.
    TOKEN_LIMIT_TRUNCATION = "token-limit-truncation"
    #: Published evidence for the row vanished before it could be read back.
    VANISHED_PUBLISHED_EVIDENCE = "vanished-published-evidence"
    #: The executing worker crashed.
    CRASH = "crash"
    #: The executing worker abandoned the row without a terminal outcome.
    ABANDONMENT = "abandonment"


# Persisted-format contract: the exact outcome-kind and cell literals of this
# table are pinned by a golden test and by the row-attribution entry in
# .defs/contracts.toml. Never derive them from enum iteration.
ROW_ATTRIBUTION_TABLE: Mapping[AttributedOutcome, AccountingCell] = (
    MappingProxyType(
        {
            AttributedOutcome.CANCELLATION: AccountingCell.MISSING,
            AttributedOutcome.ZERO_EVIDENCE: AccountingCell.MISSING,
            AttributedOutcome.INFRASTRUCTURE: AccountingCell.FAILED,
            AttributedOutcome.SEMANTIC_REJECTION: AccountingCell.INVALID,
            AttributedOutcome.EMPTY_OUTPUT: AccountingCell.INVALID,
            AttributedOutcome.EMPTY_CANDIDATE_SET: AccountingCell.INVALID,
            AttributedOutcome.TOKEN_LIMIT_TRUNCATION: AccountingCell.INVALID,
            AttributedOutcome.VANISHED_PUBLISHED_EVIDENCE: (
                AccountingCell.MISSING
            ),
            AttributedOutcome.CRASH: AccountingCell.FAILED,
            AttributedOutcome.ABANDONMENT: AccountingCell.FAILED,
        }
    )
)


def attribute_outcome(outcome: AttributedOutcome) -> AccountingCell:
    """The one cell the pinned table assigns to a named outcome kind."""
    return ROW_ATTRIBUTION_TABLE[outcome]


def _unscored_row_value(outcome: AttributedOutcome) -> RowValue:
    from whetstone.evaluation.aggregate import RowValue

    cell = attribute_outcome(outcome)
    if cell is AccountingCell.MISSING:
        return RowValue(missing=True)
    if cell is AccountingCell.FAILED:
        return RowValue(failed=True)
    if cell is AccountingCell.INVALID:
        return RowValue(invalid=True)
    raise ValueError(
        f"outcome {outcome.value!r} attributes to a scored cell; a scored "
        "row requires its measured value"
    )


def _outcome_for_generated_failure(
    failure_code: str | None,
) -> AttributedOutcome:
    """Map a coordinator failure code to the pinned unscored outcome kind."""
    if failure_code == SemanticFailureClass.PROVIDER_REJECTION.value:
        return AttributedOutcome.SEMANTIC_REJECTION
    if failure_code == SemanticFailureClass.BLANK_PROVIDER_GENERATION.value:
        return AttributedOutcome.EMPTY_OUTPUT
    return AttributedOutcome.INFRASTRUCTURE


def attribute_generated_row(
    *,
    row_state: ExecutedRowState,
    score: float | None,
    failure_code: str | None = None,
) -> RowValue:
    """Attribute one generated primary row (direct and encdec flows).

    A planned row that never produced evidence is missing. An executed row
    without a score attributes through the pinned table using the row's
    provider failure class when present; otherwise it failed on
    infrastructure. A measured value — numeric zero included — is scored.
    """
    from whetstone.evaluation.aggregate import RowValue

    if row_state is ExecutedRowState.MISSING:
        return _unscored_row_value(AttributedOutcome.ZERO_EVIDENCE)
    if row_state is ExecutedRowState.FAILED or score is None:
        return _unscored_row_value(
            _outcome_for_generated_failure(failure_code)
        )
    return RowValue(value=float(score))


def attribute_compression_row(
    *,
    row_state: ExecutedRowState,
    compression_value: float | None,
) -> RowValue:
    """Attribute one encdec compression row.

    A planned row that never produced evidence is missing. A row without a
    compression measurement failed when its execution failed, and is
    otherwise invalid (measured-but-not-a-number, e.g. a zero-denominator
    Compression Ratio from empty output). A measured value — numeric zero
    included — is scored.
    """
    from whetstone.evaluation.aggregate import RowValue

    if row_state is ExecutedRowState.MISSING:
        return _unscored_row_value(AttributedOutcome.ZERO_EVIDENCE)
    if compression_value is None:
        if row_state is ExecutedRowState.FAILED:
            return _unscored_row_value(AttributedOutcome.INFRASTRUCTURE)
        return _unscored_row_value(AttributedOutcome.EMPTY_OUTPUT)
    return RowValue(value=float(compression_value))


def attribute_published_row(
    *,
    score: float | None,
    failed: bool,
    missing: bool,
    invalid: bool,
) -> RowValue:
    """Re-derive one published output row's cell (evidence read-back flow).

    The wire row already carries its exclusive cell; this derivation
    revalidates exclusivity and projects the recorded cell without
    reclassifying it.
    """
    from whetstone.evaluation.aggregate import RowValue

    require_exclusive_row_state(
        scored=score is not None,
        failed=failed,
        missing=missing,
        invalid=invalid,
    )
    if failed:
        return RowValue(failed=True)
    if missing:
        return RowValue(missing=True)
    if invalid:
        return RowValue(invalid=True)
    assert score is not None
    return RowValue(value=score)


def require_exclusive_row_state(
    *,
    scored: bool,
    failed: bool,
    missing: bool,
    invalid: bool,
) -> None:
    """Enforce the one exclusivity invariant on a row's cell flags.

    A scored row carries no unscored state; an unscored row carries exactly
    one of failed / missing / invalid.
    """
    state_count = sum((failed, missing, invalid))
    if scored and state_count:
        raise ValueError("a scored row must be present")
    if not scored and state_count != 1:
        raise ValueError(
            "an unscored row requires exactly one of failed / missing / "
            "invalid"
        )


def require_exhaustive_row_accounting(
    *,
    planned: int,
    present: int,
    missing: int,
    failed: int,
    invalid: int,
) -> None:
    """Enforce the one exhaustiveness invariant on matrix row accounting.

    Every planned cell is accounted for exactly once across the four cells.
    """
    counts = (present, missing, failed, invalid)
    if planned < 0 or any(count < 0 for count in counts):
        raise ValueError("evaluation row accounting cannot be negative")
    if planned != sum(counts):
        raise ValueError("evaluation row accounting is not exhaustive")
