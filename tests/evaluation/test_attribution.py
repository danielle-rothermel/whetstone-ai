"""Golden pins and behavior-preservation tests for F4 row attribution."""

from __future__ import annotations

import pytest

from whetstone.evaluation.aggregate import RowValue
from whetstone.evaluation.attribution import (
    ROW_ATTRIBUTION_TABLE,
    AccountingCell,
    AttributedOutcome,
    attribute_compression_row,
    attribute_generated_row,
    attribute_outcome,
    attribute_published_row,
    require_exclusive_row_state,
    require_exhaustive_row_accounting,
)
from whetstone.evaluation.traces import ExecutedRowState


def test_attribution_table_literals_are_golden() -> None:
    """The six named mappings, pinned byte-for-byte."""
    assert {
        outcome.value: cell.value
        for outcome, cell in ROW_ATTRIBUTION_TABLE.items()
    } == {
        "cancellation": "missing",
        "zero-evidence": "missing",
        "infrastructure": "failed",
        "semantic-rejection": "invalid",
        "empty-output": "invalid",
        "empty-candidate-set": "invalid",
        "token-limit-truncation": "invalid",
        "vanished-published-evidence": "missing",
        "crash": "failed",
        "abandonment": "failed",
    }
    assert [cell.value for cell in AccountingCell] == [
        "scored",
        "missing",
        "failed",
        "invalid",
    ]


def test_every_outcome_kind_attributes_to_exactly_one_cell() -> None:
    assert set(ROW_ATTRIBUTION_TABLE) == set(AttributedOutcome)
    for outcome in AttributedOutcome:
        assert attribute_outcome(outcome) is ROW_ATTRIBUTION_TABLE[outcome]


def test_numeric_zero_remains_a_scored_value() -> None:
    assert attribute_generated_row(
        row_state=ExecutedRowState.SUCCESS, score=0.0
    ) == RowValue(value=0.0)
    assert attribute_compression_row(
        row_state=ExecutedRowState.SUCCESS, compression_value=0.0
    ) == RowValue(value=0.0)
    assert attribute_published_row(
        score=0.0, failed=False, missing=False, invalid=False
    ) == RowValue(value=0.0)


@pytest.mark.parametrize(
    ("row_state", "score", "failure_code", "expected"),
    [
        (ExecutedRowState.MISSING, None, None, RowValue(missing=True)),
        (
            ExecutedRowState.FAILED,
            None,
            "transport-error",
            RowValue(failed=True),
        ),
        (
            ExecutedRowState.FAILED,
            None,
            "provider-rejection",
            RowValue(invalid=True),
        ),
        (
            ExecutedRowState.FAILED,
            None,
            "blank-provider-generation",
            RowValue(invalid=True),
        ),
        (ExecutedRowState.FAILED, None, None, RowValue(failed=True)),
        (ExecutedRowState.SUCCESS, None, None, RowValue(failed=True)),
        (ExecutedRowState.SUCCESS, 0.75, None, RowValue(value=0.75)),
    ],
)
def test_generated_row_attributes_through_the_pinned_table(
    row_state: ExecutedRowState,
    score: float | None,
    failure_code: str | None,
    expected: RowValue,
) -> None:
    assert (
        attribute_generated_row(
            row_state=row_state, score=score, failure_code=failure_code
        )
        == expected
    )


@pytest.mark.parametrize(
    ("row_state", "compression_value", "expected"),
    [
        # The retired encdec compression ladder, cell for cell:
        # missing -> missing; absent value -> failed when the row failed,
        # else invalid; else scored.
        (ExecutedRowState.MISSING, None, RowValue(missing=True)),
        (ExecutedRowState.FAILED, None, RowValue(failed=True)),
        (ExecutedRowState.SUCCESS, None, RowValue(invalid=True)),
        (ExecutedRowState.SUCCESS, 1.25, RowValue(value=1.25)),
    ],
)
def test_compression_row_matches_the_retired_ladder(
    row_state: ExecutedRowState,
    compression_value: float | None,
    expected: RowValue,
) -> None:
    assert (
        attribute_compression_row(
            row_state=row_state, compression_value=compression_value
        )
        == expected
    )


@pytest.mark.parametrize(
    ("score", "failed", "missing", "invalid", "expected"),
    [
        (None, True, False, False, RowValue(failed=True)),
        (None, False, True, False, RowValue(missing=True)),
        (None, False, False, True, RowValue(invalid=True)),
        (0.5, False, False, False, RowValue(value=0.5)),
    ],
)
def test_published_row_matches_the_retired_read_side_ladder(
    score: float | None,
    failed: bool,
    missing: bool,
    invalid: bool,
    expected: RowValue,
) -> None:
    assert (
        attribute_published_row(
            score=score, failed=failed, missing=missing, invalid=invalid
        )
        == expected
    )


def test_published_row_rejects_nonexclusive_states() -> None:
    with pytest.raises(ValueError, match="scored row must be present"):
        attribute_published_row(
            score=0.5, failed=True, missing=False, invalid=False
        )
    with pytest.raises(ValueError, match="exactly one of"):
        attribute_published_row(
            score=None, failed=True, missing=True, invalid=False
        )
    with pytest.raises(ValueError, match="exactly one of"):
        attribute_published_row(
            score=None, failed=False, missing=False, invalid=False
        )


def test_exclusive_row_state_is_the_one_invariant() -> None:
    require_exclusive_row_state(
        scored=True, failed=False, missing=False, invalid=False
    )
    require_exclusive_row_state(
        scored=False, failed=False, missing=True, invalid=False
    )
    with pytest.raises(ValueError, match="scored row must be present"):
        require_exclusive_row_state(
            scored=True, failed=False, missing=True, invalid=False
        )
    with pytest.raises(ValueError, match="exactly one of"):
        require_exclusive_row_state(
            scored=False, failed=True, missing=False, invalid=True
        )


def test_row_value_enforces_the_shared_exclusivity_invariant() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        RowValue(failed=True, missing=True)
    with pytest.raises(ValueError, match="scored row must be present"):
        RowValue(value=1.0, invalid=True)
    with pytest.raises(ValueError, match="exactly one of"):
        RowValue()


def test_accounting_models_share_the_exhaustiveness_invariant() -> None:
    from pydantic import ValidationError

    from whetstone.evaluation.schema import RowAccounting
    from whetstone.optimization.miprov2.evidence import Miprov2RowAccounting

    for model in (RowAccounting, Miprov2RowAccounting):
        model(planned=2, present=1, missing=0, failed=1, invalid=0)
        with pytest.raises(ValidationError, match="not exhaustive"):
            model(planned=2, present=0, missing=0, failed=1, invalid=0)
        with pytest.raises(ValidationError, match="cannot be negative"):
            model(planned=0, present=-1, missing=1, failed=0, invalid=0)


def test_exhaustive_row_accounting_is_the_one_invariant() -> None:
    require_exhaustive_row_accounting(
        planned=4, present=1, missing=1, failed=1, invalid=1
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        require_exhaustive_row_accounting(
            planned=1, present=-1, missing=1, failed=1, invalid=0
        )
    with pytest.raises(ValueError, match="not exhaustive"):
        require_exhaustive_row_accounting(
            planned=3, present=1, missing=1, failed=0, invalid=0
        )
