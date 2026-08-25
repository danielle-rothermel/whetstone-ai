from __future__ import annotations

from collections.abc import Mapping
from enum import UNIQUE, StrEnum, verify
from types import MappingProxyType
from typing import TYPE_CHECKING

from whetstone.eval.traces import ExecutedRowState
from whetstone.provider.classification import SemanticFailureClass

if TYPE_CHECKING:
    from whetstone.eval.aggregate import RowValue

__all__ = [
    "ROW_ATTRIBUTION_TABLE",
    "AccountingCell",
    "AttributedOutcome",
    "attribute_generated_row",
    "attribute_generated_row_cell",
    "attribute_outcome",
    "attribute_published_row",
    "require_exclusive_row_state",
    "require_exhaustive_row_accounting",
]


@verify(UNIQUE)
class AccountingCell(StrEnum):
    SCORED = "scored"
    MISSING = "missing"
    FAILED = "failed"
    INVALID = "invalid"


@verify(UNIQUE)
class AttributedOutcome(StrEnum):
    CANCELLATION = "cancellation"

    ZERO_EVIDENCE = "zero-evidence"

    INFRASTRUCTURE = "infrastructure"

    SEMANTIC_REJECTION = "semantic-rejection"

    EMPTY_OUTPUT = "empty-output"

    EMPTY_CANDIDATE_SET = "empty-candidate-set"

    TOKEN_LIMIT_TRUNCATION = "token-limit-truncation"

    VANISHED_PUBLISHED_EVIDENCE = "vanished-published-evidence"

    CRASH = "crash"

    ABANDONMENT = "abandonment"


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
    return ROW_ATTRIBUTION_TABLE[outcome]


def _unscored_row_value(outcome: AttributedOutcome) -> RowValue:
    from whetstone.eval.aggregate import RowValue

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


#: Failure codes the eval contract attributes to a specific outcome. A code
#: absent here has no contract attribution and is infrastructure.
_OUTCOME_BY_FAILURE_CODE: Mapping[str, AttributedOutcome] = MappingProxyType(
    {
        SemanticFailureClass.PROVIDER_REJECTION.value: (
            AttributedOutcome.SEMANTIC_REJECTION
        ),
        SemanticFailureClass.BLANK_PROVIDER_GENERATION.value: (
            AttributedOutcome.EMPTY_OUTPUT
        ),
    }
)


def _outcome_for_generated_failure(
    failure_code: str | None,
) -> AttributedOutcome:
    if failure_code is None:
        return AttributedOutcome.INFRASTRUCTURE
    return _OUTCOME_BY_FAILURE_CODE.get(
        failure_code, AttributedOutcome.INFRASTRUCTURE
    )


def attribute_generated_row_cell(failure_code: str) -> AccountingCell | None:
    """Return the accounting cell one failure code attributes to.

    ``None`` means the code carries no contract attribution, leaving the
    decision to the caller's own fallback rather than silently claiming the
    row is an infrastructure failure.
    """
    outcome = _OUTCOME_BY_FAILURE_CODE.get(failure_code)
    if outcome is None:
        return None
    return attribute_outcome(outcome)


def attribute_generated_row(
    *,
    row_state: ExecutedRowState,
    score: float | None,
    failure_code: str | None = None,
) -> RowValue:
    from whetstone.eval.aggregate import RowValue

    if row_state is ExecutedRowState.MISSING:
        return _unscored_row_value(AttributedOutcome.ZERO_EVIDENCE)
    if row_state is ExecutedRowState.FAILED or score is None:
        return _unscored_row_value(
            _outcome_for_generated_failure(failure_code)
        )
    return RowValue(value=float(score))


def attribute_published_row(
    *,
    score: float | None,
    failed: bool,
    missing: bool,
    invalid: bool,
) -> RowValue:
    from whetstone.eval.aggregate import RowValue

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
    counts = (present, missing, failed, invalid)
    if planned < 0 or any(count < 0 for count in counts):
        raise ValueError("evaluation row accounting cannot be negative")
    if planned != sum(counts):
        raise ValueError("evaluation row accounting is not exhaustive")
