"""Rollout Aggregate: provenance binding + ABTPR + Mean Compression Ratio.

Proves the Rollout Aggregate binds the pure dr-code Aggregation Output to
``(graph_hash, eval_config_hash)``, the complete planned matrix, and the
Evaluation Context; that Average Binary Test Pass Rate is the per-task mean
then the unweighted cross-task mean; and that failed / missing / invalid rows
are handled explicitly (never silently dropped) under the declared policy.
"""

from __future__ import annotations

import pytest
from dr_code.eval import AggregationOutput, AggregationStatus

from whetstone.code_eval import (
    RolloutAggregate,
    RowPolicy,
    RowValue,
    TaskRows,
    average_binary_test_pass_rate,
    mean_compression_ratio,
)

from .support import FULL_HASH

CTX = "c" * 64


def _abtpr(task_rows, repeat_count=3, policy=RowPolicy.PROPAGATE):
    return average_binary_test_pass_rate(
        graph_hash=FULL_HASH,
        eval_config_hash=FULL_HASH,
        evaluation_context_id=CTX,
        task_rows=task_rows,
        repeat_count=repeat_count,
        policy=policy,
    )


# --- Provenance binding ----------------------------------------------------


def test_aggregate_binds_pure_output_to_identity_and_context() -> None:
    t1 = TaskRows(
        task_identity="t1",
        expected_repeats=2,
        rows=(RowValue(value=1.0), RowValue(value=0.0)),
    )
    agg = _abtpr((t1,), repeat_count=2)
    assert isinstance(agg, RolloutAggregate)
    # Identity ( graph_hash, eval_config_hash ) + Evaluation Context.
    assert agg.graph_hash == FULL_HASH
    assert agg.eval_config_hash == FULL_HASH
    assert agg.evaluation_context_id == CTX
    # The bound value is the *pure* dr-code Aggregation Output.
    assert isinstance(agg.aggregation_output, AggregationOutput)
    # Complete planned matrix shape.
    assert agg.task_count == 1
    assert agg.repeat_count == 2


# --- Average Binary Test Pass Rate: two-stage mean -------------------------


def test_abtpr_is_per_task_then_unweighted_cross_task_mean() -> None:
    # t1: [1,0,1] -> 2/3 ; t2: [0,0,0] -> 0. Cross-task unweighted mean = 1/3.
    t1 = TaskRows(
        task_identity="t1",
        expected_repeats=3,
        rows=(RowValue(value=1.0), RowValue(value=0.0), RowValue(value=1.0)),
    )
    t2 = TaskRows(
        task_identity="t2",
        expected_repeats=3,
        rows=(RowValue(value=0.0), RowValue(value=0.0), RowValue(value=0.0)),
    )
    agg = _abtpr((t1, t2))
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(1 / 3)
    assert agg.rows_present == 6
    assert agg.rows_missing == 0


def test_abtpr_is_unweighted_regardless_of_repeat_counts() -> None:
    # Both tasks contribute equally to the cross-task mean (unweighted), even
    # though per-task means come from the same repeat_count here.
    t_all_pass = TaskRows(
        task_identity="t1",
        expected_repeats=3,
        rows=(RowValue(value=1.0),) * 3,
    )
    t_all_fail = TaskRows(
        task_identity="t2",
        expected_repeats=3,
        rows=(RowValue(value=0.0),) * 3,
    )
    agg = _abtpr((t_all_pass, t_all_fail))
    # (1.0 + 0.0) / 2 = 0.5 unweighted.
    assert agg.aggregation_output.value == pytest.approx(0.5)


# --- Missing / failed rows are never silently dropped ----------------------


def test_missing_rows_padded_and_counted_propagate() -> None:
    # A task with fewer rows than repeat_count: the shortfall is explicit
    # missing rows. Under PROPAGATE the aggregate is MISSING_DATA.
    t1 = TaskRows(
        task_identity="t1", expected_repeats=3, rows=(RowValue(value=1.0),)
    )
    agg = _abtpr((t1,))
    assert agg.rows_missing == 2
    assert agg.rows_present == 1
    # Matrix fully accounted for.
    assert (
        agg.rows_present
        + agg.rows_missing
        + agg.rows_failed
        + agg.rows_invalid
        == agg.task_count * agg.repeat_count
    )
    assert agg.aggregation_output.status is AggregationStatus.MISSING_DATA
    assert agg.aggregation_output.value is None


def test_missing_rows_skip_policy_excludes_but_counts() -> None:
    t1 = TaskRows(
        task_identity="t1",
        expected_repeats=3,
        rows=(RowValue(value=1.0), RowValue(value=0.0)),
    )
    agg = _abtpr((t1,), policy=RowPolicy.SKIP)
    # Under SKIP: per-task mean over the 2 present rows = 0.5. Missing row is
    # excluded from the denominator but still counted in provenance.
    assert agg.rows_missing == 1
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(0.5)


def test_failed_row_propagates_missing_data() -> None:
    t1 = TaskRows(
        task_identity="t1",
        expected_repeats=2,
        rows=(RowValue(value=1.0), RowValue(failed=True)),
    )
    agg = _abtpr((t1,), repeat_count=2)
    assert agg.rows_failed == 1
    assert agg.aggregation_output.status is AggregationStatus.MISSING_DATA


# --- Mean Compression Ratio ------------------------------------------------


def _mcr(rows, task_count=1, repeat_count=2, policy=RowPolicy.PROPAGATE):
    return mean_compression_ratio(
        graph_hash=FULL_HASH,
        eval_config_hash=FULL_HASH,
        evaluation_context_id=CTX,
        rows=rows,
        task_count=task_count,
        repeat_count=repeat_count,
        policy=policy,
    )


def test_mcr_complete_matrix_mean() -> None:
    agg = _mcr((RowValue(value=0.4), RowValue(value=0.6)))
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(0.5)
    assert agg.rows_present == 2


def test_mcr_requires_complete_planned_matrix() -> None:
    with pytest.raises(ValueError):
        _mcr((RowValue(value=0.4),), task_count=1, repeat_count=2)


def test_mcr_invalid_denominator_excluded_but_counted() -> None:
    # An invalid (zero-denominator) Compression Ratio is not-applicable: it is
    # excluded from the mean but explicitly counted, never dropped silently.
    agg = _mcr((RowValue(value=0.4), RowValue(invalid=True)))
    assert agg.rows_invalid == 1
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(0.4)


def test_mcr_failed_row_propagates() -> None:
    agg = _mcr((RowValue(value=0.4), RowValue(failed=True)))
    assert agg.rows_failed == 1
    assert agg.aggregation_output.status is AggregationStatus.MISSING_DATA


def test_mcr_all_invalid_is_not_applicable_never_fabricated() -> None:
    agg = _mcr((RowValue(invalid=True), RowValue(invalid=True)))
    # Every row invalid => not-applicable, value is None (never fabricated).
    assert agg.aggregation_output.status is AggregationStatus.NOT_APPLICABLE
    assert agg.aggregation_output.value is None
    assert agg.rows_invalid == 2


# --- RowValue explicitness -------------------------------------------------


def test_row_value_requires_explicit_state() -> None:
    # A bare None value is rejected: absence must be declared explicitly.
    with pytest.raises(ValueError):
        RowValue()
    # A row cannot be two absence kinds at once.
    with pytest.raises(ValueError):
        RowValue(failed=True, missing=True)
    # A present value cannot also be flagged absent.
    with pytest.raises(ValueError):
        RowValue(value=0.5, invalid=True)


def test_aggregate_rejects_incomplete_accounting() -> None:
    # Constructing a RolloutAggregate whose row counts do not cover the
    # planned matrix is rejected (the matrix must be complete).
    from dr_code.eval import AggregationOutput as _AO

    with pytest.raises(ValueError):
        RolloutAggregate(
            name="x",
            graph_hash=FULL_HASH,
            eval_config_hash=FULL_HASH,
            evaluation_context_id=CTX,
            task_count=2,
            repeat_count=3,
            aggregation_output=_AO(
                status=AggregationStatus.OK,
                value=0.5,
                count_total=1,
                count_applicable=1,
                count_present=1,
            ),
            rows_present=1,
            rows_missing=0,
            rows_failed=0,
            rows_invalid=0,
        )
