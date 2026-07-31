"""Rollout Aggregate: provenance binding and scalar reductions.

Proves the Rollout Aggregate binds the pure dr-code Aggregation Output to
``(graph_hash, eval_config_hash)``, the complete planned matrix, and the
Evaluation Context; that the generic task reducer is the per-task mean then
the unweighted cross-task mean; and that failed / missing / invalid rows are
handled explicitly (never silently dropped) under the declared policy.
"""

from __future__ import annotations

from typing import cast

import pytest
from dr_code.eval import AggregationOutput, AggregationStatus

from whetstone.code_eval import (
    CompletenessPolicy,
    RolloutAggregate,
    RowPolicy,
    RowValue,
    TaskRows,
    aggregation_definition,
    mean_compression_ratio,
    unweighted_task_mean,
)
from whetstone.code_eval.aggregate import SKIP_TOLERANCE_VARIABLE

from .support import FULL_HASH

CTX = "c" * 64
AGGREGATE_NAME = "custom_scalar_quality"
PROPAGATE = CompletenessPolicy()


def _task_mean(
    task_rows: tuple[TaskRows, ...],
    repeat_count: int = 3,
    policy: CompletenessPolicy = PROPAGATE,
    aggregate_name: str = AGGREGATE_NAME,
) -> RolloutAggregate:
    return unweighted_task_mean(
        aggregate_name=aggregate_name,
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
    agg = _task_mean((t1,), repeat_count=2)
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


# --- Generic scalar task mean: two-stage mean ------------------------------


def test_unweighted_task_mean_accepts_generic_scalar_values() -> None:
    # t1: [2,4,6] -> 4; t2: [10,14,18] -> 14. Cross-task mean = 9.
    t1 = TaskRows(
        task_identity="t1",
        expected_repeats=3,
        rows=(RowValue(value=2.0), RowValue(value=4.0), RowValue(value=6.0)),
    )
    t2 = TaskRows(
        task_identity="t2",
        expected_repeats=3,
        rows=(
            RowValue(value=10.0),
            RowValue(value=14.0),
            RowValue(value=18.0),
        ),
    )
    agg = _task_mean((t1, t2))
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(9.0)
    assert agg.rows_present == 6
    assert agg.rows_missing == 0


def test_unweighted_task_mean_is_two_stage_not_row_weighted() -> None:
    full_task = TaskRows(
        task_identity="t1",
        expected_repeats=3,
        rows=(RowValue(value=2.0), RowValue(value=4.0), RowValue(value=6.0)),
    )
    sparse_task = TaskRows(
        task_identity="t2",
        expected_repeats=3,
        rows=(RowValue(value=20.0),),
    )
    agg = _task_mean(
        (full_task, sparse_task),
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP,
            max_skip_fraction=1 / 3,
        ),
    )
    # Per-task means are 4 and 20, so the unweighted task mean is 12. A flat
    # row mean would be 8 and would violate the two-stage contract.
    assert agg.aggregation_output.value == pytest.approx(12.0)


def test_unweighted_task_mean_propagates_caller_owned_name() -> None:
    task = TaskRows(
        task_identity="t1",
        expected_repeats=1,
        rows=(RowValue(value=3.5),),
    )
    aggregate_name = "latency_adjusted_quality"
    agg = _task_mean((task,), repeat_count=1, aggregate_name=aggregate_name)
    assert agg.name == aggregate_name


@pytest.mark.parametrize("aggregate_name", ["", " ", "\t\n"])
def test_unweighted_task_mean_rejects_blank_name(
    aggregate_name: str,
) -> None:
    with pytest.raises(ValueError, match="aggregate_name must be nonblank"):
        _task_mean((), repeat_count=1, aggregate_name=aggregate_name)


def test_unweighted_task_mean_rejects_non_string_name() -> None:
    with pytest.raises(TypeError, match="aggregate_name must be a string"):
        _task_mean(
            (),
            repeat_count=1,
            aggregate_name=cast(str, 42),
        )


# --- Missing / failed rows are never silently dropped ----------------------


def test_missing_rows_padded_and_counted_propagate() -> None:
    # A task with fewer rows than repeat_count: the shortfall is explicit
    # missing rows. Under PROPAGATE the aggregate is MISSING_DATA.
    t1 = TaskRows(
        task_identity="t1", expected_repeats=3, rows=(RowValue(value=1.0),)
    )
    agg = _task_mean((t1,))
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
    agg = _task_mean(
        (t1,),
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP,
            max_skip_fraction=1 / 3,
        ),
    )
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
    agg = _task_mean((t1,), repeat_count=2)
    assert agg.rows_failed == 1
    assert agg.aggregation_output.status is AggregationStatus.MISSING_DATA


# --- Bounded completeness tolerance (declared max_skip_fraction) ------------


def _rows_with_skips(n_tasks: int, skipped: int) -> tuple[TaskRows, ...]:
    """``n_tasks`` single-repeat tasks; the first ``skipped`` are failed."""
    return tuple(
        TaskRows(
            task_identity=f"t{i}",
            expected_repeats=1,
            rows=(RowValue(failed=True),)
            if i < skipped
            else (RowValue(value=1.0),),
        )
        for i in range(n_tasks)
    )


def test_skip_within_tolerance_certifies_a_value() -> None:
    # 1 of 100 rows skipped (1%) under a declared 2% tolerance: certified.
    task_rows = _rows_with_skips(100, skipped=1)
    agg = _task_mean(
        task_rows,
        repeat_count=1,
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP, max_skip_fraction=0.02
        ),
    )
    assert agg.rows_failed == 1
    assert agg.rows_present == 99
    # Within tolerance -> a real value over the surviving rows (all 1.0).
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(1.0)


def test_skip_over_tolerance_forced_incomplete_but_counts_kept() -> None:
    # 3 of 100 rows skipped (3%) exceeds the 2% bound: forced MISSING_DATA,
    # value None (an incomplete arm), yet the skipped rows stay counted.
    task_rows = _rows_with_skips(100, skipped=3)
    agg = _task_mean(
        task_rows,
        repeat_count=1,
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP, max_skip_fraction=0.02
        ),
    )
    assert agg.rows_failed == 3
    assert agg.rows_present == 97
    assert agg.aggregation_output.status is AggregationStatus.MISSING_DATA
    assert agg.aggregation_output.value is None


def test_skip_exactly_at_tolerance_certifies() -> None:
    # 2 of 100 (exactly 2%) is within the inclusive bound.
    agg = _task_mean(
        _rows_with_skips(100, skipped=2),
        repeat_count=1,
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP, max_skip_fraction=0.02
        ),
    )
    assert agg.aggregation_output.status is AggregationStatus.OK


def test_skip_tolerance_participates_in_aggregation_config_identity() -> None:
    definition = aggregation_definition("test.aggregate")
    shared = {
        "reduction": "mean",
        "missing_data": "skip",
        "zero_denominator": "not_applicable",
    }
    one_percent = definition.materialize(
        {**shared, SKIP_TOLERANCE_VARIABLE: "0.0100"}
    )
    two_percent = definition.materialize(
        {**shared, SKIP_TOLERANCE_VARIABLE: "0.0200"}
    )
    assert one_percent.config_identity_hash != two_percent.config_identity_hash


def test_skip_requires_an_explicit_bounded_completeness_policy() -> None:
    # A bare SKIP declares no completeness bound and is rejected. Callers must
    # provide an identity-bearing max_skip_fraction.
    with pytest.raises(TypeError, match="CompletenessPolicy"):
        _task_mean(
            _rows_with_skips(100, skipped=1),
            repeat_count=1,
            policy=cast(CompletenessPolicy, RowPolicy.SKIP),
        )


# --- Mean Compression Ratio ------------------------------------------------


def _mcr(
    rows,
    task_count=1,
    repeat_count=2,
    policy=PROPAGATE,
):
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


def test_aggregate_reuses_foundation_hash_validation() -> None:
    with pytest.raises(ValueError, match="graph_hash"):
        RolloutAggregate(
            name="x",
            graph_hash="short",
            eval_config_hash=FULL_HASH,
            evaluation_context_id=CTX,
            task_count=0,
            repeat_count=1,
            aggregation_output=AggregationOutput(
                status=AggregationStatus.NOT_APPLICABLE,
                value=None,
                count_total=0,
                count_applicable=0,
                count_present=0,
            ),
            rows_present=0,
            rows_missing=0,
            rows_failed=0,
            rows_invalid=0,
        )
