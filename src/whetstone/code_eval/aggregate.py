"""Rollout Aggregate: provenance-bearing binding of pure aggregation output.

dr-code's :func:`~dr_code.eval.aggregate` is a pure, provenance-free function
over an explicitly complete tuple of inputs. Whetstone binds that value into a
**provenance-bearing** Rollout Aggregate: it attaches the identity
``(graph_hash, eval_config_hash)``, the complete planned Rollout Result matrix,
and the stated Evaluation Context. Whetstone owns provenance/context; dr-code
owns the numeric reduction.

Two Rollout Aggregates are derived here:

* **Average Binary Test Pass Rate** — the mean Binary Test Pass Score across
  Repeat IDs *per Task*, followed by the configured **unweighted** mean across
  the **complete** Task Set. The per-task mean is a first reduction; the
  cross-task mean is the second. Every planned cell is accounted for.
* **Mean Compression Ratio** — the configured complete-matrix mean of measured
  Compression Ratios.

Both handle failed rows, missing rows, and invalid (zero-denominator)
Compression Ratios **explicitly** via the declared :class:`RowPolicy`: rows are
never silently dropped. Under ``PROPAGATE`` any incomplete/failed/invalid cell
makes the whole aggregate ``MISSING_DATA``; under ``SKIP`` such cells are
excluded but their exclusion is recorded in the provenance counts, and a
wholly-empty reduction is an explicit ``ZERO_DENOMINATOR`` / ``MISSING_DATA``
rather than a fabricated value.
"""

from __future__ import annotations

from enum import StrEnum

from dr_code.eval import (
    AggregationConfig,
    AggregationDefinition,
    AggregationInput,
    AggregationOutput,
    AggregationStatus,
    aggregate,
)
from pydantic import BaseModel, ConfigDict, model_validator


class RowPolicy(StrEnum):
    """Explicit policy for failed / missing / invalid rows.

    ``PROPAGATE`` (default): any such row makes the aggregate ``MISSING_DATA``
    — the aggregate is not reported over an incomplete matrix. ``SKIP``:
    exclude such rows from the reduction, recording the exclusion counts; an
    empty reduction is an explicit non-OK status, never a fabricated value.
    """

    PROPAGATE = "propagate"
    SKIP = "skip"


class RowValue(BaseModel):
    """One planned cell's contribution to an aggregate.

    Exactly one of ``value`` is present, or the row is explicitly not present
    (``missing``) / failed (``failed``) / invalid (``invalid``). None of these
    are inferred from a bare ``None``: each is a declared state so no row is
    silently dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The measured numeric value, when the row produced one.
    value: float | None = None
    #: The row's Rollout failed (e.g. exhausted causal failure / rollout
    #: failure from infrastructure-unknown correctness).
    failed: bool = False
    #: The planned row is absent from the observed matrix.
    missing: bool = False
    #: The row produced an invalid value (e.g. zero-denominator Compression
    #: Ratio) — measured-but-not-a-number.
    invalid: bool = False

    @model_validator(mode="after")
    def _validate(self) -> RowValue:
        flags = (self.failed, self.missing, self.invalid)
        if sum(flags) > 1:
            raise ValueError(
                "a row is at most one of failed / missing / invalid"
            )
        if any(flags) and self.value is not None:
            raise ValueError(
                "a failed / missing / invalid row carries no value"
            )
        if not any(flags) and self.value is None:
            raise ValueError(
                "a present row requires a value (use missing/failed/invalid "
                "to declare absence explicitly)"
            )
        return self

    @property
    def is_present(self) -> bool:
        return self.value is not None

    def to_aggregation_input(self) -> AggregationInput:
        """Project onto a dr-code ``AggregationInput``.

        A present row contributes its value (applicable, present). A missing
        or failed row is applicable-but-absent (``value=None``), so a
        ``propagate`` reduction sees the incompleteness. An invalid row is
        marked not-applicable (it was measured but is not a usable number).
        """

        if self.is_present:
            return AggregationInput(value=self.value, applicable=True)
        if self.invalid:
            return AggregationInput(value=None, applicable=False)
        # missing or failed: applicable slot with no present value.
        return AggregationInput(value=None, applicable=True)


class TaskRows(BaseModel):
    """All planned Repeat-ID rows for one Task.

    ``expected_repeats`` is the planned repeat count; a row list shorter than
    it declares the shortfall as ``missing`` rows so the per-task mean sees the
    full planned denominator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_identity: str
    expected_repeats: int
    rows: tuple[RowValue, ...]

    @model_validator(mode="after")
    def _validate(self) -> TaskRows:
        if self.expected_repeats < 1:
            raise ValueError("expected_repeats must be at least 1")
        if len(self.rows) > self.expected_repeats:
            raise ValueError(
                "more rows than the planned expected_repeats"
            )
        return self

    def completed_rows(self) -> tuple[RowValue, ...]:
        """Rows padded to ``expected_repeats`` with explicit missing rows."""

        shortfall = self.expected_repeats - len(self.rows)
        return self.rows + tuple(
            RowValue(missing=True) for _ in range(shortfall)
        )


class RolloutAggregate(BaseModel):
    """A provenance-bearing Rollout Aggregate.

    Binds a pure dr-code :class:`AggregationOutput` to the aggregate identity
    ``(graph_hash, eval_config_hash)``, the complete planned matrix
    (``task_count`` by ``repeat_count``), and the stated Evaluation Context.
    The numeric reduction stays in the pure ``aggregation_output``; provenance
    is Whetstone's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    graph_hash: str
    eval_config_hash: str
    evaluation_context_id: str
    #: Complete planned matrix shape.
    task_count: int
    repeat_count: int
    #: The pure dr-code output (provenance-free).
    aggregation_output: AggregationOutput
    #: Explicit accounting so no row is silently dropped.
    rows_present: int
    rows_missing: int
    rows_failed: int
    rows_invalid: int

    @model_validator(mode="after")
    def _validate(self) -> RolloutAggregate:
        planned = self.task_count * self.repeat_count
        accounted = (
            self.rows_present
            + self.rows_missing
            + self.rows_failed
            + self.rows_invalid
        )
        if accounted != planned:
            raise ValueError(
                "row accounting does not cover the complete planned matrix: "
                f"{accounted} != {planned}"
            )
        return self


def _row_counts(rows: tuple[RowValue, ...]) -> tuple[int, int, int, int]:
    present = sum(1 for r in rows if r.is_present)
    missing = sum(1 for r in rows if r.missing)
    failed = sum(1 for r in rows if r.failed)
    invalid = sum(1 for r in rows if r.invalid)
    return present, missing, failed, invalid


def _aggregation_config(
    reduction: str, policy: RowPolicy
) -> AggregationConfig:
    definition = AggregationDefinition(
        definition_id="whetstone.rollout_aggregate",
        version="1",
    )
    missing_data = "propagate" if policy is RowPolicy.PROPAGATE else "skip"
    return definition.materialize(
        {
            "reduction": reduction,
            "missing_data": missing_data,
            "zero_denominator": "not_applicable",
        }
    )


def average_binary_test_pass_rate(
    *,
    graph_hash: str,
    eval_config_hash: str,
    evaluation_context_id: str,
    task_rows: tuple[TaskRows, ...],
    repeat_count: int,
    policy: RowPolicy = RowPolicy.PROPAGATE,
) -> RolloutAggregate:
    """Average Binary Test Pass Rate over the complete Task Set.

    Two staged reductions:

    1. **Per Task**: the mean Binary Test Pass Score across the task's Repeat
       IDs. Each task's planned rows are padded to ``repeat_count`` with
       explicit missing rows, so the per-task denominator is the full plan.
    2. **Across the complete Task Set**: the configured unweighted mean of the
       per-task means.

    ``policy`` governs failed / missing rows. Under ``PROPAGATE`` any such row
    makes a task's mean (and hence the aggregate) ``MISSING_DATA``. Under
    ``SKIP`` those rows are excluded from the per-task denominator, and a task
    with no usable rows contributes a not-applicable slot to the cross-task
    mean. No row is silently dropped: all are counted in the aggregate's
    provenance.
    """

    per_task_config = _aggregation_config("mean", policy)

    all_rows: list[RowValue] = []
    per_task_inputs: list[AggregationInput] = []
    for task in task_rows:
        if task.expected_repeats != repeat_count:
            raise ValueError(
                f"task {task.task_identity} expected_repeats "
                f"{task.expected_repeats} != plan repeat_count {repeat_count}"
            )
        completed = task.completed_rows()
        all_rows.extend(completed)
        task_output = aggregate(
            per_task_config,
            tuple(row.to_aggregation_input() for row in completed),
        )
        # The per-task mean feeds the cross-task reduction. A non-OK per-task
        # status is carried explicitly: propagate -> the missing per-task value
        # flows as an applicable-but-absent slot; not-applicable (no usable
        # rows under skip) -> a not-applicable slot.
        if task_output.status is AggregationStatus.OK:
            per_task_inputs.append(
                AggregationInput(value=task_output.value, applicable=True)
            )
        elif task_output.status is AggregationStatus.NOT_APPLICABLE:
            per_task_inputs.append(
                AggregationInput(value=None, applicable=False)
            )
        else:
            # MISSING_DATA or ZERO_DENOMINATOR: an applicable slot with no
            # present value, so a propagate cross-task reduction sees it.
            per_task_inputs.append(
                AggregationInput(value=None, applicable=True)
            )

    cross_task_config = _aggregation_config("mean", policy)
    output = aggregate(cross_task_config, tuple(per_task_inputs))

    present, missing, failed, invalid = _row_counts(tuple(all_rows))
    return RolloutAggregate(
        name="average_binary_test_pass_rate",
        graph_hash=graph_hash,
        eval_config_hash=eval_config_hash,
        evaluation_context_id=evaluation_context_id,
        task_count=len(task_rows),
        repeat_count=repeat_count,
        aggregation_output=output,
        rows_present=present,
        rows_missing=missing,
        rows_failed=failed,
        rows_invalid=invalid,
    )


def mean_compression_ratio(
    *,
    graph_hash: str,
    eval_config_hash: str,
    evaluation_context_id: str,
    rows: tuple[RowValue, ...],
    task_count: int,
    repeat_count: int,
    policy: RowPolicy = RowPolicy.PROPAGATE,
) -> RolloutAggregate:
    """Mean Compression Ratio over the complete planned matrix.

    A single configured complete-matrix mean over measured Compression Ratios.
    ``rows`` MUST be the complete ``task_count`` by ``repeat_count`` matrix,
    each cell an explicit :class:`RowValue` (present value, or an explicit
    ``missing`` / ``failed`` / ``invalid`` — the last being a zero-denominator
    Compression Ratio). Under ``PROPAGATE`` any non-present cell makes the
    aggregate ``MISSING_DATA``; under ``SKIP`` invalid cells are excluded as
    not-applicable and missing/failed as applicable-but-absent, and a wholly
    empty reduction is an explicit non-OK status. No cell is silently dropped.
    """

    planned = task_count * repeat_count
    if len(rows) != planned:
        raise ValueError(
            "mean_compression_ratio requires the complete planned matrix: "
            f"{len(rows)} rows != {planned}"
        )

    config = _aggregation_config("mean", policy)
    output = aggregate(
        config,
        tuple(row.to_aggregation_input() for row in rows),
    )

    present, missing, failed, invalid = _row_counts(rows)
    return RolloutAggregate(
        name="mean_compression_ratio",
        graph_hash=graph_hash,
        eval_config_hash=eval_config_hash,
        evaluation_context_id=evaluation_context_id,
        task_count=task_count,
        repeat_count=repeat_count,
        aggregation_output=output,
        rows_present=present,
        rows_missing=missing,
        rows_failed=failed,
        rows_invalid=invalid,
    )


__all__ = [
    "RolloutAggregate",
    "RowPolicy",
    "RowValue",
    "TaskRows",
    "average_binary_test_pass_rate",
    "mean_compression_ratio",
]
