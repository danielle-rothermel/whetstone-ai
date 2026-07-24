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

from dataclasses import dataclass
from enum import StrEnum

from dr_code.eval import (
    AggregationConfig,
    AggregationDefinition,
    AggregationInput,
    AggregationOutput,
    AggregationStatus,
    VariableSpec,
    aggregate,
)

from whetstone.result.schema import require_full_hash


class RowPolicy(StrEnum):
    """Explicit policy for failed / missing / invalid rows.

    ``PROPAGATE`` (default): any such row makes the aggregate ``MISSING_DATA``
    — the aggregate is not reported over an incomplete matrix. ``SKIP``:
    exclude such rows from the reduction, recording the exclusion counts; an
    empty reduction is an explicit non-OK status, never a fabricated value.
    """

    PROPAGATE = "propagate"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class CompletenessPolicy:
    """A declared missing-data policy with an optional bounded skip tolerance.

    ``row_policy`` is the dr-code ``missing_data`` rule. ``max_skip_fraction``
    is the DECLARED completeness tolerance: under ``SKIP`` the aggregate is
    only certified when the fraction of skipped (missing + failed + invalid)
    rows over the complete planned matrix is at or below this bound; beyond it
    the aggregate is forced ``MISSING_DATA`` (an incomplete arm), never a value
    reduced over an out-of-tolerance matrix. Under ``PROPAGATE`` the bound is
    inert (any skipped row already makes the aggregate missing).

    The tolerance is identity-bearing: it is folded into the Aggregation Config
    identity (a distinct ``max_skip_fraction`` yields a distinct
    ``eval_config_hash``). ``0.0`` is exact completeness — SKIP with a ``0.0``
    bound certifies only a fully complete matrix, matching PROPAGATE's numeric
    result while remaining a declared, distinct config identity.
    """

    row_policy: RowPolicy = RowPolicy.PROPAGATE
    max_skip_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.row_policy, RowPolicy):
            raise TypeError("row_policy must be a RowPolicy")
        if not 0.0 <= self.max_skip_fraction <= 1.0:
            raise ValueError(
                "max_skip_fraction must be in [0.0, 1.0]; got "
                f"{self.max_skip_fraction}"
            )

    @property
    def missing_data(self) -> str:
        return (
            "propagate" if self.row_policy is RowPolicy.PROPAGATE else "skip"
        )

    def skip_fraction_token(self) -> str:
        """The canonical, identity-bearing string form of the tolerance.

        A fixed 4-decimal token so ``0.02`` and ``0.0200`` are one identity
        and float formatting never perturbs the config hash.
        """
        return f"{self.max_skip_fraction:.4f}"

    def within_tolerance(self, *, skipped: int, planned: int) -> bool:
        """Whether ``skipped`` of ``planned`` rows is within the bound.

        Only meaningful under ``SKIP``; under ``PROPAGATE`` any skip is
        already fatal to the scalar via the dr-code reduction, so this is not
        consulted.
        """
        if planned <= 0:
            return True
        return (skipped / planned) <= self.max_skip_fraction


_PROPAGATE_POLICY = CompletenessPolicy()


@dataclass(frozen=True, slots=True)
class RowValue:
    """One planned cell's contribution to an aggregate.

    Exactly one of ``value`` is present, or the row is explicitly not present
    (``missing``) / failed (``failed``) / invalid (``invalid``). None of these
    are inferred from a bare ``None``: each is a declared state so no row is
    silently dropped.
    """

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

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class TaskRows:
    """All planned Repeat-ID rows for one Task.

    ``expected_repeats`` is the planned repeat count; a row list shorter than
    it declares the shortfall as ``missing`` rows so the per-task mean sees the
    full planned denominator.
    """

    task_identity: str
    expected_repeats: int
    rows: tuple[RowValue, ...]

    def __post_init__(self) -> None:
        if self.expected_repeats < 1:
            raise ValueError("expected_repeats must be at least 1")
        if len(self.rows) > self.expected_repeats:
            raise ValueError("more rows than the planned expected_repeats")

    def completed_rows(self) -> tuple[RowValue, ...]:
        """Rows padded to ``expected_repeats`` with explicit missing rows."""

        shortfall = self.expected_repeats - len(self.rows)
        return self.rows + tuple(
            RowValue(missing=True) for _ in range(shortfall)
        )


@dataclass(frozen=True, slots=True)
class RolloutAggregate:
    """A provenance-bearing Rollout Aggregate.

    Binds a pure dr-code :class:`AggregationOutput` to the aggregate identity
    ``(graph_hash, eval_config_hash)``, the complete planned matrix
    (``task_count`` by ``repeat_count``), and the stated Evaluation Context.
    The numeric reduction stays in the pure ``aggregation_output``; provenance
    is Whetstone's.
    """

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

    def __post_init__(self) -> None:
        require_full_hash(self.graph_hash, field="graph_hash")
        require_full_hash(self.eval_config_hash, field="eval_config_hash")
        if not self.evaluation_context_id:
            raise ValueError("evaluation_context_id must be non-empty")
        if self.task_count < 0:
            raise ValueError("task_count cannot be negative")
        if self.repeat_count < 1:
            raise ValueError("repeat_count must be at least 1")
        planned = self.task_count * self.repeat_count
        counts = (
            self.rows_present,
            self.rows_missing,
            self.rows_failed,
            self.rows_invalid,
        )
        if any(count < 0 for count in counts):
            raise ValueError("row counts cannot be negative")
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


def _row_counts(rows: tuple[RowValue, ...]) -> tuple[int, int, int, int]:
    present = sum(1 for r in rows if r.is_present)
    missing = sum(1 for r in rows if r.missing)
    failed = sum(1 for r in rows if r.failed)
    invalid = sum(1 for r in rows if r.invalid)
    return present, missing, failed, invalid


#: The extra declared Variable that folds the bounded skip tolerance into the
#: Aggregation Config identity.
SKIP_TOLERANCE_VARIABLE = "max_skip_fraction"


def tolerance_variable_spec() -> VariableSpec:
    """The ``max_skip_fraction`` :class:`VariableSpec` (declared, defaulted).

    Returned as a builder so callers materialize an Aggregation Config whose
    identity folds in the tolerance.
    """
    return VariableSpec(
        name=SKIP_TOLERANCE_VARIABLE,
        default="0.0000",
        has_default=True,
    )


def aggregation_definition(definition_id: str) -> AggregationDefinition:
    """An Aggregation Definition that additionally declares the skip tolerance.

    The base dr-code definition declares reduction / missing_data /
    zero_denominator; this appends the identity-bearing ``max_skip_fraction``
    Variable so a declared completeness tolerance changes the config identity.
    """
    base = AggregationDefinition(definition_id=definition_id, version="1")
    return base.model_copy(
        update={"variables": (*base.variables, tolerance_variable_spec())}
    )


def _aggregation_config(
    reduction: str, policy: CompletenessPolicy
) -> AggregationConfig:
    if not isinstance(policy, CompletenessPolicy):
        raise TypeError(
            "policy must be a CompletenessPolicy with a declared "
            "max_skip_fraction"
        )
    return aggregation_definition("whetstone.rollout_aggregate").materialize(
        {
            "reduction": reduction,
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            SKIP_TOLERANCE_VARIABLE: policy.skip_fraction_token(),
        }
    )


def enforce_skip_tolerance(
    output: AggregationOutput,
    *,
    policy: CompletenessPolicy,
    skipped: int,
    planned: int,
) -> AggregationOutput:
    """Force ``MISSING_DATA`` when SKIP exceeds the declared skip tolerance.

    Under ``SKIP`` the dr-code reduction happily certifies a value over the
    surviving rows no matter how many were skipped; the DECLARED completeness
    tolerance bounds that. When the skipped fraction exceeds
    ``max_skip_fraction`` the arm is out of tolerance and its scalar is set to
    ``None`` (``MISSING_DATA``) so the incomplete-arm guard fires — the skipped
    rows are still recorded as explicit counts on the aggregate. Within the
    bound the reduced value stands unchanged.
    """
    if policy.row_policy is not RowPolicy.SKIP:
        return output
    if policy.within_tolerance(skipped=skipped, planned=planned):
        return output
    return output.model_copy(
        update={"value": None, "status": AggregationStatus.MISSING_DATA}
    )


def average_binary_test_pass_rate(
    *,
    graph_hash: str,
    eval_config_hash: str,
    evaluation_context_id: str,
    task_rows: tuple[TaskRows, ...],
    repeat_count: int,
    policy: CompletenessPolicy = _PROPAGATE_POLICY,
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
    output = enforce_skip_tolerance(
        output,
        policy=policy,
        skipped=missing + failed + invalid,
        planned=len(all_rows),
    )
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
    policy: CompletenessPolicy = _PROPAGATE_POLICY,
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
    output = enforce_skip_tolerance(
        output,
        policy=policy,
        skipped=missing + failed + invalid,
        planned=len(rows),
    )
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
    "CompletenessPolicy",
    "RolloutAggregate",
    "RowPolicy",
    "RowValue",
    "TaskRows",
    "aggregation_definition",
    "average_binary_test_pass_rate",
    "enforce_skip_tolerance",
    "mean_compression_ratio",
    "tolerance_variable_spec",
]
