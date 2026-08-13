from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from whetstone.core.identity import (
    TypedRef,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.evaluation import (
    AggregationConfig,
    AggregationDefinition,
    AggregationInput,
    AggregationOutput,
    AggregationStatus,
    EvalConfig,
    SamplePlan,
    SamplingConfig,
    TaskSet,
    VariableSpec,
    aggregate,
)
from whetstone.evaluation.attribution import require_exclusive_row_state

AGGREGATE_SCHEMA = "whetstone.aggregate"


class RowPolicy(StrEnum):
    PROPAGATE = "propagate"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class CompletenessPolicy:
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
        value = float(self.max_skip_fraction)
        return "0.0" if value == 0.0 else repr(value)

    def within_tolerance(self, *, skipped: int, planned: int) -> bool:
        if planned <= 0:
            return True
        return (skipped / planned) <= self.max_skip_fraction


@dataclass(frozen=True, slots=True)
class EvaluationMatrixPlan:
    eval_config: EvalConfig
    sampling_config: SamplingConfig
    task_set: TaskSet
    sample_plan: SamplePlan
    aggregation_config: AggregationConfig

    def __post_init__(self) -> None:
        expected_types = (
            ("eval_config", self.eval_config, EvalConfig),
            ("sampling_config", self.sampling_config, SamplingConfig),
            ("task_set", self.task_set, TaskSet),
            ("sample_plan", self.sample_plan, SamplePlan),
            (
                "aggregation_config",
                self.aggregation_config,
                AggregationConfig,
            ),
        )
        for field, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{field} must be a Whetstone {expected_type.__name__}"
                )

        if (
            self.eval_config.sampling_config_hash
            != self.sampling_config.config_hash
        ):
            raise ValueError(
                "eval_config sampling_config_hash does not match "
                "sampling_config"
            )
        if (
            self.eval_config.aggregation_config_hash
            != self.aggregation_config.config_hash
        ):
            raise ValueError(
                "eval_config aggregation_config_hash does not match "
                "aggregation_config"
            )

        sampling = self.sampling_config.assignment_dict()
        if sampling.get("task_set_hash") != self.task_set.identity_hash():
            raise ValueError(
                "sampling_config task_set_hash does not match task_set"
            )
        if (
            sampling.get("sample_plan_hash")
            != self.sample_plan.identity_hash()
        ):
            raise ValueError(
                "sampling_config sample_plan_hash does not match sample_plan"
            )
        if self.task_set.selection_rule is not None:
            raise ValueError(
                "task_set must be an explicit task identity manifest"
            )
        if self.task_set.task_hashes != self.sample_plan.task_hashes:
            raise ValueError(
                "task_set and sample_plan task identities do not match"
            )

        _policy_from_aggregation_config(self.aggregation_config)

    @property
    def policy(self) -> CompletenessPolicy:

        return _policy_from_aggregation_config(self.aggregation_config)


@dataclass(frozen=True, slots=True)
class RowValue:
    value: float | None = None

    failed: bool = False

    missing: bool = False

    invalid: bool = False

    def __post_init__(self) -> None:
        require_exclusive_row_state(
            scored=self.value is not None,
            failed=self.failed,
            missing=self.missing,
            invalid=self.invalid,
        )

    @property
    def is_present(self) -> bool:
        return self.value is not None

    def to_aggregation_input(self) -> AggregationInput:

        if self.is_present:
            return AggregationInput(value=self.value, applicable=True)
        if self.invalid:
            return AggregationInput(value=None, applicable=False)

        return AggregationInput(value=None, applicable=True)


@dataclass(frozen=True, slots=True)
class TaskRows:
    task_hash: str
    rows: tuple[RowValue, ...]

    def completed_rows(self, num_samples: int) -> tuple[RowValue, ...]:

        if len(self.rows) > num_samples:
            raise ValueError(
                f"task {self.task_hash} has {len(self.rows)} rows, "
                f"more than plan num_samples {num_samples}"
            )
        shortfall = num_samples - len(self.rows)
        return self.rows + tuple(
            RowValue(missing=True) for _ in range(shortfall)
        )


@dataclass(frozen=True, slots=True)
class Aggregate:
    name: str
    graph_hash: str
    eval_config_hash: str

    task_count: int
    num_samples: int

    aggregation_output: AggregationOutput

    rows_present: int
    rows_missing: int
    rows_failed: int
    rows_invalid: int

    def __post_init__(self) -> None:
        require_full_hash(self.graph_hash, field="graph_hash")
        require_full_hash(self.eval_config_hash, field="eval_config_hash")
        if self.task_count < 0:
            raise ValueError("task_count cannot be negative")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        planned = self.task_count * self.num_samples
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

    def record_content(self) -> dict[str, object]:
        return {
            "name": self.name,
            "graph_hash": self.graph_hash,
            "eval_config_hash": self.eval_config_hash,
            "task_count": self.task_count,
            "num_samples": self.num_samples,
            "aggregation_output": self.aggregation_output.model_dump(
                mode="json"
            ),
            "rows_present": self.rows_present,
            "rows_missing": self.rows_missing,
            "rows_failed": self.rows_failed,
            "rows_invalid": self.rows_invalid,
        }

    def record_ref(self) -> TypedRef:
        return typed_ref_for_record(AGGREGATE_SCHEMA, self.record_content())


def _row_counts(rows: tuple[RowValue, ...]) -> tuple[int, int, int, int]:
    present = sum(1 for r in rows if r.is_present)
    missing = sum(1 for r in rows if r.missing)
    failed = sum(1 for r in rows if r.failed)
    invalid = sum(1 for r in rows if r.invalid)
    return present, missing, failed, invalid


SKIP_TOLERANCE_VARIABLE = "max_skip_fraction"


def tolerance_variable_spec() -> VariableSpec:
    return VariableSpec(
        name=SKIP_TOLERANCE_VARIABLE,
        default="0.0",
        has_default=True,
    )


def aggregation_definition(definition_id: str) -> AggregationDefinition:
    base = AggregationDefinition(definition_id=definition_id, version="1")
    return base.model_copy(
        update={"variables": (*base.variables, tolerance_variable_spec())}
    )


def _policy_from_aggregation_config(
    config: AggregationConfig,
) -> CompletenessPolicy:
    assignment = config.assignment_dict()
    if assignment.get("reduction") != "mean":
        raise ValueError("aggregation_config reduction must be 'mean'")
    if assignment.get("zero_denominator") != "not_applicable":
        raise ValueError(
            "aggregation_config zero_denominator must be 'not_applicable'"
        )

    missing_data = assignment.get("missing_data")
    try:
        row_policy = RowPolicy(missing_data)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "aggregation_config missing_data must be 'propagate' or 'skip'"
        ) from error

    token = assignment.get(SKIP_TOLERANCE_VARIABLE)
    if not isinstance(token, str):
        raise ValueError(
            "aggregation_config max_skip_fraction must be a canonical string"
        )
    try:
        policy = CompletenessPolicy(
            row_policy=row_policy,
            max_skip_fraction=float(token),
        )
    except ValueError as error:
        raise ValueError(
            "aggregation_config max_skip_fraction must be a finite float "
            "in [0.0, 1.0]"
        ) from error
    if token != policy.skip_fraction_token():
        raise ValueError(
            "aggregation_config max_skip_fraction must use its exact "
            "round-trippable token"
        )
    return policy


def enforce_skip_tolerance(
    output: AggregationOutput,
    *,
    policy: CompletenessPolicy,
    skipped: int,
    planned: int,
) -> AggregationOutput:
    if policy.row_policy is not RowPolicy.SKIP:
        return output
    if policy.within_tolerance(skipped=skipped, planned=planned):
        return output
    return output.model_copy(
        update={"value": None, "status": AggregationStatus.MISSING_DATA}
    )


def unweighted_task_mean(
    *,
    aggregate_name: str,
    graph_hash: str,
    task_rows: tuple[TaskRows, ...],
    plan: EvaluationMatrixPlan,
) -> Aggregate:

    if not isinstance(aggregate_name, str):
        raise TypeError("aggregate_name must be a string")
    if not aggregate_name.strip():
        raise ValueError("aggregate_name must be nonblank")
    if not isinstance(plan, EvaluationMatrixPlan):
        raise TypeError("plan must be an EvaluationMatrixPlan")

    policy = plan.policy
    num_samples = plan.sample_plan.num_samples
    planned_task_hashes = plan.sample_plan.task_hashes

    observed_by_task_hash: dict[str, TaskRows] = {}
    for task in task_rows:
        if task.task_hash in observed_by_task_hash:
            raise ValueError(
                f"duplicate observed task identity: {task.task_hash}"
            )
        observed_by_task_hash[task.task_hash] = task
    extra_task_hashes = set(observed_by_task_hash) - set(planned_task_hashes)
    if extra_task_hashes:
        extras = ", ".join(sorted(extra_task_hashes))
        raise ValueError(f"observed unplanned task identities: {extras}")

    reconciled = tuple(
        observed_by_task_hash.get(
            task_hash,
            TaskRows(task_hash=task_hash, rows=()),
        )
        for task_hash in planned_task_hashes
    )

    all_rows: list[RowValue] = []
    per_task_inputs: list[AggregationInput] = []
    for task in reconciled:
        completed = task.completed_rows(num_samples)
        all_rows.extend(completed)
        task_output = aggregate(
            plan.aggregation_config,
            tuple(row.to_aggregation_input() for row in completed),
        )

        if task_output.status is AggregationStatus.OK:
            per_task_inputs.append(
                AggregationInput(value=task_output.value, applicable=True)
            )
        elif task_output.status is AggregationStatus.NOT_APPLICABLE:
            per_task_inputs.append(
                AggregationInput(value=None, applicable=False)
            )
        else:
            per_task_inputs.append(
                AggregationInput(value=None, applicable=True)
            )

    output = aggregate(plan.aggregation_config, tuple(per_task_inputs))

    present, missing, failed, invalid = _row_counts(tuple(all_rows))
    output = enforce_skip_tolerance(
        output,
        policy=policy,
        skipped=missing + failed + invalid,
        planned=len(all_rows),
    )
    return Aggregate(
        name=aggregate_name,
        graph_hash=graph_hash,
        eval_config_hash=plan.eval_config.config_hash,
        task_count=len(planned_task_hashes),
        num_samples=num_samples,
        aggregation_output=output,
        rows_present=present,
        rows_missing=missing,
        rows_failed=failed,
        rows_invalid=invalid,
    )


__all__ = [
    "AGGREGATE_SCHEMA",
    "Aggregate",
    "CompletenessPolicy",
    "EvaluationMatrixPlan",
    "RowPolicy",
    "RowValue",
    "TaskRows",
    "aggregation_definition",
    "enforce_skip_tolerance",
    "tolerance_variable_spec",
    "unweighted_task_mean",
]
