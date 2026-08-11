from __future__ import annotations

from dataclasses import replace
from inspect import Parameter, signature
from typing import cast

import pytest
from pydantic import JsonValue

import whetstone.evaluation.code as code_eval
from tests.envs.support import tiny_experiment
from tests.evaluation.code.support import FULL_HASH
from whetstone.core.identity import typed_ref_for_record
from whetstone.evaluation import (
    AggregationConfig,
    AggregationOutput,
    AggregationStatus,
    EvalDefinition,
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    PreprocessingStepBinding,
    RepeatPlan,
    SamplingDefinition,
    SelectionRule,
    TaskSet,
)
from whetstone.evaluation.aggregate import (
    ROLLOUT_AGGREGATE_SCHEMA,
    SKIP_TOLERANCE_VARIABLE,
    CompletenessPolicy,
    EvaluationMatrixPlan,
    RolloutAggregate,
    RowPolicy,
    RowValue,
    TaskRows,
    aggregation_definition,
    unweighted_task_mean,
)

CTX = "c" * 64
AGGREGATE_NAME = "custom_scalar_quality"
PROPAGATE = CompletenessPolicy()


def _procedure_config() -> EvaluationProcedureConfig:
    preprocessing = PreprocessingDefinition(
        definition_id="test.pre",
        version="1",
        steps=(
            PreprocessingStepBinding(instance_name="all", step="return_all"),
        ),
    ).materialize(resolved_steps=(("all", "return_all", "1"),))
    metric = MetricExtractionDefinition(
        definition_id="test.metric",
        version="1",
        questions=(MetricQuestionBinding(metric="code_leakage", on="output"),),
    ).materialize(resolved_operators=(("code_leakage", "1"),))
    return EvaluationProcedureDefinition(
        definition_id="test.procedure",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric,
        assignment={"zero_denominator": "not_applicable"},
    )


def _aggregation_config(policy: CompletenessPolicy) -> AggregationConfig:
    return aggregation_definition("test.rollout_aggregate").materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            SKIP_TOLERANCE_VARIABLE: policy.skip_fraction_token(),
        }
    )


def _plan(
    task_identities: tuple[str, ...] = ("t1",),
    *,
    repeat_count: int = 3,
    policy: CompletenessPolicy = PROPAGATE,
) -> EvaluationMatrixPlan:
    task_set = TaskSet(
        manifest_id="test.tasks",
        version="1",
        dataset_revision="revision",
        task_identities=task_identities,
    )
    repeat_plan = RepeatPlan(
        plan_id="test.repeats",
        version="1",
        task_identities=task_identities,
        repeat_count=repeat_count,
    )
    sampling = SamplingDefinition(
        definition_id="test.sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )
    aggregation = _aggregation_config(policy)
    eval_config = EvalDefinition(
        definition_id="test.eval",
        version="1",
    ).materialize(
        sampling=sampling,
        evaluation_procedure=_procedure_config(),
        aggregation=aggregation,
    )
    return EvaluationMatrixPlan(
        eval_config=eval_config,
        sampling_config=sampling,
        task_set=task_set,
        repeat_plan=repeat_plan,
        aggregation_config=aggregation,
    )


def _task_mean(
    task_rows: tuple[TaskRows, ...],
    *,
    plan: EvaluationMatrixPlan | None = None,
    aggregate_name: str = AGGREGATE_NAME,
) -> RolloutAggregate:
    return unweighted_task_mean(
        aggregate_name=aggregate_name,
        graph_hash=FULL_HASH,
        evaluation_binding_hash=CTX,
        task_rows=task_rows,
        plan=plan or _plan(),
    )


def _task(task_identity: str, *rows: RowValue) -> TaskRows:
    return TaskRows(task_identity=task_identity, rows=rows)


def test_aggregate_derives_identity_binding_and_shape_from_plan() -> None:
    plan = _plan(("t1",), repeat_count=2)
    aggregate = _task_mean(
        (_task("t1", RowValue(value=1.0), RowValue(value=0.0)),),
        plan=plan,
    )

    assert aggregate.graph_hash == FULL_HASH
    assert aggregate.eval_config_hash == plan.eval_config.config_identity_hash
    assert aggregate.evaluation_binding_hash == CTX
    assert aggregate.task_count == 1
    assert aggregate.repeat_count == 2
    assert isinstance(aggregate.aggregation_output, AggregationOutput)
    parameters = tuple(signature(unweighted_task_mean).parameters.values())
    assert tuple(parameter.name for parameter in parameters) == (
        "aggregate_name",
        "graph_hash",
        "evaluation_binding_hash",
        "task_rows",
        "plan",
    )
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters
    )
    assert all(
        parameter.default is Parameter.empty for parameter in parameters
    )


def test_rollout_aggregate_wire_contract_is_pinned() -> None:
    aggregate = _task_mean(
        (_task("t1", RowValue(value=1.0), RowValue(value=0.0)),),
        plan=_plan(("t1",), repeat_count=2),
    )
    content = aggregate.record_content()

    assert ROLLOUT_AGGREGATE_SCHEMA == "whetstone.rollout_aggregate"
    assert tuple(content) == (
        "name",
        "graph_hash",
        "eval_config_hash",
        "evaluation_binding_hash",
        "task_count",
        "repeat_count",
        "aggregation_output",
        "rows_present",
        "rows_missing",
        "rows_failed",
        "rows_invalid",
    )
    assert aggregate.record_ref() == typed_ref_for_record(
        "whetstone.rollout_aggregate", content
    )


@pytest.mark.precheck
def test_unweighted_task_mean_is_two_stage_not_row_weighted() -> None:
    plan = _plan(
        ("t1", "t2"),
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP,
            max_skip_fraction=1 / 3,
        ),
    )
    aggregate = _task_mean(
        (
            _task(
                "t1",
                RowValue(value=2.0),
                RowValue(value=4.0),
                RowValue(value=6.0),
            ),
            _task("t2", RowValue(value=20.0)),
        ),
        plan=plan,
    )

    assert aggregate.aggregation_output.status is AggregationStatus.OK
    assert aggregate.aggregation_output.value == pytest.approx(12.0)
    assert aggregate.rows_present == 4
    assert aggregate.rows_missing == 2


def test_unweighted_task_mean_propagates_caller_owned_name() -> None:
    aggregate = _task_mean(
        (_task("t1", RowValue(value=3.5)),),
        plan=_plan(repeat_count=1),
        aggregate_name="latency_adjusted_quality",
    )
    assert aggregate.name == "latency_adjusted_quality"


@pytest.mark.parametrize("aggregate_name", ["", " ", "\t\n"])
def test_unweighted_task_mean_rejects_blank_name(
    aggregate_name: str,
) -> None:
    with pytest.raises(ValueError, match="aggregate_name must be nonblank"):
        _task_mean((), aggregate_name=aggregate_name)


def test_unweighted_task_mean_rejects_non_string_name() -> None:
    with pytest.raises(TypeError, match="aggregate_name must be a string"):
        _task_mean((), aggregate_name=cast(str, 42))


def test_task_rows_are_padded_to_plan_repeat_count() -> None:
    aggregate = _task_mean(
        (_task("t1", RowValue(value=1.0)),),
        plan=_plan(repeat_count=3),
    )
    assert aggregate.rows_present == 1
    assert aggregate.rows_missing == 2
    assert (
        aggregate.aggregation_output.status is AggregationStatus.MISSING_DATA
    )


def test_task_rows_reject_more_rows_than_plan() -> None:
    with pytest.raises(ValueError, match="more than plan repeat_count"):
        _task_mean(
            (_task("t1", RowValue(value=1.0), RowValue(value=2.0)),),
            plan=_plan(repeat_count=1),
        )


def test_task_rows_have_no_competing_expected_repeats_field() -> None:
    assert "expected_repeats" not in TaskRows.__dataclass_fields__


def test_failed_row_propagates_missing_data() -> None:
    aggregate = _task_mean(
        (_task("t1", RowValue(value=1.0), RowValue(failed=True)),),
        plan=_plan(repeat_count=2),
    )
    assert aggregate.rows_failed == 1
    assert (
        aggregate.aggregation_output.status is AggregationStatus.MISSING_DATA
    )


def test_failed_rows_still_visible_in_provenance() -> None:
    experiment = tiny_experiment("c18")
    sampling = experiment.eval_configs.internal
    task_rows = tuple(
        TaskRows(
            task_identity=task_identity,
            rows=(RowValue(failed=True), RowValue(failed=True)),
        )
        for task_identity in sampling.task_set.task_identities
    )
    agg = unweighted_task_mean(
        aggregate_name="env_exact_match",
        graph_hash=experiment.rollout_definition.graph_hash,
        evaluation_binding_hash="c" * 64,
        task_rows=task_rows,
        plan=sampling.evaluation_matrix_plan,
    )
    assert agg.rows_failed == len(task_rows) * 2
    assert agg.rows_present == 0
    assert agg.aggregation_output.status is not AggregationStatus.OK


def _rows_with_skips(n_tasks: int, skipped: int) -> tuple[TaskRows, ...]:
    return tuple(
        _task(
            f"t{index}",
            RowValue(failed=True) if index < skipped else RowValue(value=1.0),
        )
        for index in range(n_tasks)
    )


def test_skip_within_tolerance_certifies_a_value() -> None:
    task_ids = tuple(f"t{index}" for index in range(100))
    aggregate = _task_mean(
        _rows_with_skips(100, skipped=1),
        plan=_plan(
            task_ids,
            repeat_count=1,
            policy=CompletenessPolicy(
                row_policy=RowPolicy.SKIP,
                max_skip_fraction=0.02,
            ),
        ),
    )
    assert aggregate.rows_failed == 1
    assert aggregate.rows_present == 99
    assert aggregate.aggregation_output.status is AggregationStatus.OK
    assert aggregate.aggregation_output.value == pytest.approx(1.0)


def test_skip_over_tolerance_forces_missing_data() -> None:
    task_ids = tuple(f"t{index}" for index in range(100))
    aggregate = _task_mean(
        _rows_with_skips(100, skipped=3),
        plan=_plan(
            task_ids,
            repeat_count=1,
            policy=CompletenessPolicy(
                row_policy=RowPolicy.SKIP,
                max_skip_fraction=0.02,
            ),
        ),
    )
    assert aggregate.rows_failed == 3
    assert (
        aggregate.aggregation_output.status is AggregationStatus.MISSING_DATA
    )
    assert aggregate.aggregation_output.value is None


def test_exact_tolerance_tokens_preserve_behavioral_identity() -> None:
    lower = CompletenessPolicy(
        row_policy=RowPolicy.SKIP,
        max_skip_fraction=0.02001,
    )
    upper = CompletenessPolicy(
        row_policy=RowPolicy.SKIP,
        max_skip_fraction=0.02004,
    )
    assert lower.skip_fraction_token() != upper.skip_fraction_token()
    assert not lower.within_tolerance(skipped=2002, planned=100_000)
    assert upper.within_tolerance(skipped=2002, planned=100_000)
    assert (
        _plan(policy=lower).eval_config.config_identity_hash
        != _plan(policy=upper).eval_config.config_identity_hash
    )


def test_equivalent_tolerances_share_one_normalized_token() -> None:
    assert CompletenessPolicy(
        max_skip_fraction=0.02
    ).skip_fraction_token() == (
        CompletenessPolicy(
            max_skip_fraction=float("0.020")
        ).skip_fraction_token()
    )
    assert CompletenessPolicy(
        max_skip_fraction=-0.0
    ).skip_fraction_token() == ("0.0")


@pytest.mark.parametrize(
    "tolerance",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_tolerance_rejects_nonfinite_and_out_of_range_values(
    tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="max_skip_fraction"):
        CompletenessPolicy(max_skip_fraction=tolerance)


def test_plan_rejects_noncanonical_tolerance_token() -> None:
    base = _plan()
    aggregation = aggregation_definition("test.rollout_aggregate").materialize(
        {
            "reduction": "mean",
            "missing_data": "propagate",
            "zero_denominator": "not_applicable",
            SKIP_TOLERANCE_VARIABLE: "0.0000",
        }
    )
    eval_config = EvalDefinition(
        definition_id="test.eval",
        version="1",
    ).materialize(
        sampling=base.sampling_config,
        evaluation_procedure=_procedure_config(),
        aggregation=aggregation,
    )
    with pytest.raises(ValueError, match="round-trippable token"):
        replace(
            base,
            eval_config=eval_config,
            aggregation_config=aggregation,
        )


def test_plan_rejects_eval_to_sampling_mismatch() -> None:
    one = _plan(("t1",))
    other = _plan(("t2",))
    with pytest.raises(ValueError, match="sampling_config_hash"):
        replace(one, sampling_config=other.sampling_config)


def test_plan_rejects_eval_to_aggregation_mismatch() -> None:
    one = _plan()
    other = _plan(
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP,
            max_skip_fraction=0.1,
        )
    )
    with pytest.raises(ValueError, match="aggregation_config_hash"):
        replace(one, aggregation_config=other.aggregation_config)


def test_plan_rejects_sampling_to_task_set_mismatch() -> None:
    plan = _plan(("t1",))
    other_task_set = TaskSet(
        manifest_id="other.tasks",
        version="1",
        dataset_revision="revision",
        task_identities=("t1",),
    )
    with pytest.raises(ValueError, match="task_set_hash"):
        replace(plan, task_set=other_task_set)


def test_plan_rejects_sampling_to_repeat_plan_mismatch() -> None:
    plan = _plan(("t1",), repeat_count=1)
    other_repeat_plan = RepeatPlan(
        plan_id="other.repeats",
        version="1",
        task_identities=("t1",),
        repeat_count=1,
    )
    with pytest.raises(ValueError, match="repeat_plan_hash"):
        replace(plan, repeat_plan=other_repeat_plan)


def test_plan_rejects_task_set_repeat_plan_identity_mismatch() -> None:
    task_set = TaskSet(
        manifest_id="test.tasks",
        version="1",
        dataset_revision="revision",
        task_identities=("t1",),
    )
    repeat_plan = RepeatPlan(
        plan_id="test.repeats",
        version="1",
        task_identities=("t2",),
        repeat_count=1,
    )
    sampling = SamplingDefinition(
        definition_id="test.sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )
    aggregation = _aggregation_config(CompletenessPolicy())
    eval_config = EvalDefinition(
        definition_id="test.eval",
        version="1",
    ).materialize(
        sampling=sampling,
        evaluation_procedure=_procedure_config(),
        aggregation=aggregation,
    )
    with pytest.raises(ValueError, match="task identities do not match"):
        EvaluationMatrixPlan(
            eval_config=eval_config,
            sampling_config=sampling,
            task_set=task_set,
            repeat_plan=repeat_plan,
            aggregation_config=aggregation,
        )


def test_plan_rejects_unresolved_task_selection_rule() -> None:
    task_set = TaskSet(
        manifest_id="test.tasks",
        version="1",
        dataset_revision="revision",
        selection_rule=SelectionRule(kind="first", params=(("count", "1"),)),
    )
    base = _plan(("t1",))
    sampling = SamplingDefinition(
        definition_id="test.sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": base.repeat_plan.identity_hash(),
        }
    )
    eval_config = EvalDefinition(
        definition_id="test.eval",
        version="1",
    ).materialize(
        sampling=sampling,
        evaluation_procedure=_procedure_config(),
        aggregation=base.aggregation_config,
    )
    with pytest.raises(ValueError, match="explicit task identity manifest"):
        EvaluationMatrixPlan(
            eval_config=eval_config,
            sampling_config=sampling,
            task_set=task_set,
            repeat_plan=base.repeat_plan,
            aggregation_config=base.aggregation_config,
        )


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        (
            {
                "reduction": "sum",
                "missing_data": "propagate",
                "zero_denominator": "not_applicable",
                SKIP_TOLERANCE_VARIABLE: "0.0",
            },
            "reduction",
        ),
        (
            {
                "reduction": "mean",
                "missing_data": "propagate",
                "zero_denominator": "error",
                SKIP_TOLERANCE_VARIABLE: "0.0",
            },
            "zero_denominator",
        ),
    ],
)
def test_plan_rejects_wrong_aggregation_semantics(
    assignment: dict[str, JsonValue],
    message: str,
) -> None:
    base = _plan()
    aggregation = aggregation_definition("test.rollout_aggregate").materialize(
        assignment
    )
    eval_config = EvalDefinition(
        definition_id="test.eval",
        version="1",
    ).materialize(
        sampling=base.sampling_config,
        evaluation_procedure=_procedure_config(),
        aggregation=aggregation,
    )
    with pytest.raises(ValueError, match=message):
        replace(
            base,
            eval_config=eval_config,
            aggregation_config=aggregation,
        )


def test_duplicate_observed_task_is_rejected() -> None:
    duplicate = _task("t1", RowValue(value=1.0))
    with pytest.raises(ValueError, match="duplicate observed task identity"):
        _task_mean((duplicate, duplicate), plan=_plan(repeat_count=1))


def test_extra_observed_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="unplanned task identities: t2"):
        _task_mean(
            (
                _task("t1", RowValue(value=1.0)),
                _task("t2", RowValue(value=2.0)),
            ),
            plan=_plan(("t1",), repeat_count=1),
        )


def test_missing_planned_task_is_synthesized_under_propagate() -> None:
    aggregate = _task_mean(
        (_task("t1", RowValue(value=1.0), RowValue(value=1.0)),),
        plan=_plan(("t1", "t2"), repeat_count=2),
    )
    assert aggregate.task_count == 2
    assert aggregate.repeat_count == 2
    assert aggregate.rows_missing == 2
    assert (
        aggregate.aggregation_output.status is AggregationStatus.MISSING_DATA
    )


def test_missing_planned_task_can_be_certified_under_tolerant_skip() -> None:
    task_ids = tuple(f"t{index}" for index in range(10))
    observed = tuple(
        _task(task_identity, RowValue(value=1.0))
        for task_identity in task_ids[:-1]
    )
    aggregate = _task_mean(
        observed,
        plan=_plan(
            task_ids,
            repeat_count=1,
            policy=CompletenessPolicy(
                row_policy=RowPolicy.SKIP,
                max_skip_fraction=0.1,
            ),
        ),
    )
    assert aggregate.task_count == 10
    assert aggregate.rows_missing == 1
    assert aggregate.aggregation_output.status is AggregationStatus.OK
    assert aggregate.aggregation_output.value == pytest.approx(1.0)


def test_input_order_does_not_change_plan_order_reduction() -> None:
    plan = _plan(("t1", "t2"), repeat_count=1)
    first = _task("t1", RowValue(value=2.0))
    second = _task("t2", RowValue(value=8.0))
    forward = _task_mean((first, second), plan=plan)
    reverse = _task_mean((second, first), plan=plan)
    assert forward == reverse
    assert forward.aggregation_output.value == pytest.approx(5.0)


def test_compression_uses_canonical_taskwise_matrix_path() -> None:
    plan = _plan(
        ("t1", "t2"),
        repeat_count=3,
        policy=CompletenessPolicy(
            row_policy=RowPolicy.SKIP,
            max_skip_fraction=0.5,
        ),
    )
    aggregate = _task_mean(
        (
            _task(
                "t1",
                RowValue(value=0.2),
                RowValue(value=0.4),
                RowValue(value=0.6),
            ),
            _task("t2", RowValue(value=1.0)),
        ),
        plan=plan,
        aggregate_name="mean_compression_ratio",
    )
    assert aggregate.name == "mean_compression_ratio"
    assert aggregate.aggregation_output.value == pytest.approx(0.7)
    assert not hasattr(code_eval, "mean_compression_ratio")


def test_row_value_requires_explicit_state() -> None:
    with pytest.raises(ValueError):
        RowValue()
    with pytest.raises(ValueError):
        RowValue(failed=True, missing=True)
    with pytest.raises(ValueError):
        RowValue(value=0.5, invalid=True)


def test_aggregate_rejects_incomplete_accounting() -> None:
    with pytest.raises(ValueError):
        RolloutAggregate(
            name="x",
            graph_hash=FULL_HASH,
            eval_config_hash=FULL_HASH,
            evaluation_binding_hash=CTX,
            task_count=2,
            repeat_count=3,
            aggregation_output=AggregationOutput(
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


@pytest.mark.parametrize(
    "field",
    [
        "graph_hash",
        "eval_config_hash",
        "evaluation_binding_hash",
    ],
)
def test_aggregate_reuses_canonical_hash_validation(field: str) -> None:
    aggregate = _task_mean(())

    with pytest.raises(ValueError, match=field):
        replace(aggregate, **{field: "short"})
