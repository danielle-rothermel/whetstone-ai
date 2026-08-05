"""Shared builders for official-selection tests.

Constructs real Whetstone :class:`RolloutAggregate` values through the public
aggregate constructors so selection runs against certified, complete evidence
rather than stand-ins.
"""

from __future__ import annotations

from dr_code.eval import (
    EvalDefinition,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
)

from whetstone.evaluation.code.aggregate import (
    SKIP_TOLERANCE_VARIABLE,
    CompletenessPolicy,
    EvaluationMatrixPlan,
    RolloutAggregate,
    RowValue,
    TaskRows,
    aggregation_definition,
    unweighted_task_mean,
)

GRAPH_A = "a" * 64
GRAPH_B = "b" * 64
BINDING_ID = "c" * 64
SELECTION_QUALITY_AGGREGATE_NAME = "selection_quality"


def aggregate_plan(*, tasks: int, repeats: int) -> EvaluationMatrixPlan:
    task_identities = tuple(f"task-{index}" for index in range(tasks))
    task_set = TaskSet(
        manifest_id="selection.tasks",
        version="1",
        dataset_revision="selection-fixture",
        task_identities=task_identities,
    )
    repeat_plan = RepeatPlan(
        plan_id="selection.repeats",
        version="1",
        task_identities=task_identities,
        repeat_count=repeats,
    )
    sampling = SamplingDefinition(
        definition_id="selection.sampling", version="1"
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )
    policy = CompletenessPolicy()
    aggregation = aggregation_definition("selection.aggregate").materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            SKIP_TOLERANCE_VARIABLE: policy.skip_fraction_token(),
        }
    )
    preprocessing = PreprocessingDefinition(
        definition_id="selection.preprocessing", version="1", steps=()
    ).materialize()
    metric_extraction = MetricExtractionDefinition(
        definition_id="selection.metric",
        version="1",
        questions=(MetricQuestionBinding(metric="code_leakage", on="output"),),
    ).materialize()
    procedure = EvaluationProcedureDefinition(
        definition_id="selection.procedure", version="1"
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": "not_applicable"},
    )
    eval_config = EvalDefinition(
        definition_id="selection.eval", version="1"
    ).materialize(
        sampling=sampling,
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )
    return EvaluationMatrixPlan(
        eval_config=eval_config,
        sampling_config=sampling,
        task_set=task_set,
        repeat_plan=repeat_plan,
        aggregation_config=aggregation,
    )


def quality_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    value: float = 1.0,
    tasks: int = 2,
    repeats: int = 2,
) -> RolloutAggregate:
    """A complete, OK selection-quality aggregate.

    Every planned cell is present with the same value, so the two staged
    reductions produce ``value`` and the pure status is OK.
    """
    task_rows = tuple(
        TaskRows(
            task_identity=f"task-{t}",
            rows=tuple(RowValue(value=value) for _ in range(repeats)),
        )
        for t in range(tasks)
    )
    return unweighted_task_mean(
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=task_rows,
        plan=aggregate_plan(tasks=tasks, repeats=repeats),
    )


def compression_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    value: float = 2.0,
    tasks: int = 2,
    repeats: int = 2,
) -> RolloutAggregate:
    """A complete, OK Mean Compression Ratio aggregate."""
    task_rows = tuple(
        TaskRows(
            task_identity=f"task-{task_index}",
            rows=tuple(RowValue(value=value) for _ in range(repeats)),
        )
        for task_index in range(tasks)
    )
    return unweighted_task_mean(
        aggregate_name="mean_compression_ratio",
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=task_rows,
        plan=aggregate_plan(tasks=tasks, repeats=repeats),
    )


def incomplete_quality_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    tasks: int = 2,
    repeats: int = 2,
) -> RolloutAggregate:
    """An incomplete selection-quality aggregate with missing rows.

    One task is short a repeat, so under the default PROPAGATE policy the pure
    reduction is not OK (MISSING_DATA) — exactly the incomplete evidence
    official selection must refuse.
    """
    task_rows = (
        TaskRows(
            task_identity="task-0",
            rows=(RowValue(value=1.0),),  # short one repeat -> missing padded
        ),
        *(
            TaskRows(
                task_identity=f"task-{t}",
                rows=tuple(RowValue(value=1.0) for _ in range(repeats)),
            )
            for t in range(1, tasks)
        ),
    )
    return unweighted_task_mean(
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=task_rows,
        plan=aggregate_plan(tasks=tasks, repeats=repeats),
    )
