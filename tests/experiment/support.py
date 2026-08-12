from __future__ import annotations

from whetstone.evaluation import (
    EvalDefinition,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    SamplePlan,
    SamplingDefinition,
    TaskSet,
)
from whetstone.evaluation.aggregate import (
    SKIP_TOLERANCE_VARIABLE,
    Aggregate,
    CompletenessPolicy,
    EvaluationMatrixPlan,
    RowValue,
    TaskRows,
    aggregation_definition,
    unweighted_task_mean,
)

GRAPH_A = "a" * 64
GRAPH_B = "b" * 64
BINDING_ID = "c" * 64
SELECTION_QUALITY_AGGREGATE_NAME = "selection_quality"


def aggregate_plan(*, tasks: int, num_samples: int) -> EvaluationMatrixPlan:
    task_hashes = tuple(f"task-{index}" for index in range(tasks))
    task_set = TaskSet(
        manifest_id="selection.tasks",
        version="1",
        dataset_revision="selection-fixture",
        task_hashes=task_hashes,
    )
    sample_plan = SamplePlan(
        plan_id="selection.repeats",
        version="1",
        task_hashes=task_hashes,
        num_samples=num_samples,
    )
    sampling = SamplingDefinition(
        definition_id="selection.sampling", version="1"
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "sample_plan_hash": sample_plan.identity_hash(),
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
    ).materialize(resolved_operators=(("code_leakage", "1"),))
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
        sample_plan=sample_plan,
        aggregation_config=aggregation,
    )


def quality_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    value: float = 1.0,
    tasks: int = 2,
    num_samples: int = 2,
) -> Aggregate:
    task_rows = tuple(
        TaskRows(
            task_hash=f"task-{t}",
            rows=tuple(RowValue(value=value) for _ in range(num_samples)),
        )
        for t in range(tasks)
    )
    return unweighted_task_mean(
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=task_rows,
        plan=aggregate_plan(tasks=tasks, num_samples=num_samples),
    )


def compression_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    value: float = 2.0,
    tasks: int = 2,
    num_samples: int = 2,
) -> Aggregate:
    task_rows = tuple(
        TaskRows(
            task_hash=f"task-{task_index}",
            rows=tuple(RowValue(value=value) for _ in range(num_samples)),
        )
        for task_index in range(tasks)
    )
    return unweighted_task_mean(
        aggregate_name="mean_compression_ratio",
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=task_rows,
        plan=aggregate_plan(tasks=tasks, num_samples=num_samples),
    )


def incomplete_quality_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    tasks: int = 2,
    num_samples: int = 2,
) -> Aggregate:
    task_rows = (
        TaskRows(
            task_hash="task-0",
            rows=(RowValue(value=1.0),),
        ),
        *(
            TaskRows(
                task_hash=f"task-{t}",
                rows=tuple(RowValue(value=1.0) for _ in range(num_samples)),
            )
            for t in range(1, tasks)
        ),
    )
    return unweighted_task_mean(
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=task_rows,
        plan=aggregate_plan(tasks=tasks, num_samples=num_samples),
    )
