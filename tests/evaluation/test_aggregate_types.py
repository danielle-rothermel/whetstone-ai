from __future__ import annotations

from dataclasses import is_dataclass

import whetstone.evaluation as local_evaluation
from whetstone.evaluation.aggregate import (
    CompletenessPolicy,
    EvaluationMatrixPlan,
    RolloutAggregate,
    RowValue,
    TaskRows,
)


def test_internal_value_objects_are_frozen_slotted_dataclasses() -> None:
    value_objects = (
        CompletenessPolicy(),
        RowValue(value=1.0),
        TaskRows(
            task_hash="task",
            rows=(RowValue(value=1.0),),
        ),
    )
    for value in value_objects:
        assert is_dataclass(value)
        assert hasattr(type(value), "__slots__")
        assert type(value).__dataclass_params__.frozen

    assert is_dataclass(EvaluationMatrixPlan)
    assert hasattr(EvaluationMatrixPlan, "__slots__")
    assert EvaluationMatrixPlan.__dataclass_params__.frozen

    output = local_evaluation.AggregationOutput(
        status=local_evaluation.AggregationStatus.NOT_APPLICABLE,
        value=None,
        count_total=0,
        count_applicable=0,
        count_present=0,
    )
    aggregate = RolloutAggregate(
        name="x",
        graph_hash="0" * 64,
        eval_config_hash="1" * 64,
        evaluation_binding_hash="2" * 64,
        task_count=0,
        num_samples=1,
        aggregation_output=output,
        rows_present=0,
        rows_missing=0,
        rows_failed=0,
        rows_invalid=0,
    )
    assert is_dataclass(aggregate)
    assert hasattr(type(aggregate), "__slots__")
    assert type(aggregate).__dataclass_params__.frozen
