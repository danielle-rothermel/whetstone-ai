from __future__ import annotations

import importlib
import pkgutil
from dataclasses import is_dataclass

import dr_code.trace as dr_trace

import whetstone.evaluation as local_evaluation
import whetstone.evaluation.code as code_package
from whetstone.evaluation.code import (
    CompletenessPolicy,
    EvaluationMatrixPlan,
    RolloutAggregate,
    RowValue,
    TaskRows,
    compressed_description_length_fact,
    compression_ratio_score,
    select_compression_reference,
    submission,
    submission_text_artifact,
)

from .support import generation, operator_lineage


def test_boundary_reuses_released_text_artifact() -> None:
    assert submission.TextArtifact is dr_trace.TextArtifact


def test_scoring_returns_whetstone_score_and_fact_types() -> None:
    fact = compressed_description_length_fact(
        "code", lineage=operator_lineage()
    )
    assert type(fact) is local_evaluation.MetricFact

    ratio_score = compression_ratio_score(
        compressed_description_length=1,
        reference=local_evaluation.CompressionReferenceArtifact(
            content=b"abcd"
        ),
        evaluation_procedure_config_hash="0" * 64,
    )
    assert type(ratio_score) is local_evaluation.Score


def test_compression_selection_returns_generic_types() -> None:
    from pydantic import BaseModel

    class _Task(BaseModel):
        gt_code_wo_comments: str

    artifact = select_compression_reference(_Task(gt_code_wo_comments="x"))
    assert type(artifact) is local_evaluation.CompressionReferenceArtifact


def test_code_package_does_not_duplicate_evaluation_contract_types() -> None:
    evaluation_type_names = {
        name
        for name in local_evaluation.__all__
        if isinstance(getattr(local_evaluation, name), type)
    } | {"TextArtifact"}
    modules = (
        code_package,
        *(
            importlib.import_module(module_info.name)
            for module_info in pkgutil.walk_packages(
                code_package.__path__, prefix=f"{code_package.__name__}."
            )
        ),
    )
    duplicate_definitions = {
        f"{module.__name__}.{name}"
        for module in modules
        for name, value in vars(module).items()
        if name in evaluation_type_names
        and isinstance(value, type)
        and value.__module__ == module.__name__
    }

    assert duplicate_definitions == set()


def test_submission_generation_is_whetstone_owned() -> None:
    gen = generation(text="x = 1\n")
    assert gen.__class__.__module__.startswith("whetstone")
    artifact = submission_text_artifact(gen)
    assert artifact.__class__.__module__.startswith("dr_code")


def test_internal_value_objects_are_frozen_slotted_dataclasses() -> None:
    value_objects = (
        CompletenessPolicy(),
        RowValue(value=1.0),
        TaskRows(
            task_identity="task",
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
        repeat_count=1,
        aggregation_output=output,
        rows_present=0,
        rows_missing=0,
        rows_failed=0,
        rows_invalid=0,
    )
    assert is_dataclass(aggregate)
    assert hasattr(type(aggregate), "__slots__")
    assert type(aggregate).__dataclass_params__.frozen
