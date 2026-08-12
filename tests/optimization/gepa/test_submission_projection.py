from __future__ import annotations

from typing import cast

from dr_serialize import Jsonable
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.support import (
    data_instance,
    effect_context,
    evaluation_authority_binding,
)
from whetstone.core.identity import typed_ref_for_record
from whetstone.evaluation.schema import EVALUATION_COMPONENT_TRACES_SCHEMA
from whetstone.optimization.gepa.authorities import (
    CanonicalGepaEvaluationAuthority,
    _gepa_feedback,
    _gepa_prediction_failed,
    _load_component_trace_index,
)
from whetstone.optimization.gepa.contracts import (
    GepaCandidateComponent,
    GepaEffectSlot,
    GepaEvaluationEffectRequest,
)


def test_gepa_prediction_failed_for_definitive_test_failure() -> None:
    submission = {
        "kind": "humaneval",
        "score": {
            "passed": False,
            "infrastructure_unknown": False,
            "outcome": "tests_failed",
        },
    }

    assert _gepa_prediction_failed(
        failure_code="",
        submission_result=submission,
    )
    assert not _gepa_prediction_failed(
        failure_code="",
        submission_result={
            "kind": "humaneval",
            "score": {
                "passed": True,
                "infrastructure_unknown": False,
                "outcome": "passed",
            },
        },
    )


def test_gepa_feedback_includes_failed_case_summaries() -> None:
    submission = {
        "kind": "humaneval",
        "score": {
            "passed": False,
            "infrastructure_unknown": False,
            "outcome": "tests_failed",
        },
        "cases": [
            {
                "case_id": "case_0",
                "status": "failed",
                "message": "wrong value",
                "expected_output_repr": "2",
                "actual_output_repr": "3",
            }
        ],
    }

    feedback = _gepa_feedback(score=0.0, submission_result=submission)

    assert "score of 0.0" in feedback
    assert "case_0" in feedback
    assert "expected 2 got 3" in feedback


def test_load_component_trace_index_joins_rows_by_task_and_sample(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "traces.sqlite"))
    trace_record = {
        "schema_version": 2,
        "rows": [
            {
                "task_id": "task-a",
                "task_hash": "hash-a",
                "task_index": 0,
                "sample_index": 0,
                "executed_component_trace": {
                    "row_state": "success",
                    "executed_component_steps": [
                        {
                            "trace_index": 0,
                            "component_id": "alpha",
                            "input_field_names": ["prompt"],
                            "output_field_names": ["generation"],
                            "inputs": {"prompt": "hello"},
                            "outputs": {"generation": "world"},
                        }
                    ],
                },
            }
        ],
    }
    store.put(EVALUATION_COMPONENT_TRACES_SCHEMA, cast(Jsonable, trace_record))
    output_record = {
        "component_traces_ref": typed_ref_for_record(
            EVALUATION_COMPONENT_TRACES_SCHEMA,
            trace_record,
        ).model_dump(mode="json"),
    }

    index = _load_component_trace_index(store, output_record)

    assert ("hash-a", 0) in index
    assert index[("hash-a", 0)][0]["component_id"] == "alpha"


def test_project_row_uses_joined_component_traces(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "projection.sqlite"))
    task_hash = data_instance(0).data_id
    data_record = {
        "task_hash": task_hash,
        "task_id": "task-a",
        "prompt_inputs": {"question": "2+2?"},
    }
    data_ref = typed_ref_for_record("test.gepa.data", data_record)
    store.put("test.gepa.data", cast(Jsonable, data_record))
    data = data_instance(0).model_copy(
        update={"data_id": task_hash, "data_ref": data_ref}
    )

    request = GepaEvaluationEffectRequest(
        slot=GepaEffectSlot(context=effect_context(), invocation_ordinal=0),
        candidate=(GepaCandidateComponent(name="alpha", text="alpha-0"),),
        data=(data,),
        capture_traces=True,
        authority=evaluation_authority_binding(),
    )
    authority = CanonicalGepaEvaluationAuthority.__new__(
        CanonicalGepaEvaluationAuthority
    )
    authority._store = store
    row = authority._project_row(
        request=request,
        data=data,
        raw={
            "candidate_id": "candidate-a",
            "task_id": "task-a",
            "task_hash": data.data_id,
            "sample_index": 0,
            "output_text": "4",
            "failure_code": "",
            "code_submission_result": {
                "kind": "humaneval",
                "score": {
                    "passed": False,
                    "infrastructure_unknown": False,
                    "outcome": "tests_failed",
                },
                "outcome": "tests_failed",
                "function_names": ["candidate"],
                "best_function_name": "candidate",
                "total_cases": 1,
                "cases": [
                    {
                        "case_id": "case_0",
                        "status": "failed",
                        "message": "",
                        "input_repr": "(2, 2)",
                        "expected_output_repr": "4",
                        "actual_output_repr": "5",
                    }
                ],
            },
        },
        score=0.0,
        evidence_refs=(data.data_ref,),
        candidate_id="candidate-a",
        trace_steps=(
            {
                "trace_index": 0,
                "component_id": "alpha",
                "inputs": {"prompt": "2+2?"},
                "outputs": {"generation": "4"},
            },
        ),
    )

    assert row.trajectory is not None
    assert row.trajectory.prediction_failed is True
    assert "case_0" in row.trajectory.feedback
    assert row.trajectory.component_records["alpha"][0].inputs == {
        "prompt": "2+2?"
    }
    assert row.trajectory.generated_outputs["test_results"][0]["case_id"] == (
        "case_0"
    )
