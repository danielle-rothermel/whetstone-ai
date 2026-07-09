"""Minimal DBOS workflows used only by integration tests."""

from __future__ import annotations

from dbos import DBOS

from whetstone.platform.graph_workflow import (
    load_prediction_spec_step,
    persist_generation_result_step,
)
from whetstone.platform.scoring_workflow import (
    load_scoring_target_step,
    persist_score_result_step,
)

LOAD_SPEC_TEST_WORKFLOW = "dr_dspy_test_integration_load_spec_v1"
PERSIST_RESULT_TEST_WORKFLOW = "dr_dspy_test_integration_persist_result_v1"
LOAD_SCORING_TARGET_TEST_WORKFLOW = (
    "dr_dspy_test_integration_load_scoring_target_v1"
)
PERSIST_SCORE_TEST_WORKFLOW = "dr_dspy_test_integration_persist_score_v1"


@DBOS.workflow(name=LOAD_SPEC_TEST_WORKFLOW)
def integration_load_spec_workflow(
    database_url: str,
    prediction_id: str,
) -> dict:
    return load_prediction_spec_step(database_url, prediction_id)


@DBOS.workflow(name=PERSIST_RESULT_TEST_WORKFLOW)
def integration_persist_result_workflow(
    database_url: str,
    spec_payload: dict,
    generation_run_id: str,
    attempt_index: int,
    graph_result_payload: dict,
    node_step_result_payloads: list[dict],
    started_at: str,
    completed_at: str,
) -> None:
    persist_generation_result_step(
        database_url,
        spec_payload,
        generation_run_id,
        attempt_index,
        graph_result_payload,
        node_step_result_payloads,
        started_at,
        completed_at,
    )


@DBOS.workflow(name=LOAD_SCORING_TARGET_TEST_WORKFLOW)
def integration_load_scoring_target_workflow(
    database_url: str,
    generation_run_id: str,
) -> dict:
    return load_scoring_target_step(database_url, generation_run_id)


@DBOS.workflow(name=PERSIST_SCORE_TEST_WORKFLOW)
def integration_persist_score_workflow(
    database_url: str,
    score_attempt_payload: dict,
) -> dict:
    return persist_score_result_step(database_url, score_attempt_payload)
