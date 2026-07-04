from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
)
from dr_code.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
)
from dr_graph import GraphSpec
from sqlalchemy import create_engine

from tests.support.jsonl_fixtures import write_prediction_specs_jsonl
from tests.support.platform_integration_helpers import (
    fetch_batch_submit_operation_status,
    fetch_score_attempt_snapshot,
    fetch_workflow_run_snapshot,
    wait_for_workflow_result,
)
from tests.support.platform_scoring_fixtures import scoring_task
from tests.support.platform_workflow_fixtures import (
    direct_node,
    prediction_spec,
    step_success,
)
from whetstone.platform import graph_workflow, scoring_workflow, submission
from whetstone.platform.graph_workflow import platform_generation_workflow_id
from whetstone.platform.node_execution import NodeStepResult
from whetstone.platform.scoring_workflow import (
    run_score_generation_workflow_once,
)
from whetstone.records import (
    BatchSubmitOperationStatus,
    GenerationRunStatus,
    stable_generation_run_id,
    stable_score_attempt_id,
)

pytestmark = pytest.mark.integration

VALID_GENERATION = "def add_one(x):\n    return x + 1\n"


def _mock_lm_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_execute_lm_node(
        *,
        node: Any,
        node_inputs: dict[str, Any],
        **kwargs: Any,
    ) -> NodeStepResult:
        return step_success(node, VALID_GENERATION)

    monkeypatch.setattr(
        graph_workflow,
        "execute_lm_node",
        fake_execute_lm_node,
    )


def _mock_humaneval_task_step(monkeypatch: pytest.MonkeyPatch) -> None:
    task_payload = scoring_task().model_dump(
        mode="json",
        exclude={
            "ground_truth_code",
            "ground_truth_code_without_comments",
        },
    )

    def fake_load_humaneval_task_step(
        dataset_name: str,
        dataset_split: str,
        task_id: str,
    ) -> dict[str, Any]:
        assert task_id == "HumanEval/fixture"
        return task_payload

    monkeypatch.setattr(
        scoring_workflow,
        "load_humaneval_task_step",
        fake_load_humaneval_task_step,
    )


def test_jsonl_submit_enqueue_generation_and_scoring(
    app_postgres_schema,
    reset_dbos_generation_consumer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph, task_id="HumanEval/fixture")
    specs_file = tmp_path / "specs.jsonl"
    write_prediction_specs_jsonl(specs_file, (spec,))
    operation_key = "pipeline-e2e-op"

    _mock_lm_success(monkeypatch)
    _mock_humaneval_task_step(monkeypatch)
    scoring_workflow.load_humaneval_task_map.cache_clear()

    database_url = app_postgres_schema.database_url
    engine = create_engine(database_url)
    try:
        submit_result = submission.submit_prediction_specs_jsonl(
            engine,
            database_url=database_url,
            operation_key=operation_key,
            experiment_name=spec.experiment_name,
            specs_file=specs_file,
            submit_spec={"source": str(specs_file)},
            chunk_size=1,
            attempt_index=0,
        )
    finally:
        engine.dispose()

    assert submit_result.requested_count == 1
    assert submit_result.enqueued_count == 1
    assert (
        fetch_batch_submit_operation_status(database_url, operation_key)
        == BatchSubmitOperationStatus.COMPLETED.value
    )

    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    workflow_id = platform_generation_workflow_id(generation_run_id)
    wait_for_workflow_result(workflow_id)

    snapshot = fetch_workflow_run_snapshot(database_url, generation_run_id)
    assert snapshot.run_status == GenerationRunStatus.SUCCESS.value
    assert snapshot.node_count == 1

    score_result = run_score_generation_workflow_once(
        database_url,
        generation_run_id,
    )
    expected_score_id = stable_score_attempt_id(
        generation_run_id=generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    assert score_result.score_attempt_id == expected_score_id

    score_snapshot = fetch_score_attempt_snapshot(
        database_url,
        expected_score_id,
    )
    assert score_snapshot is not None
    assert score_snapshot.status == "success"
    assert score_snapshot.score == 1.0
