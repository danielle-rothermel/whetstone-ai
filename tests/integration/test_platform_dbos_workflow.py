from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from dr_dspy.eval_failures import PermanentFailureError, TransientFailureError
from dr_dspy.graph import GraphSpec
from dr_dspy.platform import graph_workflow
from dr_dspy.platform.graph_workflow import run_prediction_graph_workflow_once
from dr_dspy.platform.node_execution import NodeStepResult
from dr_dspy.records import GenerationRunStatus, stable_generation_run_id
from tests.support.platform_integration_helpers import (
    count_generation_runs,
    fetch_node_attempts,
    fetch_workflow_run_snapshot,
    seed_spec,
)
from tests.support.platform_workflow_fixtures import (
    direct_node,
    encdec_spec,
    prediction_spec,
    step_success,
)

pytestmark = pytest.mark.integration


def _mock_lm_success(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_execute_lm_node(
        *,
        spec: Any,
        node: Any,
        node_inputs: dict[str, Any],
        client_factory: Any = None,
        provider_caller: Any = None,
        raise_retryable: bool = False,
    ) -> NodeStepResult:
        calls.append((node.id, str(node_inputs.get("prompt", ""))))
        return step_success(node, f"workflow {node_inputs.get('prompt', '')}")

    monkeypatch.setattr(
        graph_workflow,
        "execute_lm_node",
        fake_execute_lm_node,
    )
    return calls


def _mock_lm_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: Exception,
    node_id: str | None = None,
) -> None:
    def failing_execute_lm_node(
        *,
        node: Any,
        **kwargs: Any,
    ) -> NodeStepResult:
        if node_id is None or node.id == node_id:
            raise error
        return step_success(node, "ok")

    monkeypatch.setattr(
        graph_workflow,
        "execute_lm_node",
        failing_execute_lm_node,
    )


def _spy_on_error_result_step(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original = graph_workflow.node_step_error_result_step

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_workflow, "node_step_error_result_step", spy)
    return calls


def test_run_prediction_graph_workflow_once_persists_success(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    seed_spec(app_postgres_schema.database_url, spec)

    calls = _mock_lm_success(monkeypatch)
    generation_run_id = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )

    assert generation_run_id == stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    assert calls == [("direct", "write add")]

    snapshot = fetch_workflow_run_snapshot(
        app_postgres_schema.database_url,
        generation_run_id,
    )
    assert snapshot.run_status == GenerationRunStatus.SUCCESS.value
    assert snapshot.node_count == 1


def test_workflow_persist_is_idempotent_on_second_run(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    seed_spec(app_postgres_schema.database_url, spec)

    _mock_lm_success(monkeypatch)
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )

    first = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )
    second = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )

    assert first == second == generation_run_id
    assert (
        count_generation_runs(
            app_postgres_schema.database_url,
            generation_run_id,
        )
        == 1
    )


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            PermanentFailureError("permanent provider failure"),
            id="permanent",
        ),
        pytest.param(
            TransientFailureError("temporary provider failure"),
            id="retry_exhausted",
        ),
    ],
)
def test_workflow_persist_is_idempotent_on_second_run_after_failure(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    seed_spec(app_postgres_schema.database_url, spec)
    _mock_lm_failure(monkeypatch, error=error)

    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )

    first = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )
    second = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )

    assert first == second == generation_run_id
    assert (
        count_generation_runs(
            app_postgres_schema.database_url,
            generation_run_id,
        )
        == 1
    )
    snapshot = fetch_workflow_run_snapshot(
        app_postgres_schema.database_url,
        generation_run_id,
    )
    assert snapshot.run_status == GenerationRunStatus.ERROR.value


@pytest.mark.parametrize(
    ("mock_kind", "error"),
    [
        ("success", None),
        ("permanent", PermanentFailureError("permanent provider failure")),
    ],
)
def test_workflow_duplicate_start_returns_existing_result(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
    mock_kind: str,
    error: Exception | None,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    seed_spec(app_postgres_schema.database_url, spec)

    if mock_kind == "success":
        _mock_lm_success(monkeypatch)
        expected_status = GenerationRunStatus.SUCCESS.value
    else:
        assert error is not None
        _mock_lm_failure(monkeypatch, error=error)
        expected_status = GenerationRunStatus.ERROR.value

    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )

    first = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )
    second = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )

    assert first == second == generation_run_id
    assert (
        count_generation_runs(
            app_postgres_schema.database_url,
            generation_run_id,
        )
        == 1
    )
    snapshot = fetch_workflow_run_snapshot(
        app_postgres_schema.database_url,
        generation_run_id,
    )
    assert snapshot.run_status == expected_status


def test_workflow_records_error_when_lm_fails_permanently(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    seed_spec(app_postgres_schema.database_url, spec)
    _mock_lm_failure(
        monkeypatch,
        error=PermanentFailureError("permanent provider failure"),
    )

    generation_run_id = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )

    snapshot = fetch_workflow_run_snapshot(
        app_postgres_schema.database_url,
        generation_run_id,
    )
    assert snapshot.run_status == GenerationRunStatus.ERROR.value
    assert snapshot.attempt_status == "error"


def test_workflow_records_error_when_lm_retries_exhausted(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    seed_spec(app_postgres_schema.database_url, spec)
    error_step_calls = _spy_on_error_result_step(monkeypatch)
    _mock_lm_failure(
        monkeypatch,
        error=TransientFailureError("temporary provider failure"),
    )

    generation_run_id = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )

    assert len(error_step_calls) == 1
    error_step_args = error_step_calls[0][0]
    started_at_arg = error_step_args[3]
    completed_at_arg = error_step_args[4]
    assert isinstance(started_at_arg, str)
    assert isinstance(completed_at_arg, str)

    snapshot = fetch_workflow_run_snapshot(
        app_postgres_schema.database_url,
        generation_run_id,
    )
    assert snapshot.run_status == GenerationRunStatus.ERROR.value
    assert snapshot.attempt_status == "error"
    assert snapshot.attempt_started_at is not None
    assert snapshot.attempt_completed_at is not None
    assert snapshot.attempt_started_at <= snapshot.attempt_completed_at
    assert snapshot.attempt_started_at == datetime.fromisoformat(
        started_at_arg
    )
    assert snapshot.attempt_completed_at == datetime.fromisoformat(
        completed_at_arg
    )


def test_workflow_records_blocked_when_upstream_node_fails(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = encdec_spec()
    seed_spec(app_postgres_schema.database_url, spec)
    _mock_lm_failure(
        monkeypatch,
        error=PermanentFailureError("encoder provider failure"),
        node_id="encoder",
    )

    generation_run_id = run_prediction_graph_workflow_once(
        app_postgres_schema.database_url,
        spec.prediction_id,
        attempt_index=0,
    )

    snapshot = fetch_workflow_run_snapshot(
        app_postgres_schema.database_url,
        generation_run_id,
    )
    assert snapshot.run_status == GenerationRunStatus.BLOCKED.value
    assert snapshot.node_count == 1
    assert fetch_node_attempts(
        app_postgres_schema.database_url,
        generation_run_id,
    ) == [("encoder", "error")]


def test_workflow_surfaces_persist_step_failure_without_writing_rows(
    app_postgres_schema,
    reset_dbos,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphSpec(nodes=(direct_node(),), terminal_node_id="direct")
    spec = prediction_spec(graph)
    seed_spec(app_postgres_schema.database_url, spec)

    _mock_lm_success(monkeypatch)

    def failing_persist(connection: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated persist failure")

    monkeypatch.setattr(
        graph_workflow,
        "persist_generation_result",
        failing_persist,
    )

    with pytest.raises(Exception, match="simulated persist failure"):
        run_prediction_graph_workflow_once(
            app_postgres_schema.database_url,
            spec.prediction_id,
            attempt_index=0,
        )

    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    assert (
        count_generation_runs(
            app_postgres_schema.database_url,
            generation_run_id,
        )
        == 0
    )
