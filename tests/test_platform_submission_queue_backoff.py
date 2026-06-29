from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy.engine import Connection

from dr_dspy.eval_failures import FailureClass, FailureSummary
from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.platform import backoff, queue_worker, submission, worker
from dr_dspy.records import (
    BatchSubmitItemStatus,
    DimensionsPayload,
    GraphSnapshotPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


class DummyConnection:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    def execute(self, statement: Any) -> list[Any]:
        self.statements.append(statement)
        return []


def _node() -> NodeSpec:
    return NodeSpec(
        id="direct",
        config=NodeConfig(
            fields=(
                FieldSpec(name="prompt", role=FieldRole.INPUT),
                FieldSpec(name="output", role=FieldRole.OUTPUT),
            ),
            input_bindings={
                "prompt": BindingRef.model_validate("task.prompt")
            },
            output_field="output",
        ),
    )


def _spec(
    *,
    task_id: str,
    model: str = "gpt-test",
    temperature: float = 0.2,
) -> PredictionSpecRecord:
    graph = GraphSpec(nodes=(_node(),), terminal_node_id="direct")
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": temperature})
    dimensions_id = dimensions_digest(dimensions)
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id=task_id,
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
    )
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        throttle_key=f"openai:responses:{model}",
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id=task_id,
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout="direct",
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id=task_id,
            inputs=TaskInputsPayload(values={"prompt": "write add"}),
        ),
        provider_configs=(provider,),
        provider_axis=provider,
        fair_order_seed="seed",
        fair_order_key=fair_order_key(
            experiment_seed="seed",
            prediction_id=prediction_id,
            provider=provider.provider_kind.value,
            endpoint_kind=provider.endpoint_kind.value,
            model=provider.model,
            throttle_key=provider.throttle_key,
            graph_layout="direct",
            task_id=task_id,
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )


def test_fair_ordering_sorts_by_stored_key() -> None:
    specs = (
        _spec(task_id="HumanEval/2"),
        _spec(task_id="HumanEval/0"),
        _spec(task_id="HumanEval/1"),
    )

    ordered = submission.fair_ordered_specs(reversed(specs))

    assert [spec.fair_order_key for spec in ordered] == sorted(
        spec.fair_order_key for spec in specs
    )


def test_submit_prediction_specs_chunks_and_records_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        _spec(task_id="HumanEval/0"),
        _spec(task_id="HumanEval/1"),
        _spec(task_id="HumanEval/2"),
    )
    chunks: list[tuple[str, ...]] = []
    item_records: list[Any] = []
    enqueue_calls: list[tuple[str, ...]] = []

    def insert_specs(
        connection: DummyConnection,
        chunk: tuple[PredictionSpecRecord, ...],
    ) -> set[str]:
        chunks.append(tuple(spec.prediction_id for spec in chunk))
        return {chunk[0].prediction_id}

    def insert_item(
        connection: DummyConnection,
        *,
        record: Any,
    ) -> None:
        item_records.append(record)

    def enqueue(
        database_url: str,
        prediction_ids: Sequence[str],
        attempt_index: int,
        queue_name: str,
    ) -> queue_worker.EnqueuePredictionWorkflowsResult:
        enqueue_calls.append(tuple(prediction_ids))
        workflows = tuple(
            queue_worker.EnqueuedPredictionWorkflow(
                prediction_id=prediction_id,
                generation_run_id=stable_generation_run_id(
                    prediction_id=prediction_id,
                    attempt_index=attempt_index,
                ),
                workflow_id=f"workflow:{prediction_id}",
                enqueued=True,
            )
            for prediction_id in prediction_ids
        )
        return queue_worker.EnqueuePredictionWorkflowsResult(
            queue_name=queue_name,
            enqueued_count=len(workflows),
            existing_count=0,
            workflows=workflows,
        )

    monkeypatch.setattr(
        submission,
        "bulk_insert_prediction_specs",
        insert_specs,
    )
    monkeypatch.setattr(submission, "insert_batch_item", insert_item)

    result = submission.submit_prediction_specs(
        cast(Connection, DummyConnection()),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs=specs,
        chunk_size=2,
        enqueue_workflows=enqueue,
    )

    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert enqueue_calls == chunks
    assert result.requested_count == 3
    assert result.inserted_count == 2
    assert result.already_present_count == 1
    assert result.enqueued_count == 3
    assert result.failed_count == 0
    assert {record.status for record in item_records} == {
        BatchSubmitItemStatus.ENQUEUED
    }
    assert item_records[0].enqueue_metadata["insert_status"] in {
        "inserted",
        "already_present",
    }


def test_submit_prediction_specs_records_chunk_enqueue_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (_spec(task_id="HumanEval/0"), _spec(task_id="HumanEval/1"))
    item_records: list[Any] = []

    monkeypatch.setattr(
        submission,
        "bulk_insert_prediction_specs",
        lambda connection, chunk: {spec.prediction_id for spec in chunk},
    )
    monkeypatch.setattr(
        submission,
        "insert_batch_item",
        lambda connection, *, record: item_records.append(record),
    )

    def enqueue(
        database_url: str,
        prediction_ids: Sequence[str],
        attempt_index: int,
        queue_name: str,
    ) -> queue_worker.EnqueuePredictionWorkflowsResult:
        raise RuntimeError("queue unavailable")

    result = submission.submit_prediction_specs(
        cast(Connection, DummyConnection()),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs=specs,
        enqueue_workflows=enqueue,
    )

    assert result.failed_count == 2
    assert result.enqueued_count == 0
    assert {record.status for record in item_records} == {
        BatchSubmitItemStatus.FAILED
    }
    assert all(record.failure is not None for record in item_records)


def test_queue_enqueue_uses_stable_workflow_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        queue_worker.DBOS,
        "get_workflow_status",
        lambda workflow_id: None,
    )
    monkeypatch.setattr(
        queue_worker.DBOS,
        "enqueue_workflow",
        lambda queue_name, workflow, *args: captured.update(
            {"queue_name": queue_name, "args": args}
        ),
    )
    prediction_id = "prediction-1"

    result = queue_worker.enqueue_prediction_graph_workflow(
        database_url="postgresql://example/db",
        prediction_id=prediction_id,
        attempt_index=2,
    )

    generation_run_id = stable_generation_run_id(
        prediction_id=prediction_id,
        attempt_index=2,
    )
    assert result.enqueued is True
    assert result.generation_run_id == generation_run_id
    assert result.workflow_id == (
        f"platform-generate-v1:{generation_run_id}"
    )
    assert captured["queue_name"] == (
        queue_worker.PLATFORM_GENERATION_QUEUE_NAME
    )
    assert captured["args"] == (
        "postgresql://example/db",
        prediction_id,
        2,
    )


def test_platform_worker_config_listens_to_v1_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    config = SimpleNamespace(database_url="postgresql://example/db")

    class FakeDbos:
        def __call__(self, *, config: dict[str, Any]) -> None:
            calls.append(("configure", config))

        def listen_queues(self, queues: list[str]) -> None:
            calls.append(("listen", queues))

        def launch(self) -> None:
            calls.append(("launch", None))

    monkeypatch.setattr(worker, "DBOS", FakeDbos())
    monkeypatch.setattr(
        worker.shared_dbos,
        "build_eval_dbos_config",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(
        worker.shared_dbos,
        "build_dbos_config",
        lambda config, app_name: {"name": app_name},
    )
    monkeypatch.setattr(
        worker,
        "listen_to_platform_generation_queue",
        lambda: calls.append(("listen_v1", None)),
    )
    monkeypatch.setattr(
        worker,
        "register_platform_generation_queue",
        lambda worker_concurrency: calls.append(
            ("register", worker_concurrency)
        ),
    )

    worker.configure_platform_dbos_runtime(
        database_url=None,
        dbos_system_database_url=None,
        worker_concurrency=3,
        consume_generation_queue=True,
    )

    assert ("listen_v1", None) in calls
    assert ("register", 3) in calls


def test_run_one_runtime_keeps_empty_queue_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    config = SimpleNamespace(database_url="postgresql://example/db")

    class FakeDbos:
        def __call__(self, *, config: dict[str, Any]) -> None:
            calls.append(("configure", config))

        def listen_queues(self, queues: list[str]) -> None:
            calls.append(("listen", queues))

        def launch(self) -> None:
            calls.append(("launch", None))

    monkeypatch.setattr(worker, "DBOS", FakeDbos())
    monkeypatch.setattr(
        worker.shared_dbos,
        "build_eval_dbos_config",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(
        worker.shared_dbos,
        "build_dbos_config",
        lambda config, app_name: {"name": app_name},
    )

    worker.configure_platform_dbos_runtime(
        database_url=None,
        dbos_system_database_url=None,
        consume_generation_queue=False,
    )

    assert ("listen", []) in calls


def test_backoff_delay_is_per_throttle_key_and_bounded() -> None:
    delay_a = backoff.next_backoff_delay_seconds(
        throttle_key="openai:model-a",
        consecutive_failures=3,
        failure_class=FailureClass.RATE_LIMITED,
        max_seconds=30.0,
    )
    delay_b = backoff.next_backoff_delay_seconds(
        throttle_key="openai:model-b",
        consecutive_failures=3,
        failure_class=FailureClass.RATE_LIMITED,
        max_seconds=30.0,
    )

    assert 0 < delay_a <= 30.0
    assert 0 < delay_b <= 30.0
    assert delay_a != delay_b
    assert backoff.next_backoff_delay_seconds(
        throttle_key="openai:model-a",
        consecutive_failures=3,
        failure_class=FailureClass.PERMANENT,
    ) == 0.0


def test_backoff_preflight_only_blocks_matching_state() -> None:
    state = backoff.ThrottleBackoffState(
        throttle_key="openai:model-a",
        blocked_until=NOW + timedelta(seconds=12),
        consecutive_failures=1,
        updated_at=NOW,
    )

    assert backoff.delay_until_unblocked_seconds(state, now=NOW) == 12.0
    assert backoff.delay_until_unblocked_seconds(None, now=NOW) == 0.0


def test_record_throttle_failure_ignores_permanent_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[Any] = []
    monkeypatch.setattr(
        backoff,
        "load_throttle_backoff_state",
        lambda connection, *, throttle_key: None,
    )
    failure = FailureSummary(
        failure_class=FailureClass.PERMANENT,
        failure_exception_type="builtins.ValueError",
        underlying_exception_type="builtins.ValueError",
        message="bad request",
    )

    result = backoff.record_throttle_failure(
        cast(
            Connection,
            SimpleNamespace(
                execute=lambda statement: executed.append(statement)
            ),
        ),
        throttle_key="openai:model-a",
        failure=failure,
        now=NOW,
    )

    assert result is None
    assert executed == []


def test_record_throttle_failure_updates_retryable_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states: list[backoff.ThrottleBackoffState] = []
    monkeypatch.setattr(
        backoff,
        "load_throttle_backoff_state",
        lambda connection, *, throttle_key: None,
    )
    monkeypatch.setattr(
        backoff,
        "upsert_throttle_backoff_state",
        lambda state: states.append(state) or object(),
    )
    failure = FailureSummary(
        failure_class=FailureClass.RATE_LIMITED,
        failure_exception_type="openai.RateLimitError",
        underlying_exception_type="openai.RateLimitError",
        message="rate limited",
    )

    result = backoff.record_throttle_failure(
        cast(Connection, SimpleNamespace(execute=lambda statement: None)),
        throttle_key="openai:model-a",
        failure=failure,
        now=NOW,
    )

    assert result is not None
    assert result.throttle_key == "openai:model-a"
    assert result.consecutive_failures == 1
    assert result.blocked_until is not None
    assert result.blocked_until > NOW
    assert states == [result]
