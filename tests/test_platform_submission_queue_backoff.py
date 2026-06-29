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
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
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


class DummyTransaction:
    def __init__(self, engine: DummyEngine) -> None:
        self.engine = engine

    def __enter__(self) -> DummyConnection:
        self.engine.in_transaction = True
        self.engine.begin_count += 1
        return self.engine.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.engine.in_transaction = False


class DummyEngine:
    def __init__(self) -> None:
        self.connection = DummyConnection()
        self.begin_count = 0
        self.in_transaction = False

    def begin(self) -> DummyTransaction:
        return DummyTransaction(self)


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
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        throttle_key=f"openai:responses:{model}",
    )
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id=task_id,
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
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


def test_fair_ordering_interleaves_model_axis() -> None:
    specs = tuple(
        _spec(task_id=f"HumanEval/{task_id}", model=model)
        for model in ("model-a", "model-b")
        for task_id in range(8)
    )

    ordered = submission.fair_ordered_specs(specs)
    prefix_models = {spec.provider_axis.model for spec in ordered[:4]}

    assert prefix_models == {"model-a", "model-b"}
    assert [spec.provider_axis.model for spec in ordered] != [
        spec.provider_axis.model for spec in specs
    ]


def test_submit_prediction_specs_chunks_and_records_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        _spec(task_id="HumanEval/0"),
        _spec(task_id="HumanEval/1"),
        _spec(task_id="HumanEval/2"),
    )
    engine = DummyEngine()
    chunks: list[tuple[str, ...]] = []
    enqueue_calls: list[str] = []
    item_updates: list[submission.SubmittedPredictionItem] = []

    def prepare(
        connection: Connection,
        *,
        operation_key: str,
        experiment_name: str,
        ordered_specs: Sequence[PredictionSpecRecord],
        submit_spec: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        assert engine.in_transaction is True
        chunks.append(tuple(spec.prediction_id for spec in ordered_specs))

    def candidates(
        connection: Connection,
        *,
        operation_key: str,
        prediction_ids: Sequence[str],
    ) -> tuple[submission.EnqueueCandidate, ...]:
        assert engine.in_transaction is True
        chunks.append(tuple(prediction_ids))
        by_id = {spec.prediction_id: spec for spec in specs}
        return tuple(
            submission.EnqueueCandidate(
                prediction_id=prediction_id,
                fair_order_key=by_id[prediction_id].fair_order_key,
                item_index=index,
                insert_status=BatchSubmitItemInsertStatus.INSERTED,
            )
            for index, prediction_id in enumerate(prediction_ids)
        )

    def enqueue(
        database_url: str,
        prediction_id: str,
        attempt_index: int,
        queue_name: str,
    ) -> queue_worker.EnqueuedPredictionWorkflow:
        assert engine.in_transaction is False
        enqueue_calls.append(prediction_id)
        return queue_worker.EnqueuedPredictionWorkflow(
            prediction_id=prediction_id,
            generation_run_id=stable_generation_run_id(
                prediction_id=prediction_id,
                attempt_index=attempt_index,
            ),
            workflow_id=f"workflow:{prediction_id}",
            enqueued=True,
        )

    def update_item(
        connection: Connection,
        *,
        operation_key: str,
        item: submission.SubmittedPredictionItem,
    ) -> None:
        assert engine.in_transaction is True
        item_updates.append(item)

    monkeypatch.setattr(
        submission,
        "prepare_submission_records",
        prepare,
    )
    monkeypatch.setattr(submission, "load_enqueue_candidates", candidates)
    monkeypatch.setattr(
        submission,
        "update_batch_item_outcome",
        update_item,
    )
    monkeypatch.setattr(
        submission,
        "update_operation_summary",
        lambda connection, *, operation_key, experiment_name, queue_name: (
            submission.SubmitPredictionSpecsResult(
                operation_key=operation_key,
                experiment_name=experiment_name,
                queue_name=queue_name,
                requested_count=len(specs),
                inserted_count=2,
                already_present_count=1,
                enqueued_count=len(item_updates),
                failed_count=0,
                items=tuple(item_updates),
            )
        ),
    )

    result = submission.submit_prediction_specs(
        cast(Any, engine),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs=specs,
        chunk_size=2,
        enqueue_workflow=enqueue,
    )

    assert [len(chunk) for chunk in chunks[1:]] == [2, 1]
    assert enqueue_calls == [item.prediction_id for item in item_updates]
    assert result.requested_count == 3
    assert result.inserted_count == 2
    assert result.already_present_count == 1
    assert result.enqueued_count == 3
    assert result.failed_count == 0
    assert {item.enqueue_status for item in item_updates} == {
        BatchSubmitItemEnqueueStatus.ENQUEUED
    }
    assert engine.begin_count == 7


def test_submit_prediction_specs_records_item_enqueue_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        _spec(task_id="HumanEval/0"),
        _spec(task_id="HumanEval/1"),
        _spec(task_id="HumanEval/2"),
    )
    engine = DummyEngine()
    item_updates: list[submission.SubmittedPredictionItem] = []
    failed_prediction_id = specs[1].prediction_id

    monkeypatch.setattr(
        submission,
        "prepare_submission_records",
        lambda connection, **kwargs: None,
    )
    monkeypatch.setattr(
        submission,
        "load_enqueue_candidates",
        lambda connection, *, operation_key, prediction_ids: tuple(
            submission.EnqueueCandidate(
                prediction_id=spec.prediction_id,
                fair_order_key=spec.fair_order_key,
                item_index=index,
                insert_status=BatchSubmitItemInsertStatus.INSERTED,
            )
            for index, spec in enumerate(specs)
            if spec.prediction_id in prediction_ids
        ),
    )
    monkeypatch.setattr(
        submission,
        "update_batch_item_outcome",
        lambda connection, *, operation_key, item: item_updates.append(item),
    )

    def enqueue(
        database_url: str,
        prediction_id: str,
        attempt_index: int,
        queue_name: str,
    ) -> queue_worker.EnqueuedPredictionWorkflow:
        if prediction_id == failed_prediction_id:
            raise RuntimeError("queue unavailable")
        return queue_worker.EnqueuedPredictionWorkflow(
            prediction_id=prediction_id,
            generation_run_id=stable_generation_run_id(
                prediction_id=prediction_id,
                attempt_index=attempt_index,
            ),
            workflow_id=f"workflow:{prediction_id}",
            enqueued=True,
        )

    monkeypatch.setattr(
        submission,
        "update_operation_summary",
        lambda connection, *, operation_key, experiment_name, queue_name: (
            submission.SubmitPredictionSpecsResult(
                operation_key=operation_key,
                experiment_name=experiment_name,
                queue_name=queue_name,
                requested_count=len(specs),
                inserted_count=len(specs),
                already_present_count=0,
                enqueued_count=sum(
                    item.enqueue_status
                    is BatchSubmitItemEnqueueStatus.ENQUEUED
                    for item in item_updates
                ),
                failed_count=sum(
                    item.enqueue_status is BatchSubmitItemEnqueueStatus.FAILED
                    for item in item_updates
                ),
                items=tuple(item_updates),
            )
        ),
    )

    result = submission.submit_prediction_specs(
        cast(Any, engine),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs=specs,
        enqueue_workflow=enqueue,
    )

    assert result.failed_count == 1
    assert result.enqueued_count == 2
    assert [item.enqueue_status for item in item_updates] == [
        BatchSubmitItemEnqueueStatus.ENQUEUED,
        BatchSubmitItemEnqueueStatus.FAILED,
        BatchSubmitItemEnqueueStatus.ENQUEUED,
    ]
    failed = next(
        item
        for item in item_updates
        if item.enqueue_status is BatchSubmitItemEnqueueStatus.FAILED
    )
    assert failed.prediction_id == failed_prediction_id
    assert failed.failure is not None


def test_submit_prediction_specs_resume_retries_only_failed_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        _spec(task_id="HumanEval/0"),
        _spec(task_id="HumanEval/1"),
        _spec(task_id="HumanEval/2"),
    )
    engine = DummyEngine()
    failed_prediction_id = specs[1].prediction_id
    prepare_operation_keys: list[str] = []
    enqueue_calls_by_run: list[list[str]] = []
    active_run_calls: list[str] = []
    item_state = {
        spec.prediction_id: (
            BatchSubmitItemEnqueueStatus.PENDING,
            {},
        )
        for spec in specs
    }
    failed_once = False

    def prepare(
        connection: Connection,
        *,
        operation_key: str,
        experiment_name: str,
        ordered_specs: Sequence[PredictionSpecRecord],
        submit_spec: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        prepare_operation_keys.append(operation_key)

    def candidates(
        connection: Connection,
        *,
        operation_key: str,
        prediction_ids: Sequence[str],
    ) -> tuple[submission.EnqueueCandidate, ...]:
        by_id = {spec.prediction_id: spec for spec in specs}
        return tuple(
            submission.EnqueueCandidate(
                prediction_id=prediction_id,
                fair_order_key=by_id[prediction_id].fair_order_key,
                item_index=index,
                insert_status=BatchSubmitItemInsertStatus.INSERTED,
            )
            for index, prediction_id in enumerate(prediction_ids)
            if submission.item_needs_enqueue(
                enqueue_status=item_state[prediction_id][0],
                enqueue_metadata=item_state[prediction_id][1],
            )
        )

    def enqueue(
        database_url: str,
        prediction_id: str,
        attempt_index: int,
        queue_name: str,
    ) -> queue_worker.EnqueuedPredictionWorkflow:
        nonlocal failed_once
        active_run_calls.append(prediction_id)
        if prediction_id == failed_prediction_id and not failed_once:
            failed_once = True
            raise RuntimeError("queue unavailable")
        return queue_worker.EnqueuedPredictionWorkflow(
            prediction_id=prediction_id,
            generation_run_id=stable_generation_run_id(
                prediction_id=prediction_id,
                attempt_index=attempt_index,
            ),
            workflow_id=f"workflow:{prediction_id}",
            enqueued=True,
        )

    def update_item(
        connection: Connection,
        *,
        operation_key: str,
        item: submission.SubmittedPredictionItem,
    ) -> None:
        metadata = (
            {"workflow_id": item.workflow_id}
            if item.workflow_id is not None
            else {}
        )
        item_state[item.prediction_id] = (item.enqueue_status, metadata)

    def summary(
        connection: Connection,
        *,
        operation_key: str,
        experiment_name: str,
        queue_name: str,
    ) -> submission.SubmitPredictionSpecsResult:
        items = tuple(
            submission.SubmittedPredictionItem(
                prediction_id=spec.prediction_id,
                fair_order_key=spec.fair_order_key,
                insert_status=BatchSubmitItemInsertStatus.INSERTED,
                enqueue_status=item_state[spec.prediction_id][0],
            )
            for spec in specs
        )
        return submission.SubmitPredictionSpecsResult(
            operation_key=operation_key,
            experiment_name=experiment_name,
            queue_name=queue_name,
            requested_count=len(specs),
            inserted_count=len(specs),
            already_present_count=0,
            enqueued_count=sum(
                item.enqueue_status is BatchSubmitItemEnqueueStatus.ENQUEUED
                for item in items
            ),
            failed_count=sum(
                item.enqueue_status is BatchSubmitItemEnqueueStatus.FAILED
                for item in items
            ),
            items=items,
        )

    monkeypatch.setattr(submission, "prepare_submission_records", prepare)
    monkeypatch.setattr(submission, "load_enqueue_candidates", candidates)
    monkeypatch.setattr(
        submission,
        "update_batch_item_outcome",
        update_item,
    )
    monkeypatch.setattr(submission, "update_operation_summary", summary)

    first = submission.submit_prediction_specs(
        cast(Any, engine),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs=specs,
        enqueue_workflow=enqueue,
    )
    enqueue_calls_by_run.append(active_run_calls)
    active_run_calls = []
    second = submission.submit_prediction_specs(
        cast(Any, engine),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs=specs,
        enqueue_workflow=enqueue,
    )
    enqueue_calls_by_run.append(active_run_calls)

    assert prepare_operation_keys == ["op-1", "op-1"]
    assert first.failed_count == 1
    assert second.failed_count == 0
    assert second.enqueued_count == 3
    assert enqueue_calls_by_run[1] == [failed_prediction_id]


def test_item_needs_enqueue_supports_resume() -> None:
    assert submission.item_needs_enqueue(
        enqueue_status=BatchSubmitItemEnqueueStatus.PENDING,
        enqueue_metadata={},
    )
    assert submission.item_needs_enqueue(
        enqueue_status=BatchSubmitItemEnqueueStatus.FAILED,
        enqueue_metadata={},
    )
    assert submission.item_needs_enqueue(
        enqueue_status=BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT,
        enqueue_metadata={},
    )
    assert not submission.item_needs_enqueue(
        enqueue_status=BatchSubmitItemEnqueueStatus.ENQUEUED,
        enqueue_metadata={},
    )
    assert not submission.item_needs_enqueue(
        enqueue_status=BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT,
        enqueue_metadata={"workflow_id": "workflow-1"},
    )


def test_prepare_submission_records_upserts_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(task_id="HumanEval/0")
    connection = DummyConnection()
    experiment_statement = object()
    operation_statement = object()

    monkeypatch.setattr(
        submission,
        "idempotent_insert_experiment",
        lambda record: experiment_statement,
    )
    monkeypatch.setattr(
        submission,
        "idempotent_insert_batch_operation",
        lambda record: operation_statement,
    )
    monkeypatch.setattr(
        submission,
        "bulk_insert_prediction_specs",
        lambda connection, chunk: {spec.prediction_id},
    )
    monkeypatch.setattr(
        submission,
        "insert_batch_item",
        lambda connection, *, record: None,
    )

    submission.prepare_submission_records(
        cast(Connection, connection),
        operation_key="op-1",
        experiment_name="exp",
        ordered_specs=(spec,),
        submit_spec={},
        metadata={},
    )

    assert connection.statements[:2] == [
        experiment_statement,
        operation_statement,
    ]


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


def test_register_platform_generation_queue_updates_existing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def register_queue(
        queue_name: str,
        *,
        worker_concurrency: int,
        on_conflict: str,
    ) -> object:
        captured.update(
            {
                "queue_name": queue_name,
                "worker_concurrency": worker_concurrency,
                "on_conflict": on_conflict,
            }
        )
        return object()

    monkeypatch.setattr(queue_worker.DBOS, "register_queue", register_queue)

    queue_worker.register_platform_generation_queue(worker_concurrency=4)

    assert captured == {
        "queue_name": queue_worker.PLATFORM_GENERATION_QUEUE_NAME,
        "worker_concurrency": 4,
        "on_conflict": "always_update",
    }


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
