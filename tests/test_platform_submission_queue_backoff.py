from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection
from typer.testing import CliRunner

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
from dr_dspy.platform import (
    backoff,
    fairness,
    queue_worker,
    submission,
    worker,
)
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

    ordered = fairness.fair_ordered_specs(reversed(specs))

    assert [spec.fair_order_key for spec in ordered] == sorted(
        spec.fair_order_key for spec in specs
    )


def test_fair_ordering_interleaves_model_axis() -> None:
    specs = tuple(
        _spec(task_id=f"HumanEval/{task_id}", model=model)
        for model in ("model-a", "model-b")
        for task_id in range(8)
    )

    ordered = fairness.fair_ordered_specs(specs)
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
    candidate_pages: list[tuple[str, ...]] = []
    enqueue_calls: list[str] = []
    item_updates: list[submission.SubmittedPredictionItem] = []
    pending_ids = {spec.prediction_id for spec in specs}
    by_id = {spec.prediction_id: spec for spec in specs}

    def prepare(
        connection: Connection,
        *,
        operation_key: str,
        experiment_name: str,
        ordered_specs: Sequence[PredictionSpecRecord],
        submit_spec: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        chunk_size: int,
        item_index_offset: int,
    ) -> None:
        assert engine.in_transaction is True
        assert chunk_size == 2
        chunks.append(tuple(spec.prediction_id for spec in ordered_specs))

    def candidates(
        connection: Connection,
        *,
        operation_key: str,
        limit: int,
    ) -> tuple[submission.EnqueueCandidate, ...]:
        assert engine.in_transaction is True
        prediction_ids = tuple(
            spec.prediction_id
            for spec in sorted(
                specs,
                key=lambda spec: (spec.fair_order_key, spec.prediction_id),
            )
            if spec.prediction_id in pending_ids
        )[:limit]
        candidate_pages.append(prediction_ids)
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
        pending_ids.discard(item.prediction_id)
        item_updates.append(item)

    monkeypatch.setattr(
        submission,
        "prepare_submission_records",
        prepare,
    )
    monkeypatch.setattr(
        submission,
        "load_pending_enqueue_candidates",
        candidates,
    )
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
                already_scheduled_count=0,
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

    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert [len(page) for page in candidate_pages] == [2, 1, 0]
    assert enqueue_calls == [item.prediction_id for item in item_updates]
    assert result.requested_count == 3
    assert result.inserted_count == 2
    assert result.already_present_count == 1
    assert result.enqueued_count == 3
    assert result.failed_count == 0
    assert {item.enqueue_status for item in item_updates} == {
        BatchSubmitItemEnqueueStatus.ENQUEUED
    }
    assert engine.begin_count == 10


def test_submit_prediction_specs_enqueues_after_all_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = tuple(
        _spec(task_id=f"HumanEval/{task_id}")
        for task_id in range(5)
    )
    engine = DummyEngine()
    yielded_count = 0
    first_enqueue_yielded_count: int | None = None

    def iter_specs() -> Iterable[PredictionSpecRecord]:
        nonlocal yielded_count
        for spec in specs:
            yielded_count += 1
            yield spec

    def prepare(
        connection: Connection,
        *,
        operation_key: str,
        experiment_name: str,
        ordered_specs: Sequence[PredictionSpecRecord],
        submit_spec: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        chunk_size: int,
        item_index_offset: int,
    ) -> None:
        assert len(ordered_specs) <= 2

    pending_ids = {spec.prediction_id for spec in specs}
    by_id = {spec.prediction_id: spec for spec in specs}

    def candidates(
        connection: Connection,
        *,
        operation_key: str,
        limit: int,
    ) -> tuple[submission.EnqueueCandidate, ...]:
        prediction_ids = tuple(
            spec.prediction_id
            for spec in sorted(
                specs,
                key=lambda spec: (spec.fair_order_key, spec.prediction_id),
            )
            if spec.prediction_id in pending_ids
        )[:limit]
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
        nonlocal first_enqueue_yielded_count
        if first_enqueue_yielded_count is None:
            first_enqueue_yielded_count = yielded_count
        return queue_worker.EnqueuedPredictionWorkflow(
            prediction_id=prediction_id,
            generation_run_id=stable_generation_run_id(
                prediction_id=prediction_id,
                attempt_index=attempt_index,
            ),
            workflow_id=f"workflow:{prediction_id}",
            enqueued=True,
        )

    monkeypatch.setattr(submission, "prepare_submission_records", prepare)
    monkeypatch.setattr(
        submission,
        "load_pending_enqueue_candidates",
        candidates,
    )
    monkeypatch.setattr(
        submission,
        "update_batch_item_outcome",
        lambda connection, *, operation_key, item: pending_ids.discard(
            item.prediction_id
        ),
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
                enqueued_count=len(specs),
                already_scheduled_count=0,
                failed_count=0,
            )
        ),
    )

    submission.submit_prediction_specs(
        cast(Any, engine),
        database_url="postgresql://example/db",
        operation_key="op-1",
        experiment_name="exp",
        specs=iter_specs(),
        chunk_size=2,
        enqueue_workflow=enqueue,
    )

    assert first_enqueue_yielded_count == 5
    assert yielded_count == 5


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_submit_prediction_specs_rejects_duplicate_prediction_ids(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int,
) -> None:
    spec = _spec(task_id="HumanEval/0")
    engine = DummyEngine()
    prepared_windows: list[tuple[str, ...]] = []

    def prepare(
        connection: Connection,
        *,
        operation_key: str,
        experiment_name: str,
        ordered_specs: Sequence[PredictionSpecRecord],
        submit_spec: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        chunk_size: int,
        item_index_offset: int,
    ) -> None:
        prepared_windows.append(
            tuple(spec.prediction_id for spec in ordered_specs)
        )

    monkeypatch.setattr(submission, "prepare_submission_records", prepare)
    monkeypatch.setattr(
        submission,
        "load_pending_enqueue_candidates",
        lambda connection, *, operation_key, limit: (),
    )

    with pytest.raises(ValueError, match="duplicate prediction_id"):
        submission.submit_prediction_specs(
            cast(Any, engine),
            database_url="postgresql://example/db",
            operation_key="op-1",
            experiment_name="exp",
            specs=(spec, spec),
            chunk_size=chunk_size,
        )

    expected_prepared_windows = (
        [(spec.prediction_id,)] if chunk_size == 1 else []
    )
    assert prepared_windows == expected_prepared_windows


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
    pending_ids = {spec.prediction_id for spec in specs}

    monkeypatch.setattr(
        submission,
        "prepare_submission_records",
        lambda connection, **kwargs: None,
    )
    monkeypatch.setattr(
        submission,
        "load_pending_enqueue_candidates",
        lambda connection, *, operation_key, limit: tuple(
            submission.EnqueueCandidate(
                prediction_id=spec.prediction_id,
                fair_order_key=spec.fair_order_key,
                item_index=index,
                insert_status=BatchSubmitItemInsertStatus.INSERTED,
            )
            for index, spec in enumerate(specs)
            if spec.prediction_id in pending_ids
        )[:limit],
    )
    monkeypatch.setattr(
        submission,
        "update_batch_item_outcome",
        lambda connection, *, operation_key, item: (
            pending_ids.discard(item.prediction_id),
            item_updates.append(item),
        ),
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
                already_scheduled_count=0,
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
    reset_operation_keys: list[str] = []
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
        chunk_size: int,
        item_index_offset: int,
    ) -> None:
        prepare_operation_keys.append(operation_key)

    by_id = {spec.prediction_id: spec for spec in specs}

    def candidates(
        connection: Connection,
        *,
        operation_key: str,
        limit: int,
    ) -> tuple[submission.EnqueueCandidate, ...]:
        prediction_ids = tuple(
            spec.prediction_id
            for spec in sorted(
                specs,
                key=lambda spec: (spec.fair_order_key, spec.prediction_id),
            )
            if item_state[spec.prediction_id][0]
            is BatchSubmitItemEnqueueStatus.PENDING
        )[:limit]
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

    def reset_failed(connection: Connection, *, operation_key: str) -> None:
        reset_operation_keys.append(operation_key)
        for prediction_id, (status, _metadata) in item_state.items():
            if status is BatchSubmitItemEnqueueStatus.FAILED:
                item_state[prediction_id] = (
                    BatchSubmitItemEnqueueStatus.PENDING,
                    {},
                )

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
            already_scheduled_count=sum(
                (
                    item.enqueue_status
                    is BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT
                )
                for item in items
            ),
            failed_count=sum(
                item.enqueue_status is BatchSubmitItemEnqueueStatus.FAILED
                for item in items
            ),
            items=items,
        )

    monkeypatch.setattr(submission, "prepare_submission_records", prepare)
    monkeypatch.setattr(
        submission,
        "load_pending_enqueue_candidates",
        candidates,
    )
    monkeypatch.setattr(submission, "reset_failed_enqueue_items", reset_failed)
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
    assert reset_operation_keys == ["op-1", "op-1"]
    assert first.failed_count == 1
    assert second.failed_count == 0
    assert second.enqueued_count == 3
    assert enqueue_calls_by_run[1] == [failed_prediction_id]


def test_pending_enqueue_selector_orders_operation_by_fair_key() -> None:
    statement = submission.select_pending_batch_items_for_enqueue(
        operation_key="op-1",
        limit=500,
    )

    rendered = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "dr_dspy_batch_submit_items.operation_key = 'op-1'" in rendered
    assert "dr_dspy_batch_submit_items.enqueue_status = 'pending'" in rendered
    assert "dr_dspy_batch_submit_items.prediction_id IN" not in rendered
    assert (
        "ORDER BY dr_dspy_batch_submit_items.fair_order_key, "
        "dr_dspy_batch_submit_items.prediction_id" in rendered
    )
    assert "LIMIT 500" in rendered


def test_update_operation_summary_counts_already_scheduled_items() -> None:
    specs = (
        _spec(task_id="HumanEval/0"),
        _spec(task_id="HumanEval/1"),
    )

    class Result:
        def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
            self.rows = rows

        def mappings(self) -> tuple[dict[str, Any], ...]:
            return self.rows

    class ConnectionWithRows:
        def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
            self.rows = rows
            self.statements: list[Any] = []

        def execute(self, statement: Any) -> Result:
            self.statements.append(statement)
            if len(self.statements) == 1:
                return Result(self.rows)
            return Result(())

    rows = tuple(
        {
            "prediction_id": spec.prediction_id,
            "fair_order_key": spec.fair_order_key,
            "insert_status": BatchSubmitItemInsertStatus.INSERTED.value,
            "enqueue_status": (
                BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT.value
            ),
            "enqueue_metadata": {
                "workflow_id": f"workflow:{spec.prediction_id}",
                "generation_run_id": stable_generation_run_id(
                    prediction_id=spec.prediction_id,
                    attempt_index=0,
                ),
            },
            "failure": None,
        }
        for spec in specs
    )

    result = submission.update_operation_summary(
        cast(Connection, ConnectionWithRows(rows)),
        operation_key="op-1",
        experiment_name="exp",
        queue_name="queue",
    )

    assert result.requested_count == 2
    assert result.enqueued_count == 0
    assert result.already_scheduled_count == 2
    assert result.failed_count == 0


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


def test_prepare_submission_records_marks_operation_enqueuing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        _spec(task_id="HumanEval/2"),
        _spec(task_id="HumanEval/3"),
    )
    connection = DummyConnection()
    operation_statuses: list[Any] = []
    requested_counts: list[int] = []
    item_indexes: list[int] = []

    monkeypatch.setattr(
        submission,
        "idempotent_insert_experiment",
        lambda record: object(),
    )

    def insert_operation(record: Any) -> object:
        operation_statuses.append(record.status)
        return object()

    monkeypatch.setattr(
        submission,
        "idempotent_insert_batch_operation",
        insert_operation,
    )
    monkeypatch.setattr(
        submission,
        "mark_operation_enqueuing",
        lambda connection, *, operation_key, requested_count: (
            requested_counts.append(requested_count)
        ),
    )
    monkeypatch.setattr(
        submission,
        "bulk_insert_prediction_specs",
        lambda connection, chunk: {spec.prediction_id for spec in chunk},
    )
    monkeypatch.setattr(
        submission,
        "insert_batch_item",
        lambda connection, *, record: item_indexes.append(record.item_index),
    )

    submission.prepare_submission_records(
        cast(Connection, connection),
        operation_key="op-1",
        experiment_name="exp",
        ordered_specs=specs,
        submit_spec={},
        metadata={},
        item_index_offset=5,
    )

    assert operation_statuses == [
        submission.BatchSubmitOperationStatus.ENQUEUING
    ]
    assert requested_counts == [7]
    assert item_indexes == [5, 6]


def test_prepare_submission_records_uses_requested_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = tuple(
        _spec(task_id=f"HumanEval/{task_id}")
        for task_id in range(5)
    )
    connection = DummyConnection()
    chunks: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        submission,
        "idempotent_insert_experiment",
        lambda record: object(),
    )
    monkeypatch.setattr(
        submission,
        "idempotent_insert_batch_operation",
        lambda record: object(),
    )

    def bulk_insert(
        connection: Connection,
        chunk: Sequence[PredictionSpecRecord],
    ) -> set[str]:
        chunks.append(tuple(spec.prediction_id for spec in chunk))
        return {spec.prediction_id for spec in chunk}

    monkeypatch.setattr(
        submission,
        "bulk_insert_prediction_specs",
        bulk_insert,
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
        ordered_specs=specs,
        submit_spec={},
        metadata={},
        chunk_size=2,
    )

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


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


def test_submit_jsonl_help_names_queue_registration_concurrency() -> None:
    result = CliRunner().invoke(worker.APP, ["submit-jsonl", "--help"])

    assert result.exit_code == 0
    assert "--queue-registration" in result.output
    assert "does not start a queue" in result.output
    assert "worker." in result.output


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
