from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from dr_dspy import batch_operation, eval_repair


class _CursorContext:
    def __init__(self, cursor: MagicMock) -> None:
        self.cursor = cursor

    def __enter__(self) -> MagicMock:
        return self.cursor

    def __exit__(self, *args: object) -> None:
        return None


class _Connection:
    def __init__(self, cursor: MagicMock) -> None:
        self._cursor = cursor

    def cursor(self) -> _CursorContext:
        return _CursorContext(self._cursor)


@contextmanager
def _connect_with_cursor(cursor: MagicMock) -> Iterator[_Connection]:
    yield _Connection(cursor)


def _progress(
    *,
    operation_kind: batch_operation.BatchOperationKind,
    total_items: int = 10,
    next_offset: int = 0,
    processed_count: int = 0,
    counters: dict[str, int] | None = None,
) -> batch_operation.BatchOperationProgress:
    return batch_operation.BatchOperationProgress(
        operation_kind=operation_kind,
        operation_key="op-key",
        experiment_name="experiment",
        script_kind="script",
        workflow_id="workflow",
        attempt=1,
        status=batch_operation.BatchOperationStatus.PENDING,
        total_items=total_items,
        next_offset=next_offset,
        metadata={},
        processed_count=processed_count,
        inserted_count=0,
        enqueued_count=0,
        existing_workflow_count=0,
        marked_count=0,
        batch_count=0,
        counters=counters or {},
        last_error=None,
        log_file="/tmp/batch-operation.log",
    )


def test_operation_key_is_stable_and_order_insensitive() -> None:
    left = {"experiment": "exp", "limit": 10, "nested": {"a": 1, "b": 2}}
    right = {"nested": {"b": 2, "a": 1}, "limit": 10, "experiment": "exp"}
    changed = {"experiment": "exp", "limit": 11, "nested": {"a": 1, "b": 2}}

    left_key = batch_operation.operation_key(left)

    assert left_key == batch_operation.operation_key(right)
    assert left_key != batch_operation.operation_key(changed)


def test_operation_workflow_id_includes_kind_key_and_attempt() -> None:
    workflow_id = batch_operation.operation_workflow_id(
        batch_operation.BatchOperationKind.REPAIR, "abc", 2
    )

    assert workflow_id == "repair:abc:2"


def test_operation_item_table_sql_references_operation_table() -> None:
    ddl = batch_operation.operation_item_table_sql()

    assert "CREATE TABLE IF NOT EXISTS dr_dspy_batch_operation_items" in ddl
    assert "REFERENCES dr_dspy_batch_operations" in ddl
    assert "ON DELETE CASCADE" in ddl


def test_record_operation_items_inserts_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    inserted: list[tuple[Any, ...]] = []

    def executemany(_query: str, params: list[tuple[Any, ...]]) -> None:
        inserted.extend(params)

    cursor.executemany.side_effect = executemany
    monkeypatch.setattr(
        batch_operation.dbos_runtime,
        "connect_db",
        lambda _database_url: _connect_with_cursor(cursor),
    )

    batch_operation.record_operation_items(
        "postgresql://x",
        operation_kind=batch_operation.BatchOperationKind.SUBMIT,
        operation_key="op-key",
        item_kind=batch_operation.BatchOperationItemKind.SAMPLE,
        payloads=[{"task_id": "a"}, {"task_id": "b"}],
    )

    assert [row[:4] for row in inserted] == [
        ("submit", "op-key", "sample", 0),
        ("submit", "op-key", "sample", 1),
    ]


def test_record_operation_items_accepts_matching_existing_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [{"task_id": "a"}, {"task_id": "b"}]
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (index, batch_operation.payload_digest(payload))
        for index, payload in enumerate(payloads)
    ]
    monkeypatch.setattr(
        batch_operation.dbos_runtime,
        "connect_db",
        lambda _database_url: _connect_with_cursor(cursor),
    )

    batch_operation.record_operation_items(
        "postgresql://x",
        operation_kind=batch_operation.BatchOperationKind.SUBMIT,
        operation_key="op-key",
        item_kind=batch_operation.BatchOperationItemKind.SAMPLE,
        payloads=payloads,
    )

    cursor.executemany.assert_not_called()


def test_record_operation_items_rejects_mismatched_existing_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = [(0, "wrong")]
    monkeypatch.setattr(
        batch_operation.dbos_runtime,
        "connect_db",
        lambda _database_url: _connect_with_cursor(cursor),
    )

    with pytest.raises(ValueError, match="operation item manifest mismatch"):
        batch_operation.record_operation_items(
            "postgresql://x",
            operation_kind=batch_operation.BatchOperationKind.SUBMIT,
            operation_key="op-key",
            item_kind=batch_operation.BatchOperationItemKind.SAMPLE,
            payloads=[{"task_id": "a"}],
        )


def test_fetch_operation_items_returns_ordered_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = [({"task_id": "a"},), ({"task_id": "b"},)]
    monkeypatch.setattr(
        batch_operation.dbos_runtime,
        "connect_db",
        lambda _database_url: _connect_with_cursor(cursor),
    )

    payloads = batch_operation.fetch_operation_items(
        "postgresql://x",
        operation_kind=batch_operation.BatchOperationKind.SUBMIT,
        operation_key="op-key",
        item_kind=batch_operation.BatchOperationItemKind.SAMPLE,
        start_index=1,
        limit=2,
    )

    assert payloads == [{"task_id": "a"}, {"task_id": "b"}]


def test_merged_counters_adds_deltas_without_losing_existing_keys() -> None:
    assert batch_operation.merged_counters(
        {"generation_processed": 3, "marked": 1},
        {"generation_processed": 2, "scoring_processed": 5},
    ) == {
        "generation_processed": 5,
        "marked": 1,
        "scoring_processed": 5,
    }


def test_repair_batch_operation_result_accounts_for_selected_work() -> None:
    progress = _progress(
        operation_kind=batch_operation.BatchOperationKind.REPAIR,
        processed_count=7,
    )
    repair_result = eval_repair.RepairApplyResult(
        repair_token="token",
        stranded_generations_marked=2,
        generation_retries_enqueued=3,
        generation_retries_existing=1,
        generation_retries_reset=4,
        stranded_scoring_marked=5,
        pending_scoring_enqueued=6,
        pending_scoring_existing=2,
        pending_scoring_marked_queued=8,
        scoring_retries_enqueued=9,
        scoring_retries_existing=3,
        scoring_retries_marked_queued=11,
    )

    result = batch_operation.repair_batch_operation_result(
        progress=progress,
        batch_size=32,
        repair_result=repair_result,
    )

    assert result == batch_operation.BatchOperationResult(
        start_offset=7,
        next_offset=31,
        batch_size=32,
        processed=24,
        enqueued=18,
        existing_workflows=6,
        marked=30,
        counters={
            "generation_processed": 4,
            "scoring_processed": 20,
            "stranded_generations_marked": 2,
            "generation_retries_reset": 4,
            "stranded_scoring_marked": 5,
            "pending_scoring_marked_queued": 8,
            "scoring_retries_marked_queued": 11,
        },
    )


def test_run_operation_dispatcher_completes_offset_work_without_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []
    marks: list[str] = []
    configured_log_files: list[Path] = []
    completion_modes = batch_operation.BatchDispatcherCompletionMode
    progress = _progress(
        operation_kind=batch_operation.BatchOperationKind.SUBMIT,
        total_items=10,
        next_offset=10,
    )

    monkeypatch.setattr(
        batch_operation,
        "fetch_operation_progress",
        lambda *_args, **_kwargs: progress,
    )
    monkeypatch.setattr(
        batch_operation,
        "mark_operation_running",
        lambda *_args, **_kwargs: marks.append("running"),
    )
    monkeypatch.setattr(
        batch_operation,
        "mark_operation_completed",
        lambda *_args, **_kwargs: marks.append("completed"),
    )

    def batch_step(_database_url: str, _operation_key: str) -> (
        batch_operation.BatchOperationResult
    ):
        raise AssertionError("batch step should not run")

    status = batch_operation.run_operation_dispatcher(
        "postgres://example",
        operation_kind=batch_operation.BatchOperationKind.SUBMIT,
        operation_key="op-key",
        configure_logging=configured_log_files.append,
        emit_log=lambda event, payload: events.append((event, payload)),
        started_event="started",
        started_payload=lambda item: {"workflow_id": item.workflow_id},
        failed_event="failed",
        batch_step=batch_step,
        completion_mode=completion_modes.OFFSET_TOTAL,
        completed_event="completed",
        completed_payload=lambda item: {"batches": item.batch_count},
    )

    assert status == "completed"
    assert marks == ["running", "completed"]
    assert configured_log_files == [Path("/tmp/batch-operation.log")]
    assert events == [
        ("started", {"workflow_id": "workflow"}),
        ("completed", {"batches": 0}),
    ]


def test_run_operation_dispatcher_completes_after_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marks: list[str] = []
    completion_modes = batch_operation.BatchDispatcherCompletionMode
    progress = _progress(
        operation_kind=batch_operation.BatchOperationKind.ENQUEUE_SCORES,
        total_items=10,
        processed_count=0,
    )

    monkeypatch.setattr(
        batch_operation,
        "fetch_operation_progress",
        lambda *_args, **_kwargs: progress,
    )
    monkeypatch.setattr(
        batch_operation,
        "mark_operation_running",
        lambda *_args, **_kwargs: marks.append("running"),
    )
    monkeypatch.setattr(
        batch_operation,
        "mark_operation_completed",
        lambda *_args, **_kwargs: marks.append("completed"),
    )

    status = batch_operation.run_operation_dispatcher(
        "postgres://example",
        operation_kind=batch_operation.BatchOperationKind.ENQUEUE_SCORES,
        operation_key="op-key",
        configure_logging=lambda _path: None,
        emit_log=lambda _event, _payload: None,
        started_event="started",
        started_payload=lambda _progress: {},
        failed_event="failed",
        batch_step=lambda _database_url, _operation_key: (
            batch_operation.BatchOperationResult(
                start_offset=0,
                next_offset=0,
                batch_size=0,
                processed=0,
            )
        ),
        completion_mode=completion_modes.PROCESSED_TOTAL_OR_EMPTY_BATCH,
    )

    assert status == "completed"
    assert marks == ["running", "completed"]


def test_run_operation_dispatcher_marks_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []
    failures: list[str] = []
    completion_modes = batch_operation.BatchDispatcherCompletionMode
    progress = _progress(
        operation_kind=batch_operation.BatchOperationKind.REPAIR,
        total_items=10,
        processed_count=0,
    )

    monkeypatch.setattr(
        batch_operation,
        "fetch_operation_progress",
        lambda *_args, **_kwargs: progress,
    )
    monkeypatch.setattr(
        batch_operation,
        "mark_operation_running",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        batch_operation,
        "mark_operation_failed",
        lambda *args, **kwargs: failures.append(str(kwargs["error"])),
    )

    def batch_step(
        _database_url: str, _operation_key: str
    ) -> batch_operation.BatchOperationResult:
        raise RuntimeError("batch failed")

    with pytest.raises(RuntimeError, match="batch failed"):
        batch_operation.run_operation_dispatcher(
            "postgres://example",
            operation_kind=batch_operation.BatchOperationKind.REPAIR,
            operation_key="op-key",
            configure_logging=lambda _path: None,
            emit_log=lambda event, payload: events.append((event, payload)),
            started_event="started",
            started_payload=lambda _progress: {},
            failed_event="failed",
            batch_step=batch_step,
            completion_mode=completion_modes.PROCESSED_TOTAL_OR_EMPTY_BATCH,
        )

    assert failures == ["RuntimeError('batch failed')"]
    assert events == [
        ("started", {}),
        (
            "failed",
            {
                "operation_key": "op-key",
                "error": "RuntimeError('batch failed')",
            },
        ),
    ]
