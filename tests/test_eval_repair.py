from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dr_dspy import eval_repair
from dr_dspy.eval_repair import (
    RepairRetryCandidate,
    RepairRetrySelection,
    fetch_prediction_retry_selection,
    unique_prediction_ids,
)
from dr_dspy.failure_policy import FailureClass
from dr_dspy.prediction_status import GENERATION_RETRY_STATUSES


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


def test_retry_selection_summary_counts_recoverable_and_excluded() -> None:
    selection = RepairRetrySelection(
        candidates=[
            RepairRetryCandidate(
                prediction_id="recoverable",
                failure_class=FailureClass.TRANSIENT.value,
            ),
        ],
        excluded_count=3,
    )

    assert selection.prediction_ids == ["recoverable"]
    assert selection.summary.recoverable_count == 1
    assert selection.summary.excluded_count == 3
    assert selection.summary.retryable_count == 1


def test_retry_candidate_requires_failure_class() -> None:
    with pytest.raises(ValidationError):
        RepairRetryCandidate.model_validate({"prediction_id": "legacy"})


def test_fetch_retry_selection_excludes_null_failure_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("recoverable", FailureClass.TRANSIENT.value)
    ]
    cursor.fetchone.return_value = (2,)
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def execute(query: str, params: tuple[Any, ...]) -> None:
        calls.append((query, params))

    cursor.execute.side_effect = execute
    monkeypatch.setattr(
        eval_repair.dbos_runtime,
        "connect_db",
        lambda database_url: _connect_with_cursor(cursor),
    )

    selection = fetch_prediction_retry_selection(
        "postgresql://x",
        prediction_table="predictions",
        experiment_name="exp",
        status_column="generation_status",
        failure_class_column="generation_failure_class",
        retry_statuses=GENERATION_RETRY_STATUSES,
        order_columns=("prediction_id",),
        limit=10,
    )

    assert selection.prediction_ids == ["recoverable"]
    assert selection.summary.recoverable_count == 1
    assert selection.summary.excluded_count == 2
    retry_query = calls[0][0]
    excluded_query = calls[1][0]
    assert "generation_failure_class = ANY(%s)" in retry_query
    assert "generation_failure_class IS NULL" not in retry_query
    assert "generation_failure_class IS NULL" in excluded_query


def test_unique_prediction_ids_preserves_order() -> None:
    assert unique_prediction_ids(["p1", "p2", "p1", "p3"]) == [
        "p1",
        "p2",
        "p3",
    ]
