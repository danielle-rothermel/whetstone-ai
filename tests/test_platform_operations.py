from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import typer

from whetstone.platform import operations


class _Engine:
    def dispose(self) -> None:
        pass


def test_list_uses_the_canonical_whetstone_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setattr(operations, "_engine", _Engine)

    def fake_list_operations(**kwargs: Any) -> tuple[Any, ...]:
        observed.update(kwargs)
        return ()

    monkeypatch.setattr(operations, "list_operations", fake_list_operations)
    operations.operation_list()

    assert observed["schema"] is operations.PLATFORM_SCHEMA
    assert operations.PLATFORM_SCHEMA.prefix == "whetstone"


def test_tuple_json_output_is_one_valid_array(
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = (
        operations.AttemptPreview(
            item_id="a",
            source_attempt=0,
            workflow_id="w-a",
            execution_state="active",
        ),
        operations.AttemptPreview(
            item_id="b",
            source_attempt=1,
            workflow_id="w-b",
            execution_state="succeeded",
        ),
    )

    operations._emit(values, as_json=True)

    assert json.loads(capsys.readouterr().out) == [
        values[0].model_dump(mode="json"),
        values[1].model_dump(mode="json"),
    ]


def test_dbos_error_preserves_authoritative_terminal_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class Client:
        def list_workflows(self, **kwargs: Any) -> list[Any]:
            calls.append(kwargs["load_output"])
            if kwargs.get("parent_workflow_id"):
                return []
            return [
                SimpleNamespace(
                    status="ERROR",
                    error=ValueError("invalid persisted recipe")
                    if kwargs["load_output"]
                    else None,
                )
            ]

    canceller = object.__new__(operations.WhetstoneDbosCanceller)
    canceller.client = Client()

    result = canceller.inspect(workflow_id="workflow")

    assert result.disposition.value == "error"
    assert result.failure is not None
    assert result.failure.failure_class.value == "permanent"
    assert result.retry_disposition is not None
    assert result.retry_disposition.value == "permanent"
    assert calls == [False, True, False]


def test_dbos_error_fails_closed_without_classified_failure() -> None:
    class Client:
        def list_workflows(self, **kwargs: Any) -> list[Any]:
            return [SimpleNamespace(status="ERROR", error=None)]

    canceller = object.__new__(operations.WhetstoneDbosCanceller)
    canceller.client = Client()

    with pytest.raises(RuntimeError, match="classification is unavailable"):
        canceller.inspect(workflow_id="workflow")


def test_cancel_preview_does_not_mutate_and_drift_blocks_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine()
    preview = operations._mutation_preview(
        command="cancel",
        request_identity={
            "operation_key": "operation",
            "request_id": "request",
            "requested_by": "operator",
        },
        operation_key="operation",
        platform_cut_version=3,
        affected_attempts=(),
        eligible=True,
        exhausted=False,
        rejection_detail=None,
    )
    mutation_calls = 0

    class Canceller:
        client = SimpleNamespace(destroy=lambda: None)

    def mutate(*args: Any, **kwargs: Any) -> None:
        nonlocal mutation_calls
        mutation_calls += 1

    monkeypatch.setattr(operations, "_engine", lambda: engine)
    monkeypatch.setattr(operations, "WhetstoneDbosCanceller", Canceller)
    monkeypatch.setattr(
        operations, "_cancel_preview", lambda **kwargs: preview
    )
    monkeypatch.setattr(operations, "cancel_operation", mutate)

    operations.cancel("operation", "request", "operator")
    assert mutation_calls == 0

    with pytest.raises(typer.BadParameter, match="preview drift"):
        operations.cancel(
            "operation",
            "request",
            "operator",
            confirm=True,
            preview_digest="stale",
        )
    assert mutation_calls == 0

    operations.cancel(
        "operation",
        "request",
        "operator",
        confirm=True,
        preview_digest=preview.preview_digest,
    )
    assert mutation_calls == 1
