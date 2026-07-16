from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from dr_platform import FailureClass as PlatformFailureClass

from whetstone.eval_failures import TransientFailureError
from whetstone.platform import graph_workflow


class _FakeEngine:
    @contextmanager
    def begin(self) -> Iterator[object]:
        yield object()

    def dispose(self) -> None:
        pass


def test_throttle_record_uses_platform_failure_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def record_failure(_connection: object, **values: Any) -> None:
        recorded.update(values)

    monkeypatch.setattr(
        graph_workflow, "create_engine", lambda _url: _FakeEngine()
    )
    monkeypatch.setattr(
        graph_workflow,
        "resolve_application_database_url",
        lambda: "postgresql://unused",
    )
    monkeypatch.setattr(
        graph_workflow, "record_throttle_failure", record_failure
    )

    graph_workflow.record_throttle_failure_state(
        throttle_key="provider:test",
        error=TransientFailureError("provider unavailable"),
    )

    assert recorded["failure_class"] is PlatformFailureClass.TRANSIENT
