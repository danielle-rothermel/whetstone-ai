from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from dr_platform.reconciliation_runtime import (
    DbosLifecycleReader,
    ReconciliationObservationDisposition,
)

from whetstone.platform import publication
from whetstone.platform.operations import WhetstoneDbosCanceller

APPLICATION_URL = "postgresql://user:secret@localhost:5432/whetstone"
SQLITE_SYSTEM_URL = "sqlite:////tmp/dbos-system.sqlite"


def _patch_publication_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime: SimpleNamespace,
    facade: SimpleNamespace,
    runtime_calls: list[dict[str, Any]],
    engines: list[SimpleNamespace],
) -> None:
    @contextmanager
    def fake_runtime(**kwargs: Any) -> Iterator[SimpleNamespace]:
        runtime_calls.append(kwargs)
        yield runtime

    def fake_create_engine(url: str) -> SimpleNamespace:
        engine = SimpleNamespace(url=url, disposed=False)
        engine.dispose = lambda: setattr(engine, "disposed", True)
        engines.append(engine)
        return engine

    monkeypatch.setattr(
        publication, "platform_enqueue_runtime", fake_runtime
    )
    monkeypatch.setattr(publication, "InProcessDbosApi", lambda: facade)
    monkeypatch.setattr(publication, "create_engine", fake_create_engine)


def test_export_dependencies_run_over_the_in_process_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reader, admission, and canceller all share the launched runtime."""
    runtime = SimpleNamespace(
        queue_lookup=object(),
        enqueue_adapter=object(),
        workflow_observer=object(),
    )
    lookups: list[dict[str, Any]] = []
    facade = SimpleNamespace(
        list_workflows=lambda **kwargs: (lookups.append(kwargs), [])[1]
    )
    runtime_calls: list[dict[str, Any]] = []
    engines: list[SimpleNamespace] = []
    _patch_publication_runtime(
        monkeypatch,
        runtime=runtime,
        facade=facade,
        runtime_calls=runtime_calls,
        engines=engines,
    )
    resolver = object()
    monkeypatch.setattr(publication, "target_registry", lambda: resolver)

    with publication.build_export_reconciliation_dependencies(
        application_database_url=APPLICATION_URL,
        dbos_system_database_url=SQLITE_SYSTEM_URL,
    ) as dependencies:
        assert runtime_calls == [
            {
                "application_database_url": APPLICATION_URL,
                "system_database_url": SQLITE_SYSTEM_URL,
            }
        ]
        assert dependencies.resolver is resolver
        assert dependencies.queue_lookup is runtime.queue_lookup
        assert dependencies.enqueue_adapter is runtime.enqueue_adapter
        assert dependencies.recovery_observer is runtime.workflow_observer
        assert isinstance(dependencies.reader, DbosLifecycleReader)
        observation = dependencies.reader.observe(workflow_id="wf-absent")
        assert (
            observation.disposition
            is ReconciliationObservationDisposition.ABSENT
        )
        assert lookups == [
            {
                "workflow_ids": ["wf-absent"],
                "limit": 2,
                "load_input": False,
                "load_output": False,
            }
        ]
        assert isinstance(
            dependencies.compensation_canceller, WhetstoneDbosCanceller
        )
        assert dependencies.compensation_canceller.client is facade


def test_export_reconciled_cut_clock_reads_the_application_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dbos_engine feeds PostgreSQL-only proof SQL, never the SQLite store."""
    runtime = SimpleNamespace(
        queue_lookup=object(),
        enqueue_adapter=object(),
        workflow_observer=object(),
    )
    engines: list[SimpleNamespace] = []
    _patch_publication_runtime(
        monkeypatch,
        runtime=runtime,
        facade=SimpleNamespace(),
        runtime_calls=[],
        engines=engines,
    )
    monkeypatch.setattr(publication, "target_registry", lambda: object())

    with publication.build_export_reconciliation_dependencies(
        application_database_url=APPLICATION_URL,
        dbos_system_database_url=SQLITE_SYSTEM_URL,
    ) as dependencies:
        assert dependencies.dbos_engine is engines[0]
        assert engines[0].url == APPLICATION_URL
        assert not engines[0].disposed

    assert len(engines) == 1
    assert engines[0].disposed
