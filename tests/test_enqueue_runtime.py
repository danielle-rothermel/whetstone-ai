from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from whetstone.platform import enqueue_runtime
from whetstone.platform.enqueue_runtime import (
    InProcessDbosApi,
    RegisteredQueueLookup,
    platform_enqueue_runtime,
)

APPLICATION_URL = "postgresql://user:secret@localhost:5432/whetstone"
SQLITE_SYSTEM_URL = "sqlite:////tmp/dbos-system.sqlite"


def _recording_dbos() -> type[Any]:
    """A DBOS stand-in recording lifecycle calls; unit tests never launch."""

    class RecordingDbos:
        calls: ClassVar[list[tuple[str, Any]]] = []

        def __init__(self, *, config: dict[str, Any]) -> None:
            type(self).calls.append(("init", dict(config)))

        @classmethod
        def listen_queues(cls, queues: list[str]) -> None:
            cls.calls.append(("listen_queues", list(queues)))

        @classmethod
        def launch(cls) -> None:
            cls.calls.append(("launch", None))

    return RecordingDbos


def _patch_runtime_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    dbos: type[Any],
    registrations: list[dict[str, Any]],
    shutdowns: list[bool],
) -> None:
    monkeypatch.setattr(enqueue_runtime, "DBOS", dbos)
    monkeypatch.setattr(
        enqueue_runtime,
        "register_execution_queues",
        lambda **kwargs: registrations.append(kwargs),
    )
    monkeypatch.setattr(
        enqueue_runtime,
        "shutdown_dbos_runtime",
        lambda: shutdowns.append(True),
    )


def test_registered_queue_lookup_resolves_member_queues_through_dbos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission needs the database-backed Queue object, never a bare name."""
    resolved = object()
    monkeypatch.setattr(
        enqueue_runtime,
        "DBOS",
        SimpleNamespace(retrieve_queue=lambda name: resolved),
    )
    lookup = RegisteredQueueLookup(names=frozenset({"member"}))
    assert lookup.retrieve_queue("member") is resolved
    assert lookup.retrieve_queue("other") is None


def test_runtime_pins_empty_listen_set_and_never_updates_worker_queues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator process must never consume or reconfigure paid queues."""
    dbos = _recording_dbos()
    registrations: list[dict[str, Any]] = []
    shutdowns: list[bool] = []
    _patch_runtime_boundaries(monkeypatch, dbos, registrations, shutdowns)

    with platform_enqueue_runtime(
        application_database_url=APPLICATION_URL,
        system_database_url=SQLITE_SYSTEM_URL,
    ) as runtime:
        assert runtime.queue_lookup.names == frozenset(
            {"whetstone-generation", "whetstone-scoring"}
        )

    assert [name for name, _ in dbos.calls] == [
        "init",
        "listen_queues",
        "launch",
    ]
    assert dbos.calls[1][1] == []
    config = dbos.calls[0][1]
    assert config["name"] == "whetstone"
    assert config["system_database_url"] == SQLITE_SYSTEM_URL
    assert config["executor_id"].startswith("whetstone-enqueue-")
    assert registrations == [
        {"worker_concurrency": 1, "on_conflict": "never_update"}
    ]
    assert shutdowns == [True]


def test_runtime_uses_a_unique_executor_id_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct executor IDs keep launch recovery away from worker work."""
    dbos = _recording_dbos()
    _patch_runtime_boundaries(monkeypatch, dbos, [], [])

    for _invocation in range(2):
        with platform_enqueue_runtime(
            application_database_url=APPLICATION_URL,
            system_database_url=SQLITE_SYSTEM_URL,
        ):
            pass

    executor_ids = [
        config["executor_id"] for name, config in dbos.calls if name == "init"
    ]
    assert len(executor_ids) == 2
    assert executor_ids[0] != executor_ids[1]


def test_runtime_shuts_down_after_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dbos = _recording_dbos()
    shutdowns: list[bool] = []
    _patch_runtime_boundaries(monkeypatch, dbos, [], shutdowns)

    def failing_launch() -> None:
        raise RuntimeError("launch failed")

    monkeypatch.setattr(dbos, "launch", failing_launch)

    with pytest.raises(RuntimeError, match="launch failed"):
        with platform_enqueue_runtime(
            application_database_url=APPLICATION_URL,
            system_database_url=SQLITE_SYSTEM_URL,
        ):
            pass  # pragma: no cover -- launch fails before the yield

    assert shutdowns == [True]


def test_in_process_api_exposes_sys_db_and_delegates_workflow_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facade serves the exact DBOSClient surface lifecycle readers use."""
    sys_db = SimpleNamespace(engine=object())
    monkeypatch.setattr(
        enqueue_runtime,
        "_get_dbos_instance",
        lambda: SimpleNamespace(_sys_db=sys_db),
    )
    lookups: list[dict[str, Any]] = []
    rows = [SimpleNamespace(status="SUCCESS")]
    monkeypatch.setattr(
        enqueue_runtime,
        "DBOS",
        SimpleNamespace(
            list_workflows=lambda **kwargs: (lookups.append(kwargs), rows)[1]
        ),
    )

    api = InProcessDbosApi()

    assert api._sys_db is sys_db
    assert (
        api.list_workflows(workflow_ids=["wf-a"], limit=2, load_output=False)
        == rows
    )
    assert lookups == [
        {"workflow_ids": ["wf-a"], "limit": 2, "load_output": False}
    ]


def test_in_process_api_delegates_cancellation_with_children_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation forwards the DBOSClient.cancel_workflow signature."""
    monkeypatch.setattr(
        enqueue_runtime,
        "_get_dbos_instance",
        lambda: SimpleNamespace(_sys_db=SimpleNamespace()),
    )
    cancellations: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        enqueue_runtime,
        "DBOS",
        SimpleNamespace(
            cancel_workflow=lambda workflow_id, *, cancel_children: (
                cancellations.append((workflow_id, cancel_children))
            )
        ),
    )

    api = InProcessDbosApi()
    api.cancel_workflow("wf-a", cancel_children=True)
    api.cancel_workflow("wf-b")

    assert cancellations == [("wf-a", True), ("wf-b", False)]
