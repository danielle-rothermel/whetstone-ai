"""Shared operator-process DBOS runtime that never consumes paid queues.

Platform operations need a launched in-process DBOS to enqueue and observe
workflows, but they must never claim paid generation/scoring work or clobber
the long-lived worker's queue configuration.  ``DBOSClient`` cannot open SQLite
system databases (it applies pool keyword arguments SQLite rejects), so all
system-database access runs over the launched in-process runtime instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from dbos import DBOS
from dbos._dbos import _get_dbos_instance
from dr_platform.enqueue_runtime import (
    DbosEnqueueAdapter,
    DbosWorkflowObserver,
)

from whetstone.platform.runtime import (
    build_whetstone_dbos_config,
    dbos_config,
    shutdown_dbos_runtime,
)
from whetstone.platform.targets import (
    GENERATION_QUEUE_NAME,
    SCORING_QUEUE_NAME,
    register_execution_queues,
)

DBOS_APP_NAME = "whetstone"
EXECUTOR_ID_PREFIX = "whetstone-enqueue"
OPERATOR_RUNTIME_CONCURRENCY = 1


class InProcessDbosApi:
    """DBOSClient-shaped facade over the launched in-process DBOS.

    ``DBOSClient`` cannot open SQLite system databases (it applies pool
    keyword arguments SQLite rejects), so lifecycle readers and cancellers
    run over the already-launched runtime instead.  ``DbosLifecycleReader``
    needs exactly ``list_workflows`` and ``_sys_db.engine``;
    ``WhetstoneDbosCanceller`` additionally needs ``cancel_workflow``, whose
    in-process signature matches ``DBOSClient.cancel_workflow`` exactly.
    ``_get_dbos_instance`` is a private dbos API pinned by the dbos 2.26
    dependency.
    """

    def __init__(self) -> None:
        self._sys_db = _get_dbos_instance()._sys_db

    def list_workflows(self, **kwargs: Any) -> list[Any]:
        return DBOS.list_workflows(**kwargs)

    def cancel_workflow(
        self, workflow_id: str, *, cancel_children: bool = False
    ) -> None:
        DBOS.cancel_workflow(workflow_id, cancel_children=cancel_children)


@dataclass(frozen=True)
class RegisteredQueueLookup:
    """In-process admission for exactly the queues whetstone registers.

    Kernel admission requires a database-backed queue object exposing
    ``database_backed_queue`` and ``priority_enabled``; a bare name fails
    closed.  ``DBOSClient`` cannot open SQLite system databases, so the
    queue is resolved through the launched in-process runtime instead.
    """

    names: frozenset[str]

    def retrieve_queue(self, name: str) -> object | None:
        if name not in self.names:
            return None
        return DBOS.retrieve_queue(name)


@dataclass(frozen=True)
class EnqueueRuntime:
    queue_lookup: RegisteredQueueLookup
    enqueue_adapter: DbosEnqueueAdapter
    workflow_observer: DbosWorkflowObserver


@contextmanager
def platform_enqueue_runtime(
    *,
    application_database_url: str | None = None,
    system_database_url: str | None = None,
) -> Iterator[EnqueueRuntime]:
    """DBOS runtime that can enqueue but never consumes execution queues.

    ``DbosEnqueueAdapter`` requires a launched in-process DBOS, and DBOS
    2.26 gives a launched process two consumption paths that must both be
    closed explicitly.  A process that never calls ``listen_queues`` polls
    every registered queue, so ``DBOS.listen_queues([])`` pins the listen
    set to nothing before launch.  Launch recovery re-executes PENDING
    workflows owned by this process's executor ID, and the worker runs as
    the default executor ``local``, so a unique per-invocation executor ID
    makes recovery a no-op here.  Worker pickup of the enqueued workflows
    requires an equal computed application version, which holds exactly
    when the worker serves the same workflow sources as this checkout
    (both processes register the identical workflow set via
    ``whetstone.platform.targets``); restart the worker after deploys that
    touch workflow code.
    """
    config = build_whetstone_dbos_config(
        database_url=application_database_url,
        system_database_url=system_database_url,
        generation_concurrency=OPERATOR_RUNTIME_CONCURRENCY,
        scoring_concurrency=OPERATOR_RUNTIME_CONCURRENCY,
    )
    runtime_config = dbos_config(config, app_name=DBOS_APP_NAME)
    runtime_config["executor_id"] = f"{EXECUTOR_ID_PREFIX}-{uuid4().hex}"
    try:
        DBOS(config=runtime_config)
        DBOS.listen_queues([])
        DBOS.launch()
        # never_update: create the queues if absent, but never clobber the
        # worker-owned configuration in the shared system database.
        register_execution_queues(
            worker_concurrency=OPERATOR_RUNTIME_CONCURRENCY,
            on_conflict="never_update",
        )
        yield EnqueueRuntime(
            queue_lookup=RegisteredQueueLookup(
                names=frozenset(
                    {GENERATION_QUEUE_NAME, SCORING_QUEUE_NAME}
                )
            ),
            enqueue_adapter=DbosEnqueueAdapter(),
            workflow_observer=DbosWorkflowObserver(),
        )
    finally:
        shutdown_dbos_runtime()
