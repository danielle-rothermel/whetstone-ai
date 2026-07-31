"""Spawn-process helpers for SQLite Tool admission tests."""

from __future__ import annotations

from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from dr_store import ObjectStore, SqliteBackend

from whetstone.optimization.effect_authority import EffectAuthority
from whetstone.optimization.identity import typed_ref_for_record
from whetstone.optimization.tool_store import (
    ToolAdmissionAuthority,
    ToolCallStore,
)
from whetstone.optimization.tools import (
    RUN_CAPACITY_SUBJECT_SCHEMA,
    ToolCall,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    tool_config_reference,
)


class _TransactionSignals:
    def __init__(self, attempted: Any, acquired: Any) -> None:
        self._attempted = attempted
        self._acquired = acquired

    def transaction_attempted(self) -> None:
        self._attempted.set()

    def transaction_acquired(self) -> None:
        self._acquired.set()


def admit_once(
    object_database: str,
    admission_database: str,
    config_payload: dict[str, Any],
    call_id: str,
    template: str,
    ready: Any,
    start: Any,
    attempted: Any,
    acquired: Any,
    queue: Queue,
) -> None:
    """Open independent stores, admit one call, and report stable evidence."""
    try:
        config = ToolConfig.model_validate(config_payload)
        store = ToolCallStore(
            ObjectStore(SqliteBackend(Path(object_database))),
            ToolAdmissionAuthority.sqlite(
                admission_database,
                _transaction_observer=_TransactionSignals(
                    attempted,
                    acquired,
                ),
            ),
            EffectAuthority.memory(),
        )
        call = ToolCall(
            call_id=call_id,
            tool_config=tool_config_reference(config),
            capacity_binding=ToolCapacityBinding(
                scope=config.capacity.scope,
                subject_ref=(
                    None
                    if config.capacity.scope is ToolCapacityScope.GLOBAL
                    else typed_ref_for_record(
                        RUN_CAPACITY_SUBJECT_SCHEMA,
                        {"subject": "run-1"},
                    )
                ),
            ),
            args={"model_route": "route", "template": template},
        )
        ready.set()
        if not start.wait(timeout=60):
            raise TimeoutError("admission worker was not released")
        entry = store.admit(call, config)
        queue.put(
            {
                "state": entry.state.value,
                "ordinal": entry.capacity_debit_ordinal,
            }
        )
    except BaseException as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def load_terminal_result_once(
    database: str,
    call_payload: dict[str, Any],
    queue: Queue,
) -> None:
    """Load one completion through independently reopened durable stores."""
    try:
        call = ToolCall.model_validate(call_payload)
        store = ToolCallStore(
            ObjectStore(SqliteBackend(Path(database))),
            ToolAdmissionAuthority.sqlite(database),
            EffectAuthority.sqlite(database),
        )
        entry = store.get(call)
        if entry is None:
            raise RuntimeError("completed Tool Call disappeared")
        result = store.load_terminal_result(entry)
        queue.put(
            {
                "entry": entry.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
            }
        )
    except BaseException as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})
