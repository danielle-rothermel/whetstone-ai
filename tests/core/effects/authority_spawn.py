from __future__ import annotations

from datetime import timedelta
from functools import partial
from pathlib import Path
from queue import Empty
from typing import Any

from tests.postgres import (
    PostgresOperationGate,
    connect_in_postgres_schema,
)
from whetstone.core.effects.authority import (
    EffectAuthority,
    EffectRequest,
)
from whetstone.core.identity import TypedRef


def spawn_result(queue: Any, *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        return queue.get(timeout=timeout)
    except Empty as exc:
        raise AssertionError(
            "spawned authority worker produced no result"
        ) from exc


class _TransactionSignals:
    def __init__(self, attempted: Any, acquired: Any) -> None:
        self._attempted = attempted
        self._acquired = acquired

    def transaction_attempted(self) -> None:
        self._attempted.set()

    def transaction_acquired(self) -> None:
        self._acquired.set()


def race_acquire(
    database_path: str,
    request_payload: dict[str, Any],
    owner_id: str,
    attempt_id: str,
    lease_seconds: float,
    ready: Any,
    start: Any,
    attempted: Any,
    acquired: Any,
    output: Any,
) -> None:
    try:
        authority = EffectAuthority.sqlite(
            Path(database_path),
            _transaction_observer=_TransactionSignals(attempted, acquired),
        )
        request = EffectRequest.model_validate(request_payload)
        ready.set()
        if not start.wait(timeout=60):
            raise TimeoutError("authority race worker was not released")
        result = authority.acquire(
            request,
            owner_id=owner_id,
            attempt_id=attempt_id,
            lease_duration=timedelta(seconds=lease_seconds),
        )
        output.put(result.model_dump(mode="json"))
    except BaseException as exc:
        output.put({"error": f"{type(exc).__name__}: {exc}"})


def acquire_then_exit(
    database_path: str,
    request_payload: dict[str, Any],
    owner_id: str,
    attempt_id: str,
    lease_seconds: float,
    output: Any,
) -> None:
    authority = EffectAuthority.sqlite(Path(database_path))
    result = authority.acquire(
        EffectRequest.model_validate(request_payload),
        owner_id=owner_id,
        attempt_id=attempt_id,
        lease_duration=timedelta(seconds=lease_seconds),
    )
    output.put(result.model_dump(mode="json"))


def race_postgresql_acquire(
    dsn: str,
    schema: str,
    request_json: str,
    owner_id: str,
    attempt_id: str,
    role: str,
    ready: Any,
    start: Any,
    query_reached: Any,
    release: Any,
    backend_pid: Any,
    output: Any,
) -> None:
    try:
        connect_gate = (
            PostgresOperationGate(
                schema=schema,
                backend_pid=backend_pid,
                after_query="INSERT INTO whetstone_effect_authority (",
                after_query_reached=query_reached,
                release=release,
            )
            if role == "holder"
            else PostgresOperationGate(
                schema=schema,
                backend_pid=backend_pid,
                before_query=(
                    "SELECT request_identity_hash, replay_policy, state"
                ),
                before_query_reached=query_reached,
            )
        )
        authority = EffectAuthority.postgresql(
            dsn,
            _connect=connect_gate,
        )
        request = EffectRequest.model_validate_json(request_json)
        ready.set()
        if not start.wait(timeout=60):
            raise TimeoutError("PostgreSQL authority worker was not released")
        result = authority.acquire(
            request,
            owner_id=owner_id,
            attempt_id=attempt_id,
            lease_duration=timedelta(minutes=5),
        )
        output.put(result.model_dump(mode="json"))
    except BaseException as exc:
        output.put({"error": f"{type(exc).__name__}: {exc}"})


def postgresql_acquire_and_succeed_once(
    dsn: str,
    schema: str,
    request_json: str,
    result_ref_json: str,
    output: Any,
) -> None:
    try:
        authority = EffectAuthority.postgresql(
            dsn,
            _connect=partial(
                connect_in_postgres_schema,
                schema=schema,
            ),
        )
        acquired = authority.acquire(
            EffectRequest.model_validate_json(request_json),
            owner_id="terminal-writer",
            attempt_id="terminal-attempt",
            lease_duration=timedelta(minutes=5),
        )
        if acquired.lease is None:
            raise RuntimeError("terminal writer did not acquire the effect")
        terminal = authority.succeed(
            acquired.lease,
            result_ref=TypedRef.model_validate_json(result_ref_json),
        )
        output.put(terminal.model_dump(mode="json"))
    except BaseException as exc:
        output.put({"error": f"{type(exc).__name__}: {exc}"})


def replay_postgresql_effect_once(
    dsn: str,
    schema: str,
    request_json: str,
    output: Any,
) -> None:
    try:
        authority = EffectAuthority.postgresql(
            dsn,
            _connect=partial(
                connect_in_postgres_schema,
                schema=schema,
            ),
        )
        replay = authority.acquire(
            EffectRequest.model_validate_json(request_json),
            owner_id="terminal-reader",
            attempt_id="replay-attempt",
            lease_duration=timedelta(minutes=5),
        )
        output.put(replay.model_dump(mode="json"))
    except BaseException as exc:
        output.put({"error": f"{type(exc).__name__}: {exc}"})
