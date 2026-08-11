from __future__ import annotations

import multiprocessing
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

import pytest
from dr_serialize import StrictJsonDecodeError

from tests.core.effects.authority_spawn import acquire_then_exit, spawn_result
from tests.core.effects.authority_sqlite_scenarios import (
    run_spawned_authority_contention,
    run_spawned_same_owner_different_attempts_arbitrate_once,
)
from tests.core.effects.authority_support import (
    _acquire,
    _request,
    _result_ref,
)
from tests.optimization.processes import (
    in_process_start_methods,
    join_processes,
    terminate_processes,
)
from tests.optimization.sqlite_time import wait_for_sqlite_authority_after
from whetstone.core.effects import _storage as storage_module
from whetstone.core.effects.authority import (
    AcquireOutcome,
    EffectAuthority,
    EffectAuthorityError,
    EffectLease,
    ReplayPolicy,
    StaleLeaseError,
    TerminalOutcome,
)


@pytest.mark.parametrize(
    "terminal_json",
    ['{"value":1,"value":2}', '{"value":NaN}'],
)
def test_terminal_row_rejects_non_strict_json(terminal_json: str) -> None:
    raw = (
        "a" * 64,
        ReplayPolicy.IDEMPOTENT.value,
        TerminalOutcome.SUCCEEDED.value,
        "owner",
        "attempt",
        1,
        None,
        terminal_json,
    )

    with pytest.raises(StrictJsonDecodeError):
        storage_module._decode_row("semantic-key", raw)


def test_persisted_authority_literals_and_sqlite_schema_are_pinned(
    tmp_path: Path,
) -> None:
    assert (
        ReplayPolicy.IDEMPOTENT.value,
        ReplayPolicy.DURABLE_WORKFLOW.value,
        ReplayPolicy.NO_REDRIVE.value,
    ) == ("idempotent", "durable_workflow", "no_redrive")
    assert (
        TerminalOutcome.SUCCEEDED.value,
        TerminalOutcome.FAILED.value,
        TerminalOutcome.RECOVERY_REQUIRED.value,
    ) == ("succeeded", "failed", "recovery_required")
    assert (
        AcquireOutcome.ACQUIRED.value,
        AcquireOutcome.BUSY.value,
        AcquireOutcome.SUCCEEDED.value,
        AcquireOutcome.FAILED.value,
        AcquireOutcome.REQUEST_CONFLICT.value,
        AcquireOutcome.RECOVERY_REQUIRED.value,
    ) == (
        "acquired",
        "busy",
        "succeeded",
        "failed",
        "request_conflict",
        "recovery_required",
    )

    database = tmp_path / "schema.sqlite"
    EffectAuthority.sqlite(database)
    with sqlite3.connect(database) as connection:
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(whetstone_effect_authority)"
            )
        ]
        metadata_columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(whetstone_effect_authority_metadata)"
            )
        ]
        version = connection.execute(
            """
            SELECT singleton, schema_version
            FROM whetstone_effect_authority_metadata
            """
        ).fetchall()
        table_sql = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'whetstone_effect_authority'
            """
        ).fetchone()
        metadata_sql = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table'
              AND name = 'whetstone_effect_authority_metadata'
            """
        ).fetchone()
    assert columns == [
        "semantic_key",
        "request_hash",
        "replay_policy",
        "state",
        "owner_id",
        "attempt_id",
        "fence",
        "expires_at",
        "terminal_json",
    ]
    assert metadata_columns == ["singleton", "schema_version"]
    assert version == [(1, 2)]
    assert table_sql is not None
    assert metadata_sql is not None
    compact_table_sql = "".join(table_sql[0].split())
    compact_metadata_sql = "".join(metadata_sql[0].split())
    for column in (
        "semantic_key",
        "request_hash",
        "replay_policy",
        "state",
        "owner_id",
        "attempt_id",
        "fence",
        "expires_at",
        "terminal_json",
    ):
        assert f"typeof({column})=" in compact_table_sql
    for column in ("singleton", "schema_version"):
        assert f"typeof({column})=" in compact_metadata_sql


@pytest.mark.parametrize(
    ("field", "poison", "message"),
    [
        ("fence", 1.5, "fence must have integer storage"),
        (
            "owner_id",
            sqlite3.Binary(b"owner"),
            "owner_id must have text storage",
        ),
    ],
)
def test_sqlite_effect_decode_rejects_wrong_storage_classes(
    tmp_path: Path,
    field: str,
    poison: object,
    message: str,
) -> None:
    database = tmp_path / f"poison-{field}.sqlite"
    authority = EffectAuthority.sqlite(database)
    request = _request()
    _acquire(authority, request)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            f"""
            UPDATE whetstone_effect_authority
            SET {field} = ?
            WHERE semantic_key = ?
            """,
            (poison, str(request.semantic_key)),
        )

    with pytest.raises(EffectAuthorityError, match=message):
        _acquire(authority, request)


def test_sqlite_effect_metadata_rejects_real_schema_version(
    tmp_path: Path,
) -> None:
    database = tmp_path / "real-schema-version.sqlite"
    EffectAuthority.sqlite(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE whetstone_effect_authority_metadata
            SET schema_version = 2.5
            """
        )

    with pytest.raises(EffectAuthorityError, match="schema version"):
        EffectAuthority.sqlite(database)


@pytest.mark.sqlite_time_integration
def test_sqlite_heartbeat_keeps_real_time_work_publishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "heartbeat.sqlite"
    authority = EffectAuthority.sqlite(database)
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=180),
    )
    assert acquired.lease is not None
    initial_expiry = acquired.lease.expires_at
    renewal_observed_past_original_expiry = Event()
    real_renew = authority.renew

    def recording_renew(
        lease: EffectLease,
        *,
        lease_duration: timedelta,
    ) -> EffectLease:
        renewed = real_renew(lease, lease_duration=lease_duration)
        renewal_authority_time = renewed.expires_at - lease_duration
        if renewal_authority_time > initial_expiry:
            renewal_observed_past_original_expiry.set()
        return renewed

    monkeypatch.setattr(authority, "renew", recording_renew)
    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(milliseconds=180),
    ) as maintenance:
        assert renewal_observed_past_original_expiry.wait(timeout=10)
        maintenance.check()
        contender = _acquire(
            EffectAuthority.sqlite(database),
            _request(),
            owner="contender",
            attempt="contender-attempt",
            duration=timedelta(milliseconds=180),
        )
        assert contender.outcome is AcquireOutcome.BUSY
        terminal = maintenance.succeed(
            result_ref=_result_ref("sqlite-heartbeat")
        )
    assert terminal.outcome is TerminalOutcome.SUCCEEDED


def test_sqlite_rejects_submillisecond_lease_durations(
    tmp_path: Path,
) -> None:
    authority = EffectAuthority.sqlite(tmp_path / "duration.sqlite")
    with pytest.raises(
        ValueError,
        match="at least 1 millisecond for SQLite authority clock precision",
    ):
        _acquire(
            authority,
            _request(),
            duration=timedelta(microseconds=999),
        )

    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=10),
    )
    assert acquired.lease is not None
    with pytest.raises(
        ValueError,
        match="at least 1 millisecond for SQLite authority clock precision",
    ):
        authority.maintain(
            acquired.lease,
            lease_duration=timedelta(microseconds=999),
        )


@pytest.mark.sqlite_time_integration
def test_sqlite_maintenance_terminalization_surfaces_renewal_loss(
    tmp_path: Path,
) -> None:
    database = tmp_path / "renewal-loss.sqlite"
    authority = EffectAuthority.sqlite(database)
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=1),
    )
    assert acquired.lease is not None
    wait_for_sqlite_authority_after(database, acquired.lease.expires_at)

    with pytest.raises(StaleLeaseError, match="effect lease is stale"):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(milliseconds=1),
        ) as maintenance:
            assert maintenance._stop.wait(timeout=1)
            maintenance.succeed(result_ref=_result_ref("too-late"))


def test_sqlite_rejects_malformed_preexisting_schema_and_version(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.sqlite"
    with sqlite3.connect(malformed) as connection:
        connection.execute(
            "CREATE TABLE whetstone_effect_authority (semantic_key TEXT)"
        )
        connection.execute(
            """
            CREATE TABLE whetstone_effect_authority_metadata (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO whetstone_effect_authority_metadata
            VALUES (1, 1)
            """
        )
    with pytest.raises(EffectAuthorityError, match="incompatible SQLite"):
        EffectAuthority.sqlite(malformed)

    wrong_version = tmp_path / "wrong-version.sqlite"
    EffectAuthority.sqlite(wrong_version)
    with sqlite3.connect(wrong_version) as connection:
        connection.execute(
            """
            UPDATE whetstone_effect_authority_metadata
            SET schema_version = 1
            """
        )
    with pytest.raises(EffectAuthorityError, match="schema version"):
        EffectAuthority.sqlite(wrong_version)


def test_sqlite_initialization_is_idempotent_and_terminal_write_atomic(
    tmp_path: Path,
) -> None:
    database = tmp_path / "atomic.sqlite"
    authority = EffectAuthority.sqlite(database)
    EffectAuthority.sqlite(database)
    acquired = _acquire(authority, _request())
    assert acquired.lease is not None
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_success
            BEFORE UPDATE ON whetstone_effect_authority
            WHEN NEW.state = 'succeeded'
            BEGIN
                SELECT RAISE(ABORT, 'injected terminal failure');
            END
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        authority.succeed(acquired.lease, result_ref=_result_ref())
    replay = _acquire(
        authority,
        _request(),
        owner="worker-2",
        attempt="attempt-2",
    )
    assert replay.outcome is AcquireOutcome.BUSY


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
@pytest.mark.process_integration
def test_spawned_same_owner_different_attempts_arbitrate_once(
    tmp_path: Path,
    start_method: str,
) -> None:
    run_spawned_same_owner_different_attempts_arbitrate_once(
        tmp_path,
        start_method,
    )


@pytest.mark.sqlite_time_integration
@pytest.mark.sqlite_contention
@pytest.mark.process_integration
def test_spawned_sqlite_owner_exit_allows_authority_timed_takeover(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    database = tmp_path / "takeover.sqlite"
    request = _request()
    output = context.Queue()
    process = context.Process(
        target=acquire_then_exit,
        args=(
            str(database),
            request.model_dump(),
            "crashed-worker",
            "crashed-attempt",
            1.2,
            output,
        ),
    )
    process_started = False
    try:
        process.start()
        process_started = True
        first = spawn_result(output)
        join_processes((process,), timeout=10)
    finally:
        if process_started:
            terminate_processes((process,), timeout=10)
    assert first["lease"]["fence"] == 1
    first_expiry = datetime.fromisoformat(first["lease"]["expires_at"])
    wait_for_sqlite_authority_after(database, first_expiry)

    takeovers = run_spawned_authority_contention(
        database,
        start_method="spawn",
    )
    assert sorted(result["outcome"] for result in takeovers) == [
        AcquireOutcome.ACQUIRED.value,
        AcquireOutcome.BUSY.value,
    ]
    acquired = next(
        result
        for result in takeovers
        if result["outcome"] == AcquireOutcome.ACQUIRED
    )
    busy = next(
        result
        for result in takeovers
        if result["outcome"] == AcquireOutcome.BUSY
    )
    assert acquired["lease"]["fence"] == 2
    assert busy["busy_expires_at"] == acquired["lease"]["expires_at"]
