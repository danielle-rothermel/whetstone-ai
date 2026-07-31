from __future__ import annotations

import inspect
import multiprocessing
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty
from threading import Event, Lock, Thread
from typing import Any, LiteralString, cast
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tests.optimization.effect_authority_spawn import (
    acquire_then_exit,
    race_acquire,
)
from whetstone.optimization import (
    effect_authority as effect_authority_module,
)
from whetstone.optimization.effect_authority import (
    AcquireOutcome,
    EffectAuthority,
    EffectAuthorityError,
    EffectAuthoritySchemaMismatchError,
    EffectLease,
    EffectRequest,
    EffectTerminal,
    ReplayPolicy,
    StaleLeaseError,
    TerminalConflictError,
    TerminalFailure,
    TerminalOutcome,
)
from whetstone.optimization.identity import (
    IdentityHash,
    NonEmptyId,
    OpaqueKey,
    TypedRef,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_NOW = datetime(2026, 7, 31, 12, tzinfo=UTC)
_LEASE_DURATION = timedelta(milliseconds=100)


class _FakeClock:
    def __init__(self, now: datetime = _NOW) -> None:
        self._now = now
        self._lock = Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, duration: timedelta) -> None:
        with self._lock:
            self._now += duration


@dataclass(frozen=True, slots=True)
class _Backend:
    authority: EffectAuthority
    clock: _FakeClock | None

    def advance_past(self, duration: timedelta) -> None:
        if self.clock is not None:
            self.clock.advance(duration + timedelta(microseconds=1))
        else:
            time.sleep(duration.total_seconds() + 0.04)


class _CoordinatedAuthority(EffectAuthority):
    def __init__(self, authority: EffectAuthority) -> None:
        self._authority = authority
        self.release_renewal = Event()
        self.renewal_entered = Event()
        self.terminal_entered = Event()
        self.renew_calls = 0
        self.terminal_lease: EffectLease | None = None

    def renew(
        self,
        lease: EffectLease,
        *,
        lease_duration: timedelta,
    ) -> EffectLease:
        self.renew_calls += 1
        self.renewal_entered.set()
        if not self.release_renewal.wait(timeout=2):
            raise TimeoutError("test did not release coordinated renewal")
        return self._authority.renew(
            lease,
            lease_duration=lease_duration,
        )

    def _validate_lease_duration(self, value: timedelta) -> timedelta:
        return self._authority._validate_lease_duration(value)

    def succeed(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
    ) -> EffectTerminal:
        self.terminal_lease = lease
        self.terminal_entered.set()
        return self._authority.succeed(lease, result_ref=result_ref)

    def fail(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
        failure: TerminalFailure,
    ) -> EffectTerminal:
        self.terminal_lease = lease
        self.terminal_entered.set()
        return self._authority.fail(
            lease,
            result_ref=result_ref,
            failure=failure,
        )


@pytest.fixture(params=("memory", "sqlite"))
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> _Backend:
    if request.param == "memory":
        clock = _FakeClock()
        return _Backend(EffectAuthority.memory(clock=clock), clock)
    return _Backend(
        EffectAuthority.sqlite(tmp_path / "authority.sqlite"),
        None,
    )


def _result_ref(label: str = "result") -> TypedRef:
    return TypedRef(
        schema_name=f"whetstone.test.{label}",
        content_hash="c" * 64,
    )


def _request(
    *,
    key: str = "evaluation:run-1:intent-1",
    identity_hash: str = _HASH_A,
    policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
) -> EffectRequest:
    return EffectRequest(
        semantic_key=OpaqueKey(key),
        request_identity_hash=IdentityHash(identity_hash),
        replay_policy=policy,
    )


def _acquire(
    authority: EffectAuthority,
    request: EffectRequest,
    *,
    owner: str = "worker-1",
    attempt: str = "attempt-1",
    duration: timedelta = _LEASE_DURATION,
):
    return authority.acquire(
        request,
        owner_id=owner,
        attempt_id=attempt,
        lease_duration=duration,
    )


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
        "request_identity_hash",
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
        "request_identity_hash",
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


def test_public_transitions_expose_duration_but_no_process_time() -> None:
    assert list(inspect.signature(EffectAuthority.acquire).parameters) == [
        "self",
        "request",
        "owner_id",
        "attempt_id",
        "lease_duration",
    ]
    assert list(inspect.signature(EffectAuthority.renew).parameters) == [
        "self",
        "lease",
        "lease_duration",
    ]
    assert list(inspect.signature(EffectAuthority.succeed).parameters) == [
        "self",
        "lease",
        "result_ref",
    ]
    assert list(inspect.signature(EffectAuthority.fail).parameters) == [
        "self",
        "lease",
        "result_ref",
        "failure",
    ]
    authority = EffectAuthority.memory(clock=_FakeClock())
    with pytest.raises(TypeError, match="unexpected keyword argument 'now'"):
        authority.acquire(
            _request(),
            owner_id="worker",
            attempt_id="attempt",
            lease_duration=_LEASE_DURATION,
            now=datetime.max.replace(  # ty: ignore[unknown-argument]
                tzinfo=UTC
            ),
        )


@pytest.mark.parametrize(
    "duration",
    (timedelta(0), timedelta(microseconds=-1)),
)
def test_lease_duration_must_be_positive(duration: timedelta) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _acquire(
            EffectAuthority.memory(clock=_FakeClock()),
            _request(),
            duration=duration,
        )


def test_same_attempt_replays_but_same_owner_new_attempt_is_busy(
    backend: _Backend,
) -> None:
    request = _request()
    first = _acquire(backend.authority, request)
    replay = _acquire(backend.authority, request)
    competing = _acquire(
        backend.authority,
        request,
        owner="worker-1",
        attempt="attempt-2",
    )

    assert first.outcome is AcquireOutcome.ACQUIRED
    assert replay == first
    assert first.lease is not None
    assert first.lease.attempt_id == "attempt-1"
    assert competing.outcome is AcquireOutcome.BUSY
    assert competing.busy_expires_at == first.lease.expires_at


def test_renew_and_takeover_share_backend_authority_time(
    backend: _Backend,
) -> None:
    request = _request()
    first = _acquire(backend.authority, request)
    assert first.lease is not None

    if backend.clock is not None:
        backend.clock.advance(timedelta(milliseconds=30))
    else:
        time.sleep(0.03)
    renewed = backend.authority.renew(
        first.lease,
        lease_duration=_LEASE_DURATION,
    )
    assert renewed.fence == first.lease.fence
    assert renewed.expires_at > first.lease.expires_at
    with pytest.raises(StaleLeaseError):
        backend.authority.renew(
            first.lease,
            lease_duration=_LEASE_DURATION,
        )

    backend.advance_past(_LEASE_DURATION)
    takeover = _acquire(
        backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert takeover.outcome is AcquireOutcome.ACQUIRED
    assert takeover.lease is not None
    assert takeover.lease.owner_id == "worker-2"
    assert takeover.lease.attempt_id == "attempt-2"
    assert takeover.lease.fence == 2


@pytest.mark.parametrize(
    "policy",
    (ReplayPolicy.IDEMPOTENT, ReplayPolicy.DURABLE_WORKFLOW),
)
def test_stale_attempt_cannot_terminalize_after_takeover(
    backend: _Backend,
    policy: ReplayPolicy,
) -> None:
    request = _request(policy=policy)
    first = _acquire(backend.authority, request)
    assert first.lease is not None
    backend.advance_past(_LEASE_DURATION)
    takeover = _acquire(
        backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert takeover.lease is not None

    with pytest.raises(StaleLeaseError):
        backend.authority.succeed(
            first.lease,
            result_ref=_result_ref("first"),
        )
    terminal = backend.authority.succeed(
        takeover.lease,
        result_ref=_result_ref("second"),
    )
    assert terminal.attempt_id == "attempt-2"
    assert terminal.result_ref == _result_ref("second")


def test_no_redrive_expiry_becomes_immutable_recovery_required(
    backend: _Backend,
) -> None:
    request = _request(policy=ReplayPolicy.NO_REDRIVE)
    first = _acquire(backend.authority, request)
    assert first.lease is not None
    backend.advance_past(_LEASE_DURATION)

    recovered = _acquire(
        backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert recovered.outcome is AcquireOutcome.RECOVERY_REQUIRED
    assert recovered.terminal is not None
    assert recovered.terminal.failure is not None
    assert recovered.terminal.failure.code.startswith("effect-recovery:")
    assert recovered.terminal.failure.details["owner_id"] == "worker-1"
    assert recovered.terminal.failure.details["attempt_id"] == "attempt-1"

    replay = _acquire(
        backend.authority,
        request,
        owner="worker-3",
        attempt="attempt-3",
    )
    assert replay == recovered
    with pytest.raises(TerminalConflictError):
        backend.authority.succeed(
            first.lease,
            result_ref=_result_ref("late"),
        )


def test_request_identity_and_policy_are_immutable(
    backend: _Backend,
) -> None:
    original = _request()
    _acquire(backend.authority, original)

    divergent_identity = _acquire(
        backend.authority,
        _request(identity_hash=_HASH_B),
        owner="worker-2",
        attempt="attempt-2",
    )
    assert divergent_identity.outcome is AcquireOutcome.REQUEST_CONFLICT
    assert divergent_identity.existing_request_identity_hash == _HASH_A
    assert divergent_identity.existing_replay_policy is ReplayPolicy.IDEMPOTENT

    divergent_policy = _acquire(
        backend.authority,
        _request(policy=ReplayPolicy.DURABLE_WORKFLOW),
        owner="worker-2",
        attempt="attempt-2",
    )
    assert divergent_policy.outcome is AcquireOutcome.REQUEST_CONFLICT


def test_success_and_failure_are_exact_immutable_and_replayed(
    backend: _Backend,
) -> None:
    success_request = _request(key="effect:success")
    success = _acquire(backend.authority, success_request)
    assert success.lease is not None
    terminal = backend.authority.succeed(
        success.lease,
        result_ref=_result_ref("result-a"),
    )
    assert (
        backend.authority.succeed(
            success.lease,
            result_ref=_result_ref("result-a"),
        )
        == terminal
    )
    with pytest.raises(TerminalConflictError):
        backend.authority.succeed(
            success.lease,
            result_ref=_result_ref("result-b"),
        )
    replay = _acquire(
        backend.authority,
        success_request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert replay.outcome is AcquireOutcome.SUCCEEDED
    assert replay.terminal == terminal

    failure_request = _request(key="effect:failure")
    failed = _acquire(backend.authority, failure_request)
    assert failed.lease is not None
    failure = TerminalFailure(
        code="provider_timeout",
        message="provider timed out",
        details={"provider": "test"},
    )
    failed_terminal = backend.authority.fail(
        failed.lease,
        result_ref=_result_ref("failed-result"),
        failure=failure,
    )
    failed_replay = _acquire(
        backend.authority,
        failure_request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert failed_replay.outcome is AcquireOutcome.FAILED
    assert failed_replay.terminal == failed_terminal
    with pytest.raises(TerminalConflictError):
        backend.authority.fail(
            failed.lease,
            result_ref=_result_ref("different"),
            failure=failure,
        )


def test_terminal_outcome_shapes_and_serialized_lease_are_exact() -> None:
    request = _request()
    failure = TerminalFailure(code="failed", message="failed")
    common = {
        "request": request,
        "owner_id": NonEmptyId("worker"),
        "attempt_id": NonEmptyId("attempt"),
        "fence": 1,
    }
    with pytest.raises(ValidationError, match="succeeded terminal"):
        EffectTerminal.model_validate(
            {
                **common,
                "outcome": TerminalOutcome.SUCCEEDED,
                "result_ref": _result_ref(),
                "failure": failure,
            }
        )
    with pytest.raises(ValidationError, match="failed terminal"):
        EffectTerminal.model_validate(
            {
                **common,
                "outcome": TerminalOutcome.FAILED,
                "failure": failure,
            }
        )
    lease = EffectLease(
        request=request,
        owner_id=NonEmptyId("worker"),
        attempt_id=NonEmptyId("attempt"),
        fence=1,
        expires_at=_NOW + _LEASE_DURATION,
    )
    assert EffectLease.model_validate_json(lease.model_dump_json()) == lease
    with pytest.raises(ValidationError):
        EffectLease(
            request=request,
            owner_id=NonEmptyId("worker"),
            attempt_id=NonEmptyId("attempt"),
            fence=True,
            expires_at=lease.expires_at,
        )
    with pytest.raises(ValidationError):
        lease.attempt_id = NonEmptyId(  # ty: ignore[invalid-assignment]
            "changed"
        )


def test_heartbeat_renews_across_multiple_durations() -> None:
    clock = _FakeClock()
    authority = EffectAuthority.memory(clock=clock)
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=90),
    )
    assert acquired.lease is not None
    initial_expiry = acquired.lease.expires_at

    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(milliseconds=90),
    ) as maintenance:
        for _ in range(4):
            clock.advance(timedelta(milliseconds=30))
            time.sleep(0.045)
            maintenance.check()
        maintained_lease = maintenance.lease
        assert maintained_lease.expires_at >= (
            initial_expiry + timedelta(milliseconds=90)
        )
        terminal = maintenance.succeed(result_ref=_result_ref("heartbeat"))
    assert terminal.outcome is TerminalOutcome.SUCCEEDED


def test_heartbeat_reports_lease_loss_on_clean_exit() -> None:
    clock = _FakeClock()
    authority = EffectAuthority.memory(clock=clock)
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=60),
    )
    assert acquired.lease is not None

    with pytest.raises(StaleLeaseError):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(milliseconds=60),
        ):
            clock.advance(timedelta(milliseconds=61))
            time.sleep(0.04)


def test_maintained_success_waits_for_renewal_and_uses_latest_lease() -> None:
    clock = _FakeClock()
    base_authority = EffectAuthority.memory(clock=clock)
    authority = _CoordinatedAuthority(base_authority)
    acquired = _acquire(
        base_authority,
        _request(),
        duration=timedelta(milliseconds=90),
    )
    assert acquired.lease is not None
    initial_lease = acquired.lease
    terminals: list[EffectTerminal] = []
    errors: list[BaseException] = []

    with authority.maintain(
        initial_lease,
        lease_duration=timedelta(milliseconds=90),
    ) as maintenance:
        maintenance_thread = maintenance._thread
        assert authority.renewal_entered.wait(timeout=1)
        clock.advance(timedelta(milliseconds=30))

        def publish() -> None:
            try:
                terminals.append(
                    maintenance.succeed(result_ref=_result_ref("renewal-won"))
                )
            except BaseException as exc:
                errors.append(exc)

        publisher = Thread(target=publish, name="test-terminal-publisher")
        publisher.start()
        assert maintenance._stop.wait(timeout=1)
        assert not authority.terminal_entered.is_set()
        authority.release_renewal.set()
        publisher.join(timeout=2)
        assert not publisher.is_alive()
        assert not errors

    assert not maintenance_thread.is_alive()
    assert len(terminals) == 1
    terminal = terminals[0]
    assert authority.terminal_lease is not None
    assert authority.terminal_lease.expires_at > initial_lease.expires_at
    assert terminal == EffectTerminal(
        request=initial_lease.request,
        outcome=TerminalOutcome.SUCCEEDED,
        owner_id=initial_lease.owner_id,
        attempt_id=initial_lease.attempt_id,
        fence=initial_lease.fence,
        result_ref=_result_ref("renewal-won"),
    )


def test_maintained_success_stops_renewal_before_publish() -> None:
    base_authority = EffectAuthority.memory(clock=_FakeClock())
    authority = _CoordinatedAuthority(base_authority)
    acquired = _acquire(
        base_authority,
        _request(),
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None

    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(seconds=30),
    ) as maintenance:
        maintenance_thread = maintenance._thread
        terminal = maintenance.succeed(result_ref=_result_ref("terminal-won"))

    assert terminal.outcome is TerminalOutcome.SUCCEEDED
    assert authority.renew_calls == 0
    assert not maintenance_thread.is_alive()


def test_maintained_failure_is_exact_and_rejects_double_publication() -> None:
    authority = EffectAuthority.memory(clock=_FakeClock())
    request = _request()
    acquired = _acquire(
        authority,
        request,
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None
    failure = TerminalFailure(
        code="provider_rejected",
        message="provider rejected the request",
        details={"provider": "test"},
    )

    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(seconds=30),
    ) as maintenance:
        terminal = maintenance.fail(
            result_ref=_result_ref("failed-maintained-result"),
            failure=failure,
        )
        with pytest.raises(RuntimeError, match="already terminalized"):
            maintenance.succeed(result_ref=_result_ref("second-terminal"))

    assert terminal == EffectTerminal(
        request=request,
        outcome=TerminalOutcome.FAILED,
        owner_id=acquired.lease.owner_id,
        attempt_id=acquired.lease.attempt_id,
        fence=acquired.lease.fence,
        result_ref=_result_ref("failed-maintained-result"),
        failure=failure,
    )


def test_terminal_publication_requires_an_entered_live_context() -> None:
    authority = EffectAuthority.memory(clock=_FakeClock())
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None
    maintenance = authority.maintain(
        acquired.lease,
        lease_duration=timedelta(seconds=30),
    )

    with pytest.raises(RuntimeError, match="active maintenance context"):
        maintenance.succeed(result_ref=_result_ref("before-enter"))

    with pytest.raises(LookupError, match="body failed"):
        with maintenance:
            raise LookupError("body failed")

    with pytest.raises(RuntimeError, match="active maintenance context"):
        maintenance.fail(
            result_ref=_result_ref("after-exit"),
            failure=TerminalFailure(code="failed", message="failed"),
        )


def test_clean_maintenance_exit_requires_terminal_publication() -> None:
    authority = EffectAuthority.memory(clock=_FakeClock())
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None

    with pytest.raises(
        RuntimeError,
        match="clean lease maintenance exit requires terminal publication",
    ):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(seconds=30),
        ):
            pass


def test_exception_exit_stops_renewer_and_preserves_observed_loss() -> None:
    clock = _FakeClock()
    base_authority = EffectAuthority.memory(clock=clock)
    authority = _CoordinatedAuthority(base_authority)
    acquired = _acquire(
        base_authority,
        _request(),
        duration=timedelta(milliseconds=90),
    )
    assert acquired.lease is not None

    with pytest.raises(LookupError, match="body failed"):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(milliseconds=90),
        ) as maintenance:
            maintenance_thread = maintenance._thread
            assert authority.renewal_entered.wait(timeout=1)
            clock.advance(timedelta(milliseconds=91))
            authority.release_renewal.set()
            assert maintenance._stop.wait(timeout=1)
            raise LookupError("body failed")

    assert not maintenance_thread.is_alive()
    with pytest.raises(StaleLeaseError):
        maintenance.check()


def test_sqlite_heartbeat_keeps_real_time_work_publishable(
    tmp_path: Path,
) -> None:
    authority = EffectAuthority.sqlite(tmp_path / "heartbeat.sqlite")
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=180),
    )
    assert acquired.lease is not None
    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(milliseconds=180),
    ) as maintenance:
        time.sleep(0.48)
        maintenance.check()
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


def test_sqlite_maintenance_terminalization_surfaces_renewal_loss(
    tmp_path: Path,
) -> None:
    authority = EffectAuthority.sqlite(tmp_path / "renewal-loss.sqlite")
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=1),
    )
    assert acquired.lease is not None
    time.sleep(0.01)

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


def _spawn_result(queue: Any, *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        return queue.get(timeout=timeout)
    except Empty as exc:
        raise AssertionError(
            "spawned authority worker produced no result"
        ) from exc


def test_spawned_same_owner_different_attempts_arbitrate_once(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    database = tmp_path / "race.sqlite"
    payload = _request().model_dump()
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=race_acquire,
            args=(
                str(database),
                payload,
                "shared-worker",
                attempt,
                0.25,
                start,
                output,
            ),
        )
        for attempt in ("attempt-1", "attempt-2")
    ]
    for process in processes:
        process.start()
    start.set()
    results = [_spawn_result(output) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted(result["outcome"] for result in results) == [
        AcquireOutcome.ACQUIRED.value,
        AcquireOutcome.BUSY.value,
    ]


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
            0.12,
            output,
        ),
    )
    process.start()
    first = _spawn_result(output)
    process.join(timeout=10)
    assert process.exitcode == 0
    assert first["lease"]["fence"] == 1
    time.sleep(0.16)

    start = context.Event()
    takeover_output = context.Queue()
    replacements = [
        context.Process(
            target=race_acquire,
            args=(
                str(database),
                request.model_dump(),
                owner,
                attempt,
                0.25,
                start,
                takeover_output,
            ),
        )
        for owner, attempt in (
            ("replacement-1", "attempt-1"),
            ("replacement-2", "attempt-2"),
        )
    ]
    for replacement in replacements:
        replacement.start()
    start.set()
    takeovers = [_spawn_result(takeover_output) for _ in replacements]
    for replacement in replacements:
        replacement.join(timeout=10)
        assert replacement.exitcode == 0
    assert sorted(result["outcome"] for result in takeovers) == [
        AcquireOutcome.ACQUIRED.value,
        AcquireOutcome.BUSY.value,
    ]
    acquired = next(
        result
        for result in takeovers
        if result["outcome"] == AcquireOutcome.ACQUIRED
    )
    assert acquired["lease"]["fence"] == 2


class _RecordingCursor:
    def __init__(self, recorder: _PostgresRecorder) -> None:
        self._recorder = recorder
        self._one: tuple[Any, ...] | None = None
        self._many: list[tuple[Any, ...]] = []
        self.rowcount = -1

    def execute(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> None:
        self._recorder.queries.append((query, params))
        normalized = " ".join(query.split())
        self._one = None
        self._many = []
        self.rowcount = -1
        if normalized == "SHOW server_encoding":
            self._one = (self._recorder.server_encoding,)
        elif "FROM information_schema.tables" in query:
            self._many = [(name,) for name in sorted(self._recorder.tables)]
        elif normalized.startswith("CREATE TABLE"):
            self._recorder.tables.add(normalized.split()[2])
        elif "FROM information_schema.columns" in query:
            self._many = [
                (table, *column)
                for table in sorted(self._recorder.tables)
                for column in self._recorder.columns[table]
            ]
        elif "FROM pg_catalog.pg_constraint" in query:
            self._many = [
                row
                for row in self._recorder.constraints
                if row[0] in self._recorder.tables
            ]
        elif (
            "SELECT singleton, schema_version" in query and "metadata" in query
        ):
            self._many = [(1, self._recorder.schema_version)]
        elif (
            normalized.startswith(
                "INSERT INTO whetstone_effect_authority_metadata"
            )
            and params is not None
        ):
            self._recorder.schema_version = int(params[1])
        elif "SELECT request_identity_hash" in query:
            self._one = self._recorder.row
        elif "clock_timestamp()" in query:
            self._one = (
                self._recorder.now.isoformat(timespec="microseconds"),
            )
        elif "RETURNING 1" in query:
            assert params is not None
            if self._recorder.row is None:
                self._recorder.row = tuple(params[1:])
                self._one = (1,)
        elif normalized.startswith("UPDATE"):
            assert params is not None
            if self._recorder.row == tuple(params[9:]):
                self._recorder.row = tuple(params[:8])
                self.rowcount = 1
            else:
                self.rowcount = 0

    def fetchone(self) -> tuple[Any, ...] | None:
        value = self._one
        self._one = None
        return value

    def fetchall(self) -> list[tuple[Any, ...]]:
        values = self._many
        self._many = []
        return values

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _RecordingConnection:
    def __init__(self, recorder: _PostgresRecorder) -> None:
        self._recorder = recorder

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self._recorder)

    def __enter__(self) -> _RecordingConnection:
        self._recorder.entered += 1
        return self

    def __exit__(self, *args: object) -> None:
        self._recorder.exited += 1


class _PostgresRecorder:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[Any, ...] | None]] = []
        self.tables: set[str] = set()
        self.schema_version = 2
        self.server_encoding = "UTF8"
        text_collation = ("pg_catalog", "C", "c", True, -1)
        no_collation = (None, None, None, None, None)
        self.columns: dict[str, tuple[tuple[Any, ...], ...]] = {
            "whetstone_effect_authority": (
                ("semantic_key", "text", "NO", 1, *text_collation),
                (
                    "request_identity_hash",
                    "text",
                    "NO",
                    2,
                    *text_collation,
                ),
                ("replay_policy", "text", "NO", 3, *text_collation),
                ("state", "text", "NO", 4, *text_collation),
                ("owner_id", "text", "NO", 5, *text_collation),
                ("attempt_id", "text", "NO", 6, *text_collation),
                ("fence", "bigint", "NO", 7, *no_collation),
                ("expires_at", "text", "YES", 8, *text_collation),
                ("terminal_json", "text", "YES", 9, *text_collation),
            ),
            "whetstone_effect_authority_metadata": (
                ("singleton", "integer", "NO", 1, *no_collation),
                ("schema_version", "integer", "NO", 2, *no_collation),
            ),
        }
        self.row: tuple[Any, ...] | None = None
        self.constraints: list[tuple[Any, ...]] = [
            (
                "whetstone_effect_authority",
                "c",
                ("state", "expires_at", "terminal_json"),
                "state = 'leased'::text AND expires_at IS NOT NULL AND "
                "terminal_json IS NULL OR state <> 'leased'::text AND "
                "expires_at IS NULL AND terminal_json IS NOT NULL",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "c",
                ("fence",),
                "fence > 0",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "c",
                ("replay_policy",),
                "replay_policy = ANY (ARRAY['idempotent'::text, "
                "'durable_workflow'::text, 'no_redrive'::text])",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "c",
                ("state",),
                "state = ANY (ARRAY['leased'::text, 'succeeded'::text, "
                "'failed'::text, 'recovery_required'::text])",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "p",
                ("semantic_key",),
                None,
                False,
                False,
                True,
                True,
            ),
            (
                "whetstone_effect_authority_metadata",
                "c",
                ("singleton",),
                "singleton = 1",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority_metadata",
                "p",
                ("singleton",),
                None,
                False,
                False,
                True,
                True,
            ),
        ]
        self.now = _NOW
        self.entered = 0
        self.exited = 0

    def connect(
        self, dsn: str
    ) -> AbstractContextManager[_RecordingConnection]:
        assert dsn == "postgresql://authority-test"
        return _RecordingConnection(self)

    def advance(self, duration: timedelta) -> None:
        self.now += duration


def test_postgresql_adapter_verifies_schema_and_uses_database_time() -> None:
    recorder = _PostgresRecorder()
    authority = EffectAuthority.postgresql(
        "postgresql://authority-test",
        _connect=recorder.connect,
    )
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(minutes=5),
    )
    assert acquired.lease is not None
    recorder.advance(timedelta(minutes=1))
    renewed = authority.renew(
        acquired.lease,
        lease_duration=timedelta(minutes=5),
    )
    terminal = authority.succeed(
        renewed,
        result_ref=_result_ref("postgresql-result"),
    )
    replay = _acquire(
        authority,
        _request(),
        owner="worker-2",
        attempt="attempt-2",
        duration=timedelta(minutes=5),
    )
    assert replay.terminal == terminal
    assert recorder.entered == recorder.exited == 5
    statements = "\n".join(query for query, _ in recorder.queries)
    assert "CREATE TABLE whetstone_effect_authority" in statements
    assert "CREATE TABLE IF NOT EXISTS" not in statements
    assert "whetstone_effect_authority_metadata" in statements
    assert "information_schema.columns" in statements
    assert "pg_advisory_xact_lock(1465141076, 1)" in statements
    assert "clock_timestamp()" in statements
    assert "FOR UPDATE" in statements
    assert "ON CONFLICT (semantic_key) DO NOTHING" in statements
    assert "IS NOT DISTINCT FROM" in statements


def test_postgresql_adapter_requires_utf8_server_encoding() -> None:
    recorder = _PostgresRecorder()
    recorder.server_encoding = "SQL_ASCII"

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"requires exact server_encoding 'UTF8'.*SQL_ASCII",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )

    assert [" ".join(query.split()) for query, _ in recorder.queries] == [
        "SHOW server_encoding"
    ]


def test_postgresql_adapter_rejects_non_c_text_collation() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    columns = list(recorder.columns["whetstone_effect_authority"])
    columns[0] = (
        "semantic_key",
        "text",
        "NO",
        1,
        "public",
        "case_insensitive",
        "i",
        False,
        -1,
    )
    recorder.columns["whetstone_effect_authority"] = tuple(columns)

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"incompatible PostgreSQL effect-authority columns.*"
        r"case_insensitive",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_rejects_wrong_schema_version() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    recorder.schema_version = 3
    with pytest.raises(EffectAuthorityError, match="schema version"):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_rejects_missing_primary_key() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    recorder.constraints = [
        row
        for row in recorder.constraints
        if not (row[0] == "whetstone_effect_authority" and row[1] == "p")
    ]

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"missing .*PRIMARY KEY \(semantic_key\)",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_rejects_wrong_check_constraint() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    recorder.constraints = [
        (
            *row[:3],
            "replay_policy <> ''::text",
            *row[4:],
        )
        if row[0] == "whetstone_effect_authority"
        and row[2] == ("replay_policy",)
        else row
        for row in recorder.constraints
    ]

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"missing .*CHECK .*replay_policy.*unexpected .*<>",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason=(
        "WHETSTONE_TEST_POSTGRES_DSN is not configured; adapter SQL is "
        "covered separately, but PostgreSQL integration did not run"
    ),
)
def test_postgresql_configured_dsn_matches_memory_semantics() -> None:
    authority = EffectAuthority.postgresql(
        os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    )
    request = _request(key=f"effect-authority-test:{uuid4()}")
    acquired = _acquire(
        authority,
        request,
        duration=timedelta(minutes=5),
    )
    assert acquired.lease is not None
    terminal = authority.succeed(
        acquired.lease,
        result_ref=_result_ref("postgresql-test-result"),
    )
    replay = _acquire(
        authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
        duration=timedelta(minutes=5),
    )
    assert replay.outcome is AcquireOutcome.SUCCEEDED
    assert replay.terminal == terminal


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for live collation checks",
)
def test_postgresql_17_rejects_case_insensitive_authority_schema() -> None:
    from psycopg import connect
    from psycopg.sql import SQL, Identifier

    dsn = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    schema = f"effect_ci_{uuid4().hex}"

    @contextmanager
    def connect_in_schema(
        configured_dsn: str,
    ) -> Iterator[Any]:
        with connect(configured_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                        Identifier(schema)
                    )
                )
            yield connection

    with connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SHOW server_version_num")
            version = cursor.fetchone()
            assert version is not None
            if int(version[0]) // 10_000 != 17:
                pytest.skip(
                    "live case-insensitive check requires PostgreSQL 17"
                )
            cursor.execute(SQL("CREATE SCHEMA {}").format(Identifier(schema)))
            cursor.execute(
                SQL(
                    """
                CREATE COLLATION {}.case_insensitive (
                    provider = icu,
                    locale = 'und-u-ks-level2',
                    deterministic = false
                )
                """
                ).format(Identifier(schema))
            )
            cursor.execute(
                SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                    Identifier(schema)
                )
            )
            cursor.execute(
                SQL(
                    cast(
                        LiteralString,
                        effect_authority_module._POSTGRES_CREATE_TABLE.replace(
                            'COLLATE "C"',
                            f'COLLATE "{schema}".case_insensitive',
                        ),
                    )
                )
            )
            cursor.execute(
                effect_authority_module._POSTGRES_CREATE_METADATA_TABLE
            )
            cursor.execute(
                """
                INSERT INTO whetstone_effect_authority_metadata (
                    singleton, schema_version
                ) VALUES (%s, %s)
                """,
                (1, effect_authority_module._SCHEMA_VERSION),
            )

    try:
        with pytest.raises(
            EffectAuthoritySchemaMismatchError,
            match=r"incompatible PostgreSQL effect-authority columns.*"
            r"case_insensitive",
        ):
            EffectAuthority.postgresql(
                dsn,
                _connect=connect_in_schema,
            )
    finally:
        with connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema))
                )
