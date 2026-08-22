"""Whetstone's leasing boundary over a real PostgreSQL lease authority.

The memory and SQLite coverage lives in ``tests/test_leasing.py``. This pins
that the same whetstone usage contract -- request identity, replay policy,
whetstone-typed terminals -- holds on the PostgreSQL backend, which reads
authority time from the database rather than an injected clock.

Run locally:
    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine, make_url, text

from whetstone.core.identity import (
    ImmutableJsonObject,
    TerminalFailure,
    TypedRef,
    compute_identity_hash,
    compute_prefixed_identity_key,
    typed_ref_for_record,
)
from whetstone.core.leasing import (
    AcquireOutcome,
    EffectLeaseAuthority,
    LeaseAuthoritySchemaMismatchError,
    ReplayPolicy,
    TerminalOutcome,
    effect_request,
)

pytestmark = pytest.mark.integration

_SCHEMA = "whetstone.leasing_pg_test_effect"
_SCHEMA_VERSION = 1
_KEY_PREFIX = "whetstone.leasing_pg_test:"
_RESULT_SCHEMA = "whetstone.leasing_pg_test_result"
_LEASE = timedelta(seconds=30)


def _libpq_dsn(database_url: str) -> str:
    """Strip the SQLAlchemy driver suffix to get a plain libpq DSN."""
    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


def _request(payload: dict[str, object]):
    return effect_request(
        semantic_key=compute_prefixed_identity_key(
            schema=_SCHEMA,
            schema_version=_SCHEMA_VERSION,
            prefix=_KEY_PREFIX,
            payload=payload,
        ),
        request_hash=compute_identity_hash(
            schema=_SCHEMA,
            schema_version=_SCHEMA_VERSION,
            payload=payload,
        ),
        replay_policy=ReplayPolicy.IDEMPOTENT,
    )


def _result_ref(value: str = "ok") -> TypedRef:
    return typed_ref_for_record(_RESULT_SCHEMA, {"value": value})


def test_postgres_effect_terminals_replay_in_whetstone_types(
    clean_pg: str,
) -> None:
    """A durable terminal replays with a TypedRef and TerminalFailure."""
    authority = EffectLeaseAuthority.postgresql(_libpq_dsn(clean_pg))
    try:
        succeeded_request = _request({"call": "alpha"})
        failed_request = _request({"call": "beta"})
        result_ref = _result_ref()
        failure = TerminalFailure(
            code="leasing_pg_test_failure",
            message="the effect failed",
            details=ImmutableJsonObject({"attempt": 1}),
        )

        acquired = authority.acquire(
            succeeded_request,
            owner_id="owner-a",
            attempt_id="attempt-a",
            lease_duration=_LEASE,
        )
        assert acquired.lease is not None
        succeeded = authority.succeed(acquired.lease, result_ref=result_ref)

        acquired_failing = authority.acquire(
            failed_request,
            owner_id="owner-a",
            attempt_id="attempt-b",
            lease_duration=_LEASE,
        )
        assert acquired_failing.lease is not None
        failed = authority.fail(
            acquired_failing.lease,
            result_ref=_result_ref("failed"),
            failure=failure,
        )

        # A second holder replays the durable terminals, not a fresh lease.
        success_replay = authority.acquire(
            succeeded_request,
            owner_id="owner-b",
            attempt_id="attempt-c",
            lease_duration=_LEASE,
        )
        failure_replay = authority.acquire(
            failed_request,
            owner_id="owner-b",
            attempt_id="attempt-d",
            lease_duration=_LEASE,
        )

        assert succeeded.outcome is TerminalOutcome.SUCCEEDED
        assert succeeded.result_ref == result_ref
        assert isinstance(succeeded.result_ref, TypedRef)
        assert success_replay.outcome is AcquireOutcome.SUCCEEDED
        assert success_replay.terminal == succeeded

        assert failed.failure == failure
        assert failure_replay.outcome is AcquireOutcome.FAILED
        assert failure_replay.terminal is not None
        assert failure_replay.terminal.failure == failure

        # Both terminals stay authoritative when re-verified.
        assert authority.verify_terminal(succeeded) == succeeded
        assert authority.verify_terminal(failed) == failed
    finally:
        authority.close()


def test_postgres_busy_lease_refuses_a_second_holder(clean_pg: str) -> None:
    """A live PostgreSQL lease reports BUSY with its authority expiry."""
    authority = EffectLeaseAuthority.postgresql(_libpq_dsn(clean_pg))
    try:
        request = _request({"call": "gamma"})
        acquired = authority.acquire(
            request,
            owner_id="owner-a",
            attempt_id="attempt-a",
            lease_duration=_LEASE,
        )
        assert acquired.lease is not None

        contender = authority.acquire(
            request,
            owner_id="owner-b",
            attempt_id="attempt-b",
            lease_duration=_LEASE,
        )

        assert contender.outcome is AcquireOutcome.BUSY
        assert contender.busy_expires_at == acquired.lease.expires_at
    finally:
        authority.close()


_LEASE_TABLE = "dr_store_lease_authority"
_LEASE_METADATA_TABLE = "dr_store_lease_authority_metadata"


def _execute(database_url: str, *statements: str) -> None:
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
    finally:
        engine.dispose()


def test_postgres_drifted_lease_table_raises_at_construction(
    clean_pg: str,
) -> None:
    """A mis-shaped lease table fails EffectLeaseAuthority.postgresql eagerly."""
    _execute(
        clean_pg,
        f"""
        CREATE TABLE {_LEASE_TABLE} (
            semantic_key TEXT COLLATE "C" PRIMARY KEY,
            unexpected_column TEXT COLLATE "C" NOT NULL
        )
        """,
        f"""
        CREATE TABLE {_LEASE_METADATA_TABLE} (
            component TEXT COLLATE "C" PRIMARY KEY,
            version INTEGER NOT NULL
        )
        """,
    )

    with pytest.raises(LeaseAuthoritySchemaMismatchError) as caught:
        EffectLeaseAuthority.postgresql(_libpq_dsn(clean_pg))

    error = caught.value
    assert error.table == _LEASE_TABLE
    assert error.aspect == "columns"
    assert error.expected != error.actual
