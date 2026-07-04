from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from dr_serialize import sha256_json_digest
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from whetstone.db import schema
from whetstone.eval_failures import FailureClass, FailureSummary

DEFAULT_INITIAL_BACKOFF_SECONDS = 5.0
DEFAULT_MAX_BACKOFF_SECONDS = 300.0
DEFAULT_JITTER_SECONDS = 3.0
BACKOFF_JITTER_DIGEST_LENGTH = 8
MAX_BACKOFF_EXPONENT = 1022
RETRYABLE_BACKOFF_FAILURES = frozenset(
    {
        FailureClass.TRANSIENT,
        FailureClass.RATE_LIMITED,
    }
)


class ThrottleBackoffState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    throttle_key: StrictStr
    blocked_until: datetime | None = None
    consecutive_failures: StrictInt = 0
    failure_class: FailureClass | None = None
    last_error_type: StrictStr | None = None
    last_message: StrictStr | None = None
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)
    updated_at: datetime


def should_backoff_failure(failure_class: FailureClass) -> bool:
    return failure_class in RETRYABLE_BACKOFF_FAILURES


def delay_until_unblocked_seconds(
    state: ThrottleBackoffState | None,
    *,
    now: datetime,
) -> float:
    if state is None or state.blocked_until is None:
        return 0.0
    remaining = (state.blocked_until - now).total_seconds()
    return max(0.0, remaining)


def next_backoff_delay_seconds(
    *,
    throttle_key: str,
    consecutive_failures: int,
    failure_class: FailureClass,
    initial_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    jitter_seconds: float = DEFAULT_JITTER_SECONDS,
) -> float:
    if not should_backoff_failure(failure_class):
        return 0.0
    failure_count = max(1, consecutive_failures)
    exponent = min(failure_count - 1, MAX_BACKOFF_EXPONENT)
    exponential = initial_seconds * (2 ** exponent)
    base_delay = min(max_seconds, exponential)
    jitter = deterministic_jitter_seconds(
        throttle_key=throttle_key,
        consecutive_failures=failure_count,
        max_jitter_seconds=jitter_seconds,
    )
    return min(max_seconds, base_delay + jitter)


def deterministic_jitter_seconds(
    *,
    throttle_key: str,
    consecutive_failures: int,
    max_jitter_seconds: float,
) -> float:
    if max_jitter_seconds <= 0:
        return 0.0
    digest = sha256_json_digest(
        {
            "throttle_key": throttle_key,
            "consecutive_failures": consecutive_failures,
        },
        length=BACKOFF_JITTER_DIGEST_LENGTH,
    )
    fraction = int(digest, 16) / float(16 ** BACKOFF_JITTER_DIGEST_LENGTH - 1)
    return fraction * max_jitter_seconds


def load_throttle_backoff_state(
    connection: Connection,
    *,
    throttle_key: str,
) -> ThrottleBackoffState | None:
    row = connection.execute(
        schema.throttle_backoff.select().where(
            schema.throttle_backoff.c.throttle_key == throttle_key
        )
    ).mappings().one_or_none()
    if row is None:
        return None
    return throttle_backoff_state_from_row(dict(row))


def throttle_delay_seconds(
    connection: Connection,
    *,
    throttle_key: str,
    now: datetime,
) -> float:
    state = load_throttle_backoff_state(
        connection,
        throttle_key=throttle_key,
    )
    return delay_until_unblocked_seconds(state, now=now)


def record_throttle_failure(
    connection: Connection,
    *,
    throttle_key: str,
    failure: FailureSummary,
    now: datetime,
) -> ThrottleBackoffState | None:
    if not should_backoff_failure(failure.failure_class):
        return None

    consecutive_failures = increment_throttle_consecutive_failures(
        connection,
        throttle_key=throttle_key,
        now=now,
    )
    delay = next_backoff_delay_seconds(
        throttle_key=throttle_key,
        consecutive_failures=consecutive_failures,
        failure_class=failure.failure_class,
    )
    state = ThrottleBackoffState(
        throttle_key=throttle_key,
        blocked_until=now + timedelta(seconds=delay),
        consecutive_failures=consecutive_failures,
        failure_class=failure.failure_class,
        last_error_type=failure.failure_exception_type,
        last_message=failure.message,
        metadata=failure.failure_metadata,
        updated_at=now,
    )
    connection.execute(
        update(schema.throttle_backoff)
        .where(schema.throttle_backoff.c.throttle_key == throttle_key)
        .values(
            blocked_until=state.blocked_until,
            failure_class=(
                state.failure_class.value
                if state.failure_class is not None
                else None
            ),
            last_error_type=state.last_error_type,
            last_message=state.last_message,
            metadata=state.metadata,
            updated_at=state.updated_at,
        )
    )
    return state


def increment_throttle_consecutive_failures(
    connection: Connection,
    *,
    throttle_key: str,
    now: datetime,
) -> int:
    inserted = connection.execute(
        insert(schema.throttle_backoff)
        .values(
            {
                "throttle_key": throttle_key,
                "blocked_until": None,
                "consecutive_failures": 1,
                "failure_class": None,
                "last_error_type": None,
                "last_message": None,
                "metadata": {},
                "updated_at": now,
            }
        )
        .on_conflict_do_update(
            index_elements=["throttle_key"],
            set_={
                "consecutive_failures": (
                    schema.throttle_backoff.c.consecutive_failures + 1
                ),
                "updated_at": now,
            },
        )
        .returning(schema.throttle_backoff.c.consecutive_failures)
    )
    return int(inserted.scalar_one())


def clear_throttle_backoff(
    connection: Connection,
    *,
    throttle_key: str,
    now: datetime,
) -> None:
    state = ThrottleBackoffState(
        throttle_key=throttle_key,
        consecutive_failures=0,
        metadata={},
        updated_at=now,
    )
    connection.execute(upsert_throttle_backoff_state(state))


def upsert_throttle_backoff_state(state: ThrottleBackoffState) -> Any:
    row = throttle_backoff_row(state)
    return (
        insert(schema.throttle_backoff)
        .values(row)
        .on_conflict_do_update(
            index_elements=["throttle_key"],
            set_={
                key: value
                for key, value in row.items()
                if key != "throttle_key"
            },
        )
    )


def throttle_backoff_row(state: ThrottleBackoffState) -> dict[str, Any]:
    return {
        "throttle_key": state.throttle_key,
        "blocked_until": state.blocked_until,
        "consecutive_failures": state.consecutive_failures,
        "failure_class": (
            state.failure_class.value
            if state.failure_class is not None
            else None
        ),
        "last_error_type": state.last_error_type,
        "last_message": state.last_message,
        "metadata": state.metadata,
        "updated_at": state.updated_at,
    }


def throttle_backoff_state_from_row(
    row: dict[str, Any],
) -> ThrottleBackoffState:
    failure_class = row.get("failure_class")
    return ThrottleBackoffState(
        throttle_key=row["throttle_key"],
        blocked_until=row["blocked_until"],
        consecutive_failures=row["consecutive_failures"],
        failure_class=(
            FailureClass(failure_class) if failure_class is not None else None
        ),
        last_error_type=row["last_error_type"],
        last_message=row["last_message"],
        metadata=row["metadata"],
        updated_at=row["updated_at"],
    )


def utc_now() -> datetime:
    return datetime.now(UTC)
