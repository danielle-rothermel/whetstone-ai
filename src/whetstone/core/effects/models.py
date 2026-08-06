from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import UNIQUE, StrEnum, verify

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    IdentityHash,
    NonEmptyId,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
)

_MAX_FENCE = (1 << 63) - 1


def _require_text(value: str, *, field: str, maximum: int = 1024) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} cannot contain NUL")
    if len(value) > maximum:
        raise ValueError(f"{field} cannot exceed {maximum} characters")
    return value


def _require_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must use UTC")
    return value


def _require_lease_duration(value: timedelta) -> timedelta:
    if not isinstance(value, timedelta):
        raise TypeError("lease_duration must be a timedelta")
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("lease_duration must be positive")
    return value


@verify(UNIQUE)
class ReplayPolicy(StrEnum):
    """Whether an expired effect may be assigned to a new physical attempt.

    These values are persisted contract literals. Never iterate over this
    enum to construct a persisted payload.
    """

    IDEMPOTENT = "idempotent"
    DURABLE_WORKFLOW = "durable_workflow"
    NO_REDRIVE = "no_redrive"


@verify(UNIQUE)
class TerminalOutcome(StrEnum):
    """Persisted terminal state; never iterate it to construct a payload."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECOVERY_REQUIRED = "recovery_required"


@verify(UNIQUE)
class AcquireOutcome(StrEnum):
    """Serialized acquisition result; never iterate it to build a payload."""

    ACQUIRED = "acquired"
    BUSY = "busy"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REQUEST_CONFLICT = "request_conflict"
    RECOVERY_REQUIRED = "recovery_required"


@verify(UNIQUE)
class _StoredState(StrEnum):
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECOVERY_REQUIRED = "recovery_required"


class EffectRequest(BaseModel):
    """Immutable identity of one semantic effect."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    semantic_key: OpaqueKey
    request_identity_hash: IdentityHash
    replay_policy: ReplayPolicy

    @model_validator(mode="after")
    def _validate(self) -> EffectRequest:
        _require_text(self.semantic_key, field="semantic_key")
        return self


class EffectLease(BaseModel):
    """Exact authority token required for renewal and terminalization."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    owner_id: NonEmptyId
    attempt_id: NonEmptyId
    fence: StrictInt
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def _validate_expires_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field="expires_at")

    @model_validator(mode="after")
    def _validate(self) -> EffectLease:
        _require_text(self.owner_id, field="owner_id", maximum=255)
        _require_text(self.attempt_id, field="attempt_id", maximum=255)
        if not 1 <= self.fence <= _MAX_FENCE:
            raise ValueError("fence must be a positive signed 64-bit integer")
        return self


class EffectTerminal(BaseModel):
    """Immutable authoritative terminal outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    outcome: TerminalOutcome
    owner_id: NonEmptyId
    attempt_id: NonEmptyId
    fence: StrictInt
    result_ref: TypedRef | None = None
    failure: TerminalFailure | None = None

    @model_validator(mode="after")
    def _validate(self) -> EffectTerminal:
        _require_text(self.owner_id, field="owner_id", maximum=255)
        _require_text(self.attempt_id, field="attempt_id", maximum=255)
        if not 1 <= self.fence <= _MAX_FENCE:
            raise ValueError("fence must be a positive signed 64-bit integer")
        if self.outcome is TerminalOutcome.SUCCEEDED:
            if self.result_ref is None or self.failure is not None:
                raise ValueError(
                    "a succeeded terminal requires only result_ref"
                )
        elif self.outcome is TerminalOutcome.FAILED:
            if self.result_ref is None or self.failure is None:
                raise ValueError(
                    "a failed terminal requires result_ref and failure"
                )
        elif self.failure is None or self.result_ref is not None:
            raise ValueError(
                "a recovery-required terminal requires only failure"
            )
        return self


class AcquireResult(BaseModel):
    """Typed result of acquiring or replaying one semantic effect."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    outcome: AcquireOutcome
    lease: EffectLease | None = None
    terminal: EffectTerminal | None = None
    busy_expires_at: datetime | None = None
    existing_request_identity_hash: IdentityHash | None = None
    existing_replay_policy: ReplayPolicy | None = None

    @field_validator("busy_expires_at")
    @classmethod
    def _validate_busy_expires_at(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None:
            _require_utc(value, field="busy_expires_at")
        return value

    @model_validator(mode="after")
    def _validate(self) -> AcquireResult:
        populated = (
            self.lease is not None,
            self.terminal is not None,
            self.busy_expires_at is not None,
            self.existing_request_identity_hash is not None,
            self.existing_replay_policy is not None,
        )
        if self.outcome is AcquireOutcome.ACQUIRED:
            if populated != (True, False, False, False, False):
                raise ValueError("ACQUIRED requires only a lease")
            if self.lease is not None and self.lease.request != self.request:
                raise ValueError("acquired lease must match the request")
        elif self.outcome is AcquireOutcome.BUSY:
            if populated != (False, False, True, False, False):
                raise ValueError("BUSY requires only busy_expires_at")
        elif self.outcome is AcquireOutcome.REQUEST_CONFLICT:
            if populated != (False, False, False, True, True):
                raise ValueError(
                    "REQUEST_CONFLICT requires the existing identity "
                    "and policy"
                )
        else:
            if populated != (False, True, False, False, False):
                raise ValueError(
                    "terminal acquisition outcomes require only terminal"
                )
            expected = {
                AcquireOutcome.SUCCEEDED: TerminalOutcome.SUCCEEDED,
                AcquireOutcome.FAILED: TerminalOutcome.FAILED,
                AcquireOutcome.RECOVERY_REQUIRED: (
                    TerminalOutcome.RECOVERY_REQUIRED
                ),
            }[self.outcome]
            if self.terminal is not None:
                if self.terminal.request != self.request:
                    raise ValueError("terminal must match the request")
                if self.terminal.outcome is not expected:
                    raise ValueError("terminal outcome does not match result")
        return self


class EffectAuthorityError(RuntimeError):
    """Base error for non-acquisition authority transitions."""


class EffectAuthoritySchemaMismatchError(EffectAuthorityError):
    """The durable authority schema does not match its exact contract."""


class StaleLeaseError(EffectAuthorityError):
    """The supplied owner/fence is no longer authorized."""


class TerminalConflictError(EffectAuthorityError):
    """A different immutable terminal outcome is already authoritative."""


class _AuthorityCorruptionError(EffectAuthorityError):
    pass


@dataclass(frozen=True, slots=True)
class _EffectRow:
    request: EffectRequest
    state: _StoredState
    owner_id: NonEmptyId
    attempt_id: NonEmptyId
    fence: int
    expires_at: datetime | None
    terminal: EffectTerminal | None

    @classmethod
    def leased(cls, lease: EffectLease) -> _EffectRow:
        return cls(
            request=lease.request,
            state=_StoredState.LEASED,
            owner_id=lease.owner_id,
            attempt_id=lease.attempt_id,
            fence=lease.fence,
            expires_at=lease.expires_at,
            terminal=None,
        )

    @classmethod
    def terminalized(cls, terminal: EffectTerminal) -> _EffectRow:
        return cls(
            request=terminal.request,
            state=_StoredState(terminal.outcome.value),
            owner_id=terminal.owner_id,
            attempt_id=terminal.attempt_id,
            fence=terminal.fence,
            expires_at=None,
            terminal=terminal,
        )

    def lease(self) -> EffectLease:
        if self.state is not _StoredState.LEASED or self.expires_at is None:
            raise _AuthorityCorruptionError("row is not an active lease")
        return EffectLease(
            request=self.request,
            owner_id=self.owner_id,
            attempt_id=self.attempt_id,
            fence=self.fence,
            expires_at=self.expires_at,
        )


__all__ = [
    "AcquireOutcome",
    "AcquireResult",
    "EffectAuthorityError",
    "EffectAuthoritySchemaMismatchError",
    "EffectLease",
    "EffectRequest",
    "EffectTerminal",
    "ReplayPolicy",
    "StaleLeaseError",
    "TerminalConflictError",
    "TerminalFailure",
    "TerminalOutcome",
]
