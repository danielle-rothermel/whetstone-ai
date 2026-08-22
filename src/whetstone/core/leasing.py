"""Whetstone's domain boundary onto ``dr_store.lease.LeaseAuthority``.

The lease authority itself is dr-store's. This module owns only the
translation between whetstone's identity vocabulary and dr-store's lease
vocabulary, so call sites keep reading in whetstone terms:

- ``TypedRef`` (schema-typed whetstone reference) <-> ``ObjectReference``
  (dr-store's ``(schema, content_hash)`` pair) on terminal result refs.
- ``whetstone.core.identity.TerminalFailure`` (``ImmutableJsonObject``
  details) <-> ``dr_store.lease.TerminalFailure`` (plain JSON dict details).

``EffectLeaseAuthority`` composes a ``LeaseAuthority`` and re-exposes
``acquire`` / ``renew`` / ``succeed`` / ``fail`` / ``verify_terminal`` /
``maintain`` / ``close`` in whetstone types. It adds no leasing semantics of
its own: replay policy, fencing, takeover, and terminal authority all remain
dr-store's.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from dr_store.content_addressing import ObjectReference
from dr_store.lease import (
    AcquireOutcome,
    AcquireResult,
    Lease,
    LeaseAuthority,
    LeaseAuthoritySchemaMismatchError,
    LeaseMaintenance,
    LeaseRequest,
    ReplayPolicy,
    StaleLeaseError,
    Terminal,
    TerminalConflictError,
    TerminalOutcome,
)
from dr_store.lease import (
    TerminalFailure as _LeaseTerminalFailure,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from whetstone.core.identity import (
    ContentHash,
    IdentityHash,
    NonEmptyId,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import timedelta
    from pathlib import Path

    from dr_store.relational import TransactionObserver

__all__ = [
    "AcquireOutcome",
    "EffectAcquireResult",
    "EffectLease",
    "EffectLeaseAuthority",
    "EffectLeaseMaintenance",
    "EffectRequest",
    "EffectTerminal",
    "LeaseAuthoritySchemaMismatchError",
    "ReplayPolicy",
    "StaleLeaseError",
    "TerminalConflictError",
    "TerminalFailure",
    "TerminalOutcome",
    "effect_request",
]

# Lease identity and the lease handle carry no whetstone-typed fields, so the
# dr-store models are the whetstone models under whetstone's domain names.
EffectRequest = LeaseRequest
EffectLease = Lease


def effect_request(
    *,
    semantic_key: OpaqueKey | str,
    request_hash: IdentityHash | str,
    replay_policy: ReplayPolicy,
) -> EffectRequest:
    """Build a lease request from whetstone's identity types.

    ``OpaqueKey`` and ``IdentityHash`` are ``str`` subclasses that already
    validated at construction; dr-store re-validates the plain text and the
    64-char hash shape.
    """
    return LeaseRequest(
        semantic_key=str(semantic_key),
        request_hash=str(request_hash),
        replay_policy=replay_policy,
    )


def typed_ref(reference: ObjectReference) -> TypedRef:
    """Convert a dr-store reference back to a schema-typed whetstone ref."""
    return TypedRef(
        schema_name=NonEmptyId(reference.schema),
        content_hash=ContentHash(reference.content_hash),
    )


# dr-store's ``TerminalFailure`` validates ``code`` and ``message`` far more
# tightly than whetstone's ``NonEmptyId``, which only rejects the empty string.
# Whetstone's failure text carries arbitrary provider and exception output, so
# the conversion must be total: a message that dr-store would reject has to
# still produce an orderly FAILED terminal rather than raising inside a
# ``maintain(...)`` block and leaving the lease to expire and be redriven.
_LEASE_TEXT_LIMIT = 1024
_TRUNCATION_SUFFIX = "..."
# Boundary-owned sidecar. Not a caller diagnostic key: ``untruncated_*``
# collided with arbitrary Tool Result / checkpoint details. Values are
# ``unicode-escape`` ASCII so unpaired surrogates survive JSON.
_LEASE_ORIGINAL_KEY = "_whetstone_lease_original"


def _escape_lease_original(value: str) -> str:
    return value.encode("unicode-escape").decode("ascii")


def _unescape_lease_original(value: str) -> str:
    return value.encode("ascii").decode("unicode-escape")


def _lease_text(value: str, *, field: str, details: dict[str, Any]) -> str:
    """Coerce whetstone failure text into dr-store's accepted text shape.

    Preserves the original under ``details[_LEASE_ORIGINAL_KEY][field]``
    whenever coercion loses information, so no diagnostic text is dropped.
    """
    coerced = value.replace("\x00", "")
    coerced = "".join(
        character
        for character in coerced
        if not "\ud800" <= character <= "\udfff"
    )
    if len(coerced) > _LEASE_TEXT_LIMIT:
        coerced = (
            coerced[: _LEASE_TEXT_LIMIT - len(_TRUNCATION_SUFFIX)]
            + _TRUNCATION_SUFFIX
        )
    if not coerced.strip():
        # dr-store rejects empty and whitespace-only text; whetstone's
        # NonEmptyId admits both beyond the empty string.
        coerced = f"<unprintable {field}>"
    if coerced != value:
        envelope = details.get(_LEASE_ORIGINAL_KEY)
        if not isinstance(envelope, dict):
            envelope = {}
            details[_LEASE_ORIGINAL_KEY] = envelope
        envelope[field] = _escape_lease_original(value)
    return coerced


def _to_lease_failure(failure: TerminalFailure) -> _LeaseTerminalFailure:
    details = dict(failure.model_dump(mode="json")["details"])
    # Bind text before details so a coerced value can record its original.
    code = _lease_text(str(failure.code), field="code", details=details)
    message = _lease_text(
        str(failure.message), field="message", details=details
    )
    return _LeaseTerminalFailure(code=code, message=message, details=details)


def _lease_text_shape(value: str, *, field: str) -> str:
    """Return the dr-store text shape without recording the original."""
    return _lease_text(value, field=field, details={})


def _from_lease_failure(failure: _LeaseTerminalFailure) -> TerminalFailure:
    """Restore the whetstone failure that ``_to_lease_failure`` encoded.

    Callers persist the original ``TerminalFailure`` on the Tool Result,
    adapter checkpoint, or intent resolution, then exact-compare it to
    ``EffectTerminal.failure``. The lease-side coercion is a transport
    concern: invert it here so those comparisons stay equal. Only the
    reserved envelope is popped; caller-owned ``untruncated_*`` keys stay.
    Restore a field only when the stored text is our coercion of the
    decoded original, so a stuffed envelope cannot rewrite clean text.
    """
    data = failure.model_dump(mode="json")
    details = dict(data["details"])
    envelope = details.pop(_LEASE_ORIGINAL_KEY, None)
    if isinstance(envelope, dict):
        for field in ("code", "message"):
            encoded = envelope.get(field)
            if not isinstance(encoded, str):
                continue
            original = _unescape_lease_original(encoded)
            if _lease_text_shape(original, field=field) == data[field]:
                data[field] = original
    data["details"] = details
    return TerminalFailure.model_validate(data)


class EffectTerminal(BaseModel):
    """A dr-store lease terminal in whetstone's reference vocabulary.

    Field-for-field ``dr_store.lease.Terminal`` except that ``result_ref`` is
    a schema-typed ``TypedRef`` and ``failure`` is whetstone's
    ``TerminalFailure``. Whetstone persists this shape inside
    ``ToolCallStoreEntry``, so its JSON form is the stored form; ``to_lease``
    and ``from_lease`` are the only conversion path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    outcome: TerminalOutcome
    owner_id: NonEmptyId
    attempt_id: NonEmptyId
    fence: int
    result_ref: TypedRef | None = None
    failure: TerminalFailure | None = None

    @field_validator("fence")
    @classmethod
    def _validate_fence(cls, value: int) -> int:
        if not 1 <= value <= (1 << 63) - 1:
            raise ValueError("fence must be a positive signed 64-bit integer")
        return value

    @model_validator(mode="after")
    def _validate(self) -> EffectTerminal:
        # Delegate outcome/payload consistency to dr-store's Terminal so the
        # two models cannot drift apart.
        self.to_lease()
        return self

    def to_lease(self) -> Terminal:
        return Terminal(
            request=self.request,
            outcome=self.outcome,
            owner_id=str(self.owner_id),
            attempt_id=str(self.attempt_id),
            fence=self.fence,
            result_ref=(
                None if self.result_ref is None else self.result_ref.reference
            ),
            failure=(
                None
                if self.failure is None
                else _to_lease_failure(self.failure)
            ),
        )

    @classmethod
    def from_lease(cls, terminal: Terminal) -> EffectTerminal:
        return cls(
            request=terminal.request,
            outcome=terminal.outcome,
            owner_id=NonEmptyId(terminal.owner_id),
            attempt_id=NonEmptyId(terminal.attempt_id),
            fence=terminal.fence,
            result_ref=(
                None
                if terminal.result_ref is None
                else typed_ref(terminal.result_ref)
            ),
            failure=(
                None
                if terminal.failure is None
                else _from_lease_failure(terminal.failure)
            ),
        )


class EffectAcquireResult(BaseModel):
    """A dr-store acquire result carrying a whetstone-typed terminal."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    outcome: AcquireOutcome
    lease: EffectLease | None = None
    terminal: EffectTerminal | None = None
    busy_expires_at: datetime | None = None
    existing_request_hash: IdentityHash | None = None
    existing_replay_policy: ReplayPolicy | None = None

    @classmethod
    def _from_lease(cls, result: AcquireResult) -> EffectAcquireResult:
        return cls(
            request=result.request,
            outcome=result.outcome,
            lease=result.lease,
            terminal=(
                None
                if result.terminal is None
                else EffectTerminal.from_lease(result.terminal)
            ),
            busy_expires_at=result.busy_expires_at,
            existing_request_hash=(
                None
                if result.existing_request_hash is None
                else IdentityHash(result.existing_request_hash)
            ),
            existing_replay_policy=result.existing_replay_policy,
        )


class EffectLeaseMaintenance:
    """Whetstone-typed view of ``dr_store.lease.LeaseMaintenance``.

    Single-threaded, like the handle it wraps: enter, terminalize, and exit on
    one thread. Renewal, the terminalization state machine, and the renewer
    restart after a transient terminal-publication failure are all dr-store's.
    """

    def __init__(self, maintenance: LeaseMaintenance) -> None:
        self._maintenance = maintenance

    @property
    def lease(self) -> EffectLease:
        return self._maintenance.lease

    def check(self) -> None:
        self._maintenance.check()

    def succeed(self, *, result_ref: TypedRef) -> EffectTerminal:
        return EffectTerminal.from_lease(
            self._maintenance.succeed(result_ref=result_ref.reference)
        )

    def fail(
        self,
        *,
        result_ref: TypedRef,
        failure: TerminalFailure,
    ) -> EffectTerminal:
        return EffectTerminal.from_lease(
            self._maintenance.fail(
                result_ref=result_ref.reference,
                failure=_to_lease_failure(failure),
            )
        )

    def __enter__(self) -> EffectLeaseMaintenance:
        self._maintenance.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._maintenance.__exit__(*args)


class EffectLeaseAuthority:
    """Whetstone's effect leasing surface over ``LeaseAuthority``."""

    def __init__(self, authority: LeaseAuthority) -> None:
        self._authority = authority

    @classmethod
    def memory(
        cls,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> EffectLeaseAuthority:
        return cls(LeaseAuthority.memory(clock=clock))

    @classmethod
    def sqlite(
        cls,
        path: str | Path,
        *,
        _transaction_observer: TransactionObserver | None = None,
    ) -> EffectLeaseAuthority:
        return cls(
            LeaseAuthority.sqlite(
                path, _transaction_observer=_transaction_observer
            )
        )

    @classmethod
    def postgresql(cls, dsn: str) -> EffectLeaseAuthority:
        return cls(LeaseAuthority.postgresql(dsn))

    def acquire(
        self,
        request: EffectRequest,
        *,
        owner_id: str,
        attempt_id: str,
        lease_duration: timedelta,
    ) -> EffectAcquireResult:
        return EffectAcquireResult._from_lease(
            self._authority.acquire(
                request,
                owner_id=owner_id,
                attempt_id=attempt_id,
                lease_duration=lease_duration,
            )
        )

    def renew(
        self, lease: EffectLease, *, lease_duration: timedelta
    ) -> EffectLease:
        return self._authority.renew(lease, lease_duration=lease_duration)

    def succeed(
        self, lease: EffectLease, *, result_ref: TypedRef
    ) -> EffectTerminal:
        return EffectTerminal.from_lease(
            self._authority.succeed(lease, result_ref=result_ref.reference)
        )

    def fail(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
        failure: TerminalFailure,
    ) -> EffectTerminal:
        return EffectTerminal.from_lease(
            self._authority.fail(
                lease,
                result_ref=result_ref.reference,
                failure=_to_lease_failure(failure),
            )
        )

    def verify_terminal(self, terminal: EffectTerminal) -> EffectTerminal:
        return EffectTerminal.from_lease(
            self._authority.verify_terminal(terminal.to_lease())
        )

    def maintain(
        self, lease: EffectLease, *, lease_duration: timedelta
    ) -> EffectLeaseMaintenance:
        return EffectLeaseMaintenance(
            self._authority.maintain(lease, lease_duration=lease_duration)
        )

    def close(self) -> None:
        self._authority.close()

    def __enter__(self) -> EffectLeaseAuthority:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
