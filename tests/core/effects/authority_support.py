"""Shared effect-authority test builders."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from whetstone.core.effects.authority import (
    EffectAuthority,
    EffectRequest,
    ReplayPolicy,
)
from whetstone.core.identity import IdentityHash, OpaqueKey, TypedRef

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_NOW = datetime(2026, 7, 31, 12, tzinfo=UTC)
_LEASE_DURATION = timedelta(milliseconds=100)


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
