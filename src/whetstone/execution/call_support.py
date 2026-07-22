"""Shared inspection helpers over a terminal Provider Call Result.

Both the pilot and cell fan-out wiring need to (a) name a failed call's cause
for the loud failure summary, (b) detect a rate-limit typed failure so the
shared concurrency gate can halve, and (c) compute the runner-level guard
deadline from the execution policy's transport timeout. These are pure
functions over the already-classified :class:`ProviderCallResult`, kept in one
place so the two phases classify identically.
"""

from __future__ import annotations

from whetstone.execution.fanout import GUARD_MARGIN_SECONDS
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.classification import SemanticFailureClass
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "failure_code_of",
    "guard_deadline_seconds",
    "is_rate_limit_failure",
]


def failure_code_of(result: ProviderCallResult) -> str:
    """Name a failed call's cause: the transport ``code``, else the class.

    Prefers the causal transport failure's ``code`` (e.g.
    ``"missing_base_url"``, ``"timeout"``, ``"stalled_response"``) so a loud
    summary points at the true root cause; falls back to the semantic
    failure-class value, then ``"unknown"``. Returns ``""`` for a success.
    """
    if result.succeeded or result.semantic_failure is None:
        return ""
    failure = result.semantic_failure
    transport = failure.transport_failure
    code = getattr(transport, "code", None)
    if code:
        return str(code)
    return failure.failure_class.value


def is_rate_limit_failure(result: ProviderCallResult) -> bool:
    """Whether ANY attempt of this call hit a rate-limit (429) typed failure.

    Inspects every attempt, not just the terminal outcome: a call that hit a
    429 and then retried to success STILL observed rate limiting, and all lanes
    share one key, so it must halve the shared effective concurrency for the
    rest of the run. A terminal rate-limit failure is the exhausted case.
    """
    for attempt in result.attempts:
        if (
            attempt.failure_class
            is SemanticFailureClass.RATE_LIMIT
        ):
            return True
    return False


def guard_deadline_seconds(policy: ProviderExecutionPolicy) -> float:
    """The runner-level call-guard deadline: transport timeout + 10s margin.

    The transport now enforces its own wall-clock bound; this deadline is the
    belt-and-suspenders backstop the fan-out pool applies per call. It sums the
    referenced transport policy's ``timeout_seconds`` over ALL bounded logical
    attempts (a retrying call may legitimately run several transport timeouts)
    plus one fixed margin.
    """
    per_attempt = policy.transport_policy.timeout_seconds
    return per_attempt * policy.max_attempts + GUARD_MARGIN_SECONDS
