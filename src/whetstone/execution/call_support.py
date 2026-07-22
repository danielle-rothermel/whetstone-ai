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
    "is_transient_transport_failure",
]

#: The transient transport failure classes that a bounded re-drive may retry:
#: a wire/connection transport error, a rate limit, or a timeout. A clean
#: provider rejection, blank generation, or malformed response is NOT transient
#: (re-driving the same request will not change a deterministic "no").
_TRANSIENT_CLASSES = frozenset(
    {
        SemanticFailureClass.TRANSPORT_ERROR,
        SemanticFailureClass.RATE_LIMIT,
        SemanticFailureClass.TIMEOUT,
    }
)


def is_transient_transport_failure(result: ProviderCallResult) -> bool:
    """Whether a terminal call Result is a transient retryable transport fail.

    True when the call did NOT succeed and its terminal semantic failure class
    is transient (transport error / rate limit / timeout) -- the classes a
    bounded observation-level re-drive is allowed to retry once more. A clean
    provider rejection or a structural response defect is not transient.
    """
    if result.succeeded or result.semantic_failure is None:
        return False
    return result.semantic_failure.failure_class in _TRANSIENT_CLASSES


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
    """The runner-level call-guard deadline: transport CAP + 15s margin.

    The transport enforces its OWN absolute wall-clock cap
    (``timeout_seconds``) per single wire call, with the idle timeout as the
    primary stall detector; this runner deadline is the belt-and-suspenders
    backstop the fan-out pool applies per call, sitting just ABOVE the
    transport's single-call bound so the transport's own bound fires first.

    Aligned with the new transport semantics (guard = cap + 15s). The old
    ``cap x max_attempts + 10`` model summed the cap over every logical retry,
    which let 3 stacked semantic retries (3 x 120 + 10 = 370s) exceed the guard
    and trip it BEFORE the transport's per-call bound could -- the c23
    regression. The guard now tracks the transport's single-call cap, not the
    retry-stacked total.
    """
    return policy.transport_policy.timeout_seconds + GUARD_MARGIN_SECONDS
