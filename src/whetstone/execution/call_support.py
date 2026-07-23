"""Shared inspection helpers over a terminal Provider Call Result.

Both the pilot and cell fan-out wiring need to (a) name a failed call's cause
for the loud failure summary, (b) detect a rate-limit typed failure so the
shared concurrency gate can halve, and (c) compute the runner-level guard
deadline from the execution policy's transport timeout. These are pure
functions over the already-classified :class:`ProviderCallResult`, kept in one
place so the two phases classify identically.
"""

from __future__ import annotations

from dataclasses import dataclass

from whetstone.execution.fanout import GUARD_MARGIN_SECONDS
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.classification import SemanticFailureClass
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "CallTelemetry",
    "call_telemetry",
    "failure_code_of",
    "guard_deadline_seconds",
    "is_rate_limit_failure",
    "is_transient_transport_failure",
]


@dataclass(frozen=True, slots=True)
class CallTelemetry:
    """Per-call usage + latency telemetry (task 20).

    Every field is ``None`` when the provider did not expose it -- NEVER
    conflated with 0 (a reasoning-free model reports ``reasoning_tokens=None``,
    not 0, so an aggregate can compute over rows-with-field and report
    coverage). ``latency_s`` is the accepted attempt's wall-clock (request
    start -> completion), one number, no streaming decomposition.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None


def call_telemetry(result: ProviderCallResult | None) -> CallTelemetry:
    """Extract usage (incl. reasoning tokens) + latency from a call Result.

    Reads the accepted Generation's response usage (prompt/completion/total AND
    ``reasoning_tokens`` where the provider exposes
    ``completion_tokens_details.reasoning_tokens``) and the accepted attempt's
    wall-clock latency. Returns all-``None`` for a failed/absent call; a
    present call with no usage block or no reasoning detail leaves those None
    (coverage-honest, never 0-conflated).
    """
    if result is None or not result.succeeded or result.generation is None:
        return CallTelemetry(latency_s=_accepted_latency(result))
    usage = result.generation.response.usage
    if usage is None:
        return CallTelemetry(latency_s=_accepted_latency(result))
    return CallTelemetry(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=getattr(usage, "reasoning_tokens", None),
        latency_s=_accepted_latency(result),
    )


def _accepted_latency(result: ProviderCallResult | None) -> float | None:
    """The accepted (or last) attempt's wall-clock seconds, else ``None``."""
    if result is None or not result.attempts:
        return None
    # Prefer the accepted (generation-bearing) attempt; else the terminal one.
    chosen = next(
        (a for a in result.attempts if a.generation is not None),
        result.attempts[-1],
    )
    started, ended = chosen.started_at, chosen.ended_at
    if started is None or ended is None:
        return None
    return max(0.0, ended - started)

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


def guard_deadline_seconds(
    policy: ProviderExecutionPolicy, *, wire_calls_per_unit: int = 1
) -> float:
    """The runner-level call-guard deadline: transport CAP + 15s margin.

    The transport enforces its OWN absolute wall-clock cap
    (``timeout_seconds``) per single wire call, with the idle timeout as the
    primary stall detector; this runner deadline is the belt-and-suspenders
    backstop the fan-out pool applies per fan-out UNIT, sitting just ABOVE that
    unit's total transport-bound time so the transport's own bound fires first.

    Aligned with the new transport semantics (guard = cap + 15s). The old
    ``cap x max_attempts + 10`` model summed the cap over every logical retry,
    which let 3 stacked semantic retries (3 x 120 + 10 = 370s) exceed the guard
    and trip it BEFORE the transport's per-call bound could -- the c23
    regression. The guard now tracks the transport's single-call cap, not the
    retry-stacked total.

    ``wire_calls_per_unit`` is the number of SEQUENTIAL wire calls one fan-out
    unit makes (1 for a QA row = one call; 2 for an ed1 row = encoder THEN
    decoder). The guard scales with it so each call in the unit gets its full
    transport cap before the row-level backstop fires -- otherwise a 2-call ed1
    row under a 1-call (cap + 15s) guard trips the guard mid-second-call the
    instant the first call consumed any time, masquerading as a transport-bound
    regression (the eval:ed1:a1 hang).
    """
    cap = policy.transport_policy.timeout_seconds
    return cap * max(1, wire_calls_per_unit) + GUARD_MARGIN_SECONDS
