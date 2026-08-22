"""Run-level spend aggregated from durable evidence.

``OptimResult.cost`` reports what one optimization run spent, split by the
role that spent it: the task model evaluating candidates, and the proposer
model drafting them. The Step 10 validation protocol compares optimizers on
this record, so its wire keys are a persisted format with a golden test
pinning the exact literals.

Three rules keep the record honest rather than merely populated.

First, every total is aggregated from durable records rather than from a
counter the run happened to hold: task-model usage from the evaluation
evidence in the object store, proposer usage from the Step Results. Both are
de-duplicated by call identity, so a resumed run, a platform run, and an
in-process run of the same work report the same number.

Second, ``usd`` is present only when *every* contributing call carried a
price. dr-providers reports a per-call price only when the provider returns
one (OpenRouter does; most OpenAI-compatible endpoints do not), so a partial
sum would understate spend while looking authoritative. When any call lacks a
price the field is absent and ``priced_calls``/``unpriced_calls`` show the
split. Whetstone owns no pricing table and infers no prices.

Third, a call is counted where it was paid for, and only once. A prompt-cache
hit and a replayed GEPA reflection both come back carrying the original
call's tokens and price; each is reported separately rather than billed
again, so reuse shows up as a lower cost instead of a higher one.

Known limitation. Counting a replay once relies on the original call's Step
Result existing. If a GEPA worker dies after ``record_proposal_result``
persists a paid reflection but before that Step's ``OptimStepResult`` is
stored, the resumed Step loads the reflection from the durable effect cache,
marks it replayed, and suppresses its usage -- while no Step Result carries
the original. That reflection's spend is absent from the run report for good.
The window is a crash between two writes of one Step, so a run that completes
its Steps normally is unaffected; a run resumed from a mid-Step crash may
understate proposer spend by the reflections that Step had already paid for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

__all__ = [
    "COST_REPORT_SCHEMA",
    "COST_REPORT_SCHEMA_VERSION",
    "CostRole",
    "ProposerCallUsage",
    "RoleCost",
    "RunCostReport",
    "UsageObservation",
    "aggregate_role_cost",
]

#: Persisted-format contract. The exact wire fields and version are pinned by
#: ``tests/test_run_cost_report_golden.py``; never derive them from these
#: model attribute names.
COST_REPORT_SCHEMA = "whetstone.optim_run_cost"
COST_REPORT_SCHEMA_VERSION = 1


@verify(UNIQUE)
class CostRole(StrEnum):
    """Which model a call paid for."""

    #: Calls evaluating a candidate against the task set.
    TASK_MODEL = "task_model"
    #: Calls asking the proposer or reflection model for a new candidate.
    PROPOSER = "proposer"


class ProposerCallUsage(BaseModel):
    """Usage for one proposer-model call, carried on the Step Result.

    A proposer call's usage has no evaluation row to live on, so the Step
    Result records it directly. Every optimizer reports through this one
    shape, which keeps run-level spend re-derivable from the persisted Step
    Results alone rather than from three different state layouts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    #: Identity of the provider call this usage came from: the reflection
    #: effect's request hash for GEPA, the proposer's logical call id for
    #: COPRO and MIPROv2. Run cost de-duplicates on it exactly as the
    #: task-model side de-duplicates on an evidence ref, so a call two Step
    #: Results both report -- a replay, a resumed Step -- is counted once.
    #: Empty only for a call whose source recorded no identity, which is then
    #: counted every time it appears.
    call_id: str = ""
    #: Directional token counts, absent when the provider reported no
    #: breakdown. ``None`` is not zero: a call the provider priced without
    #: splitting tokens by direction has unknown tokens, and normalizing that
    #: to ``0`` would present an incomplete token total as complete. An
    #: absent count makes the call a ``rows_missing_token_breakdown`` row.
    prompt_tokens: StrictInt | None = None
    completion_tokens: StrictInt | None = None
    #: Provider-reported price for this call, absent when none was reported.
    usd: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> ProposerCallUsage:
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.usd is not None and self.usd < 0:
            raise ValueError("usd must be non-negative")
        return self

    def observation(self) -> UsageObservation:
        missing = self.prompt_tokens is None and self.completion_tokens is None
        return UsageObservation(
            input_tokens=self.prompt_tokens or 0,
            output_tokens=self.completion_tokens or 0,
            usd=self.usd,
            missing_token_breakdown=missing,
        )


@dataclass(frozen=True, slots=True)
class UsageObservation:
    """One observed provider call's usage, as recorded in durable evidence.

    ``usd`` is the provider-reported price for this single call, or ``None``
    when the provider reported none. It is never estimated.

    ``cached`` marks a call the prompt cache replayed. Its tokens and price
    belong to the original call, which was already counted, so a cached
    observation contributes only to ``RoleCost.cached_calls``.

    ``missing_token_breakdown`` marks a billable call the provider priced (or
    otherwise evidenced) without splitting tokens by direction. It counts as
    a call and contributes its price, but no tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None
    cached: bool = False
    missing_token_breakdown: bool = False


class RoleCost(BaseModel):
    """Totals for every call one role made during a run."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    #: Billable provider calls observed for this role: calls the run
    #: actually paid a provider for. A call the prompt cache replayed is not
    #: billable and is reported in ``cached_calls`` instead, so an optimizer
    #: that reuses cached evaluations reports fewer ``calls`` for the same
    #: evaluation volume. Compare optimizers on ``calls + cached_calls`` for
    #: work done, and on ``calls`` alone for spend.
    calls: StrictInt = 0
    input_tokens: StrictInt = 0
    output_tokens: StrictInt = 0
    #: Calls that carried a provider-reported price, and those that did not.
    #: Their sum is ``calls``, so a reader can see how much of the run the
    #: ``usd`` total would have covered even when it is absent.
    priced_calls: StrictInt = 0
    unpriced_calls: StrictInt = 0
    #: Calls the prompt cache served from a stored result. They contribute no
    #: tokens, no price, and no ``calls``, because the original call already
    #: did. Reported so a reader can tell a cheap run from a small one.
    cached_calls: StrictInt = 0
    #: Billable calls whose provider reported a price or a total but no
    #: per-direction token split. They count in ``calls`` and in ``usd``;
    #: their tokens are simply not in ``input_tokens``/``output_tokens``, so
    #: a nonzero count means the token totals understate the real usage.
    rows_missing_token_breakdown: StrictInt = 0
    #: Total price, present only when ``unpriced_calls`` is zero and at least
    #: one call was made. Absent means "not knowable", never "zero".
    usd: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> RoleCost:
        for name in (
            "calls",
            "input_tokens",
            "output_tokens",
            "priced_calls",
            "unpriced_calls",
            "cached_calls",
            "rows_missing_token_breakdown",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.priced_calls + self.unpriced_calls != self.calls:
            raise ValueError(
                "priced_calls and unpriced_calls must sum to calls"
            )
        if self.rows_missing_token_breakdown > self.calls:
            raise ValueError(
                "rows_missing_token_breakdown cannot exceed calls"
            )
        if self.usd is not None:
            if self.unpriced_calls:
                raise ValueError(
                    "usd must be absent when any call lacks a price"
                )
            if not self.calls:
                raise ValueError("usd requires at least one priced call")
            if self.usd < 0:
                raise ValueError("usd must be non-negative")
        return self


def aggregate_role_cost(observations: tuple[UsageObservation, ...]) -> RoleCost:
    """Total one role's observed calls, withholding ``usd`` when incomplete.

    A cached observation is excluded from every billable total -- calls,
    tokens, and ``usd`` alike -- and reported only as a ``cached_calls``
    count, because the call it replays was already counted when it was paid
    for. In particular a cached call never makes ``usd`` incomplete.
    """
    cached = tuple(item for item in observations if item.cached)
    billable = tuple(item for item in observations if not item.cached)
    priced = tuple(item for item in billable if item.usd is not None)
    unpriced_count = len(billable) - len(priced)
    complete = bool(billable) and not unpriced_count
    return RoleCost(
        calls=len(billable),
        input_tokens=sum(item.input_tokens for item in billable),
        output_tokens=sum(item.output_tokens for item in billable),
        priced_calls=len(priced),
        unpriced_calls=unpriced_count,
        cached_calls=len(cached),
        rows_missing_token_breakdown=sum(
            1 for item in billable if item.missing_token_breakdown
        ),
        usd=(
            sum(item.usd for item in priced if item.usd is not None)
            if complete
            else None
        ),
    )


class RunCostReport(BaseModel):
    """Per-role spend for one optimization run.

    Serialized into ``OptimResult.cost``. ``schema_version`` travels with the
    payload so a stored result stays readable after the format moves on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: StrictInt = COST_REPORT_SCHEMA_VERSION
    task_model: RoleCost = Field(default_factory=RoleCost)
    proposer: RoleCost = Field(default_factory=RoleCost)

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
