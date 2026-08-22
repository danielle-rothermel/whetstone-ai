"""Run-level spend aggregated from durable evidence.

``OptimResult.cost`` reports what one optimization run spent, split by the
role that spent it: the task model evaluating candidates, and the proposer
model drafting them. The Step 10 validation protocol compares optimizers on
this record, so its wire keys are a persisted format with a golden test
pinning the exact literals.

Two rules keep the record honest rather than merely populated.

First, every total is aggregated from evidence in the object store, never
from an in-memory counter. A resumed run, a platform run, and an in-process
run therefore report the same number, because they all re-derive it from the
same persisted records.

Second, ``usd`` is present only when *every* contributing call carried a
price. dr-providers reports a per-call price only when the provider returns
one (OpenRouter does; most OpenAI-compatible endpoints do not), so a partial
sum would understate spend while looking authoritative. When any call lacks a
price the field is absent and ``priced_calls``/``unpriced_calls`` show the
split. Whetstone owns no pricing table and infers no prices.
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

    prompt_tokens: StrictInt = 0
    completion_tokens: StrictInt = 0
    #: Provider-reported price for this call, absent when none was reported.
    usd: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> ProposerCallUsage:
        for name in ("prompt_tokens", "completion_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.usd is not None and self.usd < 0:
            raise ValueError("usd must be non-negative")
        return self

    def observation(self) -> UsageObservation:
        return UsageObservation(
            input_tokens=self.prompt_tokens,
            output_tokens=self.completion_tokens,
            usd=self.usd,
        )


@dataclass(frozen=True, slots=True)
class UsageObservation:
    """One observed provider call's usage, as recorded in durable evidence.

    ``usd`` is the provider-reported price for this single call, or ``None``
    when the provider reported none. It is never estimated.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None


class RoleCost(BaseModel):
    """Totals for every call one role made during a run."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    calls: StrictInt = 0
    input_tokens: StrictInt = 0
    output_tokens: StrictInt = 0
    #: Calls that carried a provider-reported price, and those that did not.
    #: Their sum is ``calls``, so a reader can see how much of the run the
    #: ``usd`` total would have covered even when it is absent.
    priced_calls: StrictInt = 0
    unpriced_calls: StrictInt = 0
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
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.priced_calls + self.unpriced_calls != self.calls:
            raise ValueError(
                "priced_calls and unpriced_calls must sum to calls"
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
    """Total one role's observed calls, withholding ``usd`` when incomplete."""
    priced = tuple(item for item in observations if item.usd is not None)
    unpriced_count = len(observations) - len(priced)
    complete = bool(observations) and not unpriced_count
    return RoleCost(
        calls=len(observations),
        input_tokens=sum(item.input_tokens for item in observations),
        output_tokens=sum(item.output_tokens for item in observations),
        priced_calls=len(priced),
        unpriced_calls=unpriced_count,
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
