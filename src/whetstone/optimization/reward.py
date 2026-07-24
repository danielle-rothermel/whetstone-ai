"""The reusable Reward Policy and Reward contract.

A :class:`RewardPolicy` is a reusable, versioned, optimizer-facing rule that
maps *internal* evaluation evidence — internal Rollout Aggregates or an
Objective Vector — to a named :class:`Reward`. The same policy contract is
referenced by proposal-only optimizer Configs and by tool-using Tool Configs;
it is addressed by its Identity Hash so both paths cite one policy.

Two invariants are load-bearing and tested:

* **Reward names its policy and inputs.** Every produced Reward carries the
  Reward Policy Identity Hash it was computed under and a citation of the exact
  internal inputs (the ``(name, value)`` aggregate/objective terms) it
  scalarized. A Reward can never be inferred from provider cost, a native
  metric scalar, or an anonymous tool return.

* **Official evaluation computes no Reward.** A Reward may only be produced by
  applying a Reward Policy to evidence whose Evaluation Role is ``internal``.
  :func:`apply_reward_policy` refuses any ``official`` role input, and there is
  no other constructor path for a Reward — the absence of Reward on the
  official path is enforced here, not merely by convention.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.identity import compute_identity_hash

__all__ = [
    "REWARD_POLICY_SCHEMA",
    "REWARD_POLICY_SCHEMA_VERSION",
    "MissingDataPolicy",
    "OfficialRewardError",
    "Reward",
    "RewardInputCitation",
    "RewardPolicy",
    "RewardTerm",
    "apply_reward_policy",
]

REWARD_POLICY_SCHEMA = "whetstone.reward_policy"
REWARD_POLICY_SCHEMA_VERSION = 1


class MissingDataPolicy(StrEnum):
    """How a Reward Policy treats a missing or invalid input term."""

    # A missing/invalid required term makes the Reward not computable: the
    # policy raises rather than silently substituting a value.
    FAIL = "fail"
    # A missing/invalid term contributes its declared direction-worst value.
    WORST = "worst"
    # A missing/invalid term is skipped (its weight is dropped).
    SKIP = "skip"


class RewardTerm(BaseModel):
    """One weighted, direction-bearing term of a Reward Policy.

    ``name`` selects one internal Rollout Aggregate or Objective by name;
    ``maximize`` is its direction; ``weight`` is its scalarization weight. A
    term is identity-bearing: changing any field changes the policy identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    weight: float
    maximize: StrictBool = True
    # The direction-worst value used when the missing-data policy is WORST.
    worst_value: float = 0.0

    @model_validator(mode="after")
    def _validate(self) -> RewardTerm:
        if not self.name:
            raise ValueError("RewardTerm name must be non-empty")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "maximize": self.maximize,
            "worst_value": self.worst_value,
        }


class RewardPolicy(BaseModel):
    """A reusable optimizer-facing scalarization rule.

    Maps named internal aggregate/objective terms to a single scalar Reward by
    a weighted linear combination (each term negated when it minimizes). It is
    versioned and addressed by its Identity Hash. The identity excludes nothing
    that changes the mapping and includes nothing evidence-specific.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_name: StrictStr
    reward_name: StrictStr = "reward"
    terms: tuple[RewardTerm, ...]
    missing_data: MissingDataPolicy = MissingDataPolicy.FAIL

    @model_validator(mode="after")
    def _validate(self) -> RewardPolicy:
        if not self.policy_name:
            raise ValueError("policy_name must be non-empty")
        if not self.reward_name:
            raise ValueError("reward_name must be non-empty")
        if not self.terms:
            raise ValueError("a Reward Policy must have at least one term")
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names):
            raise ValueError("Reward Policy term names must be unique")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "reward_name": self.reward_name,
            "terms": [term.identity_payload() for term in self.terms],
            "missing_data": self.missing_data.value,
        }

    def identity_hash(self) -> str:
        """The Reward Policy Identity Hash (full SHA-256)."""
        return compute_identity_hash(
            schema=REWARD_POLICY_SCHEMA,
            schema_version=REWARD_POLICY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class RewardInputCitation(BaseModel):
    """The exact internal input a Reward scalarized for one policy term."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    # None means the term was missing/invalid in the supplied evidence.
    value: float | None
    contributed: float
    was_missing: StrictBool = False


class Reward(BaseModel):
    """A named optimizer-facing value produced by applying a Reward Policy.

    Every Reward names its policy (by Identity Hash), the internal Evaluation
    Role its inputs carried, and the exact input terms it scalarized. There is
    no unnamed/anonymous scalar path: the value only exists as this typed,
    cited record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reward_name: StrictStr
    value: float
    reward_policy_hash: StrictStr
    # The Evaluation Role the evidence carried (always ``internal``).
    evidence_role: EvaluationRole
    input_citations: tuple[RewardInputCitation, ...]
    # Optional correlation to the internal evaluation evidence that fed it.
    evidence_ref_content_hash: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> Reward:
        if not self.reward_name:
            raise ValueError("reward_name must be non-empty")
        if self.evidence_role is not EvaluationRole.INTERNAL:
            raise ValueError(
                "a Reward may only cite evidence with the internal "
                "Evaluation Role; official evaluation computes no Reward"
            )
        if not self.input_citations:
            raise ValueError("a Reward must cite at least one input term")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OfficialRewardError(ValueError):
    """A Reward Policy was applied to official-role evidence."""


def apply_reward_policy(
    policy: RewardPolicy,
    *,
    aggregates: dict[str, float | None],
    evidence_role: EvaluationRole,
    evidence_ref_content_hash: str | None = None,
    provenance_ordinal: int | None = None,
) -> Reward:
    """Apply a Reward Policy to internal evaluation evidence.

    ``aggregates`` maps aggregate/objective names to their measured internal
    values (``None`` marks a missing/invalid term). ``evidence_role`` MUST be
    ``internal``: this is the sole Reward constructor, so refusing an official
    role here is what makes "official evaluation computes no Reward" true by
    construction rather than by convention.
    """
    if evidence_role is not EvaluationRole.INTERNAL:
        raise OfficialRewardError(
            "apply_reward_policy refuses official-role evidence: official "
            "evaluation computes Objective Vectors/Aggregates, never Reward"
        )

    total = 0.0
    citations: list[RewardInputCitation] = []
    for term in policy.terms:
        raw = aggregates.get(term.name)
        missing = raw is None
        if missing:
            if policy.missing_data is MissingDataPolicy.FAIL:
                raise ValueError(
                    f"Reward Policy term {term.name!r} is missing and the "
                    "missing-data policy is FAIL"
                )
            if policy.missing_data is MissingDataPolicy.SKIP:
                citations.append(
                    RewardInputCitation(
                        name=term.name,
                        value=None,
                        contributed=0.0,
                        was_missing=True,
                    )
                )
                continue
            used = term.worst_value
        else:
            used = float(raw)  # type: ignore[arg-type]
        signed = used if term.maximize else -used
        contribution = term.weight * signed
        total += contribution
        citations.append(
            RewardInputCitation(
                name=term.name,
                value=None if missing else used,
                contributed=contribution,
                was_missing=missing,
            )
        )

    return Reward(
        reward_name=policy.reward_name,
        value=total,
        reward_policy_hash=policy.identity_hash(),
        evidence_role=evidence_role,
        input_citations=tuple(citations),
        evidence_ref_content_hash=evidence_ref_content_hash,
        provenance_ordinal=provenance_ordinal,
    )
