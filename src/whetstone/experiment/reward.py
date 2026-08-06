from __future__ import annotations

import math
from collections.abc import Mapping
from enum import UNIQUE, StrEnum, verify
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    FiniteFloat,
    IdentityHash,
    NonEmptyId,
    NonNegativeInt,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.core.roles import EvaluationRole

__all__ = [
    "REWARD_POLICY_SCHEMA",
    "REWARD_POLICY_SCHEMA_VERSION",
    "REWARD_SCHEMA",
    "MissingDataPolicy",
    "OfficialRewardError",
    "Reward",
    "RewardInputCitation",
    "RewardPolicy",
    "RewardRef",
    "RewardTerm",
    "apply_reward_policy",
    "reward_reference",
]

REWARD_SCHEMA = "whetstone.reward"
REWARD_POLICY_SCHEMA = "whetstone.reward_policy"
REWARD_POLICY_SCHEMA_VERSION = 1


def _require_ordered_sequence(value: Any, info: ValidationInfo) -> Any:
    """Accept only the deliberate Python representations of a JSON array."""
    if type(value) not in (list, tuple):
        raise ValueError(
            f"{info.field_name} must be an ordered tuple or JSON array"
        )
    return value


@verify(UNIQUE)
class MissingDataPolicy(StrEnum):
    FAIL = "fail"
    WORST = "worst"
    SKIP = "skip"


class RewardTerm(BaseModel):
    """One weighted direction-bearing term."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: NonEmptyId
    weight: FiniteFloat
    maximize: StrictBool = True
    worst_value: FiniteFloat = FiniteFloat(0.0)

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are a pinned wire contract.
        return {
            "name": self.name,
            "weight": self.weight,
            "maximize": self.maximize,
            "worst_value": self.worst_value,
        }


class RewardPolicy(BaseModel):
    """Reusable finite scalarization rule addressed by Identity Hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_name: NonEmptyId
    reward_name: NonEmptyId = NonEmptyId("reward")
    terms: tuple[RewardTerm, ...]
    missing_data: MissingDataPolicy = MissingDataPolicy.FAIL

    @field_validator("terms", mode="before")
    @classmethod
    def _validate_terms_input(cls, value: Any, info: ValidationInfo) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> RewardPolicy:
        if not self.terms:
            raise ValueError("a Reward Policy must have at least one term")
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names):
            raise ValueError("Reward Policy term names must be unique")
        return self

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are a pinned wire contract.
        return {
            "policy_name": self.policy_name,
            "reward_name": self.reward_name,
            "terms": [term.identity_payload() for term in self.terms],
            "missing_data": self.missing_data.value,
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=REWARD_POLICY_SCHEMA,
            schema_version=REWARD_POLICY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class RewardInputCitation(BaseModel):
    """The exact finite or missing input used for one policy term."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: NonEmptyId
    value: FiniteFloat | None
    contributed: FiniteFloat
    was_missing: StrictBool = False

    @model_validator(mode="after")
    def _validate(self) -> RewardInputCitation:
        if self.was_missing != (self.value is None):
            raise ValueError(
                "Reward citation missingness must match its exact value"
            )
        return self


def _scalarize_term(
    term: RewardTerm,
    *,
    value: FiniteFloat | float | None,
    was_missing: bool,
    missing_data: MissingDataPolicy,
) -> float:
    if was_missing:
        if missing_data is MissingDataPolicy.FAIL:
            raise ValueError(
                "a FAIL Reward Policy cannot produce a Reward from "
                "missing input"
            )
        if missing_data is MissingDataPolicy.SKIP:
            return 0.0
        used = float(term.worst_value)
    else:
        assert value is not None
        used = float(value)
    signed = used if term.maximize else -used
    return float(term.weight) * signed


class Reward(BaseModel):
    """Named finite Reward with exact policy identity and evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reward_name: NonEmptyId
    value: FiniteFloat
    reward_policy: RewardPolicy
    evidence_role: EvaluationRole
    input_citations: tuple[RewardInputCitation, ...]
    evidence_refs: tuple[TypedRef, ...]
    provenance_ordinal: NonNegativeInt | None = None

    @field_validator("input_citations", "evidence_refs", mode="before")
    @classmethod
    def _validate_ordered_input(cls, value: Any, info: ValidationInfo) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> Reward:
        if self.evidence_role is not EvaluationRole.INTERNAL:
            raise ValueError(
                "a Reward may only cite evidence with the internal "
                "Evaluation Role; official evaluation computes no Reward"
            )
        if not self.input_citations:
            raise ValueError("a Reward must cite at least one input term")
        if not self.evidence_refs:
            raise ValueError(
                "an internal Reward must cite at least one exact evidence ref"
            )
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("Reward evidence_refs must be unique")
        if self.reward_name != self.reward_policy.reward_name:
            raise ValueError("Reward name must match its exact Reward Policy")
        expected_names = tuple(term.name for term in self.reward_policy.terms)
        actual_names = tuple(
            citation.name for citation in self.input_citations
        )
        if actual_names != expected_names:
            raise ValueError(
                "Reward input citations must exactly match Reward Policy "
                "term names and order"
            )
        expected_total = 0.0
        for term, citation in zip(
            self.reward_policy.terms,
            self.input_citations,
            strict=True,
        ):
            expected_contribution = _scalarize_term(
                term,
                value=citation.value,
                was_missing=bool(citation.was_missing),
                missing_data=self.reward_policy.missing_data,
            )
            if citation.contributed != expected_contribution:
                raise ValueError(
                    "Reward citation contribution must exactly apply its "
                    "Reward Policy term"
                )
            expected_total += expected_contribution
        if self.value != expected_total:
            raise ValueError(
                "Reward value must equal the exact scalarized citation total"
            )
        return self

    @property
    def reward_policy_hash(self) -> IdentityHash:
        return self.reward_policy.identity_hash()

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RewardRef(BaseModel):
    """An exact persisted Reward record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: Reward
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> RewardRef:
        expected = typed_ref_for_record(
            REWARD_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected:
            raise ValueError(
                "Reward record_ref must address the exact Reward record"
            )
        return self


def reward_reference(reward: Reward) -> RewardRef:
    return RewardRef(
        record=reward,
        record_ref=typed_ref_for_record(
            REWARD_SCHEMA, reward.record_content()
        ),
    )


class OfficialRewardError(ValueError):
    """A Reward Policy was applied to official-role evidence."""


def _finite_or_missing(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def apply_reward_policy(
    policy: RewardPolicy,
    *,
    aggregates: Mapping[str, float | None],
    evidence_role: EvaluationRole,
    evidence_refs: tuple[TypedRef, ...],
    provenance_ordinal: int | None = None,
) -> Reward:
    """Apply missing-data policy to absent, invalid, and non-finite inputs."""
    if evidence_role is not EvaluationRole.INTERNAL:
        raise OfficialRewardError(
            "apply_reward_policy refuses official-role evidence: official "
            "evaluation computes Objective Vectors/Aggregates, never Reward"
        )

    total = 0.0
    citations: list[RewardInputCitation] = []
    for term in policy.terms:
        raw = _finite_or_missing(aggregates.get(term.name))
        missing = raw is None
        if missing and policy.missing_data is MissingDataPolicy.FAIL:
            raise ValueError(
                f"Reward Policy term {term.name!r} is missing or invalid "
                "and the missing-data policy is FAIL"
            )
        contribution = _scalarize_term(
            term,
            value=raw,
            was_missing=missing,
            missing_data=policy.missing_data,
        )
        total += contribution
        citations.append(
            RewardInputCitation(
                name=term.name,
                value=None if missing else raw,
                contributed=contribution,
                was_missing=missing,
            )
        )

    return Reward(
        reward_name=policy.reward_name,
        value=total,
        reward_policy=policy,
        evidence_role=evidence_role,
        input_citations=tuple(citations),
        evidence_refs=evidence_refs,
        provenance_ordinal=provenance_ordinal,
    )
