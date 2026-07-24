"""Rollout measurement and execution identities.

Three settled keys, per the Target System Contract:

``RolloutKey``
    The semantic measurement-cell key
    ``(graph_hash, eval_config_hash, task_identity, repeat_id)``. Its
    config-hash fields carry *full* 64-char Identity Hashes. It is
    independent of execution authority, Evaluation Context, and
    materialization lineage.

``RolloutExecutionKey``
    ``(rollout_key, evaluation_context_id)`` — the terminal execution
    identity used as the Rollout Work Item and Result Store uniqueness key.

``EvaluationContext``
    The immutable execution-time binding: one ordinary Eval Config
    reference, one ``internal|official`` Evaluation Role field, authority
    (only when official), Evaluation Campaign, operational-policy references
    (Provider Execution Policy, Retry Policy, ...), realized-environment
    attestation, and record-local provenance fields. Its stable
    ``evaluation_context_id`` feeds the Rollout Execution Key. Semantic
    sandbox image/version lives in the Evaluation Procedure Config, never
    here.
"""

from __future__ import annotations

from enum import StrEnum

from dr_serialize import build_identity_document, identity_document_hash
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

# Whetstone-owned identity schema for the Evaluation Context id. The id is a
# full Identity Hash over the Context's identity-bearing fields.
EVALUATION_CONTEXT_SCHEMA = "whetstone.evaluation_context"
EVALUATION_CONTEXT_SCHEMA_VERSION = 1

_HEX = frozenset("0123456789abcdef")


def _require_full_hash(value: str, *, field: str) -> str:
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(
            f"{field} must be a full 64-char lowercase SHA-256 hash, "
            f"got {value!r}"
        )
    return value


class EvaluationRole(StrEnum):
    """Closed Evaluation Role field on an Evaluation Context."""

    INTERNAL = "internal"
    OFFICIAL = "official"


class RolloutKey(BaseModel):
    """Semantic measurement-cell key. Full config hashes; no lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    graph_hash: StrictStr
    eval_config_hash: StrictStr
    task_identity: StrictStr
    repeat_id: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RolloutKey:
        _require_full_hash(self.graph_hash, field="graph_hash")
        _require_full_hash(self.eval_config_hash, field="eval_config_hash")
        if not self.task_identity:
            raise ValueError("task_identity must be non-empty")
        if not self.repeat_id:
            raise ValueError("repeat_id must be non-empty")
        return self


class RolloutExecutionKey(BaseModel):
    """Terminal execution identity: ``(rollout_key, evaluation_context_id)``.

    Serves as the Rollout Work Item and Result Store uniqueness key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rollout_key: RolloutKey
    evaluation_context_id: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RolloutExecutionKey:
        if not self.evaluation_context_id:
            raise ValueError("evaluation_context_id must be non-empty")
        return self


class EnvironmentAttestation(BaseModel):
    """Realized-environment attestation carried by an Evaluation Context.

    Records the released dependency package versions, resolved code
    provenance revision, and the realized runtime environment identity.
    Semantic sandbox image/version is NOT here — that is Evaluation
    Procedure Config.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dependency_versions: tuple[tuple[str, str], ...] = ()
    code_revision: StrictStr | None = None
    runtime_identity: StrictStr | None = None


class EvaluationContext(BaseModel):
    """Immutable execution-time binding for one Rollout Variant execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Ordinary Eval Config reference (the exact same one under either role).
    eval_config_hash: StrictStr
    # One closed internal|official field. Officialness comes from authority.
    role: EvaluationRole
    # Authority principal, required only when the role is official.
    authority: StrictStr | None = None
    # Evaluation Campaign namespace.
    campaign: StrictStr
    # Operational-policy references (not identity-bearing config).
    provider_execution_policy_ref: StrictStr | None = None
    retry_policy_ref: StrictStr | None = None
    operational_policy_refs: tuple[str, ...] = ()
    # Realized-environment attestation.
    environment: EnvironmentAttestation = Field(
        default_factory=EnvironmentAttestation
    )
    # Record-local provenance fields (typed, local to this schema; no
    # universal provenance class).
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> EvaluationContext:
        _require_full_hash(self.eval_config_hash, field="eval_config_hash")
        if not self.campaign:
            raise ValueError("campaign must be non-empty")
        if self.role is EvaluationRole.OFFICIAL and not self.authority:
            raise ValueError(
                "authority is required when the Evaluation Role is official"
            )
        if self.role is EvaluationRole.INTERNAL and self.authority:
            raise ValueError(
                "authority must be absent when the Evaluation Role is internal"
            )
        return self

    def identity_payload(self) -> dict[str, object]:
        """The identity-bearing fields of the Evaluation Context."""
        return {
            "eval_config_hash": self.eval_config_hash,
            "role": self.role.value,
            "authority": self.authority,
            "campaign": self.campaign,
            "provider_execution_policy_ref": (
                self.provider_execution_policy_ref
            ),
            "retry_policy_ref": self.retry_policy_ref,
            "operational_policy_refs": list(self.operational_policy_refs),
            "environment": {
                "dependency_versions": [
                    list(pair) for pair in self.environment.dependency_versions
                ],
                "code_revision": self.environment.code_revision,
                "runtime_identity": self.environment.runtime_identity,
            },
            "provenance_note": self.provenance_note,
            "provenance_ordinal": self.provenance_ordinal,
        }

    def evaluation_context_id(self) -> str:
        """The stable Evaluation Context id (a full Identity Hash)."""
        document = build_identity_document(
            schema=EVALUATION_CONTEXT_SCHEMA,
            schema_version=EVALUATION_CONTEXT_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )
        return identity_document_hash(document)


def rollout_execution_key(
    *,
    rollout_key: RolloutKey,
    context: EvaluationContext,
) -> RolloutExecutionKey:
    """Build the Rollout Execution Key from a measurement cell and Context.

    The Context's ordinary Eval Config MUST match the Rollout Key's
    ``eval_config_hash`` (they name the same measurement cell).
    """
    if context.eval_config_hash != rollout_key.eval_config_hash:
        raise ValueError(
            "Evaluation Context eval_config_hash "
            f"{context.eval_config_hash!r} does not match Rollout Key "
            f"eval_config_hash {rollout_key.eval_config_hash!r}"
        )
    return RolloutExecutionKey(
        rollout_key=rollout_key,
        evaluation_context_id=context.evaluation_context_id(),
    )


__all__ = [
    "EVALUATION_CONTEXT_SCHEMA",
    "EVALUATION_CONTEXT_SCHEMA_VERSION",
    "EnvironmentAttestation",
    "EvaluationContext",
    "EvaluationRole",
    "RolloutExecutionKey",
    "RolloutKey",
    "rollout_execution_key",
]
