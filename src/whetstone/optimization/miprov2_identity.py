"""MIPROv2 algorithm-local identities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.optimization.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)

MIPROV2_DEMO_SET_SCHEMA = "whetstone.miprov2_demo_set"


class DemoPair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rendered_input: StrictStr
    observed_output: StrictStr


class DemoSetIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: tuple[DemoPair, ...] = ()

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="miprov2.demo-set",
            schema_version=1,
            payload={
                "pairs": [pair.model_dump(mode="json") for pair in self.pairs]
            },
        )


class DemoSetArtifact(BaseModel):
    """Persisted observed demonstrations and their exact source evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    demo_set: DemoSetIdentity
    source_evidence_ref: TypedRef


class DemoSetArtifactRef(BaseModel):
    """A demo identity composed with its persisted artifact reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_hash: StrictStr
    artifact_ref: TypedRef

    @model_validator(mode="after")
    def _identity(self) -> DemoSetArtifactRef:
        require_full_hash(self.identity_hash, field="identity_hash")
        if self.artifact_ref.schema_name != MIPROV2_DEMO_SET_SCHEMA:
            raise ValueError("artifact_ref must reference a MIPROv2 demo set")
        return self


class InstructionIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction_text: StrictStr

    @model_validator(mode="after")
    def _not_empty(self) -> InstructionIdentity:
        if not self.instruction_text:
            raise ValueError("instruction_text must be non-empty")
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="miprov2.instruction",
            schema_version=1,
            payload={"instruction_text": self.instruction_text},
        )


class TrialCombinationIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction_hash: StrictStr
    demo_set_hash: StrictStr

    @model_validator(mode="after")
    def _hashes(self) -> TrialCombinationIdentity:
        require_full_hash(self.instruction_hash, field="instruction_hash")
        require_full_hash(self.demo_set_hash, field="demo_set_hash")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "instruction_hash": self.instruction_hash,
            "demo_set_hash": self.demo_set_hash,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="miprov2.trial-combination",
            schema_version=1,
            payload=self.identity_payload(),
        )


__all__ = [
    "MIPROV2_DEMO_SET_SCHEMA",
    "DemoPair",
    "DemoSetArtifact",
    "DemoSetArtifactRef",
    "DemoSetIdentity",
    "InstructionIdentity",
    "TrialCombinationIdentity",
]
