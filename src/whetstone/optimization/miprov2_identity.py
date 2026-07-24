"""MIPROv2's three algorithm-local versioned identity domains.

Per ``optimizer-briefs.md`` §3 ("Identity domains") and ``miprov2-run.html``
("Algorithm-local identity"), MIPROv2 owns three versioned identity domains,
each ``{schema, schema_version, payload}`` hashed with the full lowercase
SHA-256 over canonical identity JSON (through the same dr-serialize lane every
harness identity uses):

* **Demo Set Identity** (``schema=miprov2.demo-set``, v1) — ordered demo pairs,
  each with exact ``ground_truth_code`` and ``encoded_representation``.
* **Instruction Identity** (``schema=miprov2.instruction``, v1) — the exact
  validated ``instruction_text``; excludes attempt nonce, evidence, and cost.
* **Trial Combination Identity** (``schema=miprov2.trial-combination``, v1) —
  the full instruction + demo-set Identity Hashes; excludes trial ID and
  scores, so it groups repeated trials and establishes combination equality.

A Content Hash covers a persisted object for integrity and NEVER substitutes
for these identities.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.optimization.identity import (
    compute_identity_hash,
    require_full_hash,
)

__all__ = [
    "DEMO_SET_SCHEMA",
    "DEMO_SET_SCHEMA_VERSION",
    "INSTRUCTION_SCHEMA",
    "INSTRUCTION_SCHEMA_VERSION",
    "TRIAL_COMBINATION_SCHEMA",
    "TRIAL_COMBINATION_SCHEMA_VERSION",
    "DemoPair",
    "DemoSetIdentity",
    "InstructionIdentity",
    "TrialCombinationIdentity",
]

DEMO_SET_SCHEMA = "miprov2.demo-set"
DEMO_SET_SCHEMA_VERSION = 1
INSTRUCTION_SCHEMA = "miprov2.instruction"
INSTRUCTION_SCHEMA_VERSION = 1
TRIAL_COMBINATION_SCHEMA = "miprov2.trial-combination"
TRIAL_COMBINATION_SCHEMA_VERSION = 1


class DemoPair(BaseModel):
    """One ordered demo pair: exact ground-truth code + encoded rep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ground_truth_code: StrictStr
    encoded_representation: StrictStr


class DemoSetIdentity(BaseModel):
    """A demo-set identity over ordered demo pairs (incl. the empty set)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: tuple[DemoPair, ...] = ()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "pairs": [
                {
                    "ground_truth_code": p.ground_truth_code,
                    "encoded_representation": p.encoded_representation,
                }
                for p in self.pairs
            ]
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=DEMO_SET_SCHEMA,
            schema_version=DEMO_SET_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class InstructionIdentity(BaseModel):
    """An instruction identity over the exact validated instruction text.

    Excludes attempt nonce, proposer evidence, and cost by construction: only
    ``instruction_text`` is identity-bearing, so two proposal-LM attempts that
    produce the same text share one instruction identity (a duplicate consumes
    an attempt but adds no pool member).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction_text: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> InstructionIdentity:
        if self.instruction_text == "":
            raise ValueError("instruction_text must be non-empty")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {"instruction_text": self.instruction_text}

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=INSTRUCTION_SCHEMA,
            schema_version=INSTRUCTION_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class TrialCombinationIdentity(BaseModel):
    """A trial-combination identity over instruction + demo-set hashes.

    Excludes trial ID and scores, so repeated trials that select the same
    (instruction, demo set) share one combination identity — the equality key
    the TPE study groups minibatch observations by.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction_hash: StrictStr
    demo_set_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> TrialCombinationIdentity:
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
            schema=TRIAL_COMBINATION_SCHEMA,
            schema_version=TRIAL_COMBINATION_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )
