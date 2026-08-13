from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from whetstone.core.identity import ImmutableJsonObject

__all__ = [
    "SandboxCandidateMutation",
    "SandboxCoproSeedTranscript",
    "SandboxProposalCall",
    "SandboxProposalDraft",
    "SandboxRoundPlan",
]


class SandboxRoundPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    iteration: StrictInt
    proposal_mode: StrictStr
    proposal_count: StrictInt
    include_initial_candidate: bool = False


class SandboxProposalCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_mode: StrictStr
    request_ordinal: StrictInt
    mutation_field: StrictStr
    base_template: StrictStr
    prompt: StrictStr
    context_keys: tuple[str, ...]


class SandboxProposalDraft(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: StrictInt
    template: StrictStr
    failed: bool = False
    failure_message: StrictStr | None = None


class SandboxCandidateMutation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: StrictInt
    candidate_id: StrictStr
    template: StrictStr
    disposition: StrictStr
    reason: StrictStr | None = None


class SandboxCoproSeedTranscript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_prompt: StrictStr
    breadth: StrictInt
    depth: StrictInt
    round_plan: SandboxRoundPlan
    proposal_call: SandboxProposalCall
    drafts: tuple[SandboxProposalDraft, ...]
    mutations: tuple[SandboxCandidateMutation, ...]
    contract: ImmutableJsonObject
