"""The proposer route: a Provider Call Config the optimizer owns.

Both proposal-only optimizers (COPRO's Proposal LM, MIPROv2's proposal LM)
call a model to draft new encoder ``user_prompt_template`` text. That model is
reached through a **proposer route** — a Provider Call Config reference
*distinct* from the encoder/decoder routes inside an evaluated Rollout graph.

The load-bearing identity fact, per ``copro-run.html`` / ``miprov2-run.html``
and the Workstream-7 table: the proposer route lives in the **optimizer Config
identity**, never in the **graph identity**. A Rollout Variant's ``graph_hash``
folds the encoder/decoder Provider Call Configs; it MUST NOT fold the proposer
route, because changing which model *drafts* a template does not change the
identity of the *materialized* graph that a template is evaluated under. This
module gives the proposer route its own typed Config so the two identity
domains stay separate and testable.

A :class:`FakeProposerTransport` is provided for deterministic-harness tests: a
scripted transport keyed by ``(proposal_mode, request_ordinal)`` that returns
canned template drafts with no network. It never participates in identity.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.optimization.identity import (
    compute_identity_hash,
    require_full_hash,
)

__all__ = [
    "PROPOSER_CONFIG_SCHEMA",
    "PROPOSER_CONFIG_SCHEMA_VERSION",
    "FakeProposerTransport",
    "ProposalDraft",
    "ProposalRequest",
    "ProposerConfig",
    "ProposerTransport",
]

PROPOSER_CONFIG_SCHEMA = "whetstone.proposer_config"
PROPOSER_CONFIG_SCHEMA_VERSION = 1


class ProposerConfig(BaseModel):
    """The optimizer-owned proposer route (a Provider Call Config reference).

    Carries the typed reference and Identity Hash of the **Provider Call
    Config** the Proposal/proposal LM is reached through, plus the generation
    temperature the algorithm pins for drafting. It is deliberately a small
    identity-bearing Config so it can be folded into the **optimizer Config
    identity** — and only there. It is never folded into a graph identity: the
    encoder/decoder routes inside the evaluated graph are separate Provider
    Call Configs with their own hashes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Typed reference + Identity Hash of the proposer's Provider Call Config.
    provider_call_config_ref: StrictStr
    provider_call_config_hash: StrictStr
    # The drafting temperature the algorithm pins (COPRO init_temperature,
    # MIPROv2 proposal_temperature). Part of the proposer route identity.
    temperature: float = 1.0

    @model_validator(mode="after")
    def _validate(self) -> ProposerConfig:
        if not self.provider_call_config_ref:
            raise ValueError("provider_call_config_ref must be non-empty")
        require_full_hash(
            self.provider_call_config_hash,
            field="provider_call_config_hash",
        )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "provider_call_config_ref": self.provider_call_config_ref,
            "provider_call_config_hash": self.provider_call_config_hash,
            "temperature": self.temperature,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=PROPOSER_CONFIG_SCHEMA,
            schema_version=PROPOSER_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class ProposalRequest(BaseModel):
    """A single call to the proposer route to draft template text.

    It carries the algorithm's proposal mode/purpose, the ordinal within the
    run (so a scripted transport can key deterministic responses), the base
    candidate's current template text, and an opaque, JSON-only ``context``
    (e.g. the Reward-ranked history summary COPRO conditions on, or the seed
    instruction MIPROv2 mutates). No runtime handle: strict JSON only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_mode: StrictStr
    request_ordinal: StrictInt
    base_ref: StrictStr
    base_template: StrictStr = ""
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ProposalRequest:
        if not self.proposal_mode:
            raise ValueError("proposal_mode must be non-empty")
        if self.request_ordinal < 0:
            raise ValueError("request_ordinal cannot be negative")
        return self


class ProposalDraft(BaseModel):
    """One successful template or one explicitly failed draft slot.

    ``template`` is the mutated ``user_prompt_template`` text. The evidence
    fields carry the Proposal LM request/response/usage/cost provenance the
    Step Result records — never a score and never a Reward.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    template: StrictStr = ""
    failed: bool = False
    failure_detail: StrictStr | None = None
    request_evidence: dict[str, Any] = Field(default_factory=dict)
    response_evidence: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> ProposalDraft:
        if self.failed:
            if self.template:
                raise ValueError("a failed ProposalDraft carries no template")
            if not self.failure_detail:
                raise ValueError("a failed ProposalDraft requires detail")
        elif not self.template:
            raise ValueError(
                "a successful ProposalDraft requires a non-empty template"
            )
        elif self.failure_detail is not None:
            raise ValueError(
                "a successful ProposalDraft carries no failure detail"
            )
        return self

    @classmethod
    def failure(
        cls,
        *,
        detail: str,
        request_evidence: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> ProposalDraft:
        return cls(
            failed=True,
            failure_detail=detail,
            request_evidence=request_evidence or {},
            response_evidence={"finish": "failed"},
            usage=usage or {},
            cost=0.0,
        )


class ProposerTransport(Protocol):
    """The proposer route transport: draft ``count`` templates for a request.

    A real transport folds the :class:`ProposerConfig`'s Provider Call Config
    and drives dr-providers; the test double is scripted. Either way it returns
    exactly ``count`` :class:`ProposalDraft`s and performs no evaluation.
    """

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]: ...


class FakeProposerTransport:
    """A scripted, deterministic proposer transport for harness tests.

    Responses are keyed by ``(proposal_mode, request_ordinal)`` -> a tuple of
    template strings; the transport slices/pads to the requested ``count`` and
    records every drafting call so a test can assert the proposer route (not an
    encoder/decoder route) produced the templates. It touches no network and is
    never part of any identity.
    """

    def __init__(
        self,
        script: dict[tuple[str, int], tuple[str, ...]],
        *,
        default: tuple[str, ...] = (),
    ) -> None:
        self._script = dict(script)
        self._default = default
        self.calls: list[tuple[str, ProposalRequest, int]] = []

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]:
        # Record which proposer route was used (by its Identity Hash) so a test
        # proves the proposer route is distinct from encoder/decoder routes.
        self.calls.append((config.identity_hash(), request, count))
        templates = self._script.get(
            (request.proposal_mode, request.request_ordinal), self._default
        )
        evidence_base = {
            "proposal_mode": request.proposal_mode,
            "request_ordinal": request.request_ordinal,
            "temperature": config.temperature,
        }
        drafts: list[ProposalDraft] = []
        for index in range(count):
            if index < len(templates):
                text = templates[index]
            else:
                # Deterministic pad so a short script never underfills.
                text = (
                    f"{request.base_template}::pad::"
                    f"{request.request_ordinal}:{index}"
                )
            if not text:
                drafts.append(
                    ProposalDraft.failure(
                        detail="scripted proposer produced an empty draft",
                        request_evidence={
                            **evidence_base,
                            "draft_index": index,
                        },
                        usage={"proposer_calls": 1},
                    )
                )
                continue
            drafts.append(
                ProposalDraft(
                    template=text,
                    request_evidence=evidence_base,
                    response_evidence={"draft_index": index},
                    usage={"proposer_calls": 1},
                    cost=0.0,
                )
            )
        return tuple(drafts)
