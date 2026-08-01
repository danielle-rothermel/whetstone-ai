"""Optimizer-owned proposer route and immutable proposal evidence."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.optimization.identity import (
    FiniteFloat,
    IdentityHash,
    IdentityRef,
    ImmutableJsonObject,
    NonEmptyId,
    NonNegativeInt,
    TerminalFailure,
    compute_identity_hash,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.schema import CandidateRef

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
    """Exact proposer Provider Call Config plus finite draft temperature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_call_config: IdentityRef
    temperature: FiniteFloat = FiniteFloat(1.0)

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity contract: spell every key explicitly and pin the
        # complete payload plus digest in golden tests.
        return {
            "provider_call_config": {
                "record_ref": {
                    "schema_name": str(
                        self.provider_call_config.record_ref.schema_name
                    ),
                    "content_hash": str(
                        self.provider_call_config.record_ref.content_hash
                    ),
                },
                "identity_hash": str(self.provider_call_config.identity_hash),
            },
            "temperature": float(self.temperature),
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=PROPOSER_CONFIG_SCHEMA,
            schema_version=PROPOSER_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class ProposalRequest(BaseModel):
    """One proposer request with exact base and immutable JSON context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_mode: NonEmptyId
    request_ordinal: NonNegativeInt
    base_candidate: CandidateRef
    context: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    @model_validator(mode="after")
    def _validate(self) -> ProposalRequest:
        _ = self.base_template
        return self

    @property
    def base_template(self) -> str:
        try:
            value = self.base_candidate.record.payload[MUTATION_FIELD]
        except KeyError as error:
            raise ValueError(
                "base candidate payload must contain the "
                f"{MUTATION_FIELD!r} mutation field"
            ) from error
        if type(value) is not str:
            raise ValueError(
                f"base candidate {MUTATION_FIELD!r} mutation field "
                "must be a string"
            )
        return value


class ProposalDraft(BaseModel):
    """Exactly one successful draft or one shared terminal failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template: str = ""
    terminal_failure: TerminalFailure | None = None
    request_evidence: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    response_evidence: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    usage: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    cost: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate(self) -> ProposalDraft:
        if self.terminal_failure is not None:
            if self.template:
                raise ValueError("a failed ProposalDraft carries no template")
        elif not self.template:
            raise ValueError(
                "a successful ProposalDraft requires a non-empty template"
            )
        return self

    @property
    def failed(self) -> bool:
        return self.terminal_failure is not None

    @classmethod
    def failure(
        cls,
        *,
        detail: str,
        request_evidence: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> ProposalDraft:
        return cls(
            terminal_failure=TerminalFailure(
                code="proposal_failed",
                message=detail,
            ),
            request_evidence=request_evidence or {},
            response_evidence={"finish": "failed"},
            usage=usage or {},
            cost=0.0,
        )


class ProposerTransport(Protocol):
    """Draft exactly ``count`` proposals without performing evaluation."""

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]: ...


class FakeProposerTransport:
    """Scripted deterministic proposer transport for contract tests."""

    def __init__(
        self,
        script: dict[tuple[str, int], tuple[str, ...]],
        *,
        default: tuple[str, ...] = (),
    ) -> None:
        self._script = dict(script)
        self._default = default
        self.calls: list[tuple[IdentityHash, ProposalRequest, int]] = []

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]:
        if type(count) is not int or count < 0:
            raise ValueError("proposal draft count must be nonnegative")
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
