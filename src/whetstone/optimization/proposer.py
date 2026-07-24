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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.lm.boundary import (
    PlainPromptAdapter,
    provider_call_request_from_parameters,
    provider_result_from_response,
)
from whetstone.optimization.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.provider.driver import (
    Clock,
    Sleep,
    TransportCall,
    run_provider_call,
)
from whetstone.provider.policy import ProviderExecutionPolicy

if TYPE_CHECKING:
    from dr_providers import ProviderCallConfig, ProviderCallRequest

__all__ = [
    "PROMPT_ADAPTER_SCHEMA",
    "PROMPT_ADAPTER_SCHEMA_VERSION",
    "PROPOSAL_PROMPT_SCHEMA_TAG",
    "PROPOSAL_PROMPT_SCHEMA_VERSION",
    "PROPOSAL_REQUEST_SCHEMA",
    "PROPOSAL_REQUEST_SCHEMA_VERSION",
    "PROPOSER_CONFIG_SCHEMA",
    "PROPOSER_CONFIG_SCHEMA_VERSION",
    "FakeProposerTransport",
    "ProposalDraft",
    "ProposalPromptBuilder",
    "ProposalRequest",
    "ProposerConfig",
    "ProposerTransport",
    "ProviderCallConfigResolver",
    "ProviderProposerTransport",
    "fold_prompt_schema_tag",
    "prompt_adapter_identity_hash",
]

PROPOSER_CONFIG_SCHEMA = "whetstone.proposer_config"
PROPOSER_CONFIG_SCHEMA_VERSION = 1
# Generic proposer-protocol version. Algorithm prompt identities are separate.
PROPOSAL_PROMPT_SCHEMA_VERSION = 2
PROPOSAL_PROMPT_SCHEMA_TAG = "pp2"
PROPOSAL_REQUEST_SCHEMA = "whetstone.proposal_request"
PROPOSAL_REQUEST_SCHEMA_VERSION = 1
PROMPT_ADAPTER_SCHEMA = "whetstone.proposal_prompt_adapter"
PROMPT_ADAPTER_SCHEMA_VERSION = 1


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
    run_id: StrictStr | None = None
    step_index: StrictInt | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ProposalRequest:
        if not self.proposal_mode:
            raise ValueError("proposal_mode must be non-empty")
        if self.request_ordinal < 0:
            raise ValueError("request_ordinal cannot be negative")
        if (self.run_id is None) != (self.step_index is None):
            raise ValueError("run_id and step_index must be supplied together")
        if self.run_id == "":
            raise ValueError("run_id must be non-empty")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=PROPOSAL_REQUEST_SCHEMA,
            schema_version=PROPOSAL_REQUEST_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


ProposalPromptBuilder = Callable[[ProposalRequest], str]


def fold_prompt_schema_tag(route: str) -> str:
    """Fold the reconciled proposal-prompt protocol into route identity."""
    suffix = f"#{PROPOSAL_PROMPT_SCHEMA_TAG}"
    return route if route.endswith(suffix) else f"{route}{suffix}"


def prompt_adapter_identity_hash(adapter: PlainPromptAdapter) -> str:
    """Identify the exact plain-text projection used for proposer prompts."""

    return compute_identity_hash(
        schema=PROMPT_ADAPTER_SCHEMA,
        schema_version=PROMPT_ADAPTER_SCHEMA_VERSION,
        payload=adapter.model_dump(mode="json"),
    )


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
        response_evidence: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        cost: float | None = 0.0,
    ) -> ProposalDraft:
        return cls(
            failed=True,
            failure_detail=detail,
            request_evidence=request_evidence or {},
            response_evidence=response_evidence or {"finish": "failed"},
            usage=usage or {},
            cost=cost,
        )


class ProposerTransport(Protocol):
    """The proposer route transport: draft ``count`` templates for a request.

    A real transport folds the :class:`ProposerConfig`'s Provider Call Config
    and drives dr-providers; the test double is scripted. Either way it returns
    exactly ``count`` :class:`ProposalDraft`s and performs no evaluation.
    """

    @property
    def execution_policy_hash(self) -> str: ...

    @property
    def prompt_adapter_identity_hash(self) -> str: ...

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]: ...


ProviderCallConfigResolver = Callable[[str], "ProviderCallConfig"]


class ProviderProposerTransport:
    """Production proposer route over the Whetstone provider kernel.

    dr-providers currently projects one semantic generation from one
    :class:`ProviderCallRequest`; it has no typed multi-generation result.
    Therefore one algorithm-level ``draft(..., count=N)`` invocation is
    transparently materialized as ``N`` deterministic logical provider calls
    carrying the identical prompt and controls. This transport-level shape
    differs from DSPy's use of ``n=N``, while preserving COPRO's one proposer
    invocation, exact requested candidate count, completion order, and
    temperature.

    Provider config resolution, physical transport, semantic attempt policy,
    clock, and sleep are all injected. No ambient provider registry,
    credential lookup, retry policy, or network client is consulted here.
    Every slot returns either one raw instruction or one explicit failed
    :class:`ProposalDraft`; a partial or invalid provider batch can therefore
    never be mistaken for a successful, underfilled candidate batch.
    """

    def __init__(
        self,
        *,
        resolve_provider_call_config: ProviderCallConfigResolver,
        transport: TransportCall,
        execution_policy: ProviderExecutionPolicy,
        prompt_adapter: PlainPromptAdapter | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._resolve_provider_call_config = resolve_provider_call_config
        self._transport = transport
        self._execution_policy = execution_policy
        self._prompt_adapter = prompt_adapter or PlainPromptAdapter()
        self._clock = clock
        self._sleep = sleep

    @property
    def execution_policy_hash(self) -> str:
        return self._execution_policy.identity_hash

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return prompt_adapter_identity_hash(self._prompt_adapter)

    def draft(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        if type(count) is not int or count < 1:
            raise ValueError("proposer draft count must be a positive integer")

        prompt = request.context.get("proposal_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "provider proposer requires one nonblank proposal_prompt"
            )

        provider_config = self._resolve_provider_call_config(
            config.provider_call_config_ref
        )
        if provider_config.identity_hash != config.provider_call_config_hash:
            raise ValueError(
                "resolved Provider Call Config hash does not match "
                "Proposer Config"
            )

        provider_request = provider_call_request_from_parameters(
            config=provider_config,
            messages=self._prompt_adapter.messages(user_content=prompt),
            parameters={"temperature": config.temperature},
        )

        drafts = tuple(
            self._draft_slot(
                config=config,
                proposal_request=request,
                provider_request=provider_request,
                count=count,
                slot=slot,
            )
            for slot in range(count)
        )
        if len(drafts) != count:
            raise RuntimeError(
                "provider proposer underfilled its logical batch"
            )
        return drafts

    def _draft_slot(
        self,
        *,
        config: ProposerConfig,
        proposal_request: ProposalRequest,
        provider_request: ProviderCallRequest,
        count: int,
        slot: int,
    ) -> ProposalDraft:
        logical_call_id = (
            f"proposer:{config.identity_hash()}:"
            f"{self.execution_policy_hash}:"
            f"{self.prompt_adapter_identity_hash}:"
            f"{proposal_request.identity_hash()}:{slot}"
        )
        result = run_provider_call(
            request=provider_request,
            policy=self._execution_policy,
            transport=self._transport,
            logical_call_id=logical_call_id,
            clock=self._clock,
            sleep=self._sleep,
        )
        request_evidence = {
            "logical_call_id": logical_call_id,
            "logical_batch_size": count,
            "batch_slot": slot,
            "proposal_mode": proposal_request.proposal_mode,
            "request_ordinal": proposal_request.request_ordinal,
            "proposal_request_identity_hash": (
                proposal_request.identity_hash()
            ),
            "provider_call_config_ref": config.provider_call_config_ref,
            "base_provider_call_config_hash": (
                config.provider_call_config_hash
            ),
            "materialized_provider_call_config_hash": (
                provider_request.config.identity_hash
            ),
            "provider_execution_policy_hash": (self.execution_policy_hash),
            "prompt_adapter": self._prompt_adapter.model_dump(mode="json"),
            "prompt_adapter_identity_hash": (
                self.prompt_adapter_identity_hash
            ),
            "provider_call_request": result.request_identity,
        }
        response_evidence = {
            "logical_call_id": logical_call_id,
            "provider_call_result": result.to_stable_dict(),
        }

        if result.generation is None:
            failure = result.semantic_failure
            assert failure is not None
            response = failure.rejected_response
            usage, cost = _response_accounting(response)
            return ProposalDraft.failure(
                detail=(
                    "provider proposer failed with "
                    f"{failure.failure_class.value}: {failure.message}"
                ),
                request_evidence=request_evidence,
                response_evidence=response_evidence,
                usage=usage,
                cost=cost,
            )

        provider_result = provider_result_from_response(
            result.generation.response
        )
        return ProposalDraft(
            template=provider_result.text,
            request_evidence=request_evidence,
            response_evidence={
                **response_evidence,
                "response_metadata": provider_result.response_metadata,
                "response_id": provider_result.response_id,
                "model": provider_result.model,
                "finish_reason": provider_result.finish_reason,
            },
            usage=provider_result.usage_metadata,
            cost=provider_result.provider_cost,
        )


def _response_accounting(response: Any) -> tuple[dict[str, Any], float | None]:
    """Retain usage/cost from a rejected response, when one exists."""

    if response is None:
        return {}, None
    usage = (
        response.usage.model_dump(mode="json", exclude_none=True)
        if response.usage is not None
        else {}
    )
    cost = response.cost.total_cost if response.cost is not None else None
    return usage, cost


class FakeProposerTransport:
    """A scripted, deterministic proposer transport for harness tests.

    Responses are keyed by ``(proposal_mode, request_ordinal)`` -> a tuple of
    template strings. Strict mode is the default: a short script produces
    explicit failed slots instead of invented candidates. Legacy padding is
    available only with ``strict=False``. Every call records the configured
    execution-policy and prompt-adapter identities.
    """

    def __init__(
        self,
        script: dict[tuple[str, int], tuple[str, ...]],
        *,
        default: tuple[str, ...] = (),
        execution_policy_hash: str,
        prompt_adapter_identity_hash: str,
        strict: bool = True,
    ) -> None:
        require_full_hash(
            execution_policy_hash,
            field="execution_policy_hash",
        )
        require_full_hash(
            prompt_adapter_identity_hash,
            field="prompt_adapter_identity_hash",
        )
        self._script = dict(script)
        self._default = default
        self._execution_policy_hash = execution_policy_hash
        self._prompt_adapter_identity_hash = prompt_adapter_identity_hash
        self._strict = strict
        self.calls: list[tuple[str, ProposalRequest, int]] = []

    @property
    def execution_policy_hash(self) -> str:
        return self._execution_policy_hash

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return self._prompt_adapter_identity_hash

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
            elif self._strict:
                drafts.append(
                    ProposalDraft.failure(
                        detail=(
                            "scripted proposer underfilled strict batch "
                            f"at slot {index} of {count}"
                        ),
                        request_evidence={
                            **evidence_base,
                            "draft_index": index,
                        },
                        usage={"proposer_calls": 0},
                    )
                )
                continue
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
