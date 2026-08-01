"""Optimizer-owned proposer route and immutable proposal evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.lm.boundary import (
    PlainPromptAdapter,
    provider_call_request_from_parameters,
    provider_result_from_response,
)
from whetstone.optimization.effect_authority import ReplayPolicy
from whetstone.optimization.identity import (
    FiniteFloat,
    IdentityHash,
    IdentityRef,
    ImmutableJsonObject,
    NonEmptyId,
    NonNegativeInt,
    TerminalFailure,
    compute_identity_hash,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.schema import CandidateRef
from whetstone.provider.driver import (
    Clock,
    Sleep,
    TransportCall,
    run_provider_call,
)
from whetstone.provider.policy import ProviderExecutionPolicy

if TYPE_CHECKING:
    from dr_providers import (
        ProviderCallConfig,
        ProviderCallRequest,
        ProviderInvocationEvidence,
    )

__all__ = [
    "PROMPT_ADAPTER_SCHEMA",
    "PROMPT_ADAPTER_SCHEMA_VERSION",
    "PROPOSAL_REQUEST_SCHEMA",
    "PROPOSAL_REQUEST_SCHEMA_VERSION",
    "PROPOSER_CONFIG_SCHEMA",
    "PROPOSER_CONFIG_SCHEMA_VERSION",
    "PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA",
    "PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION",
    "DurableProposalExecutor",
    "FakeProposerTransport",
    "IdempotentTransportCall",
    "PhysicalAttemptExecutor",
    "ProposalDraft",
    "ProposalExecutor",
    "ProposalExecutorDurabilityContract",
    "ProposalRequest",
    "ProposerConfig",
    "ProposerTransport",
    "ProviderCallConfigResolver",
    "ProviderProposerTransport",
    "RetryDurableProposerTransport",
    "prompt_adapter_identity_hash",
]

PROPOSER_CONFIG_SCHEMA = "whetstone.proposer_config"
PROPOSER_CONFIG_SCHEMA_VERSION = 1
PROPOSAL_REQUEST_SCHEMA = "whetstone.proposal_request"
PROPOSAL_REQUEST_SCHEMA_VERSION = 1
PROMPT_ADAPTER_SCHEMA = "whetstone.proposal_prompt_adapter"
PROMPT_ADAPTER_SCHEMA_VERSION = 1
PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA = (
    "whetstone.provider_proposer_transport_durability"
)
PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION = 1


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

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity contract: spell every key explicitly and pin the
        # complete payload plus digest in golden tests.
        return {
            "proposal_mode": str(self.proposal_mode),
            "request_ordinal": int(self.request_ordinal),
            "base_candidate": {
                "record_ref": {
                    "schema_name": str(
                        self.base_candidate.record_ref.schema_name
                    ),
                    "content_hash": str(
                        self.base_candidate.record_ref.content_hash
                    ),
                },
                "identity_hash": str(self.base_candidate.identity_hash),
            },
            "context": self.context.to_json(),
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=PROPOSAL_REQUEST_SCHEMA,
            schema_version=PROPOSAL_REQUEST_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


def prompt_adapter_identity_hash(adapter: PlainPromptAdapter) -> str:
    """Identify the exact plain-text projection used for proposer prompts."""

    return compute_identity_hash(
        schema=PROMPT_ADAPTER_SCHEMA,
        schema_version=PROMPT_ADAPTER_SCHEMA_VERSION,
        payload=adapter.model_dump(mode="json"),
    )


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
        response_evidence: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        cost: float | None = 0.0,
    ) -> ProposalDraft:
        return cls(
            terminal_failure=TerminalFailure(
                code="proposal_failed",
                message=detail,
            ),
            request_evidence=request_evidence or {},
            response_evidence=response_evidence or {"finish": "failed"},
            usage=usage or {},
            cost=cost,
        )


class ProposerTransport(Protocol):
    """Draft exactly ``count`` proposals without performing evaluation."""

    @property
    def execution_policy_hash(self) -> str: ...

    @property
    def prompt_adapter_identity_hash(self) -> str: ...

    @property
    def durability_identity_hash(self) -> str: ...

    def draft(
        self, config: ProposerConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]: ...


@dataclass(frozen=True, slots=True)
class ProposalExecutorDurabilityContract:
    """Immutable identity of one deliberately durable executor boundary."""

    recovery_policy: ReplayPolicy
    policy_identity_hash: str

    def __post_init__(self) -> None:
        if self.recovery_policy is not ReplayPolicy.DURABLE_WORKFLOW:
            raise ValueError(
                "durable proposal executors require durable-workflow recovery"
            )
        require_full_hash(
            self.policy_identity_hash,
            field="proposal_executor_policy_identity_hash",
        )


class _ProposalExecution(Protocol):
    def __call__(
        self,
        *,
        config: ProposerConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]: ...


class ProposalExecutor(Protocol):
    """Execute one proposal call within its caller's effect authority."""

    def execute(
        self,
        *,
        config: ProposerConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]: ...


_DURABLE_PROPOSAL_EXECUTOR_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class DurableProposalExecutor:
    """Exact owned capability for the canonical durable proposal executor.

    The capability value is created only by the canonical durable provider
    factory in :mod:`whetstone.orchestration.proposal_provider`; a token
    guard keeps its declared durability contract inseparable from the
    behavior actually invoked.
    """

    __durability_contract: ProposalExecutorDurabilityContract
    __execute: _ProposalExecution

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("DurableProposalExecutor cannot be subclassed")

    def __init__(
        self,
        *,
        durability_contract: ProposalExecutorDurabilityContract,
        execute: _ProposalExecution,
        _token: object,
    ) -> None:
        if _token is not _DURABLE_PROPOSAL_EXECUTOR_TOKEN:
            raise TypeError(
                "DurableProposalExecutor is created only by its canonical "
                "durable provider factory"
            )
        object.__setattr__(
            self,
            "_DurableProposalExecutor__durability_contract",
            durability_contract,
        )
        object.__setattr__(self, "_DurableProposalExecutor__execute", execute)

    @property
    def durability_contract(self) -> ProposalExecutorDurabilityContract:
        """Return the frozen contract snapshot stored by the capability."""

        return self.__durability_contract

    @property
    def policy_identity_hash(self) -> str:
        return self.__durability_contract.policy_identity_hash

    @property
    def recovery_policy(self) -> ReplayPolicy:
        return self.__durability_contract.recovery_policy

    def execute(
        self,
        *,
        config: ProposerConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        """Execute one proposal call inside the durable effect authority."""

        return self.__execute(
            config=config,
            request=request,
            transport=transport,
            count=count,
        )


def _durable_proposal_executor(
    *,
    durability_contract: ProposalExecutorDurabilityContract,
    execute: _ProposalExecution,
) -> DurableProposalExecutor:
    """Mint the capability from its canonical durable provider factory."""

    return DurableProposalExecutor(
        durability_contract=durability_contract,
        execute=execute,
        _token=_DURABLE_PROPOSAL_EXECUTOR_TOKEN,
    )


ProviderCallConfigResolver = Callable[[IdentityRef], "ProviderCallConfig"]
PhysicalAttemptExecutor = Callable[
    [
        TransportCall,
        "ProviderCallRequest",
        str,
        int,
    ],
    "ProviderInvocationEvidence",
]


@runtime_checkable
class IdempotentTransportCall(Protocol):
    """A physical provider transport accepting a stable attempt key."""

    @property
    def idempotency_policy_identity_hash(self) -> str: ...

    def invoke_idempotent(
        self,
        request: ProviderCallRequest,
        *,
        idempotency_key: str,
    ) -> ProviderInvocationEvidence: ...


@runtime_checkable
class RetryDurableProposerTransport(Protocol):
    """A proposer whose physical attempts can be checkpointed separately."""

    @property
    def supports_provider_idempotency(self) -> bool: ...

    @property
    def durability_identity_hash(self) -> str: ...

    def draft_with_attempt_executor(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
        *,
        execute_attempt: PhysicalAttemptExecutor,
        sleep: Sleep,
    ) -> tuple[ProposalDraft, ...]: ...

    def invoke_physical_attempt(
        self,
        request: ProviderCallRequest,
        *,
        idempotency_key: str | None,
    ) -> ProviderInvocationEvidence: ...


@dataclass(frozen=True, slots=True)
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

    _resolve_provider_call_config: ProviderCallConfigResolver
    _transport: TransportCall
    _execution_policy: ProviderExecutionPolicy
    _prompt_adapter: PlainPromptAdapter
    _clock: Clock | None
    _sleep: Sleep | None

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("ProviderProposerTransport cannot be subclassed")

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
        object.__setattr__(
            self,
            "_resolve_provider_call_config",
            resolve_provider_call_config,
        )
        object.__setattr__(self, "_transport", transport)
        object.__setattr__(self, "_execution_policy", execution_policy)
        object.__setattr__(
            self, "_prompt_adapter", prompt_adapter or PlainPromptAdapter()
        )
        object.__setattr__(self, "_clock", clock)
        object.__setattr__(self, "_sleep", sleep)

    @property
    def execution_policy_hash(self) -> str:
        return self._execution_policy.identity_hash

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return prompt_adapter_identity_hash(self._prompt_adapter)

    @property
    def supports_provider_idempotency(self) -> bool:
        """Whether the physical transport has a real idempotency contract."""

        return isinstance(self._transport, IdempotentTransportCall)

    @property
    def durability_identity_hash(self) -> str:
        """Identify the exact capability a durable executor may invoke."""

        # Persisted identity contract: spell every key explicitly and pin the
        # complete payload plus digest in golden tests.
        return compute_identity_hash(
            schema=PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA,
            schema_version=(
                PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION
            ),
            payload={
                "execution_policy_hash": self.execution_policy_hash,
                "prompt_adapter_identity_hash": (
                    self.prompt_adapter_identity_hash
                ),
            },
        )

    def draft(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        return self._draft(
            config,
            request,
            count,
            execute_attempt=None,
            sleep=self._sleep,
        )

    def draft_with_attempt_executor(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
        *,
        execute_attempt: PhysicalAttemptExecutor,
        sleep: Sleep,
    ) -> tuple[ProposalDraft, ...]:
        """Run retries with one caller-owned boundary per physical attempt."""

        return self._draft(
            config,
            request,
            count,
            execute_attempt=execute_attempt,
            sleep=sleep,
        )

    def invoke_physical_attempt(
        self,
        request: ProviderCallRequest,
        *,
        idempotency_key: str | None,
    ) -> ProviderInvocationEvidence:
        """Invoke exactly one physical transport attempt."""

        if idempotency_key is None:
            return self._transport(request)
        if not isinstance(self._transport, IdempotentTransportCall):
            raise RuntimeError(
                "physical proposer transport does not support idempotency"
            )
        return self._transport.invoke_idempotent(
            request,
            idempotency_key=idempotency_key,
        )

    def _draft(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
        *,
        execute_attempt: PhysicalAttemptExecutor | None,
        sleep: Sleep | None,
    ) -> tuple[ProposalDraft, ...]:
        if type(count) is not int or count < 1:
            raise ValueError("proposer draft count must be a positive integer")

        prompt = request.context.get("proposal_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "provider proposer requires one nonblank proposal_prompt"
            )

        provider_config = self._resolve_provider_call_config(
            config.provider_call_config
        )
        from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA

        materialized_ref = typed_ref_for_record(
            PROVIDER_CALL_CONFIG_SCHEMA,
            provider_config.model_dump(mode="json"),
        )
        if materialized_ref != config.provider_call_config.record_ref:
            raise ValueError(
                "resolved Provider Call Config record does not match "
                "Proposer Config"
            )
        if (
            provider_config.identity_hash
            != config.provider_call_config.identity_hash
        ):
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
                execute_attempt=execute_attempt,
                sleep=sleep,
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
        execute_attempt: PhysicalAttemptExecutor | None = None,
        sleep: Sleep | None = None,
    ) -> ProposalDraft:
        logical_call_id = (
            f"proposer:{config.identity_hash()}:"
            f"{self.execution_policy_hash}:"
            f"{self.prompt_adapter_identity_hash}:"
            f"{proposal_request.identity_hash()}:{slot}"
        )
        attempt_number = 0

        def invoke(request: ProviderCallRequest) -> ProviderInvocationEvidence:
            nonlocal attempt_number
            attempt_number += 1
            if execute_attempt is None:
                return self._transport(request)
            return execute_attempt(
                self._transport,
                request,
                logical_call_id,
                attempt_number,
            )

        durable_clock = (
            None if execute_attempt is None else lambda: float(attempt_number)
        )
        result = run_provider_call(
            request=provider_request,
            policy=self._execution_policy,
            transport=invoke,
            logical_call_id=logical_call_id,
            clock=self._clock if durable_clock is None else durable_clock,
            sleep=sleep,
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
            "provider_call_config": config.provider_call_config.model_dump(
                mode="json"
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
    template strings. A short script produces explicit failed slots instead
    of invented candidates. Every call records the configured execution-policy
    and prompt-adapter identities.
    """

    def __init__(
        self,
        script: dict[tuple[str, int], tuple[str, ...]],
        *,
        default: tuple[str, ...] = (),
        execution_policy_hash: str,
        prompt_adapter_identity_hash: str,
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
        self.calls: list[tuple[IdentityHash, ProposalRequest, int]] = []

    @property
    def execution_policy_hash(self) -> str:
        return self._execution_policy_hash

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return self._prompt_adapter_identity_hash

    @property
    def supports_provider_idempotency(self) -> bool:
        """The scripted fake never carries a provider idempotency contract."""

        return False

    @property
    def durability_identity_hash(self) -> str:
        # Persisted identity contract: spell every key explicitly and pin the
        # complete payload plus digest in golden tests.
        return compute_identity_hash(
            schema=PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA,
            schema_version=(
                PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION
            ),
            payload={
                "execution_policy_hash": self.execution_policy_hash,
                "prompt_adapter_identity_hash": (
                    self.prompt_adapter_identity_hash
                ),
            },
        )

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
