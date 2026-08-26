from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.core.leasing import ReplayPolicy
from whetstone.core.identity import (
    FiniteFloat,
    IdentityHash,
    IdentityRef,
    ImmutableJsonObject,
    NonEmptyId,
    NonNegativeInt,
    TerminalFailure,
    assert_materialized_ref_matches,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.execution.call_support import (
    PROVIDER_ERROR_KEY,
    call_telemetry,
    evidences_provider_response,
)
from whetstone.experiment.candidate import CandidateRef
from whetstone.optim.cost import ProposerCallUsage
from whetstone.provider.driver import (
    Clock,
    Sleep,
    TransportCall,
)
from whetstone.provider.language_model import (
    PlainPromptAdapter,
    StructuredPromptAdapter,
    provider_result_from_response,
)
from whetstone.provider.llm_call import (
    LlmCallContext,
    build_provider_request,
    derive_rng_seed,
    execute_llm_call,
)
from whetstone.provider.policy import ProviderExecutionPolicy

if TYPE_CHECKING:
    from dr_providers import (
        PromptMessage,
        ProviderCallConfig,
    )

    from whetstone.provider.attempt import ProviderCallResult

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
    "build_inline_proposal_executor",
    "require_canonical_proposal_executor",
    "ProposalDraft",
    "ProposalExecutorDurabilityContract",
    "ProposalRequest",
    "ProposerConfig",
    "ProposerRouteConfig",
    "ProposerTransport",
    "ProviderCallConfigResolver",
    "ProviderProposerTransport",
    "prompt_adapter_identity_hash",
]

PROPOSER_CONFIG_SCHEMA = "whetstone.proposer_config"
PROPOSER_CONFIG_SCHEMA_VERSION = 1
PROPOSAL_REQUEST_SCHEMA = "whetstone.proposal_request"
PROPOSAL_REQUEST_SCHEMA_VERSION = 2
PROMPT_ADAPTER_SCHEMA = "whetstone.proposal_prompt_adapter"
PROMPT_ADAPTER_SCHEMA_VERSION = 1
PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA = (
    "whetstone.provider_proposer_transport_durability"
)
PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION = 1


class ProposerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_call_config: IdentityRef
    temperature: FiniteFloat | None = FiniteFloat(1.0)

    def identity_payload(self) -> dict[str, Any]:

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
                "identity_hash": str(self.provider_call_config.record_hash),
            },
            "temperature": (
                None if self.temperature is None else float(self.temperature)
            ),
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=PROPOSER_CONFIG_SCHEMA,
            schema_version=PROPOSER_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class ProposerRouteConfig(Protocol):
    def identity_payload(self) -> dict[str, Any]: ...

    def identity_hash(self) -> str: ...


class ProposalRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_mode: NonEmptyId
    request_ordinal: NonNegativeInt
    proposal_authority_identity_hash: IdentityHash
    mutation_field: NonEmptyId
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
        field = self.mutation_field
        try:
            value = self.base_candidate.record.payload[field]
        except KeyError as error:
            raise ValueError(
                "base candidate payload must contain the "
                f"{field!r} mutation field"
            ) from error
        if type(value) is not str:
            raise ValueError(
                f"base candidate {field!r} mutation field must be a string"
            )
        return value

    def identity_payload(self) -> dict[str, Any]:

        return {
            "proposal_mode": str(self.proposal_mode),
            "request_ordinal": int(self.request_ordinal),
            "proposal_authority_identity_hash": str(
                self.proposal_authority_identity_hash
            ),
            "mutation_field": str(self.mutation_field),
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

    return compute_identity_hash(
        schema=PROMPT_ADAPTER_SCHEMA,
        schema_version=PROMPT_ADAPTER_SCHEMA_VERSION,
        payload=adapter.model_dump(mode="json"),
    )


def _usage_tokens(usage: Mapping[str, Any], key: str) -> int | None:
    """One directional token count, ``None`` when the provider omitted it.

    Absence is preserved rather than normalized to zero, so a call the
    provider priced without a token breakdown reports unknown tokens instead
    of an incomplete total that reads as complete.
    """
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(0, value)


class ProposalDraft(BaseModel):
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

    def call_usage(self) -> ProposerCallUsage | None:
        """Project this draft's provider usage onto the run cost contract.

        ``None`` when this draft was never billed, which run cost then records
        as nothing at all. Two things have to be true for a draft to count.

        It must carry a ``logical_call_id``: a batch slot the proposer never
        filled has no call identity and must not appear as a zero-dollar call
        inflating ``calls`` or completing a ``usd`` total.

        It must also evidence work the provider actually did. A *failed*
        draft carrying neither usage, nor a price, nor a rejected response is
        a transport failure: the logical call id was minted before the
        request left, and nothing came back, so nothing was billed. Counting
        it would add an unpriced call and withhold the role's ``usd`` for
        spend that never happened.

        A failed draft that carries usage, a price, or a rejected response --
        a blank or malformed generation the provider produced and charged
        for -- is counted like any other call. The rejected response is what
        makes a response-level semantic failure legible when the provider
        reported no telemetry at all: it is the same signal, from the same
        owner, that lets a rejected task-model row count. Every successful
        draft counts too: a template came back, so the model ran, whether or
        not the transport reported any telemetry alongside it. A counted
        draft without telemetry (the Codex CLI transport reports none) is an
        identified, unpriced call with an unknown token breakdown.

        ``usage`` is a dr-providers ``TokenUsage`` dump, so its keys are the
        provider contract rather than local spelling. A directional count the
        provider omitted stays absent rather than reading as zero, so the call
        lands in ``rows_missing_token_breakdown`` instead of presenting an
        incomplete token total as complete. ``cost`` stays ``None`` when the
        provider reported no price, which keeps the run total honest about
        what it does not know.

        The call identity is derived from the proposal request and batch slot
        and so is stable across a re-drive. Run cost de-duplicates on it, so a
        Step Result replayed after a crash reports the same proposer spend as
        the uninterrupted one.
        """
        call_id = self.logical_call_id
        if not call_id:
            return None
        usage = self.usage.to_json()
        prompt_tokens = _usage_tokens(usage, "prompt_tokens")
        completion_tokens = _usage_tokens(usage, "completion_tokens")
        if (
            self.failed
            and prompt_tokens is None
            and completion_tokens is None
            and self.cost is None
            and not self.response_rejected
        ):
            # A failure with no usage, no price, and no rejected response:
            # nothing came back from the provider, so there is no billed
            # call to report.
            return None
        return ProposerCallUsage(
            call_id=call_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd=self.cost,
        )

    @property
    def logical_call_id(self) -> str:
        """This draft's provider call identity, empty when unrecorded."""
        value = self.request_evidence.to_json().get("logical_call_id")
        return value if isinstance(value, str) else ""

    @property
    def response_rejected(self) -> bool:
        """The provider answered and the classifier rejected the answer.

        Reads the failure body the transport persisted on this draft's
        response evidence, which is the same body an ``EvalOutputRow``
        carries, so both cost roles decide "did the provider answer?" from
        one signal.
        """
        error = self.response_evidence.to_json().get(PROVIDER_ERROR_KEY)
        return evidences_provider_response(error)

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
    @property
    def execution_policy_hash(self) -> str: ...

    @property
    def prompt_adapter_identity_hash(self) -> str: ...

    @property
    def durability_identity_hash(self) -> str: ...

    def draft(
        self, config: ProposerRouteConfig, request: ProposalRequest, count: int
    ) -> tuple[ProposalDraft, ...]: ...


@dataclass(frozen=True, slots=True)
class ProposalExecutorDurabilityContract:
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
        config: ProposerRouteConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]: ...


_DURABLE_PROPOSAL_EXECUTOR_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class DurableProposalExecutor:
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
        config: ProposerRouteConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]:

        return self.__execute(
            config=config,
            request=request,
            transport=transport,
            count=count,
        )


def require_canonical_proposal_executor(
    executor: object,
    *,
    algorithm: str,
    purpose: str,
) -> DurableProposalExecutor:
    if type(executor) is not DurableProposalExecutor:
        raise TypeError(
            f"{algorithm} requires the canonical DurableProposalExecutor "
            f"capability for its {purpose}"
        )
    return executor


def _durable_proposal_executor(
    *,
    durability_contract: ProposalExecutorDurabilityContract,
    execute: _ProposalExecution,
) -> DurableProposalExecutor:

    return DurableProposalExecutor(
        durability_contract=durability_contract,
        execute=execute,
        _token=_DURABLE_PROPOSAL_EXECUTOR_TOKEN,
    )


def _inline_proposal_execution(
    *,
    config: ProposerRouteConfig,
    request: ProposalRequest,
    transport: ProposerTransport,
    count: int,
) -> tuple[ProposalDraft, ...]:
    return transport.draft(config, request, count)


def build_inline_proposal_executor(
    *,
    policy_identity_hash: str,
) -> DurableProposalExecutor:
    """Build the canonical in-process proposal executor.

    The draft call runs inline on the caller's thread with no external
    durability layer. ``policy_identity_hash`` names the caller's inline
    durability policy and must match the optimizer control that binds it.
    """
    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=policy_identity_hash,
        ),
        execute=_inline_proposal_execution,
    )


ProviderCallConfigResolver = Callable[[IdentityRef], "ProviderCallConfig"]


@dataclass(frozen=True, slots=True)
class ProviderProposerTransport:
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
    def durability_identity_hash(self) -> str:

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
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        if not isinstance(config, ProposerConfig):
            raise TypeError(
                "provider proposer requires provider ProposerConfig"
            )
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

        assert_materialized_ref_matches(
            record=provider_config,
            ref=config.provider_call_config,
            schema=PROVIDER_CALL_CONFIG_SCHEMA,
        )

        raw_messages = request.context.get("proposal_messages")
        if raw_messages is None:
            messages = self._prompt_adapter.messages(user_content=prompt)
        else:
            if not isinstance(self._prompt_adapter, StructuredPromptAdapter):
                raise ValueError(
                    "structured proposer messages require the identity-bound "
                    "StructuredPromptAdapter"
                )
            if not isinstance(raw_messages, list | tuple):
                raise ValueError(
                    "proposal_messages must be an ordered list of JSON "
                    "message records"
                )
            records: list[dict[str, Any]] = []
            for record in raw_messages:
                if not isinstance(record, Mapping):
                    raise ValueError(
                        "proposal_messages must be an ordered list of JSON "
                        "message records"
                    )
                records.append(
                    {str(key): value for key, value in record.items()}
                )
            messages = self._prompt_adapter.messages_from_records(
                tuple(records)
            )
        parameters = (
            {}
            if config.temperature is None
            else {"temperature": config.temperature}
        )

        drafts = tuple(
            self._draft_slot(
                config=config,
                proposal_request=request,
                provider_config=provider_config,
                messages=messages,
                parameters=parameters,
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
        provider_config: ProviderCallConfig,
        messages: tuple[PromptMessage, ...],
        parameters: dict[str, object],
        count: int,
        slot: int,
    ) -> ProposalDraft:
        logical_call_id = (
            f"proposer:{config.identity_hash()}:"
            f"{self.execution_policy_hash}:"
            f"{self.prompt_adapter_identity_hash}:"
            f"{proposal_request.identity_hash()}:{slot}"
        )
        rng_seed = derive_rng_seed(proposal_request.identity_hash(), slot)
        provider_request = build_provider_request(
            provider_config=provider_config,
            rng_seed=rng_seed,
            messages=messages,
            parameters=parameters,
            prompt_adapter=self._prompt_adapter,
        )
        execution = execute_llm_call(
            context=LlmCallContext(
                execution_policy=self._execution_policy,
                transport=self._transport,
                prompt_adapter=self._prompt_adapter,
                clock=self._clock,
                sleep=self._sleep,
                prompt_cache=None,
            ),
            request=provider_request,
            logical_call_id=logical_call_id,
        )
        result = execution.result
        request_evidence = {
            "logical_call_id": logical_call_id,
            "logical_batch_size": count,
            "batch_slot": slot,
            "proposal_mode": proposal_request.proposal_mode,
            "request_ordinal": proposal_request.request_ordinal,
            "proposal_request_hash": (proposal_request.identity_hash()),
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
            "provider_call_request": result.request_hash,
        }
        response_evidence = {
            "logical_call_id": logical_call_id,
            "provider_call_result": result.to_stable_dict(),
        }

        # Accounting spans every attempt the provider was paid for, not just
        # the terminal one. When the execution policy retries a rejected
        # response, that rejected generation was produced and billed, and
        # ``call_telemetry`` sums it in under the same partial-price rule the
        # task-model rows obey: an unpriced billed attempt withholds the
        # price entirely, and a billed attempt without a token breakdown
        # withholds the tokens.
        usage, cost, provider_error = _billed_accounting(result)

        if result.provider_generation is None:
            failure = result.semantic_failure
            assert failure is not None
            return ProposalDraft.failure(
                detail=(
                    "provider proposer failed with "
                    f"{failure.failure_class.value}: {failure.message}"
                ),
                request_evidence=request_evidence,
                response_evidence={
                    **response_evidence,
                    # The same failure body the task-model rows persist, and
                    # from the same owner, so both roles decide "did the
                    # provider answer?" on one signal: ``rejected_response``
                    # is present exactly when a generation came back and only
                    # the classifier turned it down. Without it a rejected
                    # response the provider reported no usage for would be
                    # indistinguishable from a transport failure and dropped
                    # from run cost.
                    PROVIDER_ERROR_KEY: provider_error,
                },
                usage=usage,
                cost=cost,
            )

        provider_result = provider_result_from_response(
            result.provider_generation.response
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
            usage=usage,
            cost=cost,
        )


def _billed_accounting(
    result: ProviderCallResult,
) -> tuple[dict[str, Any], float | None, dict[str, Any] | None]:
    """One logical proposer call's accounting, over every billed attempt.

    Delegates to :func:`call_telemetry`, the single owner of what a logical
    call was billed for, so the proposer side and the task-model side account
    for retries identically. A field the aggregate could not complete stays
    out of the returned usage map rather than appearing as a partial total.

    Returns the usage map, the summed price, and the failure body describing
    what came back -- the third of which is what lets a failed draft say
    whether the provider answered at all.
    """
    telemetry = call_telemetry(result)
    usage: dict[str, Any] = {}
    for key, value in (
        ("prompt_tokens", telemetry.prompt_tokens),
        ("completion_tokens", telemetry.completion_tokens),
        ("total_tokens", telemetry.total_tokens),
        ("reasoning_tokens", telemetry.reasoning_tokens),
    ):
        if value is not None:
            usage[key] = value
    provider_error = telemetry.provider_error
    return (
        usage,
        telemetry.provider_cost,
        None if provider_error is None else dict(provider_error),
    )


class FakeProposerTransport:
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
        self.calls: list[tuple[str, ProposalRequest, int]] = []

    @property
    def execution_policy_hash(self) -> str:
        return self._execution_policy_hash

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return self._prompt_adapter_identity_hash

    @property
    def durability_identity_hash(self) -> str:

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
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
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
            "proposer_config": config.identity_payload(),
        }

        def call_id(slot: int) -> str:
            """The scripted stand-in for a provider call's identity.

            Same shape and same inputs as the provider transport's, so the
            toy path exercises run cost's de-duplication instead of silently
            disabling it: two re-drives of one request mint one id, and two
            batch slots mint two.
            """
            return (
                f"fake-proposer:{config.identity_hash()}:"
                f"{self.execution_policy_hash}:"
                f"{self.prompt_adapter_identity_hash}:"
                f"{request.identity_hash()}:{slot}"
            )
        drafts: list[ProposalDraft] = []
        for index in range(count):
            if index < len(templates):
                text = templates[index]
            else:
                # No provider call was made for this slot, so it carries
                # no call identity and no price: run cost must record
                # nothing rather than a zero-dollar phantom call.
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
                        cost=None,
                    )
                )
                continue
            if not text:
                # A call was made and came back empty: billed, and failed.
                drafts.append(
                    ProposalDraft.failure(
                        detail="scripted proposer produced an empty draft",
                        request_evidence={
                            **evidence_base,
                            "draft_index": index,
                            "logical_call_id": call_id(index),
                        },
                        usage={"proposer_calls": 1},
                    )
                )
                continue
            drafts.append(
                ProposalDraft(
                    template=text,
                    request_evidence={
                        **evidence_base,
                        "logical_call_id": call_id(index),
                    },
                    response_evidence={"draft_index": index},
                    usage={"proposer_calls": 1},
                    cost=0.0,
                )
            )
        return tuple(drafts)
