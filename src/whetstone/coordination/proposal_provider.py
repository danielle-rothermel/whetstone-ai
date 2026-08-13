from __future__ import annotations

from threading import Lock
from typing import NamedTuple

try:
    from dbos import DBOS, SetWorkflowID
except ImportError as exc:
    raise ImportError(
        "DBOS coordination requires the optional dbos extra: "
        "pip install 'whetstone-ai[dbos]'"
    ) from exc

from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optim.codex.proposer import (
    CodexCliProposerConfig,
    CodexCliProposerTransport,
)
from whetstone.optim.proposal.proposer import (
    DurableProposalExecutor,
    ProposalDraft,
    ProposalExecutorDurabilityContract,
    ProposalRequest,
    ProposerConfig,
    ProposerRouteConfig,
    ProposerTransport,
    ProviderProposerTransport,
    _durable_proposal_executor,
)

PROPOSAL_DBOS_POLICY_SCHEMA = "whetstone.proposal_dbos_policy"
PROPOSAL_DBOS_POLICY_VERSION = 2
PROPOSAL_DBOS_WORKFLOW_SCHEMA = "whetstone.proposal_dbos_workflow"
PROPOSAL_DBOS_WORKFLOW_VERSION = 1


PROPOSAL_DURABILITY_MODE = "at_least_once"

type DurableProposerConfig = ProposerConfig | CodexCliProposerConfig
type DurableProposerTransport = (
    ProviderProposerTransport | CodexCliProposerTransport
)


class ProposalProviderError(RuntimeError):
    pass


class _ProposalTransportRegistry:
    __slots__ = ("_lock", "_transports")

    def __init__(self) -> None:
        self._lock = Lock()
        self._transports: dict[str, DurableProposerTransport] = {}

    def bind(self, transport: DurableProposerTransport) -> str:
        if type(transport) not in (
            ProviderProposerTransport,
            CodexCliProposerTransport,
        ):
            raise ProposalProviderError(
                "durable proposal registration requires "
                "a supported proposer transport"
            )

        registry_key = transport.durability_identity_hash
        require_full_hash(
            registry_key,
            field="transport_durability_identity_hash",
        )
        with self._lock:
            existing = self._transports.get(registry_key)
            if existing is not None and existing is not transport:
                raise ProposalProviderError(
                    "proposal transport key is already bound"
                )
            self._transports[registry_key] = transport
        return registry_key

    def resolve(self, registry_key: str) -> DurableProposerTransport:
        require_full_hash(registry_key, field="transport_registry_key")
        with self._lock:
            try:
                transport = self._transports[registry_key]
            except KeyError:
                raise ProposalProviderError(
                    "proposal transport is not registered before DBOS launch"
                ) from None
            current_hash = transport.durability_identity_hash
            require_full_hash(
                current_hash,
                field="transport_durability_identity_hash",
            )
            if current_hash != registry_key:
                raise ProposalProviderError(
                    "registered proposal transport durability identity changed"
                )
            return transport


_TRANSPORT_REGISTRY = _ProposalTransportRegistry()


def register_proposal_transport(
    transport: DurableProposerTransport,
) -> str:

    return _TRANSPORT_REGISTRY.bind(transport)


def _registered_transport(registry_key: str) -> DurableProposerTransport:
    return _TRANSPORT_REGISTRY.resolve(registry_key)


def _proposal_policy_identity_payload(
    registry_key: str,
) -> dict[str, bool | str]:

    return {
        "automatic_dbos_retries": False,
        "durability_mode": PROPOSAL_DURABILITY_MODE,
        "logical_call_boundary": "one_retry_disabled_dbos_step",
        "retry_owner": "proposer_transport",
        "transport_durability_identity_hash": registry_key,
    }


def _proposal_policy_hash(registry_key: str) -> str:
    return compute_identity_hash(
        schema=PROPOSAL_DBOS_POLICY_SCHEMA,
        schema_version=PROPOSAL_DBOS_POLICY_VERSION,
        payload=_proposal_policy_identity_payload(registry_key),
    )


def _proposal_workflow_identity_payload(
    *,
    registry_key: str,
    policy_identity_hash: str,
    config: DurableProposerConfig,
    request: ProposalRequest,
    count: int,
) -> dict[str, int | str]:

    return {
        "count": count,
        "policy_identity_hash": policy_identity_hash,
        "proposal_request_hash": request.identity_hash(),
        "proposer_config_hash": config.identity_hash(),
        "transport_durability_identity_hash": registry_key,
    }


def _proposal_workflow_hash(
    *,
    registry_key: str,
    policy_identity_hash: str,
    config: DurableProposerConfig,
    request: ProposalRequest,
    count: int,
) -> str:
    return compute_identity_hash(
        schema=PROPOSAL_DBOS_WORKFLOW_SCHEMA,
        schema_version=PROPOSAL_DBOS_WORKFLOW_VERSION,
        payload=_proposal_workflow_identity_payload(
            registry_key=registry_key,
            policy_identity_hash=policy_identity_hash,
            config=config,
            request=request,
            count=count,
        ),
    )


@DBOS.step(retries_allowed=False)
def _logical_proposal_step(
    registry_key: str,
    config: DurableProposerConfig,
    request: ProposalRequest,
    count: int,
) -> tuple[ProposalDraft, ...]:

    transport = _registered_transport(registry_key)
    return transport.draft(config, request, count)


@DBOS.workflow()
def _proposal_provider_workflow(
    registry_key: str,
    policy_identity_hash: str,
    config: DurableProposerConfig,
    request: ProposalRequest,
    count: int,
) -> tuple[ProposalDraft, ...]:

    if DBOS.step_id is not None:
        raise ProposalProviderError(
            "proposal child workflow body cannot execute inside a DBOS step"
        )
    expected_policy_hash = _proposal_policy_hash(registry_key)
    if policy_identity_hash != expected_policy_hash:
        raise ProposalProviderError(
            "proposal child workflow policy identity is inconsistent"
        )
    expected_workflow_hash = _proposal_workflow_hash(
        registry_key=registry_key,
        policy_identity_hash=policy_identity_hash,
        config=config,
        request=request,
        count=count,
    )
    if DBOS.workflow_id != expected_workflow_hash:
        raise ProposalProviderError(
            "proposal child workflow has the wrong semantic identity"
        )
    return _logical_proposal_step(registry_key, config, request, count)


class _DbosProposalExecution(NamedTuple):
    transport_registry_key: str

    def validate(self) -> None:
        require_full_hash(
            self.transport_registry_key,
            field="transport_registry_key",
        )

    def __call__(
        self,
        *,
        config: ProposerRouteConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        if type(count) is not int or count < 1:
            raise ValueError("proposer draft count must be a positive integer")
        if DBOS.workflow_id is None:
            raise ProposalProviderError(
                "proposal execution requires a DBOS workflow body context"
            )
        if DBOS.step_id is not None:
            raise ProposalProviderError(
                "proposal execution cannot start a child workflow from a "
                "DBOS step"
            )
        registered = _registered_transport(self.transport_registry_key)
        if registered is not transport:
            raise ProposalProviderError(
                "proposal transport differs from its registered identity"
            )
        if not isinstance(config, ProposerConfig | CodexCliProposerConfig):
            raise TypeError("unsupported durable proposer config")
        policy_identity_hash = _proposal_policy_hash(
            self.transport_registry_key
        )
        workflow_hash = _proposal_workflow_hash(
            registry_key=self.transport_registry_key,
            policy_identity_hash=policy_identity_hash,
            config=config,
            request=request,
            count=count,
        )
        with SetWorkflowID(workflow_hash):
            handle = DBOS.start_workflow(
                _proposal_provider_workflow,
                self.transport_registry_key,
                policy_identity_hash,
                config,
                request,
                count,
            )
        return handle.get_result()


def DbosProposalExecutor(
    *,
    transport_registry_key: str,
) -> DurableProposalExecutor:

    execution = _DbosProposalExecution(transport_registry_key)
    execution.validate()
    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=_proposal_policy_hash(transport_registry_key),
        ),
        execute=execution,
    )


__all__ = [
    "PROPOSAL_DBOS_POLICY_SCHEMA",
    "PROPOSAL_DBOS_POLICY_VERSION",
    "PROPOSAL_DBOS_WORKFLOW_SCHEMA",
    "PROPOSAL_DBOS_WORKFLOW_VERSION",
    "PROPOSAL_DURABILITY_MODE",
    "DbosProposalExecutor",
    "DurableProposalExecutor",
    "ProposalProviderError",
    "register_proposal_transport",
]
