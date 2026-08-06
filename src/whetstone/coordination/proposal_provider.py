from __future__ import annotations

from threading import Lock
from typing import NamedTuple

from dbos import DBOS, SetWorkflowID

from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    ProposalDraft,
    ProposalExecutorDurabilityContract,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
    ProviderProposerTransport,
    _durable_proposal_executor,
)

PROPOSAL_DBOS_POLICY_SCHEMA = "whetstone.proposal_dbos_policy"
PROPOSAL_DBOS_POLICY_VERSION = 1
PROPOSAL_DBOS_WORKFLOW_SCHEMA = "whetstone.proposal_dbos_workflow"
PROPOSAL_DBOS_WORKFLOW_VERSION = 1

#: Persisted contract literal for the one supported durability mode.
PROPOSAL_DURABILITY_MODE = "at_least_once"


class ProposalProviderError(RuntimeError):
    """The configured physical proposal-provider boundary is invalid."""


class _ProposalTransportRegistry:
    """Own atomic identity binding and resolution for proposal transports."""

    __slots__ = ("_lock", "_transports")

    def __init__(self) -> None:
        self._lock = Lock()
        self._transports: dict[str, ProviderProposerTransport] = {}

    def bind(self, transport: ProviderProposerTransport) -> str:
        if type(transport) is not ProviderProposerTransport:
            raise ProposalProviderError(
                "durable proposal registration requires "
                "ProviderProposerTransport"
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

    def resolve(self, registry_key: str) -> ProviderProposerTransport:
        require_full_hash(registry_key, field="transport_registry_key")
        with self._lock:
            try:
                transport = self._transports[registry_key]
            except KeyError:
                raise ProposalProviderError(
                    "proposal transport is not registered before DBOS launch"
                ) from None
            current_identity = transport.durability_identity_hash
            require_full_hash(
                current_identity,
                field="transport_durability_identity_hash",
            )
            if current_identity != registry_key:
                raise ProposalProviderError(
                    "registered proposal transport durability identity changed"
                )
            return transport


_TRANSPORT_REGISTRY = _ProposalTransportRegistry()


def register_proposal_transport(
    transport: ProviderProposerTransport,
) -> str:
    """Bind one proposer transport under its exact durability identity."""

    return _TRANSPORT_REGISTRY.bind(transport)


def _registered_transport(registry_key: str) -> ProviderProposerTransport:
    return _TRANSPORT_REGISTRY.resolve(registry_key)


def _proposal_policy_identity_payload(
    registry_key: str,
) -> dict[str, bool | str]:
    """Build the explicitly pinned persisted durability-policy payload."""

    # Persisted identity contract: keep these exact literals pinned.
    return {
        "automatic_dbos_retries": False,
        "durability_mode": PROPOSAL_DURABILITY_MODE,
        "logical_call_boundary": "one_retry_disabled_dbos_step",
        "provider_retry_owner": "provider_execution_policy",
        "transport_durability_identity_hash": registry_key,
    }


def _proposal_policy_identity(registry_key: str) -> str:
    return compute_identity_hash(
        schema=PROPOSAL_DBOS_POLICY_SCHEMA,
        schema_version=PROPOSAL_DBOS_POLICY_VERSION,
        payload=_proposal_policy_identity_payload(registry_key),
    )


def _proposal_workflow_identity_payload(
    *,
    registry_key: str,
    policy_identity_hash: str,
    config: ProposerConfig,
    request: ProposalRequest,
    count: int,
) -> dict[str, int | str]:
    """Build the explicitly pinned persisted workflow-identity payload."""

    # Persisted identity contract: keep these exact literals pinned.
    return {
        "count": count,
        "policy_identity_hash": policy_identity_hash,
        "proposal_request_identity_hash": request.identity_hash(),
        "proposer_config_identity_hash": config.identity_hash(),
        "transport_durability_identity_hash": registry_key,
    }


def _proposal_workflow_identity(
    *,
    registry_key: str,
    policy_identity_hash: str,
    config: ProposerConfig,
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
    config: ProposerConfig,
    request: ProposalRequest,
    count: int,
) -> tuple[ProposalDraft, ...]:
    """Run one whole logical proposal call, provider retries included."""

    transport = _registered_transport(registry_key)
    return transport.draft(config, request, count)


@DBOS.workflow()
def _proposal_provider_workflow(
    registry_key: str,
    policy_identity_hash: str,
    config: ProposerConfig,
    request: ProposalRequest,
    count: int,
) -> tuple[ProposalDraft, ...]:
    """Execute one exact proposal effect in its own checkpoint namespace."""

    if DBOS.step_id is not None:
        raise ProposalProviderError(
            "proposal child workflow body cannot execute inside a DBOS step"
        )
    expected_policy_identity = _proposal_policy_identity(registry_key)
    if policy_identity_hash != expected_policy_identity:
        raise ProposalProviderError(
            "proposal child workflow policy identity is inconsistent"
        )
    expected_workflow_identity = _proposal_workflow_identity(
        registry_key=registry_key,
        policy_identity_hash=policy_identity_hash,
        config=config,
        request=request,
        count=count,
    )
    if DBOS.workflow_id != expected_workflow_identity:
        raise ProposalProviderError(
            "proposal child workflow has the wrong semantic identity"
        )
    return _logical_proposal_step(registry_key, config, request, count)


class _DbosProposalExecution(NamedTuple):
    """Canonical behavior wrapped by the exact durable capability value."""

    transport_registry_key: str

    def validate(self) -> None:
        require_full_hash(
            self.transport_registry_key,
            field="transport_registry_key",
        )

    def __call__(
        self,
        *,
        config: ProposerConfig,
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
        policy_identity_hash = _proposal_policy_identity(
            self.transport_registry_key
        )
        workflow_identity = _proposal_workflow_identity(
            registry_key=self.transport_registry_key,
            policy_identity_hash=policy_identity_hash,
            config=config,
            request=request,
            count=count,
        )
        with SetWorkflowID(workflow_identity):
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
    """Create the one canonical durable proposal-execution capability."""

    execution = _DbosProposalExecution(transport_registry_key)
    execution.validate()
    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=_proposal_policy_identity(
                transport_registry_key
            ),
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
