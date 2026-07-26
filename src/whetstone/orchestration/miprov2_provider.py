"""DBOS durability boundary for MIPROv2 proposal-model effects."""

from __future__ import annotations

from typing import Literal

from dbos import DBOS

from whetstone.optimization.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.miprov2 import Miprov2ProposalEffectExecutor
from whetstone.optimization.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
    RetryDurableProposerTransport,
)

MIPROV2_PROPOSAL_DBOS_POLICY_SCHEMA = "whetstone.miprov2_proposal_dbos_policy"
MIPROV2_PROPOSAL_DBOS_POLICY_VERSION = 2
MIPROV2_PROPOSAL_ATTEMPT_SCHEMA = "whetstone.miprov2_proposal_physical_attempt"
Miprov2ProposalDurabilityMode = Literal[
    "at_least_once",
    "provider_idempotent",
]


class Miprov2ProposalDurabilityError(RuntimeError):
    """The configured proposal crash-safety contract cannot be satisfied."""


_PROPOSAL_TRANSPORTS: dict[str, ProposerTransport] = {}


def register_miprov2_proposal_transport(
    registry_key: str,
    transport: ProposerTransport,
) -> None:
    """Register one stable transport dependency before DBOS launch."""

    require_full_hash(registry_key, field="transport_registry_key")
    existing = _PROPOSAL_TRANSPORTS.get(registry_key)
    if existing is not None and existing is not transport:
        raise Miprov2ProposalDurabilityError(
            "MIPROv2 proposal transport registry key is already bound"
        )
    _PROPOSAL_TRANSPORTS[registry_key] = transport


def _registered_transport(registry_key: str) -> ProposerTransport:
    try:
        return _PROPOSAL_TRANSPORTS[registry_key]
    except KeyError:
        raise Miprov2ProposalDurabilityError(
            "MIPROv2 proposal transport is not registered; configure the "
            "identity-keyed registry before DBOS launch"
        ) from None


@DBOS.step(retries_allowed=False)
def _proposal_provider_attempt_step(
    registry_key,
    request,
    logical_call_id: str,
    attempt_number: int,
    attempt_identity: str,
    durability_mode: Miprov2ProposalDurabilityMode,
):
    del logical_call_id, attempt_number
    transport = _registered_transport(registry_key)
    if not isinstance(transport, RetryDurableProposerTransport):
        raise Miprov2ProposalDurabilityError(
            "registered proposal transport has no physical-attempt boundary"
        )
    return transport.invoke_physical_attempt(
        request,
        idempotency_key=(
            attempt_identity
            if durability_mode == "provider_idempotent"
            else None
        ),
    )


@DBOS.step(retries_allowed=False)
def _proposal_provider_whole_call_step(
    registry_key: str,
    config: ProposerConfig,
    request: ProposalRequest,
    count: int,
) -> tuple[ProposalDraft, ...]:
    return _registered_transport(registry_key).draft(config, request, count)


class DbosMiprov2ProposalEffectExecutor(Miprov2ProposalEffectExecutor):
    """Durably execute proposal transport effects under an explicit contract.

    Provider-backed proposers expose their retry loop so every physical
    attempt is one retry-disabled DBOS step and every retry delay is a durable
    ``DBOS.sleep``.  ``provider_idempotent`` additionally requires the
    physical transport to accept a stable idempotency key.  The default
    ``at_least_once`` mode is honest about the irreducible crash window between
    a provider accepting a request and DBOS committing that attempt's
    checkpoint.

    Non-provider test transports do not expose physical attempts.  They retain
    a retry-disabled whole-call checkpoint fallback, which is also
    ``at_least_once`` and is rejected in ``provider_idempotent`` mode.
    """

    def __init__(
        self,
        *,
        transport_registry_key: str,
        durability_mode: Miprov2ProposalDurabilityMode = "at_least_once",
    ) -> None:
        require_full_hash(
            transport_registry_key,
            field="transport_registry_key",
        )
        if durability_mode not in (
            "at_least_once",
            "provider_idempotent",
        ):
            raise ValueError(
                "unsupported MIPROv2 proposal durability mode "
                f"{durability_mode!r}"
            )
        self._durability_mode: Miprov2ProposalDurabilityMode = durability_mode
        self._transport_registry_key = transport_registry_key

    @property
    def durability_mode(self) -> Miprov2ProposalDurabilityMode:
        return self._durability_mode

    @property
    def durability_scope_identity_hash(self) -> str:
        workflow_id = DBOS.workflow_id
        if workflow_id is None:
            raise RuntimeError(
                "MIPROv2 proposal effects require a DBOS workflow"
            )
        return compute_identity_hash(
            schema="whetstone.miprov2_proposal_dbos_scope",
            schema_version=2,
            payload={
                "workflow_id": workflow_id,
                "durability_policy_identity_hash": (
                    self.durability_policy_identity_hash
                ),
            },
        )

    @property
    def durability_policy_identity_hash(self) -> str:
        """Identity of retry/checkpoint/idempotency behavior."""

        return compute_identity_hash(
            schema=MIPROV2_PROPOSAL_DBOS_POLICY_SCHEMA,
            schema_version=MIPROV2_PROPOSAL_DBOS_POLICY_VERSION,
            payload={
                "durability_mode": self._durability_mode,
                "automatic_dbos_retries": False,
                "retry_boundary": "one_dbos_step_per_physical_attempt",
                "backoff": "dbos_sleep",
                "whole_call_fallback": "at_least_once_test_transports_only",
                "transport_registry_key": self._transport_registry_key,
            },
        )

    def execute(
        self,
        *,
        config: ProposerConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        """Replay the exact same DBOS operation transcript on recovery."""

        registered = _registered_transport(self._transport_registry_key)
        if registered is not transport:
            raise Miprov2ProposalDurabilityError(
                "executor transport does not match its pre-registered "
                "identity key"
            )
        if isinstance(transport, RetryDurableProposerTransport):
            if (
                self._durability_mode == "provider_idempotent"
                and not transport.supports_provider_idempotency
            ):
                raise Miprov2ProposalDurabilityError(
                    "provider_idempotent proposal durability requires a "
                    "physical transport that accepts stable idempotency "
                    "evidence"
                )
            return transport.draft_with_attempt_executor(
                config,
                request,
                count,
                execute_attempt=self._execute_physical_attempt,
                sleep=self._durable_sleep,
            )
        elif self._durability_mode == "provider_idempotent":
            raise Miprov2ProposalDurabilityError(
                "provider_idempotent proposal durability requires a "
                "retry-durable proposer transport"
            )

        return _proposal_provider_whole_call_step(
            self._transport_registry_key,
            config,
            request,
            count,
        )

    def _execute_physical_attempt(
        self,
        transport,
        request,
        logical_call_id: str,
        attempt_number: int,
    ):
        del transport
        attempt_identity = compute_identity_hash(
            schema=MIPROV2_PROPOSAL_ATTEMPT_SCHEMA,
            schema_version=1,
            payload={
                "logical_call_id": logical_call_id,
                "attempt_number": attempt_number,
                "durability_policy_identity_hash": (
                    self.durability_policy_identity_hash
                ),
            },
        )
        return _proposal_provider_attempt_step(
            self._transport_registry_key,
            request,
            logical_call_id,
            attempt_number,
            attempt_identity,
            self._durability_mode,
        )

    @staticmethod
    def _durable_sleep(seconds: float) -> None:
        if seconds > 0:
            DBOS.sleep(seconds)


__all__ = [
    "DbosMiprov2ProposalEffectExecutor",
    "Miprov2ProposalDurabilityError",
    "Miprov2ProposalDurabilityMode",
    "register_miprov2_proposal_transport",
]
