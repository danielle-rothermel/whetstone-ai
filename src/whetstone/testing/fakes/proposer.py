from __future__ import annotations

from whetstone.core.identity import require_full_hash
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposalDraft,
    ProposalRequest,
    ProposerRouteConfig,
)

__all__ = ["DummyProposerTransport"]


class DummyProposerTransport(FakeProposerTransport):
    """Scripted proposer transport for sandbox optimizer previews."""

    def __init__(
        self,
        *,
        scripted_bodies: tuple[str, ...] = (),
        execution_policy_hash: str,
        prompt_adapter_identity_hash: str,
        proposal_mode: str = "seed_proposal",
        request_ordinal: int = 0,
    ) -> None:
        require_full_hash(
            execution_policy_hash,
            field="execution_policy_hash",
        )
        require_full_hash(
            prompt_adapter_identity_hash,
            field="prompt_adapter_identity_hash",
        )
        script = {
            (proposal_mode, request_ordinal): scripted_bodies,
        }
        super().__init__(
            script,
            default=scripted_bodies,
            execution_policy_hash=execution_policy_hash,
            prompt_adapter_identity_hash=prompt_adapter_identity_hash,
        )

    def propose_templates(
        self,
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        return self.draft(config, request, count)
