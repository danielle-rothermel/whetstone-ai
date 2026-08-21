"""The public inline proposal executor constructor."""

from __future__ import annotations

import pytest

from whetstone.core.leasing import ReplayPolicy
from whetstone.core.identity import compute_identity_hash
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    build_inline_proposal_executor,
    ProposalRequest,
    ProposerConfig,
    require_canonical_proposal_executor,
)

POLICY_HASH = compute_identity_hash(
    schema="whetstone.testing.inline_proposal_executor",
    schema_version=1,
    payload={"mode": "inline"},
)


def _proposer_route_ref():
    from whetstone.core.identity import IdentityRef, typed_ref_for_record

    record_ref = typed_ref_for_record(
        "whetstone.testing.provider_call_config", {"model": "fake"}
    )
    return IdentityRef(
        record_ref=record_ref,
        record_hash=record_ref.content_hash,
    )


def _request() -> ProposalRequest:
    from whetstone.core.identity import typed_ref_for_record

    base = Candidate(
        candidate_id="inline-base",
        base_ref=typed_ref_for_record(
            "whetstone.testing.inline_root_candidate", {"seed": "inline"}
        ),
        payload={"user_prompt_template": "Reply to: {prompt}"},
    )
    return ProposalRequest(
        proposal_mode="gepa_reflection",
        request_ordinal=0,
        proposal_authority_identity_hash="a" * 64,
        base_candidate=candidate_reference(base),
        mutation_field="user_prompt_template",
    )


def test_inline_executor_is_the_canonical_durable_capability() -> None:
    executor = build_inline_proposal_executor(policy_identity_hash=POLICY_HASH)

    assert type(executor) is DurableProposalExecutor
    assert executor.policy_identity_hash == POLICY_HASH
    assert executor.recovery_policy is ReplayPolicy.DURABLE_WORKFLOW
    # GEPA and COPRO accept it without reaching for a private name.
    assert (
        require_canonical_proposal_executor(
            executor, algorithm="GEPA", purpose="paid reflection call"
        )
        is executor
    )


def test_inline_executor_drafts_through_the_transport() -> None:
    executor = build_inline_proposal_executor(policy_identity_hash=POLICY_HASH)
    transport = FakeProposerTransport(
        {("gepa_reflection", 0): ("Answer {prompt} plainly.",)},
        execution_policy_hash="b" * 64,
        prompt_adapter_identity_hash="c" * 64,
    )
    request = _request()
    config = ProposerConfig(provider_call_config=_proposer_route_ref())

    drafts = executor.execute(
        config=config,
        request=request,
        transport=transport,
        count=1,
    )

    assert len(drafts) == 1
    assert drafts[0].template == "Answer {prompt} plainly."
    assert transport.calls == [(config.identity_hash(), request, 1)]


def test_inline_executor_requires_a_full_policy_hash() -> None:
    with pytest.raises(ValueError, match="policy_identity_hash"):
        build_inline_proposal_executor(policy_identity_hash="short")
