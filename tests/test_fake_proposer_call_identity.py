"""The fake proposer transport carries call identity and prices honestly.

Two properties run cost depends on, both of which the toy/test path used to
break:

* every draft that made a provider call carries a ``logical_call_id``, so a
  Step Result reported twice -- a replay, a resumed Step -- de-duplicates
  instead of double-counting the same spend;
* a draft that made *no* provider call carries no usage at all, so a scripted
  underfill cannot appear as a priced zero-dollar call and invent a complete
  ``usd`` total.
"""

from __future__ import annotations

from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposalRequest,
    ProposerConfig,
)

_POLICY_HASH = "b" * 64
_ADAPTER_HASH = "c" * 64


def _config() -> ProposerConfig:
    record_ref = typed_ref_for_record(
        "whetstone.testing.provider_call_config", {"model": "fake"}
    )
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=record_ref, record_hash=record_ref.content_hash
        )
    )


def _request(ordinal: int = 0) -> ProposalRequest:
    base = Candidate(
        candidate_id="fake-base",
        base_ref=typed_ref_for_record(
            "whetstone.testing.fake_root_candidate", {"seed": "fake"}
        ),
        payload={"user_prompt_template": "Reply to: {prompt}"},
    )
    return ProposalRequest(
        proposal_mode="copro_instruction",
        request_ordinal=ordinal,
        proposal_authority_identity_hash="a" * 64,
        base_candidate=candidate_reference(base),
        mutation_field="user_prompt_template",
    )


def _transport(script) -> FakeProposerTransport:
    return FakeProposerTransport(
        script,
        execution_policy_hash=_POLICY_HASH,
        prompt_adapter_identity_hash=_ADAPTER_HASH,
    )


def test_a_drafted_proposal_carries_a_logical_call_id() -> None:
    transport = _transport({("copro_instruction", 0): ("First.", "Second.")})

    drafts = transport.draft(_config(), _request(), 2)

    assert [draft.logical_call_id for draft in drafts] != ["", ""]
    assert all(draft.logical_call_id for draft in drafts)
    assert all(draft.call_usage().call_id for draft in drafts)


def test_each_batch_slot_gets_a_distinct_call_id() -> None:
    transport = _transport({("copro_instruction", 0): ("First.", "Second.")})

    drafts = transport.draft(_config(), _request(), 2)

    ids = {draft.logical_call_id for draft in drafts}
    assert len(ids) == 2


def test_the_call_id_is_stable_across_a_redrive() -> None:
    # De-duplication only works if a re-drive mints the same identity.
    first = _transport({("copro_instruction", 0): ("First.",)})
    second = _transport({("copro_instruction", 0): ("First.",)})

    a = first.draft(_config(), _request(), 1)[0]
    b = second.draft(_config(), _request(), 1)[0]

    assert a.logical_call_id == b.logical_call_id


def test_a_different_request_gets_a_different_call_id() -> None:
    transport = _transport(
        {
            ("copro_instruction", 0): ("First.",),
            ("copro_instruction", 1): ("Second.",),
        }
    )

    a = transport.draft(_config(), _request(ordinal=0), 1)[0]
    b = transport.draft(_config(), _request(ordinal=1), 1)[0]

    assert a.logical_call_id
    assert a.logical_call_id != b.logical_call_id


def test_an_underfilled_slot_made_no_call_and_carries_no_usage() -> None:
    # The script ran out: this slot never reached a provider, so it must not
    # become a priced zero-dollar call in the run's proposer total.
    transport = _transport({("copro_instruction", 0): ("Only one.",)})

    drafts = transport.draft(_config(), _request(), 2)

    underfilled = drafts[1]
    assert underfilled.terminal_failure is not None
    assert underfilled.cost is None
    assert underfilled.call_usage() is None


def test_an_empty_draft_did_make_a_call_and_keeps_its_usage() -> None:
    # An empty response is a real, billed call that simply failed.
    transport = _transport({("copro_instruction", 0): ("",)})

    draft = transport.draft(_config(), _request(), 1)[0]

    assert draft.terminal_failure is not None
    usage = draft.call_usage()
    assert usage is not None
    assert usage.call_id


def test_a_successful_draft_keeps_its_usage() -> None:
    transport = _transport({("copro_instruction", 0): ("First.",)})

    draft = transport.draft(_config(), _request(), 1)[0]

    usage = draft.call_usage()
    assert usage is not None
    assert usage.call_id
