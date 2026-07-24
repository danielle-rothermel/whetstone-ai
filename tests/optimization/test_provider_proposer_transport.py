from __future__ import annotations

import pytest
from dr_providers import (
    CostInfo,
    FailureClass,
    ProviderTransportOutcome,
    ProviderTransportResponse,
    TokenUsage,
    openrouter_chat_config,
)

from tests.provider import support as provider_support
from whetstone.optimization.proposer import (
    FakeProposerTransport,
    ProposalRequest,
    ProposerConfig,
    ProviderProposerTransport,
)


def _proposal_request(
    *, prompt: object = "Improve this prompt."
) -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        base_ref="route-a",
        base_template="Initial {input}",
        context={"proposal_prompt": prompt},
    )


def _transport(
    *outcomes: ProviderTransportOutcome,
    temperature: float = 1.4,
    max_attempts: int = 1,
):
    provider_config = openrouter_chat_config(model="proposal-model")
    transport_policy = provider_support.build_transport_policy()
    recording = provider_support.RecordingTransport(
        request=provider_support.build_request(),
        transport_policy=transport_policy,
        outcomes=list(outcomes),
    )
    resolved_refs: list[str] = []

    def resolve(ref: str):
        resolved_refs.append(ref)
        return provider_config

    proposer = ProviderProposerTransport(
        resolve_provider_call_config=resolve,
        transport=recording,
        execution_policy=provider_support.build_execution_policy(
            max_attempts=max_attempts,
            transport_policy=transport_policy,
        ),
        clock=provider_support.FakeClock(),
        sleep=provider_support.SleepRecorder(),
    )
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash=provider_config.identity_hash,
        temperature=temperature,
    )
    return proposer, config, recording, resolved_refs


def test_exact_batch_uses_identical_prompt_and_temperature() -> None:
    proposer, config, recording, resolved_refs = _transport(
        provider_support.response_outcome(text="candidate one"),
        provider_support.response_outcome(text="candidate two"),
        provider_support.response_outcome(text="candidate three"),
        temperature=1.4,
    )
    request = _proposal_request(prompt="Optimize exactly this.")

    drafts = proposer.draft(config, request, 3)

    assert [draft.template for draft in drafts] == [
        "candidate one",
        "candidate two",
        "candidate three",
    ]
    assert len(recording.served) == 3
    assert resolved_refs == ["provider://proposal"]
    assert all(
        served.config.controls.temperature == 1.4
        for served in recording.served
    )
    assert all(
        served.transcript.messages[0].content == "Optimize exactly this."
        for served in recording.served
    )
    assert [draft.request_evidence["batch_slot"] for draft in drafts] == [
        0,
        1,
        2,
    ]
    assert all(
        draft.request_evidence["logical_batch_size"] == 3 for draft in drafts
    )
    assert (
        len({draft.request_evidence["logical_call_id"] for draft in drafts})
        == 3
    )


def test_preserves_provider_response_usage_cost_and_attempt_evidence() -> None:
    response = ProviderTransportResponse(
        text="one instruction",
        raw_body={"id": "resp-1", "output": "one instruction"},
        response_id="resp-1",
        model="proposal-model",
        finish_reason="stop",
        usage=TokenUsage(total_tokens=17),
        cost=CostInfo(total_cost=0.031),
    )
    proposer, config, recording, _ = _transport(response)

    (draft,) = proposer.draft(config, _proposal_request(), 1)

    assert draft.template == "one instruction"
    assert draft.usage == {"total_tokens": 17}
    assert draft.cost == 0.031
    assert draft.request_evidence["provider_call_config_ref"] == (
        "provider://proposal"
    )
    assert (
        draft.request_evidence["base_provider_call_config_hash"]
        == config.provider_call_config_hash
    )
    assert (
        draft.request_evidence["materialized_provider_call_config_hash"]
        == recording.served[0].config.identity_hash
    )
    assert draft.request_evidence["provider_execution_policy_hash"] == (
        proposer.execution_policy_hash
    )
    assert draft.request_evidence["prompt_adapter_identity_hash"] == (
        proposer.prompt_adapter_identity_hash
    )
    result_evidence = draft.response_evidence["provider_call_result"]
    assert len(result_evidence["attempts"]) == 1
    assert draft.response_evidence["response_metadata"]["id"] == "resp-1"
    assert draft.response_evidence["response_id"] == "resp-1"


def test_invalid_generation_is_an_explicit_failed_slot_not_an_underfill() -> (
    None
):
    proposer, config, recording, _ = _transport(
        provider_support.response_outcome(text="valid"),
        provider_support.response_outcome(text="   "),
    )

    drafts = proposer.draft(config, _proposal_request(), 2)

    assert len(drafts) == 2
    assert drafts[0].template == "valid"
    assert not drafts[0].failed
    assert drafts[1].failed
    assert drafts[1].template == ""
    assert "blank-generation" in (drafts[1].failure_detail or "")
    assert len(recording.served) == 2
    assert (
        drafts[1].response_evidence["provider_call_result"][
            "semantic_failure"
        ]["failure_class"]
        == "blank-generation"
    )


def test_injected_attempt_policy_retries_within_one_batch_slot() -> None:
    transient = provider_support.failure_outcome(
        failure_class=FailureClass.TRANSIENT
    )
    proposer, config, recording, _ = _transport(
        transient,
        provider_support.response_outcome(text="after retry"),
        max_attempts=2,
    )

    (draft,) = proposer.draft(config, _proposal_request(), 1)

    assert draft.template == "after retry"
    assert len(recording.served) == 2
    result_evidence = draft.response_evidence["provider_call_result"]
    assert len(result_evidence["attempts"]) == 2


def test_resolved_provider_config_hash_must_match_proposer_identity() -> None:
    proposer, config, recording, _ = _transport(
        provider_support.response_outcome(text="unused")
    )
    mismatched = config.model_copy(
        update={"provider_call_config_hash": "f" * 64}
    )

    with pytest.raises(ValueError, match="hash does not match"):
        proposer.draft(mismatched, _proposal_request(), 1)

    assert recording.served == []


@pytest.mark.parametrize("count", [0, -1, True])
def test_rejects_nonpositive_or_boolean_count(count: int) -> None:
    proposer, config, recording, _ = _transport(
        provider_support.response_outcome(text="unused")
    )

    with pytest.raises(ValueError, match="positive integer"):
        proposer.draft(config, _proposal_request(), count)

    assert recording.served == []


@pytest.mark.parametrize("prompt", [None, "", "   ", ["not", "text"]])
def test_rejects_missing_or_invalid_prompt(prompt: object) -> None:
    proposer, config, recording, _ = _transport(
        provider_support.response_outcome(text="unused")
    )

    with pytest.raises(ValueError, match="nonblank proposal_prompt"):
        proposer.draft(config, _proposal_request(prompt=prompt), 1)

    assert recording.served == []


def test_logical_call_identity_binds_full_proposal_request() -> None:
    proposer, config, _, _ = _transport(
        provider_support.response_outcome(text="first"),
        provider_support.response_outcome(text="second"),
        provider_support.response_outcome(text="third"),
    )
    base = _proposal_request(prompt="prompt a")
    different = _proposal_request(prompt="prompt b")

    (first,) = proposer.draft(config, base, 1)
    (replay,) = proposer.draft(config, base, 1)
    (other,) = proposer.draft(config, different, 1)

    assert (
        first.request_evidence["logical_call_id"]
        == (replay.request_evidence["logical_call_id"])
    )
    assert (
        first.request_evidence["logical_call_id"]
        != (other.request_evidence["logical_call_id"])
    )


def test_proposal_request_optionally_binds_run_and_step_identity() -> None:
    first = _proposal_request().model_copy(
        update={"run_id": "run-a", "step_index": 0}
    )
    second = first.model_copy(update={"run_id": "run-b"})

    assert first.identity_hash() != second.identity_hash()


def test_fake_transport_strict_mode_never_invents_padding_candidates() -> None:
    transport = FakeProposerTransport(
        {("seed_proposal", 0): ("only one",)},
        execution_policy_hash="b" * 64,
        prompt_adapter_identity_hash="c" * 64,
        strict=True,
    )
    config = ProposerConfig(
        provider_call_config_ref="provider://proposal",
        provider_call_config_hash="a" * 64,
        temperature=1.4,
    )

    drafts = transport.draft(config, _proposal_request(), 2)

    assert drafts[0].template == "only one"
    assert drafts[1].failed
    assert drafts[1].template == ""
    assert "underfilled strict batch" in (drafts[1].failure_detail or "")
