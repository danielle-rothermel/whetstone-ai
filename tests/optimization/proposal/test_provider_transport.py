from __future__ import annotations

import pytest
from dr_providers import (
    CostInfo,
    FailureClass,
    ProviderCallConfig,
    ProviderTransportOutcome,
    ProviderTransportResponse,
    TokenUsage,
    openrouter_chat_config,
)

from tests.provider import support as provider_support
from whetstone.core.identity import (
    IdentityRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.experiment.candidate import (
    Candidate,
    candidate_reference,
)
from whetstone.optimization.proposal.proposer import (
    PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA,
    PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION,
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
        proposal_authority_identity_hash="f" * 64,
        base_candidate=candidate_reference(
            Candidate(
                candidate_id="base",
                base_ref=typed_ref_for_record(
                    "test.candidate_parent", {"id": "parent"}
                ),
                payload={"user_prompt_template": "Initial {input}"},
            )
        ),
        context={"proposal_prompt": prompt},
    )


def _transport(
    *outcomes: ProviderTransportOutcome,
    temperature: float | None = 1.4,
    max_attempts: int = 1,
    resolved_provider_config: ProviderCallConfig | None = None,
):
    provider_config = openrouter_chat_config(model="proposal-model")
    transport_policy = provider_support.build_transport_policy()
    recording = provider_support.RecordingTransport(
        request=provider_support.build_request(),
        transport_policy=transport_policy,
        outcomes=list(outcomes),
    )
    resolved_refs: list[IdentityRef] = []

    def resolve(ref: IdentityRef):
        resolved_refs.append(ref)
        if resolved_provider_config is not None:
            return resolved_provider_config
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
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                provider_config.model_dump(mode="json"),
            ),
            record_hash=provider_config.identity_hash,
        ),
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
    assert resolved_refs == [config.provider_call_config]
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


def test_unset_temperature_is_not_added_to_provider_controls() -> None:
    proposer, config, recording, _ = _transport(
        provider_support.response_outcome(text="candidate"),
        temperature=None,
    )

    proposer.draft(config, _proposal_request(), 1)

    assert recording.served[0].config.controls.temperature is None


def test_preserves_provider_response_usage_cost_and_attempt_evidence() -> None:
    response = ProviderTransportResponse(
        text="one instruction",
        response_body={"id": "resp-1", "output": "one instruction"},
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
    assert draft.request_evidence["provider_call_config"] == (
        config.provider_call_config.model_dump(mode="json")
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
    assert drafts[1].terminal_failure is not None
    assert "blank-provider-generation" in drafts[1].terminal_failure.message
    assert len(recording.served) == 2
    assert (
        drafts[1].response_evidence["provider_call_result"][
            "semantic_failure"
        ]["failure_class"]
        == "blank-provider-generation"
    )


def test_rejected_response_retains_accounting_and_failure_evidence() -> None:
    response = ProviderTransportResponse(
        text="   ",
        response_body={"id": "rejected-1", "output": "   "},
        response_id="rejected-1",
        model="proposal-model",
        finish_reason="stop",
        usage=TokenUsage(total_tokens=23),
        cost=CostInfo(total_cost=0.047),
    )
    proposer, config, recording, _ = _transport(response)

    (draft,) = proposer.draft(config, _proposal_request(), 1)

    assert draft.failed
    assert draft.template == ""
    assert draft.usage == {"total_tokens": 23}
    assert draft.cost == 0.047
    assert draft.terminal_failure is not None
    assert draft.terminal_failure.model_dump(mode="json") == {
        "code": "proposal_failed",
        "message": (
            "provider proposer failed with blank-provider-generation: "
            "provider returned a blank or whitespace-only generation"
        ),
        "details": {},
    }

    result_evidence = draft.response_evidence["provider_call_result"]
    assert (
        result_evidence["logical_call_id"]
        == (draft.request_evidence["logical_call_id"])
    )
    assert result_evidence["provider_generation"] is None
    assert result_evidence["semantic_failure"] == {
        "failure_class": "blank-provider-generation",
        "message": "provider returned a blank or whitespace-only generation",
        "transport_failure": None,
        "rejected_response": response.model_dump(mode="json"),
    }
    assert (
        result_evidence["attempts"][0]["semantic_failure"]
        == (result_evidence["semantic_failure"])
    )
    assert "response_metadata" not in draft.response_evidence
    assert "response_id" not in draft.response_evidence
    assert len(recording.served) == 1


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
        update={
            "provider_call_config": config.provider_call_config.model_copy(
                update={"record_hash": "f" * 64}
            )
        }
    )

    with pytest.raises(ValueError, match="hash does not match"):
        proposer.draft(mismatched, _proposal_request(), 1)

    assert recording.served == []


def test_resolved_provider_config_record_ref_must_match_before_call() -> None:
    resolved = openrouter_chat_config(model="proposal-model")
    claimed_identity_hash = resolved.identity_hash
    wrong_record = resolved.model_copy(
        update={
            "controls": resolved.controls.model_copy(
                update={"temperature": 0.25}
            )
        }
    )
    assert wrong_record.identity_hash == claimed_identity_hash

    proposer, config, recording, resolved_refs = _transport(
        provider_support.response_outcome(text="unused"),
        resolved_provider_config=wrong_record,
    )
    wrong_ref = typed_ref_for_record(
        str(config.provider_call_config.record_ref.schema_name),
        wrong_record.model_dump(mode="json"),
    )
    assert wrong_ref != config.provider_call_config.record_ref
    assert (
        wrong_record.identity_hash == config.provider_call_config.record_hash
    )

    with pytest.raises(ValueError, match="record does not match"):
        proposer.draft(config, _proposal_request(), 1)

    assert resolved_refs == [config.provider_call_config]
    assert recording.served == []
    assert recording.produced == []


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


def test_proposal_request_binds_exact_base_candidate() -> None:
    first = _proposal_request()
    other_base = first.base_candidate.record.model_copy(
        update={"candidate_id": "other"}
    )
    second = ProposalRequest(
        proposal_mode=first.proposal_mode,
        request_ordinal=first.request_ordinal,
        proposal_authority_identity_hash=(
            first.proposal_authority_identity_hash
        ),
        base_candidate=candidate_reference(other_base),
        context=first.context.to_json(),
    )

    assert first.identity_hash() != second.identity_hash()


def test_fake_transport_never_invents_padding_candidates() -> None:
    transport = FakeProposerTransport(
        {("seed_proposal", 0): ("only one",)},
        execution_policy_hash="b" * 64,
        prompt_adapter_identity_hash="c" * 64,
    )
    config = ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config", {"route": "proposal"}
            ),
            record_hash="a" * 64,
        ),
        temperature=1.4,
    )

    drafts = transport.draft(config, _proposal_request(), 2)

    assert drafts[0].template == "only one"
    assert drafts[1].failed
    assert drafts[1].template == ""
    assert drafts[1].terminal_failure is not None
    assert "underfilled strict batch" in drafts[1].terminal_failure.message


def test_provider_transport_cannot_be_subclassed_or_mutated() -> None:
    proposer, _config, recording, _refs = _transport(
        provider_support.response_outcome(text="must not run"),
    )

    assert not hasattr(proposer, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        proposer.draft = lambda *_args, **_kwargs: ()
    with pytest.raises((AttributeError, TypeError)):
        proposer._execution_policy = None
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("ProposerOverride", (ProviderProposerTransport,), {})

    assert recording.served == []


def test_transport_durability_schema_constants() -> None:
    assert PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA == (
        "whetstone.provider_proposer_transport_durability"
    )
    assert PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION == 1


def test_transport_durability_identity_payload_and_digest_are_golden() -> None:

    golden = "862c33a44ec69924e65f4556e5077ba61e0afbcfcb129e4cbd32ce7fa75b767c"
    assert (
        compute_identity_hash(
            schema=PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA,
            schema_version=(
                PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION
            ),
            payload={
                "execution_policy_hash": "a" * 64,
                "prompt_adapter_identity_hash": "b" * 64,
            },
        )
        == golden
    )

    scripted = FakeProposerTransport(
        {},
        execution_policy_hash="a" * 64,
        prompt_adapter_identity_hash="b" * 64,
    )
    assert scripted.durability_identity_hash == golden


def test_transport_durability_identity_binds_policy_and_prompt_adapter() -> (
    None
):
    outcome = provider_support.response_outcome(text="identity fixture")
    base, _c, _r, _refs = _transport(outcome)
    changed_policy, _c2, _r2, _refs2 = _transport(outcome, max_attempts=3)
    changed_adapter = FakeProposerTransport(
        {},
        execution_policy_hash=base.execution_policy_hash,
        prompt_adapter_identity_hash="d" * 64,
    )

    assert base.durability_identity_hash == compute_identity_hash(
        schema=PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA,
        schema_version=PROVIDER_PROPOSER_TRANSPORT_DURABILITY_SCHEMA_VERSION,
        payload={
            "execution_policy_hash": base.execution_policy_hash,
            "prompt_adapter_identity_hash": (
                base.prompt_adapter_identity_hash
            ),
        },
    )
    assert (
        len(
            {
                base.durability_identity_hash,
                changed_policy.durability_identity_hash,
                changed_adapter.durability_identity_hash,
            }
        )
        == 3
    )
