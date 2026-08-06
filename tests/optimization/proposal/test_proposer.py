from __future__ import annotations

import pytest

from whetstone.core.identity import (
    IdentityRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.optimization.proposal.proposer import (
    PROPOSAL_REQUEST_SCHEMA,
    PROPOSAL_REQUEST_SCHEMA_VERSION,
    PROPOSER_CONFIG_SCHEMA,
    PROPOSER_CONFIG_SCHEMA_VERSION,
    FakeProposerTransport,
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
)


def _pc(route: str, *, temperature: float = 1.0) -> ProposerConfig:
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                {"route": route},
            ),
            identity_hash="f" * 64,
        ),
        temperature=temperature,
    )


def _base_candidate_ref(
    payload: dict[str, object] | None = None,
) -> CandidateRef:
    return candidate_reference(
        Candidate(
            candidate_id="base",
            base_ref=typed_ref_for_record(
                "whetstone.test.candidate_parent",
                {"id": "parent"},
            ),
            payload=(
                {"user_prompt_template": "base"}
                if payload is None
                else payload
            ),
        )
    )


def test_proposer_config_identity_is_stable_and_route_sensitive() -> None:
    a = _pc("pcc://openai/gpt-5.4-proposer")
    b = _pc("pcc://openai/gpt-5.4-proposer")
    assert a.identity_hash() == b.identity_hash()
    assert len(a.identity_hash()) == 64
    assert a.identity_hash() != _pc("pcc://other").identity_hash()
    hotter = _pc("pcc://openai/gpt-5.4-proposer", temperature=1.4)
    assert a.identity_hash() != hotter.identity_hash()


def test_proposer_config_identity_payload_is_golden() -> None:
    assert _pc("pcc://openai/gpt-5.4-proposer").identity_payload() == {
        "provider_call_config": {
            "record_ref": {
                "schema_name": "dr_providers.provider_call_config",
                "content_hash": (
                    "bc64b6c4fbffa36e113544f2b60bd4e5"
                    "fcbbbc1f73b0185f04a0db2597cdc625"
                ),
            },
            "identity_hash": "f" * 64,
        },
        "temperature": 1.0,
    }


def test_proposer_config_identity_digest_is_golden() -> None:
    assert (
        _pc("pcc://openai/gpt-5.4-proposer").identity_hash()
        == "2df0fa4fb993fb4085df0e6c77bda9ac0dfef7cea8eef8d88cc747bea45d62a3"
    )


def test_proposer_config_rejects_invalid_provider_identity_hash() -> None:
    with pytest.raises(ValueError, match="identity hash"):
        ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    {"route": "r"},
                ),
                identity_hash="short",
            )
        )


def test_proposer_config_folds_into_optimizer_config_not_graph() -> None:
    def optimizer_config_hash(proposer: ProposerConfig) -> str:
        return compute_identity_hash(
            schema="whetstone.test.optimizer_config",
            schema_version=1,
            payload={
                "algorithm": "copro",
                "breadth": 4,
                "depth": 2,
                "proposer_config": proposer.identity_hash(),
            },
        )

    base = _pc("pcc://openai/gpt-5.4-proposer")
    alt = _pc("pcc://openai/gpt-5.4-proposer-b")
    assert optimizer_config_hash(base) != optimizer_config_hash(alt)

    def graph_hash(encoder_route: str, decoder_route: str) -> str:
        return compute_identity_hash(
            schema="whetstone.test.graph",
            schema_version=1,
            payload={
                "encoder_route": encoder_route,
                "decoder_route": decoder_route,
                "user_prompt_template": "describe concisely",
            },
        )

    graph_a = graph_hash("pcc://enc", "pcc://dec")
    assert graph_a == graph_hash("pcc://enc", "pcc://dec")
    assert base.identity_hash() != graph_a
    assert alt.identity_hash() != graph_a


def test_proposer_route_distinct_from_encoder_decoder_routes() -> None:
    proposer = _pc("pcc://openai/gpt-5.4-proposer")
    encoder = Candidate(
        candidate_id="A",
        base_ref=typed_ref_for_record(
            "dr_providers.provider_call_config",
            {"route": "pcc://enc"},
        ),
        payload={"user_prompt_template": "x"},
    )
    assert proposer.provider_call_config.record_ref != encoder.base_ref
    assert proposer.identity_hash() != encoder.base_ref


def test_proposer_config_schema_constants() -> None:
    assert PROPOSER_CONFIG_SCHEMA == "whetstone.proposer_config"
    assert PROPOSER_CONFIG_SCHEMA_VERSION == 1


def test_proposal_request_schema_constants() -> None:
    assert PROPOSAL_REQUEST_SCHEMA == "whetstone.proposal_request"
    assert PROPOSAL_REQUEST_SCHEMA_VERSION == 2


def _golden_proposal_request() -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=3,
        optimization_run_identity_hash="f" * 64,
        base_candidate=_base_candidate_ref(),
        context={"proposal_prompt": "Improve this prompt."},
    )


def test_proposal_request_identity_payload_is_golden() -> None:
    assert _golden_proposal_request().identity_payload() == {
        "proposal_mode": "seed_proposal",
        "request_ordinal": 3,
        "optimization_run_identity_hash": "f" * 64,
        "base_candidate": {
            "record_ref": {
                "schema_name": "whetstone.optimization_candidate",
                "content_hash": (
                    "a9ad4c9ed294a02a6158db2c0d5685e0"
                    "3054f87676644da547fcc2beecfc455b"
                ),
            },
            "identity_hash": (
                "026534070c4cca3d8446ec47ab424e8e"
                "a452db63804d23387a88d975940b5b30"
            ),
        },
        "context": {"proposal_prompt": "Improve this prompt."},
    }


def test_proposal_request_identity_digest_is_golden() -> None:
    assert (
        _golden_proposal_request().identity_hash()
        == "518f1bd891b318dd306a9f6bf091141a30fc519cbc31d2ddedfa823f49ea2d47"
    )


def test_proposal_request_identity_payload_carries_no_extra_keys() -> None:
    payload = _golden_proposal_request().identity_payload()

    assert sorted(payload) == [
        "base_candidate",
        "context",
        "optimization_run_identity_hash",
        "proposal_mode",
        "request_ordinal",
    ]
    assert sorted(payload["base_candidate"]) == [
        "identity_hash",
        "record_ref",
    ]


def test_proposal_request_identity_ignores_unaddressed_base_payload() -> None:

    payload = _golden_proposal_request().identity_payload()

    assert "record" not in payload["base_candidate"]


def test_fake_transport_is_scripted_and_records_calls() -> None:
    transport = FakeProposerTransport(
        {("seed_proposal", 0): ("t1", "t2")},
        default=("d",),
        execution_policy_hash="a" * 64,
        prompt_adapter_identity_hash="b" * 64,
    )
    pc = _pc("pcc://proposer")
    request = ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        optimization_run_identity_hash="f" * 64,
        base_candidate=_base_candidate_ref(),
    )
    drafts = transport.draft(pc, request, 3)
    assert [d.template for d in drafts[:2]] == ["t1", "t2"]
    assert drafts[2].failed is True
    assert transport.calls[0][0] == pc.identity_hash()
    other = ProposalRequest(
        proposal_mode="history_proposal",
        request_ordinal=7,
        optimization_run_identity_hash="f" * 64,
        base_candidate=_base_candidate_ref(),
    )
    fallback = transport.draft(pc, other, 1)
    assert fallback[0].template == "d"


def test_proposal_context_evidence_and_usage_are_deeply_immutable() -> None:
    context = {"history": [{"score": 1.0}]}
    request = ProposalRequest(
        proposal_mode="history_proposal",
        request_ordinal=0,
        optimization_run_identity_hash="f" * 64,
        base_candidate=_base_candidate_ref(),
        context=context,
    )
    context["history"][0]["score"] = 0.0
    assert (
        request.model_dump(mode="json")["context"]["history"][0]["score"]
        == 1.0
    )

    evidence = {"request": {"messages": ["a"]}}
    usage = {"tokens": {"input": 10}}
    draft = FakeProposerTransport(
        {},
        execution_policy_hash="a" * 64,
        prompt_adapter_identity_hash="b" * 64,
    ).draft(
        _pc("pcc://proposer"),
        request,
        0,
    )
    assert draft == ()
    result = ProposalDraft(
        template="draft",
        request_evidence=evidence,
        response_evidence={"response": ["b"]},
        usage=usage,
        cost=0.0,
    )
    evidence["request"]["messages"].append("changed")
    usage["tokens"]["input"] = 999
    dumped = result.model_dump(mode="json")
    assert dumped["request_evidence"]["request"]["messages"] == ["a"]
    assert dumped["usage"]["tokens"]["input"] == 10


@pytest.mark.parametrize("temperature", [float("nan"), float("inf")])
def test_proposer_rejects_nonfinite_temperature(temperature) -> None:
    with pytest.raises(ValueError, match="finite"):
        _pc("pcc://proposer", temperature=temperature)


def test_proposal_request_owns_exact_base_and_derives_template() -> None:
    base_candidate = _base_candidate_ref(
        {"user_prompt_template": "derived", "fixed": "same"}
    )
    request = ProposalRequest(
        proposal_mode="proposal",
        request_ordinal=0,
        optimization_run_identity_hash="f" * 64,
        base_candidate=base_candidate,
    )
    assert request.base_candidate == base_candidate
    assert request.base_template == "derived"


@pytest.mark.parametrize(
    "payload",
    (
        {"fixed": "same"},
        {"user_prompt_template": 1},
    ),
)
def test_proposal_request_requires_string_mutation_field(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="mutation field"):
        ProposalRequest(
            proposal_mode="proposal",
            request_ordinal=0,
            optimization_run_identity_hash="f" * 64,
            base_candidate=_base_candidate_ref(payload),
        )


def test_proposal_request_rejects_independent_base_inputs() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        ProposalRequest.model_validate(
            {
                "proposal_mode": "proposal",
                "request_ordinal": 0,
                "base_candidate": _base_candidate_ref(),
                "base_ref": typed_ref_for_record(
                    "whetstone.test.base", {"id": "b"}
                ),
                "base_template": "unrelated",
            }
        )


def test_proposal_request_rejects_invalid_ordinal() -> None:
    with pytest.raises(ValueError):
        ProposalRequest(
            proposal_mode="proposal",
            request_ordinal=True,
            optimization_run_identity_hash="f" * 64,
            base_candidate=_base_candidate_ref(),
        )
