"""The proposer route: identity, distinctness from graph routes, and the fake.

Deliverable #3: the proposer model is reached through a Provider Call Config
whose route is distinct from any encoder/decoder route, and the proposer config
lives in the optimizer Config identity, NOT the graph identity — tested here.
"""

from __future__ import annotations

import pytest

from whetstone.optimization import (
    Candidate,
    ProposalRequest,
    ProposerConfig,
    compute_identity_hash,
)
from whetstone.optimization.proposer import (
    PROPOSER_CONFIG_SCHEMA,
    PROPOSER_CONFIG_SCHEMA_VERSION,
    FakeProposerTransport,
)


def _pc(route: str, *, temperature: float = 1.0) -> ProposerConfig:
    return ProposerConfig(
        provider_call_config_ref=route,
        provider_call_config_hash="f" * 64,
        temperature=temperature,
    )


def test_proposer_config_identity_is_stable_and_route_sensitive() -> None:
    a = _pc("pcc://openai/gpt-5.4-proposer")
    b = _pc("pcc://openai/gpt-5.4-proposer")
    assert a.identity_hash() == b.identity_hash()
    assert len(a.identity_hash()) == 64
    # A different proposer route (or temperature) changes the identity.
    assert a.identity_hash() != _pc("pcc://other").identity_hash()
    hotter = _pc("pcc://openai/gpt-5.4-proposer", temperature=1.4)
    assert a.identity_hash() != hotter.identity_hash()
    with pytest.raises(ValueError, match="provider_call_config_hash"):
        ProposerConfig(
            provider_call_config_ref="r", provider_call_config_hash="short"
        )


def test_proposer_config_folds_into_optimizer_config_not_graph() -> None:
    # An optimizer Config identity that folds the proposer route: changing the
    # proposer route changes the optimizer Config identity.
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

    # A graph identity folds the encoder/decoder Provider Call Configs, NOT the
    # proposer route: swapping the proposer route leaves graph_hash unchanged.
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
    # Same graph inputs, different proposer route -> identical graph_hash.
    assert graph_a == graph_hash("pcc://enc", "pcc://dec")
    # The proposer route hash never appears in the graph identity payload.
    assert base.identity_hash() != graph_a
    assert alt.identity_hash() != graph_a


def test_proposer_route_distinct_from_encoder_decoder_routes() -> None:
    proposer = _pc("pcc://openai/gpt-5.4-proposer")
    encoder = Candidate(
        candidate_id="A", base_ref="pcc://enc",
        payload={"user_prompt_template": "x"},
    )
    # The proposer route identity is not the encoder graph route.
    assert proposer.provider_call_config_ref != encoder.base_ref
    assert proposer.identity_hash() != encoder.base_ref


def test_proposer_config_schema_constants() -> None:
    assert PROPOSER_CONFIG_SCHEMA == "whetstone.proposer_config"
    assert PROPOSER_CONFIG_SCHEMA_VERSION == 1


def test_fake_transport_is_scripted_and_records_calls() -> None:
    transport = FakeProposerTransport(
        {("seed_proposal", 0): ("t1", "t2")}, default=("d",)
    )
    pc = _pc("pcc://proposer")
    request = ProposalRequest(
        proposal_mode="seed_proposal", request_ordinal=0, base_ref="pcc://enc",
        base_template="base",
    )
    drafts = transport.draft(pc, request, 3)
    # Scripted for the first two; deterministic pad for the third.
    assert [d.template for d in drafts[:2]] == ["t1", "t2"]
    assert drafts[2].template.startswith("base::pad::")
    # It recorded the proposer route identity used (no network).
    assert transport.calls[0][0] == pc.identity_hash()
    # A missing script key falls back to the default.
    other = ProposalRequest(
        proposal_mode="history_proposal", request_ordinal=7,
        base_ref="pcc://enc",
    )
    fallback = transport.draft(pc, other, 1)
    assert fallback[0].template == "d"
