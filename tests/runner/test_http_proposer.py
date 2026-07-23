"""The live-path OpenRouter proposer transport (scripted, no live call).

``_HttpProposerTransport`` drafts template variants by driving the proposer
route through the bounded dr-providers attempt loop. Here the transport is
injected as a scripted fake so the drafting logic is exercised with no network:
a successful draft returns the completion text as the template; a failed draft
returns the base template unchanged (which the optimizer diff-check rejects, so
a failed call never becomes a fabricated candidate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dr_providers import (
    FailureClass,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    RawHttpRequest,
)

from tests.envs.support import FakeTransport, _response, transport_policy
from whetstone.optimization.proposer import ProposalRequest
from whetstone.runner.cli import _HttpProposerTransport, _proposal_prompt
from whetstone.runner.routes import route_for

_REQUEST = ProposalRequest(
    proposal_mode="seed_proposal",
    request_ordinal=0,
    base_ref="run-1",
    base_template="Answer the question: {input}",
)


def _proposer_route():
    return route_for("openrouter", role="proposer", temperature=1.0)


def _config():
    from whetstone.optimization.proposer import ProposerConfig

    route = _proposer_route()
    return ProposerConfig(
        provider_call_config_ref="pcc://openai/gpt-5.4-nano",
        provider_call_config_hash=route.call_config.identity_hash,
        temperature=1.0,
    )


def test_proposal_prompt_preserves_placeholder_and_asks_for_rewrite() -> None:
    prompt = _proposal_prompt(_REQUEST)
    assert "{input}" in prompt
    assert "REWRITTEN TEMPLATE" in prompt
    assert _REQUEST.base_template in prompt


def test_successful_draft_returns_completion_as_template() -> None:
    transport = FakeTransport(
        reply=lambda _p: "Carefully answer this question: {input}"
    )
    proposer = _HttpProposerTransport(_proposer_route(), transport=transport)
    drafts = proposer.draft(_config(), _REQUEST, count=2)
    assert len(drafts) == 2
    for d in drafts:
        assert d.template == "Carefully answer this question: {input}"
        assert d.usage["proposer_calls"] == 1
    # Two draft calls served through the injected transport.
    assert len(transport.served) == 2


@dataclass
class _FailingTransport:
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"}, body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request, policy=self.policy, raw_request=raw_request,
            outcome=ProviderTransportFailure(
                failure_class=FailureClass.PERMANENT,
                code="empty_completion",
                message="scripted permanent failure",
                retryable=False,
            ),
        )


def test_failed_draft_returns_base_template_unchanged() -> None:
    proposer = _HttpProposerTransport(
        _proposer_route(), transport=_FailingTransport()
    )
    drafts = proposer.draft(_config(), _REQUEST, count=1)
    assert len(drafts) == 1
    # A failed draft returns the base unchanged (the diff check then rejects
    # it) -- never a fabricated non-base candidate from a failed call.
    assert drafts[0].template == _REQUEST.base_template
    assert drafts[0].request_evidence["failed"] is True


def test_blank_completion_falls_back_to_base_template() -> None:
    # A whitespace-only completion is not a usable template: fall back to the
    # base (rejected by the diff check) rather than emit an empty template.
    transport = FakeTransport(reply=lambda _p: "   ")
    proposer = _HttpProposerTransport(_proposer_route(), transport=transport)
    drafts = proposer.draft(_config(), _REQUEST, count=1)
    assert drafts[0].template == _REQUEST.base_template


def test_response_helper_is_used_for_success_shape() -> None:
    # Sanity: the fake success response carries stop/text so the draft path's
    # success branch (result.succeeded, generation.text) is what runs.
    resp = _response("x")
    assert resp.finish_reason == "stop"
