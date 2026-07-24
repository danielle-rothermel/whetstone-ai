"""The live-path OpenRouter proposer transport (scripted, no live call).

``_HttpProposerTransport`` drafts template variants by driving the proposer
route through the bounded dr-providers attempt loop. Here the transport is
injected as a scripted fake so the drafting logic is exercised with no network:
a successful draft returns the completion text as the template; a failed OR
empty-completion draft is a TYPED FAILURE (``failed=True``, empty template) --
the base template is NEVER echoed back, so a failed call can never become a
fabricated candidate.
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
    assert "PROPOSED INSTRUCTION TEMPLATE" in prompt
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


def test_failed_draft_is_typed_failure_not_base_template() -> None:
    proposer = _HttpProposerTransport(
        _proposer_route(), transport=_FailingTransport()
    )
    drafts = proposer.draft(_config(), _REQUEST, count=1)
    assert len(drafts) == 1
    # A failed draft is a TYPED FAILURE with NO template -- the base is NEVER
    # echoed back (no fabricated candidate from a failed call).
    assert drafts[0].failed is True
    assert drafts[0].template == ""
    assert drafts[0].template != _REQUEST.base_template
    assert drafts[0].failure_detail and "failed" in drafts[0].failure_detail
    assert drafts[0].request_evidence["failed"] is True


def test_blank_completion_is_typed_failure_not_base_template() -> None:
    # A whitespace-only completion is not a usable template: it is a TYPED
    # FAILURE (empty template), never an echo of the base.
    transport = FakeTransport(reply=lambda _p: "   ")
    proposer = _HttpProposerTransport(_proposer_route(), transport=transport)
    drafts = proposer.draft(_config(), _REQUEST, count=1)
    assert drafts[0].failed is True
    assert drafts[0].template == ""
    assert drafts[0].template != _REQUEST.base_template
    assert drafts[0].failure_detail  # a typed reason is recorded


def test_response_helper_is_used_for_success_shape() -> None:
    # Sanity: the fake success response carries stop/text so the draft path's
    # success branch (result.succeeded, generation.text) is what runs.
    resp = _response("x")
    assert resp.finish_reason == "stop"
