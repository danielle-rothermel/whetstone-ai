"""A GEPA Step that dies on a reflection failure still reports what it spent.

A reflection call that fails for a reason the bounded retry cannot fix (a
transport or provider failure) used to raise straight out of the adapter.
Every reflection the Step had already paid for went with it: the durable
effect cache marks those calls replayed, so a resumed Step does not record
them either, and the spend simply never reached run cost.

The failure now arrives at the Step boundary as a typed error, which becomes
a terminal-failure Adapter Output carrying the accumulated ``proposer_usage``.
"""

from __future__ import annotations

import pytest

from whetstone.optim.contracts import StepStatus
from whetstone.optim.gepa.contracts import (
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)
from whetstone.optim.gepa.harness_adapter import GEPA_REFLECTION_FAILED_CODE
from whetstone.optim.gepa.upstream_adapter import GepaReflectionFailedError
from whetstone.core.identity import compute_identity_hash
from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaEffectContext,
    GepaProposalAuthorityBinding,
)
from whetstone.optim.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optim.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    WhetstoneGepaAdapter,
)

COMPONENT = "generate"
SEED = "Reply briefly to: {prompt}"
#: Missing the required {prompt} placeholder: a parser rejection.
BAD = "Answer the question in one short friendly sentence."


def _services() -> GepaPromptServices:
    return GepaPromptServices(
        descriptor=GepaPromptFormatDescriptor(
            format_name="retry_prompt_template",
            components=(
                GepaComponentFormat(
                    component_name=COMPONENT,
                    component_schema_identity_hash=compute_identity_hash(
                        schema="whetstone.testing.gepa_component",
                        schema_version=1,
                        payload={"field": "user_prompt_template"},
                    ),
                    allowed_placeholders=("prompt",),
                    required_placeholders=("prompt",),
                ),
            ),
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )

class ScriptedProposalBroker:
    """Answers each reflection request with the next scripted body."""

    def __init__(self, services: GepaPromptServices, bodies: list[str]) -> None:
        self._services = services
        self._bodies = list(bodies)
        self.prompts: list[str] = []

    def evaluate(self, request):  # pragma: no cover - unused here
        raise AssertionError("this broker only serves proposals")

    def propose(
        self, request: GepaProposalEffectRequest
    ) -> tuple[GepaProposalEffectResult, bool]:
        self.prompts.append(request.rendered_prompt.text)
        raw = self._bodies.pop(0)
        try:
            parsed = self._services.parse_replacement(
                request.component_name, raw
            )
        except (KeyError, TypeError, ValueError) as exc:
            return (
                GepaProposalEffectResult(
                    request_hash=request.identity_hash(),
                    raw_response=raw,
                    failed=True,
                    rejected_by_parser=True,
                    failure_detail=str(exc),
                ),
                False,
            )
        return (
            GepaProposalEffectResult(
                request_hash=request.identity_hash(),
                raw_response=raw,
                parsed_components=(
                    GepaCandidateComponent(
                        name=request.component_name, text=parsed
                    ),
                ),
                request_evidence={"scripted": True},
                response_evidence={"scripted": True},
                provider_attempt_refs=(_scripted_attempt_ref(),),
            ),
            False,
        )

def _scripted_attempt_ref():
    from whetstone.core.identity import typed_ref_for_record

    return typed_ref_for_record(
        "whetstone.gepa.proposal_provider_attempt/v2", {"scripted": True}
    )

def _adapter(bodies: list[str]) -> tuple[WhetstoneGepaAdapter, ScriptedProposalBroker]:
    services = _services()
    broker = ScriptedProposalBroker(services, bodies)
    authority = GepaProposalAuthorityBinding(
        authority_identity_hash="a" * 64,
        proposer_transport_identity_hash="b" * 64,
        prompt_binding_identity_hash=services.binding.identity_hash(),
        execution_policy_identity_hash="c" * 64,
        prompt_adapter_identity_hash="d" * 64,
        durability_policy_identity_hash="e" * 64,
        proposer_config=_proposer_config(),
    )
    from whetstone.optim.gepa.contracts import GepaEvalAuthorityBinding

    adapter = WhetstoneGepaAdapter(
        context=GepaEffectContext(
            run_id="gepa-retry-run",
            control_identity_hash="f" * 64,
            source_manifest_identity_hash="0" * 64,
            adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
        ),
        broker=broker,
        evaluation_authority=GepaEvalAuthorityBinding(
            authority_identity_hash="1" * 64,
            evaluation_config_hash="2" * 64,
            reward_policy_identity_hash="3" * 64,
            provider_route_identity_hash="4" * 64,
            execution_policy_identity_hash="5" * 64,
            prompt_adapter_identity_hash="6" * 64,
            response_parser_identity_hash="7" * 64,
            data_registry_identity_hash="8" * 64,
        ),
        proposal_authority=authority,
        prompt_services=services,
    )
    return adapter, broker


def _proposer_config():
    from whetstone.core.identity import IdentityRef, typed_ref_for_record
    from whetstone.optim.proposal.proposer import ProposerConfig

    record_ref = typed_ref_for_record(
        "whetstone.testing.provider_call_config", {"model": "fake"}
    )
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=record_ref,
            record_hash=record_ref.content_hash,
        ),
    )

def _dataset() -> dict[str, list[dict[str, object]]]:
    return {
        COMPONENT: [
            {"Inputs": {"prompt": "hi"}, "Feedback": "be friendlier"},
        ]
    }



class _BilledThenFailingBroker(ScriptedProposalBroker):
    """First reflection is billed and parser-rejected; the retry hard-fails.

    This is the shape that loses money: the run really paid for attempt one
    before attempt two failed in a way no retry can fix.
    """

    def __init__(self, services, bodies) -> None:
        super().__init__(services, bodies)
        self.calls = 0

    def propose(
        self, request: GepaProposalEffectRequest
    ) -> tuple[GepaProposalEffectResult, bool]:
        self.prompts.append(request.rendered_prompt.text)
        self.calls += 1
        if self.calls == 1:
            return (
                GepaProposalEffectResult(
                    request_hash=request.identity_hash(),
                    raw_response=BAD,
                    failed=True,
                    rejected_by_parser=True,
                    failure_detail="omitted required placeholders",
                    usage={"prompt_tokens": 120, "completion_tokens": 40},
                    cost=0.03,
                ),
                False,
            )
        return (
            GepaProposalEffectResult(
                request_hash=request.identity_hash(),
                failed=True,
                failure_detail="provider transport exploded",
                usage={"prompt_tokens": 118, "completion_tokens": 0},
                cost=0.01,
            ),
            False,
        )


def _billed_adapter():
    adapter, _ = _adapter([BAD])
    broker = _BilledThenFailingBroker(_services(), [])
    adapter._broker = broker  # noqa: SLF001
    return adapter, broker


def test_the_paid_reflections_are_recorded_before_the_failure_raises() -> None:
    adapter, broker = _billed_adapter()

    with pytest.raises(
        GepaReflectionFailedError, match="provider transport exploded"
    ):
        adapter.propose_new_texts({COMPONENT: SEED}, _dataset(), [COMPONENT])

    assert broker.calls == 2
    usage = adapter.proposer_usage
    assert len(usage) == 2
    assert sum(item.prompt_tokens for item in usage) == 238
    assert sum(item.completion_tokens for item in usage) == 40
    assert [item.usd for item in usage] == [pytest.approx(0.03), pytest.approx(0.01)]
    # Every recorded call is identified, so run cost de-duplicates a replay.
    assert all(item.call_id for item in usage)


def test_a_non_parser_failure_raises_the_typed_reflection_error() -> None:
    # Typed, not a bare RuntimeError: the Step boundary has to tell this
    # failure apart from a genuine programming error it must not swallow.
    adapter, _ = _billed_adapter()

    with pytest.raises(GepaReflectionFailedError):
        adapter.propose_new_texts({COMPONENT: SEED}, _dataset(), [COMPONENT])


def test_the_step_boundary_turns_the_failure_into_a_paid_terminal_output(
) -> None:
    # The Step fails, but it fails carrying its spend.
    from whetstone.optim.gepa.harness_adapter import GepaHarnessAdapter

    adapter, _ = _billed_adapter()
    with pytest.raises(GepaReflectionFailedError):
        adapter.propose_new_texts({COMPONENT: SEED}, _dataset(), [COMPONENT])
    recorded = adapter.proposer_usage

    class _Factory:
        def proposer_usage(self):
            return recorded

        def skipped_mutations(self):
            return ()

    class _Boundary:
        _adapter_factory = _Factory()

    output = GepaHarnessAdapter._reflection_failure_output(
        _Boundary(),
        GepaReflectionFailedError("provider transport exploded"),
    )

    assert output.proposed_status is StepStatus.FAILED
    assert output.terminal_failure is not None
    assert output.terminal_failure.code == GEPA_REFLECTION_FAILED_CODE
    assert "provider transport exploded" in output.terminal_failure.message
    assert output.proposer_usage == recorded
    assert sum(item.prompt_tokens for item in output.proposer_usage) == 238
    assert output.accepted_candidates == ()
