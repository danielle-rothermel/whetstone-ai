"""GEPA reflection: one bounded retry, then a recorded skipped mutation.

The live C19 run died on "GEPA replacement component omitted required
placeholders" and optimization simply ended. A rejected reflection response
is a model failure, not an infrastructure failure, so it gets exactly one
retry with the rejection fed back and is then recorded and skipped.
"""

from __future__ import annotations

import pytest

from whetstone.core.identity import compute_identity_hash
from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaEffectContext,
    GepaProposalAuthorityBinding,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)
from whetstone.optim.gepa.prompts import (
    GEPA_REFLECTION_RETRY_ROLE,
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    GepaReflectionRequest,
    GepaRejectedAttempt,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optim.gepa.upstream_adapter import (
    GEPA_REFLECTION_MAX_ATTEMPTS,
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    WhetstoneGepaAdapter,
)

COMPONENT = "generate"
SEED = "Reply briefly to: {prompt}"
GOOD = "Answer {prompt} in one short friendly sentence."
#: Missing the required {prompt} placeholder, the exact live failure.
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
    ) -> GepaProposalEffectResult:
        self.prompts.append(request.rendered_prompt.text)
        raw = self._bodies.pop(0)
        try:
            parsed = self._services.parse_replacement(
                request.component_name, raw
            )
        except (KeyError, TypeError, ValueError) as exc:
            return GepaProposalEffectResult(
                request_hash=request.identity_hash(),
                raw_response=raw,
                failed=True,
                rejected_by_parser=True,
                failure_detail=str(exc),
            )
        return GepaProposalEffectResult(
            request_hash=request.identity_hash(),
            raw_response=raw,
            parsed_components=(
                GepaCandidateComponent(name=request.component_name, text=parsed),
            ),
            request_evidence={"scripted": True},
            response_evidence={"scripted": True},
            provider_attempt_refs=(_scripted_attempt_ref(),),
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


# --- branch 1: the retry succeeds -----------------------------------------


def test_a_rejected_reflection_is_retried_once_and_succeeds() -> None:
    adapter, broker = _adapter([BAD, GOOD])

    replacements = adapter.propose_new_texts(
        {COMPONENT: SEED}, _dataset(), [COMPONENT]
    )

    assert replacements == {COMPONENT: GOOD}
    assert len(broker.prompts) == 2
    # The retry prompt names the rejection so the model can correct it.
    assert GEPA_REFLECTION_RETRY_ROLE in broker.prompts[1]
    assert "omitted required placeholders" in broker.prompts[1]
    assert BAD in broker.prompts[1]
    # The first rejection is still recorded, not hidden by the recovery.
    assert len(adapter.skipped_mutations) == 1
    assert adapter.skipped_mutations[0].exhausted is False


# --- branch 2: the retry also fails ---------------------------------------


def test_a_twice_rejected_reflection_is_skipped_and_recorded() -> None:
    adapter, broker = _adapter([BAD, BAD])

    replacements = adapter.propose_new_texts(
        {COMPONENT: SEED}, _dataset(), [COMPONENT]
    )

    # The component is left unchanged; the search continues.
    assert replacements == {}
    assert len(broker.prompts) == GEPA_REFLECTION_MAX_ATTEMPTS
    # Never silent: both attempts are recorded in step evidence.
    skipped = adapter.skipped_mutations
    assert len(skipped) == GEPA_REFLECTION_MAX_ATTEMPTS
    assert [item.attempt_ordinal for item in skipped] == [0, 1]
    assert [item.exhausted for item in skipped] == [False, True]
    for item in skipped:
        assert item.component_name == COMPONENT
        assert "omitted required placeholders" in item.rejection_detail
        assert item.raw_response == BAD


# --- a non-parser failure still surfaces ----------------------------------


class FailingBroker(ScriptedProposalBroker):
    def propose(
        self, request: GepaProposalEffectRequest
    ) -> GepaProposalEffectResult:
        self.prompts.append(request.rendered_prompt.text)
        return GepaProposalEffectResult(
            request_hash=request.identity_hash(),
            failed=True,
            failure_detail="provider transport exploded",
        )


def test_a_provider_failure_is_not_retried_and_still_raises() -> None:
    adapter, _broker = _adapter([GOOD])
    services = _services()
    failing = FailingBroker(services, [])
    adapter._broker = failing  # noqa: SLF001

    with pytest.raises(RuntimeError, match="provider transport exploded"):
        adapter.propose_new_texts({COMPONENT: SEED}, _dataset(), [COMPONENT])

    assert len(failing.prompts) == 1
    assert adapter.skipped_mutations == ()


# --- prompt rendering -----------------------------------------------------


def test_a_first_attempt_prompt_carries_no_retry_section() -> None:
    services = _services()
    rendered = services.reflection_builder.render(
        services.descriptor,
        GepaReflectionRequest(
            candidate={COMPONENT: SEED},
            reflective_dataset={
                COMPONENT: ({"Inputs": {"prompt": "hi"}},)
            },
            components_to_update=(COMPONENT,),
            component_name=COMPONENT,
        ),
    )

    assert GEPA_REFLECTION_RETRY_ROLE not in rendered.text


def test_a_retry_prompt_quotes_the_rejected_attempt() -> None:
    services = _services()
    rendered = services.reflection_builder.render(
        services.descriptor,
        GepaReflectionRequest(
            candidate={COMPONENT: SEED},
            reflective_dataset={
                COMPONENT: ({"Inputs": {"prompt": "hi"}},)
            },
            components_to_update=(COMPONENT,),
            component_name=COMPONENT,
            prior_attempt=GepaRejectedAttempt(
                raw_response=BAD,
                rejection_detail="omitted required placeholders ['prompt']",
            ),
        ),
    )

    assert GEPA_REFLECTION_RETRY_ROLE in rendered.text
    assert BAD in rendered.text
    assert "omitted required placeholders ['prompt']" in rendered.text


def test_the_effect_slot_advances_across_the_retry() -> None:
    """Each attempt is its own effect, so replay stays exact."""
    adapter, _broker = _adapter([BAD, GOOD])
    adapter.propose_new_texts({COMPONENT: SEED}, _dataset(), [COMPONENT])

    assert adapter.effect_count == 2


def test_reset_clears_skipped_mutations() -> None:
    adapter, _broker = _adapter([BAD, BAD])
    adapter.propose_new_texts({COMPONENT: SEED}, _dataset(), [COMPONENT])
    assert adapter.skipped_mutations

    adapter.reset_effect_ordinal()

    assert adapter.skipped_mutations == ()
