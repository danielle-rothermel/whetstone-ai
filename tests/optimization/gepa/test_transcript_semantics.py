"""Regression tests for evidence-only GEPA transcript projection."""

from __future__ import annotations

from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.test_effects import (
    _context,
    _prompt_services,
    _proposal_authority,
)
from whetstone.core.identity import typed_ref_for_record
from whetstone.optimization.gepa.contracts import (
    GepaCandidateComponent,
    GepaEffectRecorder,
    GepaEffectSlot,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)
from whetstone.optimization.gepa.prompts import GepaRenderedPrompt


def _request(
    *,
    ordinal: int,
    component_name: str,
) -> GepaProposalEffectRequest:
    services = _prompt_services()
    return GepaProposalEffectRequest(
        slot=GepaEffectSlot(
            context=_context(),
            invocation_ordinal=ordinal,
        ),
        candidate=(
            GepaCandidateComponent(name="alpha", text="unchanged"),
            GepaCandidateComponent(name="beta", text="unchanged"),
        ),
        components_to_update=("alpha", "beta"),
        component_name=component_name,
        rendered_prompt=GepaRenderedPrompt(text=f"Improve {component_name}."),
        authority=_proposal_authority(services),
    )


def _result(
    request: GepaProposalEffectRequest,
) -> GepaProposalEffectResult:
    attempt_ref = typed_ref_for_record(
        "test.gepa.proposal_attempt",
        {"request": request.identity_hash()},
    )
    return GepaProposalEffectResult(
        request_identity_hash=request.identity_hash(),
        raw_response="unchanged",
        parsed_components=(
            GepaCandidateComponent(
                name=request.component_name,
                text="unchanged",
            ),
        ),
        request_evidence={"prompt": request.rendered_prompt.text},
        response_evidence={"raw": "unchanged"},
        provider_attempt_refs=(attempt_ref,),
    )


def test_component_specific_entries_never_guess_candidate_indices(
    tmp_path,
) -> None:
    recorder = GepaEffectRecorder(
        ObjectStore(SqliteBackend(tmp_path / "transcript.sqlite"))
    )
    requests = (
        _request(ordinal=0, component_name="alpha"),
        _request(ordinal=1, component_name="beta"),
    )
    for request in requests:
        recorder.record_request(request)
        recorder.record_proposal_result(request, _result(request))

    transcript = recorder.build_transcript(
        context=_context(),
        effect_count=2,
    )

    assert tuple(entry.component_names for entry in transcript.entries) == (
        ("alpha",),
        ("beta",),
    )
    assert all(
        entry.upstream_candidate_index is None for entry in transcript.entries
    )
    assert (
        transcript.entries[0].semantic_candidate_identity_hash
        == transcript.entries[1].semantic_candidate_identity_hash
    )
