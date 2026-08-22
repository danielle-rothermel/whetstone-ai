"""Codex CLI proposer drafts carry a provider call identity.

A Codex subprocess invocation is a proposer-model call: a model ran and the
run paid for it, even though the CLI reports no tokens and no price. Without
a ``logical_call_id`` on the resulting drafts, ``OptimResult.cost.proposer``
reports zero calls and no unpriced usage for the entire proposer side of a
COPRO-with-Codex run. These tests pin that such a run is counted -- as
identified, unpriced calls with an unknown token breakdown.
"""

from __future__ import annotations

import json

from whetstone.core.identity import typed_ref_for_record
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.codex import proposer as codex_proposer
from whetstone.optim.codex.proposer import (
    CodexCliProposerConfig,
    CodexCliProposerTransport,
)
from whetstone.optim.codex.runner import CodexStructuredExecution
from whetstone.optim.proposal.proposer import ProposalRequest


def _request(ordinal: int = 0) -> ProposalRequest:
    base = Candidate(
        candidate_id="codex-base",
        base_ref=typed_ref_for_record(
            "whetstone.testing.codex_root_candidate", {"seed": "codex"}
        ),
        payload={"user_prompt_template": "Reply to: {prompt}"},
    )
    return ProposalRequest(
        proposal_mode="copro_instruction",
        request_ordinal=ordinal,
        proposal_authority_identity_hash="a" * 64,
        base_candidate=candidate_reference(base),
        mutation_field="user_prompt_template",
        context={"proposal_prompt": "Write a better instruction."},
    )


def _transport(monkeypatch, bodies: tuple[str, ...]) -> CodexCliProposerTransport:
    """A transport whose Codex subprocess is replaced by a scripted artifact."""

    class _Runner:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_structured_prompt(self, *, prompt, output_schema):
            _ = (prompt, output_schema)
            return CodexStructuredExecution(
                artifact_bytes=json.dumps({"bodies": list(bodies)}).encode(),
                stdout=b"",
                stderr="",
                isolation={},
            )

    monkeypatch.setattr(codex_proposer, "SubprocessCodexRunner", _Runner)
    return CodexCliProposerTransport(executor=object())


def test_a_codex_draft_carries_a_logical_call_id(monkeypatch) -> None:
    transport = _transport(monkeypatch, ("First.", "Second."))

    drafts = transport.draft(
        CodexCliProposerConfig(), _request(), 2
    )

    assert all(draft.logical_call_id for draft in drafts)


def test_a_different_batch_size_is_a_different_codex_invocation(
    monkeypatch,
) -> None:
    # A breadth-two request is a different subprocess run than a breadth-one
    # request for the same proposal, so it carries its own identity.
    one = _transport(monkeypatch, ("First.",))
    two = _transport(monkeypatch, ("First.", "Second."))

    a = one.draft(CodexCliProposerConfig(), _request(), 1)[0]
    b = two.draft(CodexCliProposerConfig(), _request(), 2)[0]

    assert a.logical_call_id != b.logical_call_id


def test_the_codex_call_id_is_stable_across_a_redrive(monkeypatch) -> None:
    # De-duplication only works if a re-drive mints the same identity.
    first = _transport(monkeypatch, ("First.",))
    a = first.draft(CodexCliProposerConfig(), _request(), 1)[0]
    second = _transport(monkeypatch, ("First.",))
    b = second.draft(CodexCliProposerConfig(), _request(), 1)[0]

    assert a.logical_call_id == b.logical_call_id


def test_a_different_request_gets_a_different_codex_call_id(
    monkeypatch,
) -> None:
    transport = _transport(monkeypatch, ("First.",))

    a = transport.draft(CodexCliProposerConfig(), _request(ordinal=0), 1)[0]
    b = transport.draft(CodexCliProposerConfig(), _request(ordinal=1), 1)[0]

    assert a.logical_call_id != b.logical_call_id


def test_a_codex_draft_is_an_identified_unpriced_call(monkeypatch) -> None:
    # The CLI reports no telemetry, so the call is unpriced with an unknown
    # token breakdown -- but it is a call, and run cost must see it.
    transport = _transport(monkeypatch, ("First.",))

    draft = transport.draft(CodexCliProposerConfig(), _request(), 1)[0]
    usage = draft.call_usage()

    assert usage is not None
    assert usage.call_id == draft.logical_call_id
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.usd is None
    assert usage.observation().missing_token_breakdown is True


def test_a_codex_run_reports_its_proposer_calls(monkeypatch) -> None:
    """The end the finding names: a whole Codex proposer side is counted.

    A breadth-two request is one subprocess execution, so it reports one
    unpriced call after run cost de-duplicates the shared identity -- not
    one per body.
    """
    from whetstone.optim.cost import aggregate_role_cost

    transport = _transport(monkeypatch, ("First.", "Second."))
    drafts = transport.draft(CodexCliProposerConfig(), _request(), 2)

    # The de-duplication ``aggregate_run_cost`` applies to proposer usage,
    # keyed on the call identity these drafts carry.
    seen: set[str] = set()
    observations = []
    for draft in drafts:
        usage = draft.call_usage()
        assert usage is not None
        if usage.call_id in seen:
            continue
        seen.add(usage.call_id)
        observations.append(usage.observation())
    role = aggregate_role_cost(tuple(observations))

    assert role.calls == 1
    assert role.unpriced_calls == 1
    assert role.rows_missing_token_breakdown == 1
    assert role.usd is None


def test_one_codex_invocation_is_one_call_across_all_its_bodies(
    monkeypatch,
) -> None:
    """A batch is one subprocess run, so it is one proposer call.

    ``draft(count=2)`` performs a single ``run_structured_prompt`` that
    returns both bodies. Minting a per-slot identity would make one Codex
    execution report two calls, so every body of one invocation shares the
    invocation's identity and run cost de-duplicates them back to one.
    """
    transport = _transport(monkeypatch, ("First.", "Second."))

    drafts = transport.draft(CodexCliProposerConfig(), _request(), 2)

    assert len({draft.logical_call_id for draft in drafts}) == 1
