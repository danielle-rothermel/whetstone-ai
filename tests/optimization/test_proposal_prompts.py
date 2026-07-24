"""Tests for the DSPy-faithful COPRO proposal-prompt content (task 30b).

These assert the prompts stay as close to DSPy COPRO's
``BasicGenerateInstruction`` / ``GenerateInstructionGivenAttempts`` as our
representation allows: the verbatim optimizer-persona and creativity sentences,
the increasing-score attempt ordering, the in-prompt format-rules block, and
placeholder preservation. Reference: ``dspy/teleprompt/copro_optimizer.py``
lines ~38-58 and ~299-303.
"""

from __future__ import annotations

from whetstone.optimization.proposal_prompts import copro_proposal_prompt
from whetstone.optimization.proposer import ProposalRequest

_CREATIVITY = "Don't be afraid to be creative."
_PERSONA = "You are an instruction optimizer for large language models."
_TEMPLATE = "Compress the following {placeholder} for another agent."


def _seed_request() -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        base_ref="base",
        base_template=_TEMPLATE,
        context={"seed_entries": [], "measured": []},
    )


def _history_request() -> ProposalRequest:
    # ranked_history is best-first (Reward-descending), as COPRO produces.
    return ProposalRequest(
        proposal_mode="history_proposal",
        request_ordinal=1,
        base_ref="base",
        base_template=_TEMPLATE,
        context={
            "ranked_history": [
                {"candidate_id": "P0-0", "base_ref": "b",
                 "template": "best {placeholder} template", "reward": 0.9},
                {"candidate_id": "P0-2", "base_ref": "b",
                 "template": "mid {placeholder} template", "reward": 0.5},
                {"candidate_id": "P0-1", "base_ref": "b",
                 "template": "worst {placeholder} template", "reward": 0.2},
            ]
        },
    )


# ---------------------------------------------------------------------------
# SEED mode
# ---------------------------------------------------------------------------

def test_seed_contains_dspy_mirrored_sentences() -> None:
    prompt = copro_proposal_prompt(_seed_request())
    # Verbatim optimizer persona (copro_optimizer.py L39).
    assert _PERSONA in prompt
    # Verbatim creativity sentence (copro_optimizer.py L39).
    assert _CREATIVITY in prompt
    # Adapted task framing: instruction template for a task solver.
    assert "instruction template for a task solver" in prompt
    assert "perform the task well" in prompt


def test_seed_shows_base_template_and_preserves_placeholders() -> None:
    prompt = copro_proposal_prompt(_seed_request())
    assert _TEMPLATE in prompt
    assert "{placeholder}" in prompt


def test_seed_has_format_rules_block() -> None:
    prompt = copro_proposal_prompt(_seed_request())
    assert "FORMAT RULES" in prompt
    assert "{placeholder}" in prompt
    assert "Output ONLY" in prompt
    assert "must differ" in prompt


def test_no_history_selects_seed_mode() -> None:
    # A request with no ranked_history renders the SEED body, never the
    # iteration body, even if a non-seed mode string were used.
    req = ProposalRequest(
        proposal_mode="history_proposal",
        request_ordinal=1,
        base_ref="base",
        base_template=_TEMPLATE,
        context={},
    )
    prompt = copro_proposal_prompt(req)
    assert "perform the task well" in prompt
    assert "even better" not in prompt
    assert "ATTEMPTED INSTRUCTION TEMPLATES" not in prompt


def test_seed_mode_when_no_ranked_history_key() -> None:
    prompt = copro_proposal_prompt(_seed_request())
    assert "even better" not in prompt


# ---------------------------------------------------------------------------
# ITERATION mode
# ---------------------------------------------------------------------------

def test_iteration_contains_dspy_mirrored_sentences() -> None:
    prompt = copro_proposal_prompt(_history_request())
    assert _PERSONA in prompt
    # Verbatim creativity sentence, kept in iteration mode too.
    assert _CREATIVITY in prompt
    # "even better" kept verbatim (copro_optimizer.py L51).
    assert "perform the task even better" in prompt
    assert "increasing order" in prompt


def test_iteration_lists_attempts_ascending_with_scores() -> None:
    prompt = copro_proposal_prompt(_history_request())
    # Attempts rendered increasing-score-first: worst (0.2) then 0.5 then 0.9.
    idx_worst = prompt.index("worst {placeholder} template")
    idx_mid = prompt.index("mid {placeholder} template")
    idx_best = prompt.index("best {placeholder} template")
    assert idx_worst < idx_mid < idx_best
    # Each attempt carries its validation score.
    assert "Resulting Score #1: 0.2" in prompt
    assert "Resulting Score #2: 0.5" in prompt
    assert "Resulting Score #3: 0.9" in prompt
    # Numbered templates.
    assert "Template #1: worst {placeholder} template" in prompt
    assert "Template #3: best {placeholder} template" in prompt


def test_iteration_preserves_placeholders_in_examples() -> None:
    prompt = copro_proposal_prompt(_history_request())
    assert prompt.count("{placeholder}") >= 3


def test_iteration_has_format_rules_block() -> None:
    prompt = copro_proposal_prompt(_history_request())
    assert "FORMAT RULES" in prompt
    assert "must differ from every template shown above" in prompt


def test_iteration_missing_reward_renders_unknown() -> None:
    req = ProposalRequest(
        proposal_mode="history_proposal",
        request_ordinal=1,
        base_ref="base",
        base_template=_TEMPLATE,
        context={
            "ranked_history": [
                {"candidate_id": "P0-0", "base_ref": "b",
                 "template": "no-score template", "reward": None},
            ]
        },
    )
    prompt = copro_proposal_prompt(req)
    assert "Resulting Score #1: unknown" in prompt


# ---------------------------------------------------------------------------
# Compatibility with the CodexProposer prompt_builder slot.
# ---------------------------------------------------------------------------

def test_callable_signature_matches_prompt_builder_slot() -> None:
    # Both modes return a non-empty str for a ProposalRequest.
    assert isinstance(copro_proposal_prompt(_seed_request()), str)
    assert isinstance(copro_proposal_prompt(_history_request()), str)
