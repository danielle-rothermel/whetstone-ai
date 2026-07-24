"""Tests for the single proposal-prompt seam (task 30a transport plumbing).

Before the seam, the codex-CLI and live-HTTP proposers each built their
drafting instruction from ONLY ``request.base_template`` and dropped
``request.context`` -- so COPRO's Reward-ranked score history never reached the
model (every live proposal round was score-blind). These tests pin that the
seam (a) keeps the original instruction/base template, (b) appends the ranked
history in ASCENDING score order when present, (c) stays valid when context is
absent, (d) tolerates MIPROv2's ``accepted`` shape, and (e) folds the
prompt-schema identity tag idempotently.
"""

from __future__ import annotations

from whetstone.optimization.proposal_prompts import (
    PROMPT_SCHEMA_TAG,
    PROMPT_SCHEMA_VERSION,
    copro_proposal_prompt,
    fold_prompt_schema_tag,
)
from whetstone.optimization.proposer import ProposalRequest


def _request(context: dict | None = None) -> ProposalRequest:
    return ProposalRequest(
        proposal_mode="history_proposal",
        request_ordinal=1,
        base_ref="base",
        base_template="Answer: {input}",
        context=context or {},
    )


# --- Base instruction preserved -------------------------------------------


def test_absent_context_prompt_is_valid_and_carries_base() -> None:
    prompt = copro_proposal_prompt(_request())
    # The original wording + base template are intact; no context block.
    assert "You are optimizing the instruction template" in prompt
    assert "ORIGINAL TEMPLATE:\nAnswer: {input}" in prompt
    assert prompt.rstrip().endswith("REWRITTEN TEMPLATE:")
    assert "PRIOR ATTEMPTS" not in prompt
    assert "ALREADY-PROPOSED" not in prompt


def test_absent_context_is_byte_identical_to_legacy_wording() -> None:
    # Regression guard: with no context the seam reproduces the exact prompt
    # the two transports emitted before the seam (a sibling owns rewording).
    base = "Answer: {input}"
    legacy = (
        "You are optimizing the instruction template of a prompt-based "
        "task solver. Rewrite the template below into a SINGLE improved "
        "variant that is clearer and more likely to elicit a correct answer. "
        "Rules: keep every {placeholder} token exactly as written; change the "
        "wording so the result is NOT identical to the original; output ONLY "
        "the rewritten template text with no preamble, quotes, or "
        "commentary.\n"
        f"\nORIGINAL TEMPLATE:\n{base}\n\nREWRITTEN TEMPLATE:"
    )
    assert copro_proposal_prompt(_request()) == legacy


# --- COPRO ranked history --------------------------------------------------


def test_ranked_history_present_in_prompt_ascending_order() -> None:
    # copro.rank_attempt_history yields BEST-first; the seam renders
    # WORST-first so the strongest exemplar sits nearest the instruction.
    ranked = [
        {"candidate_id": "P1-0", "template": "BEST", "reward": 0.9},
        {"candidate_id": "P0-1", "template": "MID", "reward": 0.5},
        {"candidate_id": "A", "template": "WORST", "reward": 0.1},
    ]
    prompt = copro_proposal_prompt(_request({"ranked_history": ranked}))
    assert "PRIOR ATTEMPTS" in prompt
    # Ordering: WORST before MID before BEST (ascending score).
    assert prompt.index("WORST") < prompt.index("MID") < prompt.index("BEST")
    # Scores are rendered alongside each exemplar.
    assert "[score=0.9000] BEST" in prompt
    assert "[score=0.1000] WORST" in prompt
    # The base instruction + template survive.
    assert "ORIGINAL TEMPLATE:\nAnswer: {input}" in prompt
    assert prompt.rstrip().endswith("REWRITTEN TEMPLATE:")


def test_ranked_history_missing_reward_renders_na() -> None:
    ranked = [
        {"candidate_id": "P1-0", "template": "SCORED", "reward": 0.7},
        {"candidate_id": "A", "template": "UNSCORED"},  # no reward key
    ]
    prompt = copro_proposal_prompt(_request({"ranked_history": ranked}))
    assert "[score=n/a] UNSCORED" in prompt
    assert "[score=0.7000] SCORED" in prompt


def test_ranked_history_skips_template_less_entries() -> None:
    ranked = [
        {"candidate_id": "A", "template": "", "reward": 0.4},  # empty template
        {"candidate_id": "B", "reward": 0.4},  # no template key
        {"candidate_id": "C", "template": "REAL", "reward": 0.4},
    ]
    prompt = copro_proposal_prompt(_request({"ranked_history": ranked}))
    assert "REAL" in prompt
    # Only the one real exemplar rendered.
    assert prompt.count("[score=") == 1


def test_empty_ranked_history_yields_no_block() -> None:
    prompt = copro_proposal_prompt(_request({"ranked_history": []}))
    assert "PRIOR ATTEMPTS" not in prompt
    assert prompt.rstrip().endswith("REWRITTEN TEMPLATE:")


# --- MIPROv2 accepted shape (no scores) ------------------------------------


def test_accepted_shape_renders_dedup_block() -> None:
    prompt = copro_proposal_prompt(
        _request({"accepted": ["one {input}", "two {input}"]})
    )
    assert "ALREADY-PROPOSED TEMPLATES" in prompt
    assert "- one {input}" in prompt
    assert "- two {input}" in prompt
    # No score annotation for the scoreless MIPROv2 shape.
    assert "[score=" not in prompt


def test_ranked_history_takes_precedence_over_accepted() -> None:
    prompt = copro_proposal_prompt(
        _request(
            {
                "ranked_history": [{"template": "H", "reward": 0.5}],
                "accepted": ["A"],
            }
        )
    )
    assert "PRIOR ATTEMPTS" in prompt
    assert "ALREADY-PROPOSED" not in prompt


# --- Prompt-schema identity tag --------------------------------------------


def test_fold_prompt_schema_tag_appends() -> None:
    assert (
        fold_prompt_schema_tag("codex-cli/gpt-5.4-mini")
        == f"codex-cli/gpt-5.4-mini#{PROMPT_SCHEMA_TAG}"
    )
    assert PROMPT_SCHEMA_TAG == f"pp{PROMPT_SCHEMA_VERSION}"


def test_fold_prompt_schema_tag_is_idempotent() -> None:
    once = fold_prompt_schema_tag("codex-cli/x")
    assert fold_prompt_schema_tag(once) == once


def test_untagged_and_tagged_refs_are_distinguishable() -> None:
    # An old score-blind cell recorded no tag (pp1); a new cell carries pp2.
    old = "codex-cli/gpt-5.4-mini"
    new = fold_prompt_schema_tag(old)
    assert old != new
    assert new.endswith(f"#{PROMPT_SCHEMA_TAG}")
