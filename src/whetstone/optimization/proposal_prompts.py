"""The single proposal-prompt seam every proposer transport drafts from.

TASK 30a (transport plumbing). Before this module, the codex-CLI proposer and
the live-HTTP proposer each built their drafting instruction from ONLY
``request.base_template`` and silently dropped ``request.context`` -- so the
Reward-ranked score history COPRO threads onto every History-Proposal round
(:mod:`whetstone.optimization.copro`, ``context={"ranked_history": ...}``)
never reached the model. Every live proposal round was score-blind.

This module is the ONE builder both transports call. Its job here is purely
mechanical: keep the existing instruction wording, and -- when the request
carries conditioning context -- APPEND it in a clearly delimited block so the
model sees the prior attempts and their scores. A sibling task owns the prompt
WORDING/grounding; this file only plumbs the context through faithfully.

The seam tolerates the two context shapes that exist today (render what is
present, nothing more):

* COPRO History Proposal -> ``context["ranked_history"]``: a list of Attempt
  History entries, each a dict carrying ``candidate_id``, ``template`` (the
  ``user_prompt_template`` text), and ``reward`` (a float; may be absent for an
  unmeasured entry). :func:`whetstone.optimization.copro.rank_attempt_history`
  ranks these best-first (highest reward first); this block renders them in
  ASCENDING score order (worst first, best last) so the strongest exemplar sits
  closest to the drafting instruction.
* MIPROv2 pool construction -> ``context["accepted"]``: a list of already
  accepted instruction TEXTS (no scores). Rendered as a plain de-duplication
  list so the model avoids repeating them. (MIPROv2 grounding enrichment is a
  separate future task; this seam only avoids dropping what is already passed.)

PROMPT-SCHEMA IDENTITY. Because the drafted prompt structure is now
behavior-bearing, it carries a version tag (:data:`PROMPT_SCHEMA_TAG`) that
folds into the proposer Config identity via :func:`fold_prompt_schema_tag`, the
same place ``codex-cli/<model>`` folds. Old cells (recorded with no tag) are
``pp1``; cells built after this change are ``pp2`` and are distinguishable in
the ledger.
"""

from __future__ import annotations

from typing import Any

from whetstone.optimization.proposer import ProposalRequest

__all__ = [
    "PROMPT_SCHEMA_TAG",
    "PROMPT_SCHEMA_VERSION",
    "copro_proposal_prompt",
    "fold_prompt_schema_tag",
]

#: The current prompt-schema version. ``1`` (untagged) was the score-blind
#: prompt that dropped ``request.context``; ``2`` is this context-carrying
#: seam.
PROMPT_SCHEMA_VERSION = 2

#: The identity tag folded into a behavior-bearing proposer route ref/model
#: (e.g. ``codex-cli/gpt-5.4-mini#pp2``). Untagged == pre-seam ``pp1``.
PROMPT_SCHEMA_TAG = f"pp{PROMPT_SCHEMA_VERSION}"

# The base drafting instruction. Kept byte-identical to the wording the two
# transports carried before this seam existed (a sibling task owns rewording).
_INSTRUCTION = (
    "You are optimizing the instruction template of a prompt-based "
    "task solver. Rewrite the template below into a SINGLE improved "
    "variant that is clearer and more likely to elicit a correct answer. "
    "Rules: keep every {placeholder} token exactly as written; change the "
    "wording so the result is NOT identical to the original; output ONLY "
    "the rewritten template text with no preamble, quotes, or "
    "commentary."
)


def fold_prompt_schema_tag(ref: str) -> str:
    """Fold the prompt-schema tag into a proposer route ref/model string.

    ``codex-cli/gpt-5.4-mini`` -> ``codex-cli/gpt-5.4-mini#pp2``. This is the
    minimal mechanism that makes the (now behavior-bearing) prompt structure
    part of the proposer Config identity, exactly where the lane+model folds --
    so an old score-blind cell (untagged / ``pp1``) never collides with a new
    context-carrying one. Idempotent: a ref already carrying THIS tag is
    returned unchanged.
    """
    suffix = f"#{PROMPT_SCHEMA_TAG}"
    if ref.endswith(suffix):
        return ref
    return f"{ref}{suffix}"


def _render_ranked_history(entries: list[Any]) -> str | None:
    """Render COPRO's Reward-ranked history in ASCENDING score order.

    Entries arrive best-first (see ``copro.rank_attempt_history``); we reverse
    to worst-first so the strongest exemplar is nearest the instruction. Each
    line carries the entry's ``template`` and its ``reward`` (``n/a`` when the
    entry was never measured). Non-dict / template-less entries are skipped.
    """
    lines: list[str] = []
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        template = entry.get("template")
        if not template:
            continue
        reward = entry.get("reward")
        score = (
            f"{float(reward):.4f}"
            if isinstance(reward, (int, float))
            else "n/a"
        )
        lines.append(f"[score={score}] {template}")
    if not lines:
        return None
    body = "\n".join(lines)
    return (
        "PRIOR ATTEMPTS (ascending score; the last is the best so far):\n"
        f"{body}"
    )


def _render_accepted(texts: list[Any]) -> str | None:
    """Render MIPROv2's already-accepted instruction texts (no scores)."""
    lines = [f"- {text}" for text in texts if isinstance(text, str) and text]
    if not lines:
        return None
    body = "\n".join(lines)
    return (
        "ALREADY-PROPOSED TEMPLATES (produce something DIFFERENT from "
        "these):\n"
        f"{body}"
    )


def _render_context(context: dict[str, Any]) -> str | None:
    """Render whatever conditioning context is present, or ``None``."""
    ranked = context.get("ranked_history")
    if isinstance(ranked, list) and ranked:
        return _render_ranked_history(ranked)
    accepted = context.get("accepted")
    if isinstance(accepted, list) and accepted:
        return _render_accepted(accepted)
    return None


def copro_proposal_prompt(request: ProposalRequest) -> str:
    """Build the drafting instruction for one proposer draft.

    The single seam both proposer transports (codex-CLI and live-HTTP) call.
    Emits the existing instruction over ``request.base_template`` and, when the
    request carries conditioning context, APPENDS it in a delimited block so
    the proposer sees prior attempts (and, for COPRO, their scores) instead of
    drafting blind. Absent/unknown context -> the plain instruction, unchanged.
    """
    base = request.base_template
    context_block = _render_context(request.context)
    prompt = (
        f"{_INSTRUCTION}\n"
        f"\nORIGINAL TEMPLATE:\n{base}\n"
    )
    if context_block is not None:
        prompt += f"\n{context_block}\n"
    prompt += "\nREWRITTEN TEMPLATE:"
    return prompt
