"""DSPy-faithful COPRO proposal-prompt content for the proposer route.

This module builds the drafting instruction handed to the proposer LM (reached
via the codex-CLI / HTTP proposer transports) for one template rewrite. The
goal, per the user directive, is to stay **as close to DSPy COPRO's proposal
prompts as our representation allows**, given that Whetstone does NOT represent
inputs/outputs the way DSPy does.

The DSPy source mirrored (read-only reference):
``dspy/teleprompt/copro_optimizer.py`` lines ~38-58 --

* ``BasicGenerateInstruction`` (the SEED signature) whose docstring reads:
  "You are an instruction optimizer for large language models. I will give you
  a ``signature`` of fields (inputs and outputs) in English. Your task is to
  propose an instruction that will lead a good language model to perform the
  task well. Don't be afraid to be creative."
* ``GenerateInstructionGivenAttempts`` (the ITERATION signature) whose
  docstring reads: "You are an instruction optimizer for large language
  models. I will give some task instructions I've tried, along with their
  corresponding validation scores. The instructions are arranged in increasing
  order based on their scores, where higher scores indicate better quality.
  Your task is to propose a new instruction that will lead a good language
  model to perform the task even better. Don't be afraid to be creative."

DSPy renders prior attempts (copro_optimizer.py ~lines 299-303) as, per
attempt, ``Instruction #k``, ``Prefix #k``, ``Resulting Score #k`` in
*increasing* score order. Whetstone has no separate prefix field (see
DIVERGENCES), so we render ``Template #k`` + ``Resulting Score #k`` ascending.

DIVERGENCES FROM DSPy COPRO (honest, itemized)
----------------------------------------------
Every remaining difference between these prompts and DSPy's COPRO prompts, and
*why* our representation forces it:

1. No ``proposed_prefix_for_output_field`` equivalent. DSPy's two signatures
   each emit BOTH a ``proposed_instruction`` and a
   ``proposed_prefix_for_output_field`` (the output-field prefix that seeds the
   solver's answer). Whetstone's mutation surface is a SINGLE strategy-sentence
   body inside a fixed ``ENCODER_FRAME`` (``src/whetstone/envs/ed1.py``); the
   output prefix / fenced code block / budget line are frame-owned and
   immutable. So we propose exactly one thing (the body template) and never a
   prefix.

2. Single-module only. DSPy COPRO optimizes every predictor of a (possibly
   multi-predictor) program and threads a per-predictor attempt history. The
   Whetstone proposer route drafts one ``user_prompt_template`` per request;
   the attempt history it conditions on is the run's single Reward-ranked
   history.

3. We ask over an "instruction template for a task solver", not over a DSPy
   ``signature`` of input/output fields. DSPy tells the model "I will give you
   a ``signature`` of fields (inputs and outputs) in English"; we cannot,
   because Whetstone does not expose the solver's I/O as an editable signature.
   The optimizer-persona and creativity sentences are preserved verbatim; only
   the object-of-optimization framing is adapted.

4. In-prompt FORMAT-RULES block. DSPy enforces output shape structurally via
   its Signature fields + adapter (a parsed ``proposed_instruction`` field), so
   its prompt body carries no format rules. Whetstone's transport treats the
   model's final message AS the template text, so we MUST state the mechanical
   constraints in-prompt (keep ``{placeholder}`` tokens, output only the
   template, must differ from every shown attempt). This block is intentionally
   minimal and clearly separated from the DSPy-mirrored body.

5. codex-CLI single-completion vs DSPy n-completions sampling. DSPy samples
   ``breadth`` completions at ``init_temperature`` (default 1.4) in one
   ``dspy.Predict`` call. Our proposer transport issues one single-completion
   drafting call per candidate (the codex CLI returns one final message), so
   the SAME prompt is reused across the batch; batch diversity comes from the
   transport's temperature, not from an n-sampling parameter in the prompt.

6. ``basic_instruction`` input framing. DSPy's SEED signature takes a
   ``basic_instruction`` input field ("The initial instructions before
   optimization"). We surface the equivalent as the base template shown in the
   prompt body, not as a separately-parsed input field.
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
#: prompt that dropped ``request.context``; ``2`` carries ranked history and
#: the DSPy-faithful wording.
PROMPT_SCHEMA_VERSION = 2

#: The identity tag folded into a behavior-bearing proposer route ref/model
#: (e.g. ``codex-cli/gpt-5.4-mini#pp2``). Untagged == pre-seam ``pp1``.
PROMPT_SCHEMA_TAG = f"pp{PROMPT_SCHEMA_VERSION}"


def fold_prompt_schema_tag(ref: str) -> str:
    """Fold the prompt-schema tag into a proposer route ref/model string.

    ``codex-cli/gpt-5.4-mini`` -> ``codex-cli/gpt-5.4-mini#pp2``. This makes
    the (behavior-bearing) prompt structure part of the proposer Config
    identity, exactly where the lane+model folds -- so an old score-blind
    cell (untagged / ``pp1``) never collides with a new context-carrying one.
    Idempotent: a ref already carrying THIS tag is returned unchanged.
    """
    suffix = f"#{PROMPT_SCHEMA_TAG}"
    if ref.endswith(suffix):
        return ref
    return f"{ref}{suffix}"

# The proposal_mode value the COPRO adapter sets for the first (seed) round.
# Kept as a literal to avoid importing copro.py (transport-owned; file
# ownership) -- it must match copro.SEED_PROPOSAL.
_SEED_MODE = "seed_proposal"

# ---------------------------------------------------------------------------
# DSPy-mirrored sentences (verbatim where our world allows).
# ---------------------------------------------------------------------------

# Mirrors copro_optimizer.py L39 (BasicGenerateInstruction docstring),
# optimizer persona sentence -- VERBATIM.
_OPTIMIZER_PERSONA = (
    "You are an instruction optimizer for large language models."
)

# Mirrors copro_optimizer.py L39 task-framing sentence. ADAPTED: DSPy says "I
# will give you a ``signature`` of fields (inputs and outputs) in English"; we
# have no editable I/O signature, so we frame the object of optimization as an
# instruction template for a task solver (see DIVERGENCE 3).
_SEED_TASK_FRAMING = (
    "I will give you the current instruction template for a task solver. "
    "Your task is to propose an instruction template that will lead a good "
    "language model to perform the task well."
)

# Mirrors copro_optimizer.py L39 final sentence -- VERBATIM (kept exactly).
_CREATIVITY = "Don't be afraid to be creative."

# Mirrors copro_optimizer.py L49-50 (GenerateInstructionGivenAttempts
# docstring) optimizer-persona + "instructions I've tried ... increasing order"
# framing. ADAPTED from "task instructions" to "instruction templates" (see
# DIVERGENCE 3); the ascending-score claim is preserved verbatim in spirit.
_ITER_TASK_FRAMING = (
    "I will give some instruction templates I've tried, along with their "
    "corresponding validation scores. The templates are arranged in "
    "increasing order based on their scores, where higher scores indicate "
    "better quality."
)

# Mirrors copro_optimizer.py L51 -- ADAPTED "instruction" -> "instruction
# template", KEEPS "even better" verbatim.
_ITER_PROPOSE = (
    "Your task is to propose a new instruction template that will lead a good "
    "language model to perform the task even better."
)


def _format_rules_block() -> str:
    """The minimal, clearly-separated mechanical-constraints block.

    DSPy enforces output shape via Signature fields + adapters (DIVERGENCE 4);
    Whetstone's transport treats the model's final message AS the template, so
    the constraints must be stated in-prompt. Kept short and set off from the
    DSPy-mirrored body by a rule heading.
    """
    return (
        "FORMAT RULES (Whetstone transport, not part of the task):\n"
        "- Keep every {placeholder} token exactly as written.\n"
        "- Output ONLY the instruction template text: no preamble, no quotes, "
        "no commentary, no code fences.\n"
        "- The result must differ from every template shown above."
    )


def _ascending_history(request: ProposalRequest) -> list[dict[str, Any]]:
    """Reward-ranked history rendered ascending (worst-first) for DSPy parity.

    ``request.context['ranked_history']`` is Reward-DESCENDING (best first, as
    COPRO's ``rank_attempt_history`` produces). DSPy presents attempts in
    INCREASING score order, so we reverse into worst-first here. Entries carry
    ``template`` and ``reward`` (plus ``candidate_id`` / ``base_ref``); see
    copro.py ~L225 and the ranked-history entry shape.
    """
    raw = request.context.get("ranked_history")
    if not isinstance(raw, list):
        return []
    entries = [dict(e) for e in raw if isinstance(e, dict)]
    # ranked_history is best-first -> reverse for increasing-score order.
    return list(reversed(entries))


def _is_seed(request: ProposalRequest) -> bool:
    """Seed when the mode is seed OR no ranked_history is present.

    Primary signal is ``proposal_mode`` (the adapter sets ``seed_proposal`` for
    round 1). As a defensive fallback -- e.g. a request built without a mode
    convention -- the absence of a non-empty ``ranked_history`` also selects
    seed mode, so a first-round request with no history never renders the
    iteration body with an empty attempt list.
    """
    if request.proposal_mode == _SEED_MODE:
        return True
    return not _ascending_history(request)


def _accepted_block(request: ProposalRequest) -> str:
    """MIPROv2's ``context['accepted']`` texts as a scoreless dedup list.

    MIPROv2's pool construction passes already-accepted instruction TEXTS
    (no scores). Rendering them keeps the seam from dropping context it is
    given (the MIPROv2 grounding enrichment is a separate future task).
    Empty string when absent.
    """
    texts = request.context.get("accepted") if request.context else None
    if not isinstance(texts, list):
        return ""
    lines = [f"- {t}" for t in texts if isinstance(t, str) and t]
    if not lines:
        return ""
    joined = "\n".join(lines)
    return (
        "\n\nALREADY-PROPOSED TEMPLATES (produce something DIFFERENT from "
        f"these):\n{joined}"
    )


def _seed_prompt(request: ProposalRequest) -> str:
    """SEED prompt -- mirrors BasicGenerateInstruction (L38-45)."""
    base = request.base_template
    body = f"{_OPTIMIZER_PERSONA} {_SEED_TASK_FRAMING} {_CREATIVITY}"
    # DSPy's ``basic_instruction`` input field (L41) is surfaced as the shown
    # base template rather than a separately-parsed field (DIVERGENCE 6).
    return (
        f"{body}\n\n"
        f"CURRENT INSTRUCTION TEMPLATE:\n{base}"
        f"{_accepted_block(request)}\n\n"
        f"{_format_rules_block()}\n\n"
        "PROPOSED INSTRUCTION TEMPLATE:"
    )


def _iteration_prompt(request: ProposalRequest) -> str:
    """ITERATION prompt -- mirrors GenerateInstructionGivenAttempts."""
    ascending = _ascending_history(request)
    # Render attempts increasing-score-first, mirroring copro_optimizer.py
    # L299-303 (``Instruction #k`` / ``Resulting Score #k``). We have no
    # ``Prefix #k`` (DIVERGENCE 1), so each attempt is Template + Score.
    lines: list[str] = []
    for k, entry in enumerate(ascending, start=1):
        template = str(entry.get("template", ""))
        reward = entry.get("reward")
        score = (
            f"{float(reward)}"
            if isinstance(reward, (int, float))
            else "unknown"
        )
        lines.append(f"Template #{k}: {template}")
        lines.append(f"Resulting Score #{k}: {score}")
    attempts_block = "\n".join(lines)
    body = (
        f"{_OPTIMIZER_PERSONA} {_ITER_TASK_FRAMING}\n\n"
        f"{_ITER_PROPOSE} {_CREATIVITY}"
    )
    return (
        f"{body}\n\n"
        f"ATTEMPTED INSTRUCTION TEMPLATES (increasing score):\n"
        f"{attempts_block}\n\n"
        f"{_format_rules_block()}\n\n"
        "PROPOSED INSTRUCTION TEMPLATE:"
    )


def copro_proposal_prompt(request: ProposalRequest) -> str:
    """Build the DSPy-faithful COPRO drafting prompt for one rewrite.

    SEED mode (round 1 / no ranked history) mirrors DSPy's
    ``BasicGenerateInstruction``; ITERATION mode (ranked history present)
    mirrors ``GenerateInstructionGivenAttempts``, presenting prior templates
    with their validation scores in increasing order. See the module docstring
    for the full divergence list. Compatible with ``CodexProposer``'s
    ``prompt_builder: Callable[[ProposalRequest], str]`` slot.
    """
    if _is_seed(request):
        return _seed_prompt(request)
    return _iteration_prompt(request)
