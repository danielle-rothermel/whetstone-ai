"""Algorithm-specific proposal prompts for the generic proposer transport."""

from __future__ import annotations

from whetstone.optimization.proposer import ProposalRequest

COPRO_PROPOSAL_PROMPT_SCHEMA_VERSION = 1
COPRO_PROPOSAL_PROMPT_SCHEMA_TAG = "copro-pp1"

COPRO_SEED_ROLE = (
    "You are an instruction optimizer for large language models. I will give "
    "you an initial prompt template. Your task is to propose an "
    "instruction that will lead a good language model to perform the task "
    "well. Don't be afraid to be creative."
)
COPRO_HISTORY_ROLE = (
    "You are an instruction optimizer for large language models. I will give "
    "you some task instructions I've tried, along with their corresponding "
    "validation scores. The instructions are arranged in increasing order "
    "based on their scores, where higher scores indicate better quality.\n\n"
    "Your task is to propose a new instruction that will lead a good language "
    "model to perform the task even better. Don't be afraid to be creative."
)


def copro_proposal_prompt(request: ProposalRequest) -> str:
    """Build Whetstone-native seed/history prompts with DSPy prompt topology.

    Whetstone optimizes a complete ``user_prompt_template`` and intentionally
    has no DSPy Signature field descriptions or output-prefix field.
    """

    if request.proposal_mode == "seed_proposal":
        return "\n".join(
            [
                COPRO_SEED_ROLE,
                "",
                "Initial instruction:",
                request.base_template,
                "",
                "Return only the improved instruction.",
            ]
        )
    if request.proposal_mode != "history_proposal":
        raise ValueError(
            f"unsupported COPRO proposal mode {request.proposal_mode!r}"
        )

    raw_history = request.context.get("prompt_history", [])
    if not isinstance(raw_history, list) or not raw_history:
        raise ValueError("COPRO history prompt requires selected attempts")
    lines = [COPRO_HISTORY_ROLE, ""]
    for index, entry in enumerate(raw_history, start=1):
        if not isinstance(entry, dict):
            raise ValueError("COPRO prompt history entries must be records")
        lines.extend(
            [
                f"Instruction #{index}: {entry.get('template', '')}",
                f"Resulting Score #{index}: {entry.get('reward', 'unscored')}",
            ]
        )
    lines.extend(["", "Return only the improved instruction."])
    return "\n".join(lines)


__all__ = [
    "COPRO_HISTORY_ROLE",
    "COPRO_PROPOSAL_PROMPT_SCHEMA_TAG",
    "COPRO_PROPOSAL_PROMPT_SCHEMA_VERSION",
    "COPRO_SEED_ROLE",
    "copro_proposal_prompt",
]
