from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from whetstone.core.identity import ImmutableJsonObject
from whetstone.optimization.proposal.proposer import ProposalRequest

COPRO_PROPOSAL_PROMPT_SCHEMA_VERSION = 2
COPRO_PROPOSAL_PROMPT_SCHEMA_TAG = "copro-pp2"
COPRO_INSTRUCTION_CONTRACT_KEY = "instruction_contract"
COPRO_INSTRUCTION_HISTORY_KEY = "instruction_history"

COPRO_SEED_ROLE = (
    "You are an instruction optimizer for large language models. I will give "
    "you an initial encoder instruction. Your task is to propose a "
    "replacement encoder instruction that will lead a good language model to "
    "perform the task well. Don't be afraid to be creative."
)
COPRO_HISTORY_ROLE = (
    "You are an instruction optimizer for large language models. I will give "
    "you some task instructions I've tried, along with their corresponding "
    "validation scores. The instructions are arranged in increasing order "
    "based on their scores, where higher scores indicate better quality.\n\n"
    "Your task is to propose a new instruction that will lead a good language "
    "model to perform the task even better. Don't be afraid to be creative."
)


def _instruction_contract_record(
    request: ProposalRequest,
) -> Mapping[str, Any]:
    raw_contract = request.context.get(COPRO_INSTRUCTION_CONTRACT_KEY)
    if not isinstance(raw_contract, ImmutableJsonObject):
        raise ValueError("COPRO proposal requires an instruction contract")
    return raw_contract.to_json()


def copro_proposal_prompt(request: ProposalRequest) -> str:
    """Build Whetstone-native seed/history prompts with DSPy prompt topology.

    COPRO optimizes one mutable instruction region. The exact immutable frame,
    fixed decoder, and body-only response rule are repeated in every seed and
    history request so all proposer transports receive the same semantic
    contract.
    """

    contract = _instruction_contract_record(request)
    contract_lines = [
        f"Optimization target: {contract['target_name']}",
        f"Task context: {contract['task_context']}",
        f"Budget mode: {contract['budget_mode']}",
        "Immutable encoder frame ({body} is the only mutable region):",
        contract["encoder_frame"],
        "Fixed decoder template:",
        contract["decoder_template"],
        "Output contract:",
        contract["output_rule"],
    ]

    if request.proposal_mode == "seed_proposal":
        return "\n".join(
            [
                COPRO_SEED_ROLE,
                "",
                *contract_lines,
                "",
                "Initial encoder instruction:",
                request.base_template,
                "",
                "Return only the improved encoder instruction body.",
            ]
        )
    if request.proposal_mode != "history_proposal":
        raise ValueError(
            f"unsupported COPRO proposal mode {request.proposal_mode!r}"
        )

    raw_history = request.context.get(COPRO_INSTRUCTION_HISTORY_KEY, ())
    if type(raw_history) is not tuple or not raw_history:
        raise ValueError("COPRO history prompt requires selected attempts")
    lines = [COPRO_HISTORY_ROLE, "", *contract_lines, ""]
    for index, entry in enumerate(raw_history, start=1):
        if not isinstance(entry, ImmutableJsonObject):
            raise ValueError("COPRO prompt history entries must be records")
        lines.extend(
            [
                f"Instruction #{index}: {entry.get('instruction', '')}",
                f"Resulting Score #{index}: {entry.get('reward', 'unscored')}",
            ]
        )
    lines.extend(["", "Return only the improved encoder instruction body."])
    return "\n".join(lines)


__all__ = [
    "COPRO_HISTORY_ROLE",
    "COPRO_INSTRUCTION_CONTRACT_KEY",
    "COPRO_INSTRUCTION_HISTORY_KEY",
    "COPRO_PROPOSAL_PROMPT_SCHEMA_TAG",
    "COPRO_PROPOSAL_PROMPT_SCHEMA_VERSION",
    "COPRO_SEED_ROLE",
    "copro_proposal_prompt",
]
