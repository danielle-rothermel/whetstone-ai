from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.envs.code_comp.constants import (
    DECODER_TEMPLATE,
    ENCODER_FRAME,
    ENCODER_FRAME_NO_BUDGET,
)
from whetstone.envs.code_comp.mutation_surface import validate_instruction_body

CODE_COMP_COPRO_PROPOSAL_CONTRACT_VERSION = "code_comp_encdec_instruction/v2"

CODE_COMP_COPRO_TASK_CONTEXT = (
    "HumanEval encode-decode reconstruction: the encoder describes reference "
    "Python code and the fixed decoder reconstructs functional Python code "
    "from that description."
)

CODE_COMP_COPRO_OUTPUT_RULE = (
    "Return only a replacement encoder instruction body. Do not include input "
    "code, placeholders, the budget clause, prompt headings, terminal "
    "punctuation, Markdown code fences, or a complete prompt."
)


class EncDecCoproProposalContract(BaseModel):
    """Identity-bearing mutation and output contract for ED1 COPRO."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: Literal["code_comp_encdec_instruction/v2"] = (
        CODE_COMP_COPRO_PROPOSAL_CONTRACT_VERSION
    )
    target_name: Literal["encoder_instruction"] = "encoder_instruction"
    budget_mode: Literal["budgeted", "unbudgeted"]
    task_context: StrictStr = CODE_COMP_COPRO_TASK_CONTEXT
    encoder_frame: StrictStr
    decoder_template: StrictStr = DECODER_TEMPLATE
    output_rule: StrictStr = CODE_COMP_COPRO_OUTPUT_RULE

    @model_validator(mode="after")
    def _validate(self) -> EncDecCoproProposalContract:
        expected_frame = (
            ENCODER_FRAME
            if self.budget_mode == "budgeted"
            else ENCODER_FRAME_NO_BUDGET
        )
        if self.encoder_frame != expected_frame:
            raise ValueError(
                "ED1 COPRO encoder frame conflicts with its budget mode"
            )
        if not self.task_context.strip() or not self.output_rule.strip():
            raise ValueError("ED1 COPRO proposal text must be nonblank")
        return self

    def validate_instruction(self, instruction: str) -> None:
        """Validate one proposed value against the ED1 mutation surface."""

        if not instruction.strip():
            raise ValueError("ED1 COPRO instruction must be nonblank")
        validate_instruction_body(instruction)
        if instruction.rstrip().endswith((".", "!", "?")):
            raise ValueError(
                "ED1 COPRO instruction must omit terminal punctuation"
            )
        lowered = instruction.casefold()
        if "use at most" in lowered:
            raise ValueError(
                "ED1 COPRO instruction must omit the fixed budget clause"
            )
        if "# encode" in lowered or "# decode" in lowered:
            raise ValueError(
                "ED1 COPRO instruction must omit fixed prompt headings"
            )


def encdec_copro_proposal_contract(
    *, budget_ratio: float | None
) -> EncDecCoproProposalContract:
    """Bind the exact ED1 frame selected by one experiment configuration."""

    if budget_ratio is None:
        return EncDecCoproProposalContract(
            budget_mode="unbudgeted",
            encoder_frame=ENCODER_FRAME_NO_BUDGET,
        )
    return EncDecCoproProposalContract(
        budget_mode="budgeted",
        encoder_frame=ENCODER_FRAME,
    )


__all__ = [
    "CODE_COMP_COPRO_OUTPUT_RULE",
    "CODE_COMP_COPRO_PROPOSAL_CONTRACT_VERSION",
    "CODE_COMP_COPRO_TASK_CONTEXT",
    "EncDecCoproProposalContract",
    "encdec_copro_proposal_contract",
]
