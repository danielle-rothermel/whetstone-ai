from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator


COPRO_PROPOSAL_CONTRACT_SCHEMA = "whetstone.copro_proposal_contract"
COPRO_PROPOSAL_CONTRACT_VERSION = 1


@runtime_checkable
class CoproProposalContract(Protocol):
    contract_version: str
    target_name: str
    task_context: str
    output_rule: str

    def validate_mutation(self, body: str) -> None: ...

    def prompt_context(self, *, budget_mode: str) -> dict[str, object]: ...

    def validate_instruction(self, body: str) -> None: ...


class CoproProposalContractRecord(BaseModel):
    """Identity-bearing mutation contract for COPRO proposal validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: StrictStr = "whetstone.copro_proposal/v1"
    target_name: StrictStr
    task_context: StrictStr
    output_rule: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> CoproProposalContractRecord:
        for field_name in ("target_name", "task_context", "output_rule"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        return self

    def validate_mutation(self, body: str) -> None:
        if type(body) is not str or not body.strip():
            raise ValueError("proposal mutation body must be non-empty text")

    def validate_instruction(self, body: str) -> None:
        self.validate_mutation(body)

    def prompt_context(self, *, budget_mode: str) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "task_context": self.task_context,
            "budget_mode": budget_mode,
            "output_rule": self.output_rule,
        }


__all__ = [
    "COPRO_PROPOSAL_CONTRACT_SCHEMA",
    "COPRO_PROPOSAL_CONTRACT_VERSION",
    "CoproProposalContract",
    "CoproProposalContractRecord",
]
