from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class InstructionProposalContract(Protocol):
    contract_version: str
    target_name: str
    task_context: str
    output_rule: str

    def validate_instruction(self, body: str) -> None: ...


__all__ = ["InstructionProposalContract"]
