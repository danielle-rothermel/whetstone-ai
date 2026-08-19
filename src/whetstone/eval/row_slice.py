from __future__ import annotations

from dataclasses import dataclass

from whetstone.eval.schema import EvalEvidence


@dataclass(frozen=True, slots=True)
class RowEvalSlice:
    task_id: str
    seed_index: int
    evidence: EvalEvidence


__all__ = ["RowEvalSlice"]
