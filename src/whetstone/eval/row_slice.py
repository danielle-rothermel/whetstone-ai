from __future__ import annotations

from dataclasses import dataclass

from whetstone.core.identity import TypedRef
from whetstone.eval.schema import EvalEvidence, EvalFailureEvidence
from whetstone.optim.contracts import ResolutionDetail


@dataclass(frozen=True, slots=True)
class RowEvalSlice:
    task_id: str
    seed_index: int
    evidence: EvalEvidence
    supplemental_aggregate_refs: tuple[TypedRef, ...] = ()


@dataclass(frozen=True, slots=True)
class RowEvalCompletion:
    evidence_ref: TypedRef | None = None
    rejected_detail: ResolutionDetail | None = None
    supplemental_aggregate_refs: tuple[TypedRef, ...] = ()


@dataclass(frozen=True, slots=True)
class RowEvalOutcome:
    task_id: str
    seed_index: int
    evidence_ref: TypedRef | None = None
    rejected_detail: ResolutionDetail | None = None
    supplemental_aggregate_refs: tuple[TypedRef, ...] = ()
    evidence: EvalEvidence | None = None
    failure: EvalFailureEvidence | None = None


__all__ = ["RowEvalCompletion", "RowEvalOutcome", "RowEvalSlice"]
