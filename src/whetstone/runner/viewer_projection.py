"""The immutable per-cell viewer projection published beside the ledger.

One finalized cell publishes exactly two files: ``projection.json``, the strict
summary of what the run produced, and ``rollout_outputs.jsonl``, one line per
driven evaluation output row. Both are derived, never authoritative: every
number here is recomputable from the content-addressed records the ObjectStore
already holds, and nothing in the runner ever reads a projection back to decide
what to do next.

**Serialization is the contract.** This module returns already-serialized
bytes, and the ledger commits exactly those bytes under exactly the SHA-256 it
records. Canonical JSON -- sorted keys, ASCII, no incidental whitespace --
makes the hash a function of content rather than of formatting, so republishing
an identical cell is byte-identical and the ledger's conflict check is
meaningful.

**Optimizer-agnostic by construction.** The projection reads the terminal
``OptimizationResult``, which already composes every ordered Step Result, its
request, its accepted candidates, and its resolved intents. There is no
per-algorithm extraction branch: an optimizer that wants richer reporting
records it in its own durable state, and the viewer cites that state by
reference rather than reinterpreting it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from dr_store import ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import TypedRef, require_full_hash
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.engine import EngineEvaluation
from whetstone.evaluation.schema import (
    EvaluationEvidence,
    EvaluationOutputsRecord,
)
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.experiment.candidate import CandidateRef
from whetstone.optimization.contracts import OptimizationResult, StepStatus

__all__ = [
    "VIEWER_PROJECTION_SCHEMA",
    "VIEWER_ROLLOUT_ROW_SCHEMA",
    "ViewerCandidate",
    "ViewerCellProjection",
    "ViewerEvaluationSummary",
    "ViewerRolloutRow",
    "ViewerStepSummary",
    "build_viewer_cell_projection",
]

#: Persisted-format contract: keep these exact wire keys and versions. A viewer
#: reads them to decide how to parse the file, so renaming one orphans every
#: already-published cell directory.
VIEWER_PROJECTION_SCHEMA = "whetstone.runner.viewer_projection/v1"
VIEWER_ROLLOUT_ROW_SCHEMA = "whetstone.runner.viewer_rollout_row/v1"


def _canonical_json(value: object) -> str:
    """Serialize one record so its bytes are a function of its content."""
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


class ViewerCandidate(BaseModel):
    """One ordered candidate the run proposed or accepted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: StrictInt
    candidate: CandidateRef
    #: The step that first accepted this candidate. ``None`` marks a candidate
    #: the cell supplied rather than the optimizer producing it.
    first_accepted_step_index: StrictInt | None = None

    @model_validator(mode="after")
    def _validate_candidate(self) -> ViewerCandidate:
        if self.ordinal < 0:
            raise ValueError("candidate ordinal cannot be negative")
        if (
            self.first_accepted_step_index is not None
            and self.first_accepted_step_index < 0
        ):
            raise ValueError("first_accepted_step_index cannot be negative")
        return self


class ViewerStepSummary(BaseModel):
    """One optimizer step, summarized for reporting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: StrictInt
    step_result_ref: TypedRef
    status: StepStatus
    proposed_count: StrictInt
    accepted_count: StrictInt
    resolved_intent_count: StrictInt
    tool_evidence_count: StrictInt
    #: Present only on the terminal step of a failed run.
    terminal_failure_code: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_step(self) -> ViewerStepSummary:
        if self.step_index < 0:
            raise ValueError("step index cannot be negative")
        for name in (
            "proposed_count",
            "accepted_count",
            "resolved_intent_count",
            "tool_evidence_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        return self


class ViewerEvaluationSummary(BaseModel):
    """One official arm's evidence, summarized without reinterpretation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    evidence_ref: TypedRef
    candidate_id: StrictStr
    candidate_identity_hash: StrictStr
    evaluation_role: EvaluationRole
    purpose: StrictStr
    graph_hash: StrictStr
    eval_config_hash: StrictStr
    task_hashes: tuple[StrictStr, ...]
    num_samples: StrictInt
    per_task_values: tuple[StrictFloat, ...]
    per_task_counts: tuple[StrictInt, ...]
    rows_planned: StrictInt
    rows_present: StrictInt
    rows_missing: StrictInt
    rows_failed: StrictInt
    rows_invalid: StrictInt
    aggregate_name: StrictStr
    #: ``None`` means the aggregate is genuinely unavailable, never zero.
    aggregate_value: float | None
    aggregate_status: StrictStr
    outputs_ref: TypedRef

    @model_validator(mode="after")
    def _validate_summary(self) -> ViewerEvaluationSummary:
        require_full_hash(
            self.candidate_identity_hash, field="candidate_identity_hash"
        )
        if self.evidence_ref.schema_name != EVALUATION_EVIDENCE_SCHEMA:
            raise ValueError("evidence_ref must cite evaluation evidence")
        aligned = len(self.task_hashes)
        if len(self.per_task_values) != aligned:
            raise ValueError("per_task_values must align with task identities")
        if len(self.per_task_counts) != aligned:
            raise ValueError("per_task_counts must align with task identities")
        return self


class ViewerRolloutRow(BaseModel):
    """One evaluation output row with explicit semantic task identity."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        allow_inf_nan=False,
    )

    #: Persisted-format contract: keep this exact wire key and version.
    schema_: Literal["whetstone.runner.viewer_rollout_row/v1"] = Field(
        default=VIEWER_ROLLOUT_ROW_SCHEMA,
        alias="schema",
    )
    cell_id: StrictStr
    evidence_ref: TypedRef
    candidate_id: StrictStr
    candidate_identity_hash: StrictStr
    evaluation_role: EvaluationRole
    purpose: StrictStr
    task_id: StrictStr
    task_hash: StrictStr
    sample_index: StrictInt
    rendered_prompt: StrictStr
    output_text: StrictStr | None
    score: StrictFloat | None
    failure_code: StrictStr
    finish_reason: StrictStr | None = None
    max_budget: StrictInt | None = None
    over_budget: StrictBool | None = None

    @model_validator(mode="after")
    def _validate_row(self) -> ViewerRolloutRow:
        require_full_hash(
            self.candidate_identity_hash, field="candidate_identity_hash"
        )
        if self.evidence_ref.schema_name != EVALUATION_EVIDENCE_SCHEMA:
            raise ValueError("rollout row must cite evaluation evidence")
        if self.sample_index < 0:
            raise ValueError("rollout repeat cannot be negative")
        return self

    def to_line(self) -> str:
        """One complete JSONL line, newline included."""
        return (
            _canonical_json(self.model_dump(mode="json", by_alias=True)) + "\n"
        )


class ViewerCellProjection(BaseModel):
    """The complete reporting projection for one published cell."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True
    )

    #: Persisted-format contract: keep this exact wire key and version.
    schema_: Literal["whetstone.runner.viewer_projection/v1"] = Field(
        default=VIEWER_PROJECTION_SCHEMA,
        alias="schema",
    )
    cell_id: StrictStr
    optimizer: StrictStr
    env: StrictStr
    attempt: StrictInt
    run_id: StrictStr
    optimization_result_ref: TypedRef
    terminal_status: StepStatus
    steps: tuple[ViewerStepSummary, ...]
    candidates: tuple[ViewerCandidate, ...]
    proposals: tuple[CandidateRef, ...]
    evidence_summaries: tuple[ViewerEvaluationSummary, ...]
    rollout_row_count: StrictInt

    @model_validator(mode="after")
    def _validate_projection(self) -> ViewerCellProjection:
        if self.attempt < 0:
            raise ValueError("cell attempt cannot be negative")
        if self.cell_id != f"{self.optimizer}:{self.env}:a{self.attempt}":
            raise ValueError("projection cell identity fields do not align")
        if not self.steps:
            raise ValueError("a projection composes every ordered step")
        indices = tuple(step.step_index for step in self.steps)
        if indices != tuple(range(len(self.steps))):
            raise ValueError("step indices must be contiguous from zero")
        ordinals = tuple(candidate.ordinal for candidate in self.candidates)
        if ordinals != tuple(range(len(self.candidates))):
            raise ValueError("candidate ordinals must be contiguous from zero")
        identities = tuple(
            candidate.candidate.identity_hash for candidate in self.candidates
        )
        if len(identities) != len(set(identities)):
            raise ValueError("ordered candidates must be identity-unique")
        if self.rollout_row_count < 0:
            raise ValueError("rollout_row_count cannot be negative")
        if self.steps[-1].status is not self.terminal_status:
            raise ValueError(
                "terminal_status must match the final step's status"
            )
        return self

    def to_bytes(self) -> bytes:
        """The exact bytes the ledger commits and hashes."""
        return (
            _canonical_json(self.model_dump(mode="json", by_alias=True)) + "\n"
        ).encode("utf-8")


def _evaluation_summary(
    evaluation: EngineEvaluation,
) -> ViewerEvaluationSummary:
    evidence = evaluation.evidence
    accounting = evidence.row_accounting
    binding = evidence.evaluation_binding
    return ViewerEvaluationSummary(
        evidence_ref=evaluation.evidence_ref,
        candidate_id=evidence.candidate.record.candidate_id,
        candidate_identity_hash=evidence.candidate.identity_hash,
        evaluation_role=binding.role,
        purpose=evidence.purpose,
        graph_hash=evidence.graph_hash,
        eval_config_hash=binding.eval_config.config_hash,
        task_hashes=evidence.task_hashes,
        num_samples=evidence.num_samples,
        per_task_values=evidence.per_task_values,
        per_task_counts=evidence.per_task_counts,
        rows_planned=accounting.planned,
        rows_present=accounting.present,
        rows_missing=accounting.missing,
        rows_failed=accounting.failed,
        rows_invalid=accounting.invalid,
        aggregate_name=evidence.aggregate_name,
        aggregate_value=evidence.aggregate_value,
        aggregate_status=evidence.aggregate_status,
        outputs_ref=evidence.outputs_ref,
    )


def _rollout_rows(
    *,
    cell_id: str,
    evaluation: EngineEvaluation,
    store: ObjectStore,
) -> tuple[ViewerRolloutRow, ...]:
    evidence: EvaluationEvidence = evaluation.evidence
    record = EvaluationOutputsRecord.model_validate(
        store.get(evidence.outputs_ref.reference)
    )
    candidate = evidence.candidate
    if record.candidate != candidate:
        raise ValueError("evaluation outputs belong to another candidate")
    rows: list[ViewerRolloutRow] = []
    for row in record.outputs:
        if row.candidate_id != candidate.record.candidate_id:
            raise ValueError("evaluation output row candidate is inconsistent")
        rows.append(
            ViewerRolloutRow(
                cell_id=cell_id,
                evidence_ref=evaluation.evidence_ref,
                candidate_id=row.candidate_id,
                candidate_identity_hash=candidate.identity_hash,
                evaluation_role=record.evaluation_role,
                purpose=evidence.purpose,
                task_id=row.task_id,
                task_hash=row.task_hash,
                sample_index=row.sample_index,
                rendered_prompt=row.rendered_prompt,
                output_text=row.output_text,
                score=row.score,
                failure_code=row.failure_code,
                finish_reason=row.finish_reason,
                max_budget=row.max_budget,
                over_budget=row.over_budget,
            )
        )
    return tuple(rows)


def _ordered_candidates(
    result: OptimizationResult,
) -> tuple[ViewerCandidate, ...]:
    """Every candidate the run accepted, in first-accepted order."""
    ordered: list[ViewerCandidate] = []
    seen: set[str] = set()
    for step_ref in result.step_results:
        step = step_ref.record
        for reference in step.accepted_candidates:
            if reference.identity_hash in seen:
                continue
            seen.add(reference.identity_hash)
            ordered.append(
                ViewerCandidate(
                    ordinal=len(ordered),
                    candidate=reference,
                    first_accepted_step_index=step.step_index,
                )
            )
    return tuple(ordered)


def _step_summaries(
    result: OptimizationResult,
) -> tuple[ViewerStepSummary, ...]:
    return tuple(
        ViewerStepSummary(
            step_index=step_ref.record.step_index,
            step_result_ref=step_ref.record_ref,
            status=step_ref.record.status,
            proposed_count=len(step_ref.record.proposed_candidates),
            accepted_count=len(step_ref.record.accepted_candidates),
            resolved_intent_count=len(step_ref.record.resolved_intents),
            tool_evidence_count=len(step_ref.record.tool_evidence),
            terminal_failure_code=(
                step_ref.record.terminal_failure.code
                if step_ref.record.terminal_failure is not None
                else None
            ),
        )
        for step_ref in result.step_results
    )


def build_viewer_cell_projection(
    *,
    cell_id: str,
    optimizer: str,
    env: str,
    attempt: int,
    result: OptimizationResult,
    result_ref: TypedRef,
    store: ObjectStore,
    official_evaluations: Sequence[EngineEvaluation],
) -> tuple[ViewerCellProjection, tuple[str, ...]]:
    """Build one strict viewer projection and its rollout output lines.

    Returns the projection and the already-serialized rollout lines. The caller
    passes both to :meth:`Ledger.write_viewer_publication`, which commits
    exactly these bytes under exactly the hashes it records.
    """
    if result.run.record.run_id != cell_id:
        raise ValueError("optimization result does not belong to this cell")
    steps = _step_summaries(result)
    rows: list[ViewerRolloutRow] = []
    summaries: list[ViewerEvaluationSummary] = []
    for evaluation in official_evaluations:
        summaries.append(_evaluation_summary(evaluation))
        rows.extend(
            _rollout_rows(cell_id=cell_id, evaluation=evaluation, store=store)
        )
    projection = ViewerCellProjection(
        cell_id=cell_id,
        optimizer=optimizer,
        env=env,
        attempt=attempt,
        run_id=str(result.run.record.run_id),
        optimization_result_ref=result_ref,
        terminal_status=steps[-1].status,
        steps=steps,
        candidates=_ordered_candidates(result),
        proposals=tuple(proposal.candidate for proposal in result.proposals),
        evidence_summaries=tuple(summaries),
        rollout_row_count=len(rows),
    )
    return projection, tuple(row.to_line() for row in rows)
