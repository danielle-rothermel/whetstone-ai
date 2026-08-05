"""One validation cell: the runner's unit of work and of resumability.

A cell is one optimizer against one environment at one attempt, identified by
``optimizer:env:aN``. Running it produces one durable ``cells.jsonl`` line, and
that line is the resumability record: a completed cell is skipped rather than
re-run, so relaunching a wave is safe and cheap.

**What a cell measures.** Three official arms, all evaluated through the same
official Eval Config so their scores are comparable: ``baseline`` (the naive
starting candidate), ``ceiling`` (the reference upper arm, optional), and
``best`` (the candidate the optimizer terminally proposed). The verdict is
read off ``delta_ci95``, the paired best-minus-naive bootstrap interval; the
Eval-row headroom gate is ``headroom_ci95``, the paired ceiling-minus-naive
interval.

**Order is the safety property.** Every paid boundary is preceded by a
durable spend checkpoint, so a process that dies mid-cell leaves evidence of
what it had already spent, and a resumed run enforces the same stop loss rather
than starting its budget accounting over. Official arms are bound in the
ObjectStore before they are paid for, so an arm can never be silently re-run
under different inputs: a changed binding is a conflict, not a new evaluation.

**Terminal artifacts, in a deliberate order.** The immutable viewer directory
is fsynced and committed by one atomic rename *before* the cells line is
appended.
A terminal ledger line therefore can never cite viewer evidence that was never
committed; the reverse ordering would make the line a promise instead of a
record.

**Derived, never authoritative.** Confidence intervals, the status string, the
human-readable trace, and the viewer projection are all recomputable from
content-addressed evidence. The runner recomputes them freely and never reads
them back to decide what to do next.

**Codex spend, stated honestly.** Codex steps evaluate inside a sandboxed MCP
child process. That child enforces capacity through its own admission authority
against the capacity binding passed to it, and it owns the Tool Call Store its
calls are admitted into. Harness-side ledger and step evidence therefore do not
reflect subprocess MCP spend: ``harness_store_accepted_call_count`` reads 0 for
such a run regardless of how many calls the agent made. Canonical evaluation
downstream is the authority for what a proposal is worth; per-proposal MCP
evidence is not matched against proposals.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dr_serialize import Jsonable
from dr_store import BindingConflictError, ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.code.statistics import (
    BootstrapCI,
    bootstrap_delta_ci,
)
from whetstone.evaluation.code.statistics import bootstrap_mean_ci as _mean_ci
from whetstone.evaluation.engine import EngineEvaluation, EvaluationEngine
from whetstone.evaluation.schema import EvaluationEvidence
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.experiment.binding import EvalConfigRef, EvaluationBinding
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.optimization.contracts import (
    EvaluationIntent,
    IntentOutcome,
    OptimizationResult,
    StepStatus,
)
from whetstone.runner.budget import BudgetGuard, CreditsSnapshot
from whetstone.runner.events import (
    EventStream,
    EventUnit,
    attempt_skipped_event,
    cell_failed_event,
    cell_finalized_event,
)
from whetstone.runner.ledger import (
    OFFICIAL_ANCHOR_SCHEMA,
    CellArtifacts,
    CellControls,
    CellModels,
    CellRecord,
    Ledger,
    OfficialAnchorRecord,
    SpendRecord,
)
from whetstone.runner.optimization_run import HarnessRunController
from whetstone.runner.viewer_projection import build_viewer_cell_projection

__all__ = [
    "CELL_RUN_CONTROL_SCHEMA",
    "OFFICIAL_ARM_ADMISSION_SCHEMA",
    "OFFICIAL_ARM_BINDING_SCHEMA",
    "CellBaselineFailure",
    "CellConfig",
    "CellError",
    "CellOutcome",
    "CellRunControl",
    "OfficialArmBinding",
    "bind_cell_launch",
    "prepare_cell_launch",
    "run_cell",
]

#: Persisted-format contract: the ObjectStore binding schemas that make an
#: official arm and a cell's control immutable once bound.
OFFICIAL_ARM_BINDING_SCHEMA = "whetstone.runner.official_arm_binding"
OFFICIAL_ARM_ADMISSION_SCHEMA = "whetstone.runner.official_arm_admission"
CELL_RUN_CONTROL_SCHEMA = "whetstone.runner.cell_run_control"

#: Fixed bootstrap seeds. The intervals are derived reporting numbers, so
#: they must be reproducible from the same persisted per-task vectors; a
#: per-run seed would make two readings of the same evidence disagree.
_NAIVE_CI_SEED = 17
_CEILING_CI_SEED = 19
_DELTA_CI_SEED = 23
_HEADROOM_CI_SEED = 29


class CellError(RuntimeError):
    """The cell's inputs, bindings, or evidence are invalid."""


class CellBaselineFailure(CellError):
    """The official baseline could not produce a reportable aggregate."""


class OfficialArmBinding(BaseModel):
    """The exact official-arm request, bound before its paid evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: StrictStr
    arm: StrictStr
    candidate: CandidateRef
    eval_config: EvalConfigRef
    context_id: StrictStr
    purpose: StrictStr

    def record_content(self) -> Jsonable:
        return self.model_dump(mode="json")


class CellRunControl(BaseModel):
    """The exact cell inputs, bound before paid work or terminal reuse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: StrictStr
    canonical: bool
    models: CellModels
    lane: StrictStr
    baseline: CandidateRef
    ceiling: CandidateRef | None
    run_control_identity_hash: StrictStr
    official_eval_config: EvalConfigRef

    def record_content(self) -> Jsonable:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class CellConfig:
    """Everything one cell needs, assembled by the runner's startup wiring.

    ``controller`` owns the optimizer's harness and durable control; ``driver``
    is the callable that actually enters the run's parent DBOS workflow and
    returns the terminal Optimization Result reference. Keeping the driver
    injected is what lets a cell be exercised without a DBOS runtime while the
    production path always goes through the parent workflow.
    """

    env: str
    attempt: int
    canonical: bool
    task_model: str
    proposer_model: str
    lane: str
    baseline: Candidate
    controller: HarnessRunController
    driver: Callable[[], TypedRef]
    official_engine: EvaluationEngine
    official_evaluation_binding: EvaluationBinding
    store: ObjectStore
    ledger: Ledger
    ceiling: Candidate | None = None
    budget_guard: BudgetGuard = field(default_factory=BudgetGuard)
    credits_fetcher: Callable[[], CreditsSnapshot | None] | None = None
    credits_authority_identity_hash: str | None = None
    event_stream: EventStream | None = None

    def __post_init__(self) -> None:
        expected_run_id = f"{self.optimizer}:{self.env}:a{self.attempt}"
        if self.controller.control.run_id != expected_run_id:
            raise CellError(
                "optimization run_id must equal the exact cell identity "
                f"{expected_run_id!r}; got "
                f"{self.controller.control.run_id!r}"
            )
        if self.attempt < 0:
            raise CellError("cell attempt cannot be negative")
        if (
            self.official_evaluation_binding.role
            is not EvaluationRole.OFFICIAL
        ):
            raise CellError(
                "the official binding must carry the official role"
            )
        if (
            self.official_evaluation_binding.eval_config
            != self.official_engine.eval_config_ref
        ):
            raise CellError(
                "official binding Eval Config must match the official engine"
            )
        if self.credits_authority_identity_hash is not None:
            require_full_hash(
                self.credits_authority_identity_hash,
                field="credits_authority_identity_hash",
            )

    @property
    def optimizer(self) -> str:
        return self.controller.control.adapter_key

    @property
    def cell_id(self) -> str:
        return f"{self.optimizer}:{self.env}:a{self.attempt}"

    @property
    def event_unit(self) -> EventUnit:
        return EventUnit.for_cell(
            cell_id=self.cell_id,
            env=self.env,
            optimizer=self.optimizer,
            attempt=self.attempt,
            lane=self.lane,
            model=self.task_model,
        )


@dataclass(frozen=True, slots=True)
class CellOutcome:
    """One cell's terminal ledger line and the result that produced it."""

    record: CellRecord
    result: OptimizationResult | None
    skipped: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _spend_record(
    config: CellConfig,
    *,
    phase: str,
    snapshot: CreditsSnapshot,
    event_id: str | None = None,
) -> SpendRecord:
    return SpendRecord(
        event_id=event_id or str(uuid.uuid4()),
        cell_id=config.cell_id,
        phase=phase,
        lane=config.lane,
        total_credits=snapshot.total_credits,
        total_usage=snapshot.total_usage,
        remaining_usd=snapshot.remaining_usd,
        at=snapshot.at or _now(),
    )


def _open_initial_spend(config: CellConfig) -> SpendRecord | None:
    """The cell's still-open starting snapshot, if a prior invocation left one.

    A ``before`` record already followed by an ``after`` record closed its
    invocation, so it is history rather than this invocation's stop-loss
    baseline.
    """
    records = config.ledger.spend_records()
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if record.cell_id != config.cell_id or record.phase != "before":
            continue
        closed = any(
            later.cell_id == config.cell_id and later.phase == "after"
            for later in records[index + 1 :]
        )
        return None if closed else record
    return None


def _ensure_initial_spend(config: CellConfig) -> SpendRecord | None:
    """Recover or durably create this cell's stop-loss baseline."""
    persisted = _open_initial_spend(config)
    if persisted is not None:
        return persisted
    snapshot = config.credits_fetcher() if config.credits_fetcher else None
    if snapshot is None:
        return None
    invocation_index = sum(
        record.cell_id == config.cell_id and record.phase == "before"
        for record in config.ledger.spend_records()
    )
    event_id = compute_identity_hash(
        schema="whetstone.runner.cell_initial_spend",
        schema_version=1,
        payload={
            "cell_id": config.cell_id,
            "invocation_index": invocation_index,
        },
    )
    record = _spend_record(
        config, phase="before", snapshot=snapshot, event_id=event_id
    )
    config.ledger.append_spend(record)
    return record


def _check_cell_start(config: CellConfig, initial: SpendRecord | None) -> None:
    """Apply the reserve guard, which only a rerun of *this* attempt escapes.

    A rerun is this exact ``(optimizer, env, attempt)`` starting again, so the
    money it is about to spend was already committed to once. A later attempt
    of the same env is a *fresh* paid cell no matter how many earlier attempts
    exist, so keying on any prior line for the env would let attempt N>0 start
    below the reserve -- precisely the spend the guard exists to refuse.
    """
    is_rerun = (
        config.ledger.for_attempt(config.optimizer, config.env, config.attempt)
        is not None
    )
    config.budget_guard.check_start(
        canonical=config.canonical,
        remaining_usd=(initial.remaining_usd if initial is not None else None),
        is_rerun=is_rerun,
    )


def _arm_binding(
    config: CellConfig, *, arm: str, candidate: Candidate, purpose: str
) -> OfficialArmBinding:
    return OfficialArmBinding(
        cell_id=config.cell_id,
        arm=arm,
        candidate=candidate_reference(candidate),
        eval_config=config.official_engine.eval_config_ref,
        context_id=f"{config.cell_id}:official:{arm}",
        purpose=purpose,
    )


def _bind_official_arm(
    config: CellConfig, *, arm: str, candidate: Candidate, purpose: str
) -> OfficialArmBinding:
    """Bind one official arm immutably before it is ever paid for."""
    binding = _arm_binding(
        config, arm=arm, candidate=candidate, purpose=purpose
    )
    reference, _ = config.store.put(
        OFFICIAL_ARM_BINDING_SCHEMA, binding.record_content()
    )
    key = f"{OFFICIAL_ARM_BINDING_SCHEMA}:{config.cell_id}#{arm}"
    try:
        config.store.bind(key, reference)
    except BindingConflictError as conflict:
        raise CellError(
            f"official arm {arm!r} is already bound to "
            f"{conflict.existing.content_hash}; refusing "
            f"{reference.content_hash}"
        ) from conflict
    return binding


def _evaluate_official(
    config: CellConfig, *, arm: str, candidate: Candidate, purpose: str
) -> EngineEvaluation:
    """Resolve one official arm's intent and check it against its binding."""
    binding = _arm_binding(
        config, arm=arm, candidate=candidate, purpose=purpose
    )
    resolution = EngineEvaluationService(
        store=config.store, engine=config.official_engine
    ).resolve_evaluation_intent(
        EvaluationIntent(
            intent_id=binding.context_id,
            candidate=binding.candidate,
            target_eval_config=binding.eval_config,
            evaluation_binding=config.official_evaluation_binding,
            purpose=purpose,
            run_id=config.cell_id,
            step_index=0,
        )
    )
    reference = resolution.evaluation_result_ref
    if resolution.outcome is not IntentOutcome.COMPLETED or reference is None:
        raise CellError(
            f"official arm {arm!r} did not produce canonical evidence: "
            f"{resolution.detail.message}"
        )
    if reference.schema_name != EVALUATION_EVIDENCE_SCHEMA:
        raise CellError(f"official arm {arm!r} resolved to non-evidence")
    evidence = EvaluationEvidence.model_validate(
        config.store.get(reference.reference)
    )
    evaluated = EngineEvaluation(evidence=evidence, evidence_ref=reference)
    if (
        evidence.candidate != binding.candidate
        or evidence.evaluation_binding != config.official_evaluation_binding
        or evidence.purpose != purpose
    ):
        raise CellError(
            f"official arm {arm!r} evidence does not match its binding"
        )
    if evidence.reward_ref is not None:
        raise CellError("an official evaluation must not produce a Reward")
    return evaluated


def _cell_run_control(config: CellConfig) -> CellRunControl:
    return CellRunControl(
        cell_id=config.cell_id,
        canonical=config.canonical,
        models=CellModels(
            task=config.task_model, proposer=config.proposer_model
        ),
        lane=config.lane,
        baseline=candidate_reference(config.baseline),
        ceiling=(
            candidate_reference(config.ceiling)
            if config.ceiling is not None
            else None
        ),
        run_control_identity_hash=config.controller.control.identity_hash(),
        official_eval_config=config.official_engine.eval_config_ref,
    )


def _bind_cell_run_control(config: CellConfig) -> None:
    control = _cell_run_control(config)
    reference, _ = config.store.put(
        CELL_RUN_CONTROL_SCHEMA, control.record_content()
    )
    try:
        config.store.bind(
            f"{CELL_RUN_CONTROL_SCHEMA}:{config.cell_id}", reference
        )
    except BindingConflictError as conflict:
        raise CellError(
            f"cell {config.cell_id!r} is already bound to control "
            f"{conflict.existing.content_hash}; refusing "
            f"{reference.content_hash}"
        ) from conflict
    _bind_official_arm(
        config,
        arm="baseline",
        candidate=config.baseline,
        purpose="official_baseline",
    )
    if config.ceiling is not None:
        _bind_official_arm(
            config,
            arm="ceiling",
            candidate=config.ceiling,
            purpose="official_ceiling",
        )


def _completed_record(config: CellConfig) -> CellRecord | None:
    if not config.ledger.is_completed(
        config.optimizer, config.env, config.attempt
    ):
        return None
    return config.ledger.for_attempt(
        config.optimizer, config.env, config.attempt
    )


def _verify_bound_controls(config: CellConfig) -> None:
    """Refuse to reuse a completed cell whose controls have since changed.

    Skipping a completed cell returns evidence produced under the controls
    that were bound when it ran. If the factory has since changed the
    baseline, ceiling, models, lane, run control, or official Eval Config,
    that evidence answers a different question, and returning it would make
    the skip a silent substitution. The bound control is immutable, so
    comparing this config's control against it turns the drift into the same
    loud conflict a rebinding attempt raises -- the skip path is held to
    exactly the contract the paid path is.
    """
    bound = config.store.resolve(f"{CELL_RUN_CONTROL_SCHEMA}:{config.cell_id}")
    if bound is None:
        return
    expected, _ = config.store.put(
        CELL_RUN_CONTROL_SCHEMA, _cell_run_control(config).record_content()
    )
    if bound.content_hash != expected.content_hash:
        raise CellError(
            f"cell {config.cell_id!r} completed under control "
            f"{bound.content_hash}, but the current config resolves to "
            f"{expected.content_hash}; refusing to reuse its evidence under "
            "changed controls"
        )


def prepare_cell_launch(config: CellConfig) -> CellOutcome | None:
    """Preflight a cell and short-circuit one that already completed.

    A completed cell returns its immutable ledger projection here, so the CLI
    can skip credits authorities and the DBOS runtime entirely. Preflight runs
    first regardless, because a config that cannot evaluate its own arms is
    misconfigured whether or not a prior attempt succeeded.

    The skip is only sound while the cell's controls are unchanged, so a
    completed cell's bound control is checked against this config before its
    evidence is reused. Changed controls are a conflict, not a cheap skip.
    """
    config.official_engine.preflight(config.baseline)
    if config.ceiling is not None:
        config.official_engine.preflight(config.ceiling)
    completed = _completed_record(config)
    if completed is None:
        return None
    _verify_bound_controls(config)
    if config.event_stream is not None:
        config.event_stream.emit(
            attempt_skipped_event(
                unit=config.event_unit, prior_status=completed.status
            )
        )
    return CellOutcome(record=completed, result=None, skipped=True)


def bind_cell_launch(config: CellConfig) -> None:
    """Bind the exact active-cell controls before DBOS automatic recovery."""
    _bind_cell_run_control(config)


def _reportable_score(evaluation: EngineEvaluation | None) -> float | None:
    if evaluation is None:
        return None
    evidence = evaluation.evidence
    if evidence.aggregate_status != "ok":
        return None
    return evidence.aggregate_value


def _official_anchor_record(
    config: CellConfig,
    *,
    baseline: EngineEvaluation,
    ceiling: EngineEvaluation,
) -> OfficialAnchorRecord:
    """Validate and build the viewer-only official anchor projection."""
    sampling = config.official_engine.sampling
    expected_task_identities = sampling.task_set.task_identities
    rollout_definition = config.official_engine.experiment.rollout_definition
    expected_graph_hash = rollout_definition.graph_hash
    expected_task_model = rollout_definition.provider_call_config.route.model
    if config.task_model != expected_task_model:
        raise CellError(
            "cell task_model must match the rollout definition's provider "
            f"route model {expected_task_model!r}"
        )
    aligned_count = len(expected_task_identities)
    for arm, evaluated in (("baseline", baseline), ("ceiling", ceiling)):
        evidence = evaluated.evidence
        if evidence.task_identities != expected_task_identities:
            raise CellError(
                f"official {arm} task_identities do not match sampling order"
            )
        if evidence.repeat_count != sampling.repeat_plan.repeat_count:
            raise CellError(
                f"official {arm} repeat_count does not match sampling"
            )
        if evidence.graph_hash != expected_graph_hash:
            raise CellError(
                f"official {arm} graph_hash does not match the rollout "
                "definition"
            )
        if (
            evidence.aggregate_status != "ok"
            or evidence.aggregate_value is None
        ):
            raise CellError(f"official {arm} aggregate is not reportable")
        if (
            len(evidence.per_task_values) != aligned_count
            or len(evidence.per_task_counts) != aligned_count
        ):
            raise CellError(
                f"official {arm} per-task values/counts do not align with "
                "sampling"
            )
    baseline_score = _reportable_score(baseline)
    ceiling_score = _reportable_score(ceiling)
    if baseline_score is None or ceiling_score is None:
        raise CellError("official anchors require two reportable aggregates")
    return OfficialAnchorRecord.model_validate(
        {
            "schema": OFFICIAL_ANCHOR_SCHEMA,
            "cell_id": config.cell_id,
            "env": config.env,
            "task_model": config.task_model,
            "graph_hash": expected_graph_hash,
            "eval_config_hash": (
                config.official_engine.eval_config_ref.identity_hash
            ),
            "official_instance_ids": tuple(
                str(instance.id) for instance in sampling.instances
            ),
            "official_task_identities": expected_task_identities,
            "baseline_evidence_ref": baseline.evidence_ref,
            "ceiling_evidence_ref": ceiling.evidence_ref,
            "baseline_official": baseline_score,
            "ceiling_official": ceiling_score,
            "baseline_per_task": baseline.evidence.per_task_values,
            "ceiling_per_task": ceiling.evidence.per_task_values,
            "baseline_per_task_counts": baseline.evidence.per_task_counts,
            "ceiling_per_task_counts": ceiling.evidence.per_task_counts,
            "official_repeats_used": sampling.repeat_plan.repeat_count,
        }
    )


def _status_for(
    *,
    terminal_status: StepStatus,
    best_score: float | None,
    ceiling_expected: bool,
    ceiling_score: float | None,
    delta: float | None,
    delta_ci: BootstrapCI | None,
) -> str:
    """The cell's terminal status, read off its own persisted evidence."""
    if terminal_status is StepStatus.FAILED:
        return "proposer-failure"
    if best_score is None or (ceiling_expected and ceiling_score is None):
        return "incomplete-arm"
    if delta is not None and delta > 0 and delta_ci is not None:
        return "improved" if delta_ci.low > 0 else "inconclusive"
    return "no-improvement"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _internal_evaluation_count(result: OptimizationResult) -> int:
    return sum(
        len(step.record.resolved_intents) + len(step.record.tool_evidence)
        for step in result.step_results
    )


def run_cell(config: CellConfig) -> CellOutcome:
    """Run one cell and project its canonical evidence into the ledger."""
    if completed := prepare_cell_launch(config):
        return completed
    _bind_cell_run_control(config)
    started_at = _now()
    started = time.monotonic()
    before = _ensure_initial_spend(config)
    _check_cell_start(config, before)
    initial_remaining = before.remaining_usd if before is not None else None

    def checkpoint_spend(phase: str, *, admission_id: str) -> None:
        """Record one durable spend checkpoint before a paid boundary."""
        if config.credits_fetcher is None:
            return
        checkpoint_phase = f"checkpoint:{phase}"
        existing = next(
            (
                record
                for record in config.ledger.spend_records()
                if record.event_id == admission_id
            ),
            None,
        )
        if existing is not None:
            if (
                existing.cell_id != config.cell_id
                or existing.phase != checkpoint_phase
            ):
                raise CellError(
                    "paid admission ID is already bound to another checkpoint"
                )
            remaining = existing.remaining_usd
        else:
            snapshot = config.credits_fetcher()
            if snapshot is None:
                return
            config.ledger.append_spend(
                _spend_record(
                    config,
                    phase=checkpoint_phase,
                    snapshot=snapshot,
                    event_id=admission_id,
                )
            )
            remaining = snapshot.remaining_usd
        if initial_remaining is None or remaining is None:
            return
        config.budget_guard.check_stop_loss(
            max(0.0, initial_remaining - remaining)
        )

    def evaluate_arm(
        arm: str, candidate: Candidate, purpose: str
    ) -> EngineEvaluation:
        admission_id = compute_identity_hash(
            schema=OFFICIAL_ARM_ADMISSION_SCHEMA,
            schema_version=1,
            payload={
                "cell_id": config.cell_id,
                "arm": arm,
                "candidate": candidate_reference(candidate).model_dump(
                    mode="json"
                ),
                "eval_config": (
                    config.official_engine.eval_config_ref.model_dump(
                        mode="json"
                    )
                ),
                "credits_authority_identity_hash": (
                    config.credits_authority_identity_hash
                ),
            },
        )
        checkpoint_spend(f"official:{arm}", admission_id=admission_id)
        return _evaluate_official(
            config, arm=arm, candidate=candidate, purpose=purpose
        )

    def close_invocation() -> CreditsSnapshot | None:
        """Write this invocation's closing ``after`` snapshot, once.

        Closing the pair is what makes the invocation's spend accounting
        complete and retires the open ``before`` that
        :func:`_open_initial_spend` recovers as the stop-loss baseline. Every
        exit from the paid region runs through here, so an invocation that
        stopped early is still bounded evidence rather than an open interval.
        """
        snapshot = config.credits_fetcher() if config.credits_fetcher else None
        if snapshot is not None:
            config.ledger.append_spend(
                _spend_record(config, phase="after", snapshot=snapshot)
            )
        return snapshot

    try:
        baseline = evaluate_arm(
            "baseline", config.baseline, "official_baseline"
        )
        if _reportable_score(baseline) is None:
            raise CellBaselineFailure(
                "official baseline aggregate is incomplete"
            )
        ceiling = (
            evaluate_arm("ceiling", config.ceiling, "official_ceiling")
            if config.ceiling is not None
            else None
        )
        checkpoint_spend(
            "optimization",
            admission_id=compute_identity_hash(
                schema=OFFICIAL_ARM_ADMISSION_SCHEMA,
                schema_version=1,
                payload={
                    "cell_id": config.cell_id,
                    "arm": "optimization",
                    "control": config.controller.control.identity_hash(),
                },
            ),
        )
        result_ref = config.driver()
        result = config.controller.resolve_result(result_ref)
        selected = (
            result.proposals[0].candidate.record if result.proposals else None
        )
        best = (
            evaluate_arm("best", selected, "official_best")
            if selected is not None
            else None
        )
    except Exception as exc:
        # Close the spend pair before propagating. A cell that stopped inside
        # its paid region has still spent money, and leaving its ``before``
        # open would make the next invocation recover this invocation's
        # baseline and re-trip the same durable stop-loss checkpoint forever,
        # stranding the cell with no way to record what it spent.
        close_invocation()
        if config.event_stream is not None:
            config.event_stream.emit(
                cell_failed_event(
                    unit=config.event_unit,
                    reason_class=type(exc).__name__,
                    detail=str(exc),
                )
            )
        raise

    after = close_invocation()
    spend = (
        max(0.0, before.remaining_usd - after.remaining_usd)
        if before is not None
        and after is not None
        and before.remaining_usd is not None
        and after.remaining_usd is not None
        else None
    )

    baseline_score = _reportable_score(baseline)
    if baseline_score is None:
        raise CellBaselineFailure("official baseline aggregate is incomplete")
    best_score = _reportable_score(best)
    ceiling_score = _reportable_score(ceiling)
    baseline_ci = _mean_ci(
        baseline.evidence.per_task_values, seed=_NAIVE_CI_SEED
    )
    ceiling_ci = (
        _mean_ci(ceiling.evidence.per_task_values, seed=_CEILING_CI_SEED)
        if ceiling is not None and ceiling_score is not None
        else None
    )
    delta_ci = (
        bootstrap_delta_ci(
            baseline.evidence.per_task_values,
            best.evidence.per_task_values,
            seed=_DELTA_CI_SEED,
        )
        if best is not None and best_score is not None
        else None
    )
    headroom_ci = (
        bootstrap_delta_ci(
            baseline.evidence.per_task_values,
            ceiling.evidence.per_task_values,
            seed=_HEADROOM_CI_SEED,
        )
        if ceiling is not None and ceiling_score is not None
        else None
    )
    delta = best_score - baseline_score if best_score is not None else None
    terminal_status = result.step_results[-1].record.status
    status = _status_for(
        terminal_status=terminal_status,
        best_score=best_score,
        ceiling_expected=config.ceiling is not None,
        ceiling_score=ceiling_score,
        delta=delta,
        delta_ci=delta_ci,
    )
    if status not in {
        "incomplete-arm",
        "proposer-failure",
    } and config.budget_guard.would_halt(spend):
        status = "halted"

    trace_path = config.ledger.write_optimization_trace(
        config.cell_id, result.model_dump(mode="json")
    )
    duration = time.monotonic() - started
    if ceiling is not None and ceiling_score is not None:
        config.ledger.write_official_anchor(
            _official_anchor_record(config, baseline=baseline, ceiling=ceiling)
        )
    projection, rollout_lines = build_viewer_cell_projection(
        cell_id=config.cell_id,
        optimizer=config.optimizer,
        env=config.env,
        attempt=config.attempt,
        result=result,
        result_ref=result_ref,
        store=config.store,
        official_evaluations=tuple(
            evaluation
            for evaluation in (baseline, ceiling, best)
            if evaluation is not None
        ),
    )
    # The viewer directory commits before the cells line, so a terminal line
    # can never cite evidence that was never durably published.
    viewer_publication = config.ledger.write_viewer_publication(
        cell_id=config.cell_id,
        env=config.env,
        projection_body=projection.to_bytes(),
        rollout_lines=rollout_lines,
    )
    record = CellRecord(
        cell_id=config.cell_id,
        optimizer=config.optimizer,
        env=config.env,
        attempt=config.attempt,
        canonical=config.canonical,
        models=CellModels(
            task=config.task_model, proposer=config.proposer_model
        ),
        baseline_official=baseline_score,
        ceiling_official=ceiling_score,
        best_official=best_score,
        delta=delta,
        delta_ci95=delta_ci.as_tuple() if delta_ci else None,
        naive_ci95=baseline_ci.as_tuple(),
        ceiling_ci95=ceiling_ci.as_tuple() if ceiling_ci else None,
        headroom_delta=(
            ceiling_score - baseline_score
            if ceiling_score is not None
            else None
        ),
        headroom_ci95=headroom_ci.as_tuple() if headroom_ci else None,
        official_repeats_used=(
            config.official_engine.sampling.repeat_plan.repeat_count
        ),
        pooled_observation_counts={
            "baseline": sum(baseline.evidence.per_task_counts),
            "ceiling": (
                sum(ceiling.evidence.per_task_counts)
                if ceiling is not None
                else 0
            ),
            "best": (
                sum(best.evidence.per_task_counts) if best is not None else 0
            ),
        },
        internal_evals_count=_internal_evaluation_count(result),
        optimizer_steps=len(result.step_results),
        # ``spend_usd`` is the ledger's accounting column and is always a
        # number; unknown spend contributes nothing to a sum. The finalized
        # event carries the honest ``None`` for unknown, so a reader can tell
        # "measured zero" from "never measured".
        spend_usd=spend if spend is not None else 0.0,
        wall_s=duration,
        lane=config.lane,
        status=status,
        artifacts=CellArtifacts(
            optimization_result_ref=result_ref,
            optimization_trace_ref=_relative(trace_path, config.ledger.root),
            best_candidate_id=(
                selected.candidate_id if selected is not None else None
            ),
            official_record_before=baseline.evidence_ref,
            official_record_after=(
                best.evidence_ref if best is not None else None
            ),
            viewer_publication=viewer_publication,
        ),
        graph_hash=baseline.evidence.graph_hash,
        eval_config_hash=(
            config.official_engine.eval_config_ref.identity_hash
        ),
        controls=CellControls(),
        started_at=started_at,
        finished_at=_now(),
    )
    config.ledger.append_cell(record)
    if config.event_stream is not None:
        config.event_stream.emit(
            cell_finalized_event(
                unit=config.event_unit,
                status=status,
                delta=delta,
                delta_ci95=record.delta_ci95,
                realized_spend_usd=spend,
                duration_s=duration,
            )
        )
    return CellOutcome(record=record, result=result)
