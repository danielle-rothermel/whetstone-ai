"""One live or dry validation cell over canonical durable services."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from dr_serialize import Jsonable
from dr_store import BindingConflictError
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.code_eval.statistics import (
    BootstrapCI,
    bootstrap_delta_ci,
    bootstrap_mean_ci,
)
from whetstone.evaluation import (
    EngineEvaluation,
    EngineEvaluationService,
    EvaluationEngine,
    EvaluationEvidence,
)
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    Candidate,
    CandidateRef,
    EvalConfigRef,
    EvaluationIntent,
    IntentOutcome,
    StepStatus,
    TypedRef,
    candidate_reference,
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
    CellArtifacts,
    CellControls,
    CellModels,
    CellRecord,
    Ledger,
    PromptCacheControls,
    SpendRecord,
)
from whetstone.runner.optimization_run import (
    OptimizationExecution,
    OptimizationRunControl,
    OptimizationRunServices,
    bind_optimization_control,
    run_optimization,
)

OFFICIAL_ARM_BINDING_SCHEMA = "whetstone.runner.official_arm_binding"
CELL_RUN_CONTROL_SCHEMA = "whetstone.runner.cell_run_control"


class CellBaselineFailure(RuntimeError):
    """The official baseline could not produce a reportable aggregate."""


class OfficialArmBinding(BaseModel):
    """Exact official-arm request bound before its paid evaluation."""

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
    """Exact cell-level inputs bound before paid work or terminal reuse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: StrictStr
    canonical: bool
    models: CellModels
    lane: StrictStr
    baseline: CandidateRef
    ceiling: CandidateRef | None
    optimization_control_ref: TypedRef
    official_eval_config: EvalConfigRef

    def record_content(self) -> Jsonable:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class CellConfig:
    """Internal cell value; persisted records are emitted through Ledger."""

    env: str
    attempt: int
    canonical: bool
    task_model: str
    proposer_model: str
    lane: str
    baseline: Candidate
    optimization: OptimizationRunControl
    official_engine: EvaluationEngine
    optimization_services: OptimizationRunServices
    ledger: Ledger
    ceiling: Candidate | None = None
    budget_guard: BudgetGuard = field(default_factory=BudgetGuard)
    credits_fetcher: Callable[[], CreditsSnapshot | None] | None = None
    event_stream: EventStream | None = None

    def __post_init__(self) -> None:
        expected_run_id = (
            f"{self.optimization.optimizer.value}:{self.env}:a{self.attempt}"
        )
        if self.optimization.run_id != expected_run_id:
            raise ValueError(
                "optimization run_id must equal the exact cell identity "
                f"{expected_run_id!r}"
            )

    @property
    def cell_id(self) -> str:
        return (
            f"{self.optimization.optimizer.value}:{self.env}:a{self.attempt}"
        )

    @property
    def event_unit(self) -> EventUnit:
        return EventUnit.for_cell(
            cell_id=self.cell_id,
            env=self.env,
            optimizer=self.optimization.optimizer.value,
            attempt=self.attempt,
            lane=self.lane,
            model=self.task_model,
        )


@dataclass(frozen=True, slots=True)
class CellOutcome:
    record: CellRecord
    optimization: OptimizationExecution | None
    skipped: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _spend_record(
    config: CellConfig,
    *,
    phase: str,
    snapshot: CreditsSnapshot,
) -> SpendRecord:
    return SpendRecord(
        event_id=str(uuid.uuid4()),
        cell_id=config.cell_id,
        phase=phase,
        lane=config.lane,
        total_credits=snapshot.total_credits,
        total_usage=snapshot.total_usage,
        remaining_usd=snapshot.remaining_usd,
        at=snapshot.at or _now(),
    )


def _bind_official_arm(
    config: CellConfig,
    *,
    arm: str,
    candidate: Candidate,
    purpose: str,
) -> OfficialArmBinding:
    context_id = f"{config.cell_id}:official:{arm}"
    binding = OfficialArmBinding(
        cell_id=config.cell_id,
        arm=arm,
        candidate=candidate_reference(candidate),
        eval_config=config.official_engine.eval_config_ref,
        context_id=context_id,
        purpose=purpose,
    )
    store = config.optimization_services.store
    reference, _ = store.put(
        OFFICIAL_ARM_BINDING_SCHEMA, binding.record_content()
    )
    key = f"{OFFICIAL_ARM_BINDING_SCHEMA}:{config.cell_id}#{arm}"
    try:
        store.bind(key, reference)
    except BindingConflictError as conflict:
        raise ValueError(
            f"official arm {arm!r} is already bound to "
            f"{conflict.existing.content_hash}; refusing "
            f"{reference.content_hash}"
        ) from conflict
    return binding


def _evaluate_official(
    config: CellConfig,
    *,
    arm: str,
    candidate: Candidate,
    purpose: str,
) -> EngineEvaluation:
    binding = _bind_official_arm(
        config,
        arm=arm,
        candidate=candidate,
        purpose=purpose,
    )
    store = config.optimization_services.store
    resolution = EngineEvaluationService(
        store=store,
        engine=config.official_engine,
    ).resolve_evaluation_intent(
        EvaluationIntent(
            intent_id=binding.context_id,
            candidate=binding.candidate,
            target_eval_config=binding.eval_config,
            context_role=EvaluationRole.OFFICIAL,
            purpose=purpose,
            run_id=config.cell_id,
            step_index=0,
        )
    )
    if (
        resolution.outcome is not IntentOutcome.COMPLETED
        or len(resolution.evaluation_evidence_refs) != 1
    ):
        raise ValueError(
            f"official arm {arm!r} did not produce canonical evidence: "
            f"{resolution.detail.message}"
        )
    evidence_ref = resolution.evaluation_evidence_refs[0]
    evidence = EvaluationEvidence.model_validate(
        store.get(evidence_ref.reference)
    )
    evaluated = EngineEvaluation(evidence=evidence, evidence_ref=evidence_ref)
    if (
        evidence.candidate != binding.candidate
        or evidence.eval_config != binding.eval_config
        or evidence.evaluation_role is not EvaluationRole.OFFICIAL
        or evidence.evaluation_context_id != binding.context_id
        or evidence.purpose != purpose
    ):
        raise ValueError("official arm evidence does not match its binding")
    if evaluated.evidence.reward_ref is not None:
        raise ValueError("official evaluation produced a Reward")
    return evaluated


def _bind_cell_run_control(config: CellConfig) -> None:
    store = config.optimization_services.store
    optimization_control_ref = bind_optimization_control(
        config.optimization,
        config.optimization_services,
    )
    control = CellRunControl(
        cell_id=config.cell_id,
        canonical=config.canonical,
        models=CellModels(
            task=config.task_model,
            proposer=config.proposer_model,
        ),
        lane=config.lane,
        baseline=candidate_reference(config.baseline),
        ceiling=(
            candidate_reference(config.ceiling)
            if config.ceiling is not None
            else None
        ),
        optimization_control_ref=optimization_control_ref,
        official_eval_config=config.official_engine.eval_config_ref,
    )
    reference, _ = store.put(
        CELL_RUN_CONTROL_SCHEMA,
        control.record_content(),
    )
    try:
        store.bind(
            f"{CELL_RUN_CONTROL_SCHEMA}:{config.cell_id}",
            reference,
        )
    except BindingConflictError as conflict:
        raise ValueError(
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


def _score(evaluation: EngineEvaluation) -> float | None:
    return evaluation.evidence.aggregate_value


def _ci(values: tuple[float, ...], *, seed: int) -> BootstrapCI:
    return bootstrap_mean_ci(values, seed=seed)


def _relative(path, root) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _internal_evaluation_count(execution: OptimizationExecution) -> int:
    return sum(
        len(step.dispositions) + step.tool_evidence_count
        for step in execution.trace.steps
    )


def _completed_record(
    config: CellConfig,
) -> CellRecord | None:
    if not config.ledger.is_completed(
        config.optimization.optimizer.value,
        config.env,
        config.attempt,
    ):
        return None
    return config.ledger.for_attempt(
        config.optimization.optimizer.value,
        config.env,
        config.attempt,
    )


def run_cell(config: CellConfig) -> CellOutcome:
    """Run one cell and project canonical evidence into the ledger."""
    # Every known candidate is rendered before durable binding or provider use.
    config.official_engine.preflight(config.baseline)
    if config.ceiling is not None:
        config.official_engine.preflight(config.ceiling)
    _bind_cell_run_control(config)
    if completed := _completed_record(config):
        if config.event_stream is not None:
            config.event_stream.emit(
                attempt_skipped_event(
                    unit=config.event_unit,
                    prior_status=completed.status,
                )
            )
        return CellOutcome(record=completed, optimization=None, skipped=True)

    started_at = _now()
    started = time.monotonic()
    before = config.credits_fetcher() if config.credits_fetcher else None
    if before is not None:
        config.ledger.append_spend(
            _spend_record(config, phase="before", snapshot=before)
        )
    is_rerun = (
        config.ledger.latest_for(
            config.optimization.optimizer.value,
            config.env,
        )
        is not None
    )
    config.budget_guard.check_start(
        canonical=config.canonical,
        remaining_usd=before.remaining_usd if before else None,
        is_rerun=is_rerun,
    )

    initial_remaining = before.remaining_usd if before is not None else None

    def before_paid_boundary(phase: str) -> None:
        if config.credits_fetcher is None:
            return
        snapshot = config.credits_fetcher()
        if snapshot is None:
            return
        config.ledger.append_spend(
            _spend_record(
                config,
                phase=f"checkpoint:{phase}",
                snapshot=snapshot,
            )
        )
        if initial_remaining is None or snapshot.remaining_usd is None:
            return
        config.budget_guard.check_stop_loss(
            max(0.0, initial_remaining - snapshot.remaining_usd)
        )

    def evaluate_arm(
        arm: str, candidate: Candidate, purpose: str
    ) -> EngineEvaluation:
        before_paid_boundary(f"official:{arm}")
        return _evaluate_official(
            config,
            arm=arm,
            candidate=candidate,
            purpose=purpose,
        )

    try:
        baseline = evaluate_arm(
            "baseline", config.baseline, "official_baseline"
        )
        if _score(baseline) is None:
            raise CellBaselineFailure(
                "official baseline aggregate is incomplete"
            )
        ceiling = (
            evaluate_arm("ceiling", config.ceiling, "official_ceiling")
            if config.ceiling is not None
            else None
        )

        def before_optimization_step(phase: str) -> None:
            before_paid_boundary(phase)
            if config.optimization_services.before_paid_step is not None:
                config.optimization_services.before_paid_step(phase)

        services = replace(
            config.optimization_services,
            before_paid_step=before_optimization_step,
        )
        execution = run_optimization(
            config.optimization,
            services,
        )
        selected = (
            execution.result.proposals[0].candidate.record
            if execution.result.status is StepStatus.COMPLETE
            and execution.result.proposals
            else None
        )
        best = (
            evaluate_arm("best", selected, "official_best")
            if selected is not None
            else None
        )
    except Exception as exc:
        if config.event_stream is not None:
            config.event_stream.emit(
                cell_failed_event(
                    unit=config.event_unit,
                    reason_class=type(exc).__name__,
                    detail=str(exc),
                )
            )
        raise

    after = config.credits_fetcher() if config.credits_fetcher else None
    if after is not None:
        config.ledger.append_spend(
            _spend_record(config, phase="after", snapshot=after)
        )
    spend = (
        max(0.0, before.remaining_usd - after.remaining_usd)
        if before is not None
        and after is not None
        and before.remaining_usd is not None
        and after.remaining_usd is not None
        else 0.0
    )

    baseline_score = _score(baseline)
    assert baseline_score is not None
    best_score = _score(best) if best is not None else None
    ceiling_score = _score(ceiling) if ceiling is not None else None
    baseline_ci = _ci(baseline.evidence.per_task_values, seed=17)
    ceiling_ci = (
        _ci(ceiling.evidence.per_task_values, seed=19)
        if ceiling is not None
        else None
    )
    delta_ci = (
        bootstrap_delta_ci(
            baseline.evidence.per_task_values,
            best.evidence.per_task_values,
            seed=23,
        )
        if best is not None
        else None
    )
    headroom_ci = (
        bootstrap_delta_ci(
            baseline.evidence.per_task_values,
            ceiling.evidence.per_task_values,
            seed=29,
        )
        if ceiling is not None
        else None
    )
    delta = best_score - baseline_score if best_score is not None else None
    if execution.result.status is StepStatus.FAILED:
        status = "proposer-failure"
    elif best_score is None:
        status = "incomplete-arm"
    elif delta is not None and delta > 0 and delta_ci is not None:
        status = "improved" if delta_ci.low > 0 else "inconclusive"
    else:
        status = "no-improvement"
    if config.budget_guard.would_halt(spend):
        status = "halted"

    trace_path = config.ledger.write_optimization_trace(
        config.cell_id,
        execution.trace.model_dump(mode="json"),
    )
    duration = time.monotonic() - started
    record = CellRecord(
        cell_id=config.cell_id,
        optimizer=config.optimization.optimizer.value,
        env=config.env,
        attempt=config.attempt,
        canonical=config.canonical,
        models=CellModels(
            task=config.task_model,
            proposer=config.proposer_model,
        ),
        baseline_official=baseline_score,
        ceiling_official=ceiling_score,
        best_official=best_score,
        delta=delta,
        ci95=delta_ci.as_tuple() if delta_ci else None,
        naive_ci95=baseline_ci.as_tuple(),
        ceiling_ci95=ceiling_ci.as_tuple() if ceiling_ci else None,
        delta_ci95=delta_ci.as_tuple() if delta_ci else None,
        headroom_delta=(
            ceiling_score - baseline_score
            if ceiling_score is not None
            else None
        ),
        headroom_ci95=headroom_ci.as_tuple() if headroom_ci else None,
        no_demonstrable_headroom=(
            headroom_ci.high <= 0 if headroom_ci else None
        ),
        official_repeats_used=baseline.evidence.eval_config.record.model_dump(
            mode="json"
        )
        and config.official_engine.sampling.repeat_plan.repeat_count,
        pooled_observation_counts={
            "baseline": sum(baseline.evidence.per_task_counts),
            "best": (
                sum(best.evidence.per_task_counts) if best is not None else 0
            ),
        },
        internal_evals_count=_internal_evaluation_count(execution),
        optimizer_steps=len(execution.result.step_result_refs),
        spend_usd=spend,
        wall_s=duration,
        lane=config.lane,
        status=status,
        artifacts=CellArtifacts(
            optimization_result_ref=execution.result_ref,
            optimization_trace_ref=_relative(trace_path, config.ledger.root),
            best_candidate_id=(
                selected.candidate_id if selected is not None else None
            ),
            official_record_before=baseline.evidence_ref,
            official_record_after=(
                best.evidence_ref if best is not None else None
            ),
        ),
        graph_hash=baseline.evidence.graph_hash,
        eval_config_hash=baseline.evidence.eval_config.identity_hash,
        controls=CellControls(
            prompt_cache=(
                PromptCacheControls(
                    **config.official_engine.prompt_cache.counters()
                )
                if config.official_engine.prompt_cache is not None
                else None
            )
        ),
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
    return CellOutcome(record=record, optimization=execution)


__all__ = [
    "CELL_RUN_CONTROL_SCHEMA",
    "OFFICIAL_ARM_BINDING_SCHEMA",
    "CellBaselineFailure",
    "CellConfig",
    "CellOutcome",
    "CellRunControl",
    "OfficialArmBinding",
    "run_cell",
]
