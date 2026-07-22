"""One validation cell: baseline -> optimize -> best-official -> ledger.

A **cell** is ``Cell(optimizer, env)`` per ``reports/validation-plan.md``:

1. **Baseline**: official eval of the naive Initial Candidate (+ the ceiling
   probe once per env, cached in the ledger via ``ceiling_official``).
2. **Optimize**: run the optimizer on the internal split with brief-documented
   hyperparameters scaled to the pool sizes; the optimizer sees ONLY
   internal-split evaluation (Reward).
3. **After**: official eval of the best accepted candidate on the SAME
   official-split Eval Config identity. ``delta = official(best) -
   official(naive)``; report delta + a paired bootstrap CI over tasks (cheap,
   no extra LLM calls).
4. **Persist**: append the ``cells.jsonl`` line in the EXACT schema, and append
   the OpenRouter ``spend.jsonl`` before/after snapshots when lane=openrouter.

Resumability: ``cells.jsonl`` is the ledger; a completed ``(optimizer, env,
attempt)`` cell is skipped. Budget guards refuse to start a canonical cell
below the reserve and halt a cell above the stop-loss (``status=halted``).

Every measurement runs through injected transports; nothing here makes a live
paid call by itself.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

from dr_store import MemoryBackend, ObjectStore

from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.optimization.proposer import ProposerConfig, ProposerTransport
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.budget import BudgetGuard, CreditsSnapshot
from whetstone.runner.eval_run import (
    evaluate_split,
    official_instances,
)
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import (
    CellArtifacts,
    CellModels,
    CellRecord,
    EnvOfficialCache,
    Ledger,
    SpendRecord,
)
from whetstone.runner.optimizers import run_optimize
from whetstone.runner.statistics import (
    BootstrapCI,
    bootstrap_mean_ci,
    bootstrap_paired_delta_ci,
)

__all__ = [
    "CellBaselineFailure",
    "CellConfig",
    "CellOutcome",
    "run_cell",
]


class CellBaselineFailure(RuntimeError):
    """The baseline official eval produced zero successful rollouts.

    A cell whose baseline naive candidate never once reaches the provider (or
    is rejected on every rollout) is a hard plumbing failure, not a valid
    ``baseline_official=null`` cell. Raising here stops the runner from
    appending a null-scores cell line that would silently mask the blocker;
    the CLI surfaces it as a loud non-zero exit.
    """

#: A callable returning the OpenRouter credits snapshot (injected; no network
#: in tests). ``None`` means credits are unavailable (a non-openrouter lane).
CreditsFetcher = Callable[[], CreditsSnapshot | None]


@dataclass(frozen=True, slots=True)
class CellConfig:
    """Everything one cell needs; transports are injected (no live call)."""

    optimizer: str
    env: str
    lane: str
    attempt: int
    task_model: str
    proposer_model: str
    canonical: bool
    proposer_config: ProposerConfig
    proposer_transport: ProposerTransport
    rollout_transport: TransportCall
    execution_policy: ProviderExecutionPolicy
    repeats: int = 3
    #: Official-split repeats (baseline/ceiling/best). The statistical-
    #: confidence directive raises the default 3 -> 5 for statistical power;
    #: the config surface stays overridable (pilots/dry-runs pass their own).
    official_repeats: int = 5
    pool_n_per_stratum: int | None = None
    split_sizes: tuple[int, int, int] | None = None
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS
    window_notes: str = ""


@dataclass(slots=True)
class CellOutcome:
    """The cell's terminal result (also the appended ledger record)."""

    record: CellRecord
    skipped: bool = False
    resumed: bool = False
    restarted: bool = False
    reason: str = ""


def _pool_per_task(
    scores_a: tuple[float, ...],
    counts_a: tuple[int, ...],
    scores_b: tuple[float, ...],
    counts_b: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Pool two per-task passes into a count-weighted per-task mean vector.

    Each pass reports per-task means over its own repeats plus the repeat
    counts behind them; pooling is the count-weighted mean so NO observation is
    discarded (escalation adds repeats to the existing pool). Aligned by task
    index; a task with zero total observations pools to 0.0.
    """
    lengths = {len(scores_a), len(counts_a), len(scores_b), len(counts_b)}
    if len(lengths) != 1:
        raise ValueError(
            "pooling requires aligned per-task score/count vectors"
        )
    pooled_scores: list[float] = []
    pooled_counts: list[int] = []
    for sa, ca, sb, cb in zip(
        scores_a, counts_a, scores_b, counts_b, strict=True
    ):
        total = ca + cb
        if total == 0:
            pooled_scores.append(0.0)
        else:
            pooled_scores.append((sa * ca + sb * cb) / total)
        pooled_counts.append(total)
    return tuple(pooled_scores), tuple(pooled_counts)


def _cell_seed(cell_id: str) -> int:
    """A deterministic per-cell bootstrap seed (reproducible intervals).

    The seed is fixed per cell so re-running the same cell's bootstrap yields
    identical intervals, and every interval within a cell (naive, ceiling,
    paired delta, paired headroom) shares the same seed -- so paired variants
    reuse the SAME resample indices across their two arms.
    """
    return int(hashlib.sha256(cell_id.encode()).hexdigest()[:8], 16)


def _paired_or_none(
    a_per_task: tuple[float, ...],
    b_per_task: tuple[float, ...],
    seed: int,
) -> BootstrapCI | None:
    """A paired ``b - a`` CI, or None when either arm has no tasks."""
    if not a_per_task or not b_per_task:
        return None
    if len(a_per_task) != len(b_per_task):
        return None
    return bootstrap_paired_delta_ci(a_per_task, b_per_task, seed=seed)


def _official_intervals(
    naive_per_task: tuple[float, ...],
    best_per_task: tuple[float, ...],
    baseline_score: float | None,
    best_score: float | None,
    seed: int,
) -> tuple[BootstrapCI | None, BootstrapCI | None, float | None]:
    """(naive marginal CI, paired delta CI, point delta) for an official pass.

    All three share the per-cell ``seed``; the delta CI is the paired
    best-naive bootstrap over the SAME resampled task indices as the naive
    marginal CI. Returns ``None`` intervals + delta when a score is missing.
    """
    if baseline_score is None or best_score is None:
        return None, None, None
    naive_ci = bootstrap_mean_ci(naive_per_task, seed=seed)
    delta_ci = bootstrap_paired_delta_ci(
        naive_per_task, best_per_task, seed=seed
    )
    return naive_ci, delta_ci, best_score - baseline_score


def _status_from(
    delta: float | None, delta_ci: BootstrapCI | None
) -> str:
    """The sharpened cell status from the paired delta + its CI.

    Per the "Cell statuses sharpen" directive: ``improved`` REQUIRES
    ``delta > 0`` AND the paired delta CI excluding 0; ``delta > 0`` with a CI
    spanning 0 is ``inconclusive``; ``delta <= 0`` is ``no-improvement``.
    """
    if delta is None or delta <= 0:
        return "no-improvement"
    if delta_ci is not None and delta_ci.excludes_zero():
        return "improved"
    return "inconclusive"


def _escalation_allowed(
    budget: BudgetGuard, remaining_usd: float | None
) -> tuple[bool, str]:
    """Whether an inconclusive cell may auto-escalate under the budget guard.

    Escalation runs additional paid official repeats, so it is gated behind the
    reserve check: below the reserve, escalation is skipped with a note.
    """
    if remaining_usd is not None and remaining_usd < budget.reserve_usd:
        return False, (
            f"remaining ${remaining_usd:.2f} < reserve "
            f"${budget.reserve_usd:.2f}"
        )
    return True, ""


def _spend_between(
    before: CreditsSnapshot | None, after: CreditsSnapshot | None
) -> float:
    if before is None or after is None:
        return 0.0
    b = before.remaining_usd
    a = after.remaining_usd
    if b is None or a is None:
        return 0.0
    return max(0.0, b - a)


def run_cell(
    config: CellConfig,
    *,
    ledger: Ledger,
    budget: BudgetGuard | None = None,
    credits_fetcher: CreditsFetcher | None = None,
    store: ObjectStore | None = None,
) -> CellOutcome:
    """Run one full validation cell, appending its ledger + spend lines.

    Honors resumability (skip a completed cell), the budget guards (reserve at
    start, stop-loss mid-cell), and the ceiling-once-per-env cache.
    """
    budget = budget or BudgetGuard()
    backing = store or ObjectStore(MemoryBackend())
    cell_id = f"{config.optimizer}:{config.env}:a{config.attempt}"
    is_openrouter = config.lane == "openrouter"

    # --- Resumability: skip a completed (optimizer, env, attempt) cell. ---
    if ledger.is_completed(config.optimizer, config.env, config.attempt):
        prior = ledger.latest_for(config.optimizer, config.env)
        assert prior is not None
        return CellOutcome(
            record=prior, skipped=True, reason="already completed"
        )

    resumed = ledger.latest_for(config.optimizer, config.env) is not None
    # This reduction restarts the cell (optimization state not resumable).
    restarted = resumed

    # --- Spend snapshot BEFORE (credits API when lane=openrouter). ---
    spend_before: CreditsSnapshot | None = None
    if is_openrouter and credits_fetcher is not None:
        spend_before = credits_fetcher()
        ledger.append_spend(
            SpendRecord(
                cell_id=cell_id,
                phase="before",
                lane=config.lane,
                total_credits=(
                    spend_before.total_credits if spend_before else None
                ),
                total_usage=(
                    spend_before.total_usage if spend_before else None
                ),
                remaining_usd=(
                    spend_before.remaining_usd if spend_before else None
                ),
                at=spend_before.at if spend_before else "",
            )
        )

    # --- Budget reserve guard (refuse a fresh canonical cell < reserve). ---
    # Raises ReserveError to the caller when remaining < reserve.
    remaining = spend_before.remaining_usd if spend_before else None
    budget.check_start(
        canonical=config.canonical,
        remaining_usd=remaining,
        is_rerun=config.attempt > 0,
    )

    start = time.monotonic()
    experiment: EnvExperiment = build_env_experiment(
        config.env,
        model=config.task_model,
        pool_n_per_stratum=config.pool_n_per_stratum,
        split_sizes=config.split_sizes,
    )
    naive = experiment.initial_candidate
    official = official_instances(experiment)

    official_repeats = config.official_repeats
    is_eval_row = config.optimizer == "eval"

    # --- 1. Baseline naive + ceiling official arms (per-task vectors too). ---
    # Point 6 of the statistical-confidence directive: the Eval row establishes
    # the per-env official naive/ceiling scores AND their per-task score
    # vectors in the ledger cache; every other optimizer cell REUSES that cache
    # rather than re-driving the naive/ceiling arms -- so a later paired
    # best-naive delta can be computed without re-driving naive. A non-eval
    # cell only drives them itself if no Eval row has cached the env yet.
    cache = None if is_eval_row else ledger.env_cache_for(config.env)
    ceiling_official: float | None
    if cache is not None:
        baseline_score = cache.naive_official
        naive_per_task = cache.naive_per_task
        naive_counts = tuple(cache.official_repeats_used for _ in official)
        ceiling_official = cache.ceiling_official
        ceiling_per_task = cache.ceiling_per_task
        official_repeats = cache.official_repeats_used
        baseline_before_ref = ""
    else:
        # A baseline with no successful rollout rows (every call failed
        # pre-flight or was rejected) is a plumbing failure, not a valid
        # null-scores cell. Raise CellBaselineFailure BEFORE recording any
        # ledger line so the CLI exits non-zero loudly. The empty aggregate
        # also makes the internal reduction non-computable (a bare ValueError
        # deep in evaluate_split), caught here and re-surfaced with context.
        planned = len(official) * official_repeats
        try:
            baseline = evaluate_split(
                experiment,
                candidate=naive,
                instances=official,
                split_role="official",
                transport=config.rollout_transport,
                execution_policy=config.execution_policy,
                repeats=official_repeats,
                store=backing,
                execution_mode=config.execution_mode,
            )
        except ValueError as exc:
            raise CellBaselineFailure(
                f"cell {cell_id}: baseline official eval produced no "
                f"computable score over {planned} planned rollouts (every "
                "rollout failed). This is a plumbing failure; no cell "
                "line recorded."
            ) from exc
        if baseline.aggregate.rows_present == 0:
            raise CellBaselineFailure(
                f"cell {cell_id}: baseline official eval produced "
                f"0/{planned} successful rollouts (every rollout failed). "
                "This is a plumbing failure; no cell line recorded."
            )
        baseline_score = baseline.score
        naive_per_task = baseline.per_task_scores
        naive_counts = baseline.per_task_counts
        baseline_before_ref = baseline.artifact_ref.content_hash
        # Ceiling arm: reuse the scalar cache when present; else drive it.
        ceiling_cached = ledger.ceiling_for(config.env)
        ceiling_eval = evaluate_split(
            experiment,
            candidate=experiment.ceiling_candidate,
            instances=official,
            split_role="official",
            transport=config.rollout_transport,
            execution_policy=config.execution_policy,
            repeats=official_repeats,
            store=backing,
            execution_mode=config.execution_mode,
        )
        ceiling_official = (
            ceiling_cached
            if ceiling_cached is not None
            else ceiling_eval.score
        )
        ceiling_per_task = ceiling_eval.per_task_scores

    # --- 2. Optimize on the internal split (Reward only). ---
    opt = run_optimize(
        experiment,
        optimizer=config.optimizer,
        proposer_config=config.proposer_config,
        proposer_transport=config.proposer_transport,
        rollout_transport=config.rollout_transport,
        execution_policy=config.execution_policy,
        internal_instances=experiment.eval_configs.internal.instances,
        repeats=config.repeats,
        store=backing,
        execution_mode=config.execution_mode,
    )

    # --- 3. Best-candidate official eval on the SAME official Eval Config. ---
    best = evaluate_split(
        experiment,
        candidate=opt.best_candidate,
        instances=official,
        split_role="official",
        transport=config.rollout_transport,
        execution_policy=config.execution_policy,
        repeats=official_repeats,
        store=backing,
        execution_mode=config.execution_mode,
    )
    best_score = best.score
    best_per_task = best.per_task_scores
    best_counts = best.per_task_counts

    # --- Bootstrap intervals over TASKS (seed fixed per cell). ---
    # All intervals resample the exchangeable unit (the task) via a per-cell
    # fixed seed; paired variants (delta, headroom) reuse the SAME resample
    # indices across both arms. Every interval is a pure function of already-
    # retained per-task scores -- no re-drive, no extra provider call.
    seed = _cell_seed(cell_id)
    naive_ci, delta_ci, delta = _official_intervals(
        naive_per_task, best_per_task, baseline_score, best_score, seed
    )
    ceiling_ci = (
        bootstrap_mean_ci(ceiling_per_task, seed=seed)
        if ceiling_per_task
        else None
    )
    # Headroom (paired ceiling - naive) is recorded on EVERY cell (computed
    # from the cached/driven ceiling vs naive per-task vectors); the Eval row
    # ADDITIONALLY sets the no-demonstrable-headroom gate flag it establishes
    # once per env (other cells leave the flag None, interpreting against it).
    headroom_ci = _paired_or_none(naive_per_task, ceiling_per_task, seed)
    headroom_delta = (
        headroom_ci.point if headroom_ci is not None else None
    )
    no_headroom: bool | None = None
    if is_eval_row and headroom_ci is not None:
        no_headroom = not headroom_ci.excludes_zero()

    pooled_counts = {
        "naive": sum(naive_counts),
        "best": sum(best_counts),
    }

    escalated = False
    escalation_note = ""
    status = _status_from(delta, delta_ci)

    # --- 5. Escalation: inconclusive cell auto-doubles official repeats. ---
    # Run ADDITIONAL repeats on BOTH the naive and best arms, POOL the new
    # per-task observations with the existing ones (never discard), and
    # recompute the delta + CIs ONCE. Guarded by the budget reserve: skip with
    # a recorded note when remaining credits are below the reserve.
    if status == "inconclusive":
        can_escalate, reason = _escalation_allowed(budget, remaining)
        if not can_escalate:
            escalation_note = f"escalation skipped: {reason}"
        else:
            extra_naive = evaluate_split(
                experiment, candidate=naive, instances=official,
                split_role="official", transport=config.rollout_transport,
                execution_policy=config.execution_policy,
                repeats=official_repeats, store=backing,
                execution_mode=config.execution_mode,
            )
            extra_best = evaluate_split(
                experiment, candidate=opt.best_candidate, instances=official,
                split_role="official", transport=config.rollout_transport,
                execution_policy=config.execution_policy,
                repeats=official_repeats, store=backing,
                execution_mode=config.execution_mode,
            )
            naive_per_task, naive_counts = _pool_per_task(
                naive_per_task, naive_counts,
                extra_naive.per_task_scores, extra_naive.per_task_counts,
            )
            best_per_task, best_counts = _pool_per_task(
                best_per_task, best_counts,
                extra_best.per_task_scores, extra_best.per_task_counts,
            )
            baseline_score = sum(naive_per_task) / len(naive_per_task)
            best_score = sum(best_per_task) / len(best_per_task)
            naive_ci, delta_ci, delta = _official_intervals(
                naive_per_task, best_per_task,
                baseline_score, best_score, seed,
            )
            pooled_counts = {
                "naive": sum(naive_counts),
                "best": sum(best_counts),
            }
            escalated = True
            official_repeats *= 2
            status = _status_from(delta, delta_ci)
            escalation_note = "escalated: doubled official repeats and pooled"

    wall_s = time.monotonic() - start

    # --- Spend snapshot AFTER + per-cell spend. ---
    spend_after: CreditsSnapshot | None = None
    if is_openrouter and credits_fetcher is not None:
        spend_after = credits_fetcher()
        ledger.append_spend(
            SpendRecord(
                cell_id=cell_id,
                phase="after",
                lane=config.lane,
                total_credits=(
                    spend_after.total_credits if spend_after else None
                ),
                total_usage=(
                    spend_after.total_usage if spend_after else None
                ),
                remaining_usd=(
                    spend_after.remaining_usd if spend_after else None
                ),
                at=spend_after.at if spend_after else "",
            )
        )
    spend_usd = _spend_between(spend_before, spend_after)

    # Stop-loss overrides the statistical status.
    if budget.would_halt(spend_usd):
        status = "halted"

    record = CellRecord(
        cell_id=cell_id,
        optimizer=config.optimizer,
        env=config.env,
        attempt=config.attempt,
        canonical=config.canonical,
        models=CellModels(
            task=config.task_model, proposer=config.proposer_model
        ),
        baseline_official=baseline_score,
        ceiling_official=ceiling_official,
        best_official=best_score,
        delta=delta,
        ci95=delta_ci.as_tuple() if delta_ci is not None else None,
        naive_ci95=naive_ci.as_tuple() if naive_ci is not None else None,
        ceiling_ci95=ceiling_ci.as_tuple() if ceiling_ci is not None else None,
        delta_ci95=delta_ci.as_tuple() if delta_ci is not None else None,
        headroom_delta=headroom_delta,
        headroom_ci95=(
            headroom_ci.as_tuple() if headroom_ci is not None else None
        ),
        no_demonstrable_headroom=no_headroom,
        official_repeats_used=official_repeats,
        escalated=escalated,
        escalation_note=escalation_note,
        pooled_observation_counts=pooled_counts,
        internal_evals_count=opt.internal_evals_count,
        optimizer_steps=opt.optimizer_steps,
        spend_usd=spend_usd,
        wall_s=wall_s,
        lane=config.lane,
        window_notes=config.window_notes,
        status=status,
        artifacts=CellArtifacts(
            optimization_result_ref=opt.best_candidate.candidate_id,
            official_record_before=baseline_before_ref,
            official_record_after=best.artifact_ref.content_hash,
        ),
    )
    ledger.append_cell(record)
    # The Eval row establishes the per-env official cache (naive + ceiling
    # scalars AND per-task vectors) so later optimizer cells reuse it.
    if is_eval_row and cache is None:
        ledger.append_env_cache(
            EnvOfficialCache(
                env=config.env,
                naive_official=baseline_score,
                ceiling_official=ceiling_official,
                naive_per_task=naive_per_task,
                ceiling_per_task=ceiling_per_task,
                official_repeats_used=official_repeats,
            )
        )
    return CellOutcome(
        record=record,
        resumed=resumed,
        restarted=restarted,
        reason="restarted (optimization state not resumable)"
        if restarted
        else "",
    )
