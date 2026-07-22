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
    per_task_official_scores,
)
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import (
    CellArtifacts,
    CellModels,
    CellRecord,
    Ledger,
    SpendRecord,
)
from whetstone.runner.optimizers import run_optimize
from whetstone.runner.statistics import bootstrap_delta_ci

__all__ = [
    "CellConfig",
    "CellOutcome",
    "run_cell",
]

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

    # --- 1. Baseline official eval of the naive candidate. ---
    baseline = evaluate_split(
        experiment,
        candidate=naive,
        instances=official,
        split_role="official",
        transport=config.rollout_transport,
        execution_policy=config.execution_policy,
        repeats=config.repeats,
        store=backing,
        execution_mode=config.execution_mode,
    )

    # --- Ceiling official eval (once per env; cached in the ledger). ---
    ceiling_cached = ledger.ceiling_for(config.env)
    if ceiling_cached is not None:
        ceiling_official = ceiling_cached
    else:
        ceiling_eval = evaluate_split(
            experiment,
            candidate=experiment.ceiling_candidate,
            instances=official,
            split_role="official",
            transport=config.rollout_transport,
            execution_policy=config.execution_policy,
            repeats=config.repeats,
            store=backing,
            execution_mode=config.execution_mode,
        )
        ceiling_official = ceiling_eval.score

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
        repeats=config.repeats,
        store=backing,
        execution_mode=config.execution_mode,
    )

    # --- Delta + paired bootstrap CI over official tasks. ---
    delta: float | None = None
    ci95: tuple[float, float] | None = None
    if baseline.score is not None and best.score is not None:
        naive_per_task = per_task_official_scores(
            experiment,
            candidate=naive,
            instances=official,
            transport=config.rollout_transport,
            execution_policy=config.execution_policy,
            repeats=config.repeats,
        )
        best_per_task = per_task_official_scores(
            experiment,
            candidate=opt.best_candidate,
            instances=official,
            transport=config.rollout_transport,
            execution_policy=config.execution_policy,
            repeats=config.repeats,
        )
        ci = bootstrap_delta_ci(naive_per_task, best_per_task, seed=0)
        delta = best.score - baseline.score
        ci95 = ci.as_tuple()

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

    # --- Status: halted (stop-loss) | improved | no-improvement. ---
    if budget.would_halt(spend_usd):
        status = "halted"
    elif delta is not None and delta > 0:
        status = "improved"
    else:
        status = "no-improvement"

    record = CellRecord(
        cell_id=cell_id,
        optimizer=config.optimizer,
        env=config.env,
        attempt=config.attempt,
        canonical=config.canonical,
        models=CellModels(
            task=config.task_model, proposer=config.proposer_model
        ),
        baseline_official=baseline.score,
        ceiling_official=ceiling_official,
        best_official=best.score,
        delta=delta,
        ci95=ci95,
        internal_evals_count=opt.internal_evals_count,
        optimizer_steps=opt.optimizer_steps,
        spend_usd=spend_usd,
        wall_s=wall_s,
        lane=config.lane,
        window_notes=config.window_notes,
        status=status,
        artifacts=CellArtifacts(
            optimization_result_ref=opt.best_candidate.candidate_id,
            official_record_before=baseline.artifact_ref.content_hash,
            official_record_after=best.artifact_ref.content_hash,
        ),
    )
    ledger.append_cell(record)
    return CellOutcome(
        record=record,
        resumed=resumed,
        restarted=restarted,
        reason="restarted (optimization state not resumable)"
        if restarted
        else "",
    )
