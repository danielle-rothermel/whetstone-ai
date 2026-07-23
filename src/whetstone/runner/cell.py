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
from dataclasses import dataclass, field

from dr_store import MemoryBackend, ObjectStore

from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.reward import CandidateEvaluationFailure
from whetstone.envs.sampling import Completeness, SamplingOverrides
from whetstone.execution.fanout import DEFAULT_CONCURRENCY, FanoutConfig
from whetstone.execution.partials import PartialLog
from whetstone.optimization.proposer import ProposerConfig, ProposerTransport
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.budget import BudgetGuard, CreditsSnapshot
from whetstone.runner.eval_run import (
    SplitEvaluation,
    evaluate_split,
    official_instances,
)
from whetstone.runner.execution_mode import ExecutionMode
from whetstone.runner.ledger import (
    CellArtifacts,
    CellModels,
    CellRecord,
    CellSamplingOverrides,
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
    "DEFAULT_CELL_MAX_WALL_SECONDS",
    "CellBaselineFailure",
    "CellConfig",
    "CellOutcome",
    "run_cell",
]

#: The default whole-cell wall deadline (seconds; ``--max-wall-seconds``). On
#: breach a cell finishes in-flight, persists partials, and records
#: ``status=halted`` with a halt reason. Raised 3600 -> 7200s: with the 600s
#: transport cap accommodating reasoning-model streams, a canonical cell's
#: full official + optimize matrix legitimately needs more than one hour.
DEFAULT_CELL_MAX_WALL_SECONDS = 7200.0


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
    #: Max concurrent provider calls per fan-out phase (halved once on a
    #: rate-limit failure; ``--concurrency``).
    concurrency: int = DEFAULT_CONCURRENCY
    #: Whole-cell wall deadline (seconds). On breach: finish in-flight, persist
    #: partials, exit with ``status=halted`` and a recorded halt reason.
    max_wall_seconds: float = DEFAULT_CELL_MAX_WALL_SECONDS
    #: Reduced-sampling overrides for the OFFICIAL split (``--official-n`` /
    #: ``--official-repeats``). Both fold into the composite Eval Config
    #: Identity Hash (a reduced cell is a DISTINCT Eval Config identity, so it
    #: gets a cache MISS against the full-config entry). ``official_repeats``,
    #: when set, is ALSO the count the official arms are driven at (overriding
    #: :attr:`official_repeats`). A no-op default keeps the spec-default
    #: sampling and full-config cache identity.
    sampling_overrides: SamplingOverrides = field(
        default_factory=SamplingOverrides
    )
    #: The declared missing-data completeness policy for this cell's official
    #: aggregation (the matrix default, resolved by the CLI). PROPAGATE is the
    #: strict default; a SKIP policy with ``max_skip_fraction`` tolerates that
    #: fraction of skipped rows as an explicit-count SKIP before the arm is
    #: forced incomplete. Both fold into the official Eval Config identity.
    completeness: Completeness = Completeness.PROPAGATE
    #: The declared completeness tolerance (fraction). Inert under PROPAGATE.
    max_skip_fraction: float = 0.0

    def official_repeats_effective(self) -> int:
        """Official repeats actually driven: the override, else the default."""
        if self.sampling_overrides.official_repeats is not None:
            return self.sampling_overrides.official_repeats
        return self.official_repeats

    def record_overrides(self) -> CellSamplingOverrides:
        """The ``sampling_overrides`` sub-object for this cell's line."""
        return CellSamplingOverrides(
            official_n=self.sampling_overrides.official_n,
            official_repeats=self.sampling_overrides.official_repeats,
        )


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


def _arm_incomplete_detail(
    label: str, evaluation: SplitEvaluation | None
) -> str:
    """A ``arm=rows_failed=.. rows_missing=.. ..`` detail for a failed arm.

    Uses the driven evaluation's row accounting when available (the naive arm
    may have come from cache, so its evaluation is None -- then only the label
    is reported).
    """
    if evaluation is None:
        return label
    agg = evaluation.aggregate
    return (
        f"{label}(rows_present={agg.rows_present} "
        f"rows_failed={agg.rows_failed} rows_missing={agg.rows_missing} "
        f"rows_invalid={agg.rows_invalid})"
    )


def _tolerated_skip_note(
    *evaluations: tuple[str, SplitEvaluation | None],
) -> str | None:
    """A visible-counts note for arms that CERTIFIED with some rows skipped.

    Under a bounded SKIP tolerance a certified arm (``score is not None``) may
    still have skipped (missing/failed/invalid) rows within the declared
    bound. Those skipped rows are NEVER silently dropped: this records their
    explicit counts on the cell line so a tolerant anchor honestly shows what
    it skipped. Returns ``None`` when every arm was fully complete.
    """
    parts: list[str] = []
    for label, evaluation in evaluations:
        if evaluation is None or evaluation.score is None:
            continue
        agg = evaluation.aggregate
        skipped = agg.rows_missing + agg.rows_failed + agg.rows_invalid
        if skipped == 0:
            continue
        planned = agg.task_count * agg.repeat_count
        parts.append(
            f"{label}(rows_present={agg.rows_present} "
            f"rows_missing={agg.rows_missing} rows_failed={agg.rows_failed} "
            f"rows_invalid={agg.rows_invalid} skipped={skipped}/{planned})"
        )
    if not parts:
        return None
    return "skipped rows tolerated within declared bound: " + ", ".join(parts)


def _incomplete_arm_note(
    *,
    baseline_score: float | None,
    best_score: float | None,
    baseline_eval: SplitEvaluation | None,
    best_eval: SplitEvaluation | None,
) -> str | None:
    """A note naming which official arm(s) never resolved, or None if complete.

    An official arm is INCOMPLETE when its aggregate scalar is ``None`` (under
    PROPAGATE, some planned rollout rows never landed as a present score even
    after FIX 2's bounded re-drive). A per-task vector still has one entry per
    task (absent repeats count 0), so the vector cannot be trusted to stand in
    for the arm: the cell must not emit a headroom / no-headroom determination
    or a terminal statistical status off it. Returns a human-readable note
    naming the failed arm(s) + their row accounting for the ledger, or ``None``
    when both arms resolved cleanly.
    """
    failed: list[str] = []
    if baseline_score is None:
        failed.append(_arm_incomplete_detail("naive", baseline_eval))
    if best_score is None:
        failed.append(_arm_incomplete_detail("best", best_eval))
    if not failed:
        return None
    return (
        "incomplete official arm(s) -- aggregate never resolved after the "
        "bounded re-drive; no headroom/no-improvement determination emitted "
        "(certified output requires a complete official vector): "
        + ", ".join(failed)
    )


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


def _halted_before_optimize(
    *,
    config: CellConfig,
    cell_id: str,
    ledger: Ledger,
    baseline_score: float | None,
    ceiling_official: float | None,
    baseline_before_ref: str,
    halt_reason: str,
    concurrency_halved: bool,
    spend_before: CreditsSnapshot | None,
    credits_fetcher: CreditsFetcher | None,
    is_openrouter: bool,
    wall_s: float,
    resumed: bool,
    restarted: bool,
) -> CellOutcome:
    """Record a cell halted by the wall deadline before it could optimize.

    The baseline (and ceiling) were measured; the optimize/best phases never
    ran, so ``best_official``/``delta`` are ``None`` and the status is
    ``halted``. The after-spend snapshot is still taken and the partial log is
    KEPT (not deleted) so a later resume can pick up from the recorded calls.
    """
    spend_after: CreditsSnapshot | None = None
    if is_openrouter and credits_fetcher is not None:
        spend_after = credits_fetcher()
        ledger.append_spend(
            SpendRecord(
                cell_id=cell_id, phase="after", lane=config.lane,
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
    notes = [f"halted: {halt_reason}"]
    if concurrency_halved:
        notes.append(
            "concurrency halved after a rate-limit failure (all lanes one key)"
        )
    record = CellRecord(
        cell_id=cell_id, optimizer=config.optimizer, env=config.env,
        attempt=config.attempt, canonical=config.canonical,
        models=CellModels(
            task=config.task_model, proposer=config.proposer_model
        ),
        baseline_official=baseline_score, ceiling_official=ceiling_official,
        best_official=None, delta=None, ci95=None,
        official_repeats_used=config.official_repeats_effective(),
        escalated=False,
        escalation_note="; ".join(notes), pooled_observation_counts={},
        internal_evals_count=0, optimizer_steps=0,
        spend_usd=_spend_between(spend_before, spend_after), wall_s=wall_s,
        lane=config.lane, window_notes=config.window_notes, status="halted",
        artifacts=CellArtifacts(official_record_before=baseline_before_ref),
        sampling_overrides=config.record_overrides(),
    )
    ledger.append_cell(record)
    return CellOutcome(
        record=record, resumed=resumed, restarted=restarted,
        reason=f"halted before optimize: {halt_reason}",
    )


def _incomplete_internal_arm(
    *,
    config: CellConfig,
    cell_id: str,
    ledger: Ledger,
    baseline_score: float | None,
    ceiling_official: float | None,
    baseline_before_ref: str,
    failure_detail: str,
    concurrency_halved: bool,
    spend_before: CreditsSnapshot | None,
    credits_fetcher: CreditsFetcher | None,
    is_openrouter: bool,
    wall_s: float,
    resumed: bool,
    restarted: bool,
) -> CellOutcome:
    """Finalize a cell whose INTERNAL (optimize) arm never resolved.

    A ``CandidateEvaluationFailure`` escaping the optimize phase means some
    internal-split rollouts failed and the env's FAIL missing-data policy left
    the candidate with no computable internal Reward. During a cell's optimize
    (anchor) phase this must NOT surface as a raw process exit 1: it finalizes
    as ``incomplete-arm`` carrying ``arm=internal`` and the reward-failure
    reason, records the real spend attributed to the attempt (openrouter: the
    summed credits deltas across the cell's before/after snapshots), and KEEPS
    the partial log so a resume can re-drive only the failed internal rows. The
    baseline/ceiling arms that DID measure are preserved on the line; the
    best/delta phases never ran so they stay ``None`` and NO headroom /
    statistical determination is emitted (the same guarantee as an incomplete
    OFFICIAL arm). The per-env official cache is NOT written (an incomplete
    cell must never poison downstream optimizer cells).
    """
    spend_after: CreditsSnapshot | None = None
    if is_openrouter and credits_fetcher is not None:
        spend_after = credits_fetcher()
        ledger.append_spend(
            SpendRecord(
                cell_id=cell_id, phase="after", lane=config.lane,
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
    if is_openrouter and credits_fetcher is not None:
        spend_usd, _gaps = ledger.spend_for_cell(cell_id)
    else:
        spend_usd = _spend_between(spend_before, spend_after)
    notes = [
        "incomplete internal arm -- the optimize phase could not score the "
        "candidate into a Reward (some internal-split rollouts failed under "
        "the FAIL missing-data policy); no best/delta/headroom determination "
        "emitted (certified output requires a resolved internal arm): "
        f"internal({failure_detail})"
    ]
    if concurrency_halved:
        notes.append(
            "concurrency halved after a rate-limit failure (all lanes one key)"
        )
    record = CellRecord(
        cell_id=cell_id, optimizer=config.optimizer, env=config.env,
        attempt=config.attempt, canonical=config.canonical,
        models=CellModels(
            task=config.task_model, proposer=config.proposer_model
        ),
        baseline_official=baseline_score, ceiling_official=ceiling_official,
        best_official=None, delta=None, ci95=None,
        official_repeats_used=config.official_repeats_effective(),
        escalated=False,
        escalation_note="; ".join(notes), pooled_observation_counts={},
        internal_evals_count=0, optimizer_steps=0,
        spend_usd=spend_usd, wall_s=wall_s,
        lane=config.lane, window_notes=config.window_notes,
        status="incomplete-arm",
        artifacts=CellArtifacts(official_record_before=baseline_before_ref),
        sampling_overrides=config.record_overrides(),
    )
    ledger.append_cell(record)
    # KEEP the partial log (an incomplete-arm cell is resumable): the caller
    # must not delete it. The per-env official cache is NOT written here.
    return CellOutcome(
        record=record, resumed=resumed, restarted=restarted,
        reason=f"incomplete internal arm: {failure_detail}",
    )


def run_cell(
    config: CellConfig,
    *,
    ledger: Ledger,
    budget: BudgetGuard | None = None,
    credits_fetcher: CreditsFetcher | None = None,
    store: ObjectStore | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CellOutcome:
    """Run one full validation cell, appending its ledger + spend lines.

    Honors resumability (skip a completed cell), the budget guards (reserve at
    start, stop-loss mid-cell), the ceiling-once-per-env cache, and the
    whole-cell wall deadline (``config.max_wall_seconds``, measured on the
    injectable ``clock``).
    """
    budget = budget or BudgetGuard()
    backing = store or ObjectStore(MemoryBackend())
    cell_id = f"{config.optimizer}:{config.env}:a{config.attempt}"
    is_openrouter = config.lane == "openrouter"

    # Per-cell incremental persistence: each provider call is appended to this
    # partial log as it completes, so a crash/interrupt leaves a resumable
    # record. A resumed cell restores already-recorded (instance, candidate,
    # repeat) observations instead of re-driving them.
    partial_log = PartialLog(
        path=ledger.root / "partials" / f"{cell_id}.partial.jsonl"
    )

    # --- Whole-cell wall deadline: shared across every fan-out phase. ---
    cell_start = clock()

    def _fanout() -> FanoutConfig:
        """A FanoutConfig carrying the REMAINING whole-cell wall budget.

        Computed fresh per phase so the shared deadline decrements as the cell
        proceeds; once exhausted, dispatch stops (in-flight calls finish) and
        the phase reports ``deadline_reached``.
        """
        remaining = config.max_wall_seconds - (clock() - cell_start)
        return FanoutConfig(
            concurrency=config.concurrency,
            max_wall_seconds=max(0.0, remaining),
        )

    halt_reason = ""
    concurrency_halved = False

    def _observe(evaluation: object) -> None:
        """Fold a split evaluation's fan-out flags into the cell-level halt."""
        nonlocal halt_reason, concurrency_halved
        if getattr(evaluation, "concurrency_halved", False):
            concurrency_halved = True
        if getattr(evaluation, "deadline_reached", False) and not halt_reason:
            halt_reason = (
                f"whole-cell wall deadline {config.max_wall_seconds:.0f}s "
                "reached; dispatch stopped, in-flight calls finished"
            )

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

    start = clock()
    experiment: EnvExperiment = build_env_experiment(
        config.env,
        model=config.task_model,
        pool_n_per_stratum=config.pool_n_per_stratum,
        completeness=config.completeness,
        max_skip_fraction=config.max_skip_fraction,
        split_sizes=config.split_sizes,
        overrides=config.sampling_overrides,
    )
    naive = experiment.initial_candidate
    official = official_instances(experiment)

    official_repeats = config.official_repeats_effective()
    is_eval_row = config.optimizer == "eval"
    # The composite Eval Config Identity Hash of the official split -- folds in
    # the (possibly overridden) official Task Set + Repeat Plan, so a reduced-
    # sampling cell has a DISTINCT identity from the full-config one. It is
    # part of the env cache key: a reduced cell gets a MISS against a
    # full-config entry for the same (env, task-model). ``default_config`` (no
    # overrides) additionally lets a full-config read match OLD sentinel lines.
    eval_config_hash = (
        experiment.eval_configs.official.eval_config.config_identity_hash
    )
    default_config = config.sampling_overrides.is_noop()

    # --- 1. Baseline naive + ceiling official arms (per-task vectors too). ---
    # Point 6 of the statistical-confidence directive: the Eval row establishes
    # the per-env official naive/ceiling scores AND their per-task score
    # vectors in the ledger cache; every other optimizer cell REUSES that cache
    # rather than re-driving the naive/ceiling arms -- so a later paired
    # best-naive delta can be computed without re-driving naive. A non-eval
    # cell only drives them itself if no Eval row has cached the env yet.
    # Cache is keyed by (env, task-model): a cell must never pair against
    # cached vectors measured under a DIFFERENT task model (different graph
    # identity). A deepseek cell gets a MISS on a nano-cached env and drives
    # its own naive/ceiling arms (FIX 7).
    cache = (
        None
        if is_eval_row
        else ledger.env_cache_for(
            config.env,
            task_model=config.task_model,
            eval_config_hash=eval_config_hash,
            default_config=default_config,
        )
    )
    ceiling_official: float | None
    # The driven baseline evaluation (None when the naive arm came from cache):
    # carries the row-completeness accounting used by the incomplete-arm guard.
    baseline_eval: SplitEvaluation | None = None
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
        # null-scores cell. Detect it via the aggregate's row counts and raise
        # CellBaselineFailure BEFORE recording any ledger line so the CLI exits
        # non-zero loudly. An official eval derives NO Reward, so a merely
        # INCOMPLETE aggregate (some observations timed out) no longer crashes
        # here: it is visible incompleteness (score=None, per-task rows record
        # the failures), distinct from a zero-success plumbing blocker.
        planned = len(official) * official_repeats
        baseline = evaluate_split(
            experiment,
            candidate=naive,
            instances=official,
            split_role="official",
            transport=config.rollout_transport,
            execution_policy=config.execution_policy,
            policy=experiment.completeness_policy,
            repeats=official_repeats,
            store=backing,
            execution_mode=config.execution_mode,
            fanout=_fanout(),
            partial_log=partial_log,
        )
        _observe(baseline)
        if baseline.aggregate.rows_present == 0:
            raise CellBaselineFailure(
                f"cell {cell_id}: baseline official eval produced "
                f"0/{planned} successful rollouts (every rollout failed). "
                "This is a plumbing failure; no cell line recorded."
            )
        baseline_eval = baseline
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
            policy=experiment.completeness_policy,
            repeats=official_repeats,
            store=backing,
            execution_mode=config.execution_mode,
            fanout=_fanout(),
            partial_log=partial_log,
        )
        _observe(ceiling_eval)
        ceiling_official = (
            ceiling_cached
            if ceiling_cached is not None
            else ceiling_eval.score
        )
        ceiling_per_task = ceiling_eval.per_task_scores

    # --- Whole-cell deadline check BEFORE the optimize/best phases. ---
    # A deadline reached during (or by the end of) the baseline/ceiling phases
    # must halt HERE: driving a zero-budget optimize phase would dispatch no
    # calls and produce an uncomputable aggregate (a bare ValueError). We
    # instead record a halted cell carrying the baseline scores we DID measure
    # and keep the partials for a resume. best=None (never optimized).
    deadline_before_optimize = (
        halt_reason != ""
        or (clock() - cell_start) >= config.max_wall_seconds
    )
    if deadline_before_optimize:
        if not halt_reason:
            halt_reason = (
                f"whole-cell wall deadline {config.max_wall_seconds:.0f}s "
                "reached before the optimize phase"
            )
        return _halted_before_optimize(
            config=config, cell_id=cell_id, ledger=ledger,
            baseline_score=baseline_score, ceiling_official=ceiling_official,
            baseline_before_ref=baseline_before_ref, halt_reason=halt_reason,
            concurrency_halved=concurrency_halved,
            spend_before=spend_before, credits_fetcher=credits_fetcher,
            is_openrouter=is_openrouter, wall_s=clock() - start,
            resumed=resumed, restarted=restarted,
        )

    # --- 2. Optimize on the internal split (Reward only). ---
    # Residual hardening: a searching optimizer's internal measurement can
    # still raise a typed CandidateEvaluationFailure when internal rollouts
    # fail under the env's FAIL missing-data policy (the identity optimizer no
    # longer derives a Reward at all -- FIX 1 -- but COPRO/MIPROv2/GEPA/Codex
    # do). During a CELL's anchor phase that failure must NEVER escape as a raw
    # process exit 1: it finalizes the cell typed as ``incomplete-arm``
    # (arm=internal) with real spend and retained partials, exactly as an
    # incomplete OFFICIAL arm does.
    try:
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
            fanout=_fanout(),
        )
    except CandidateEvaluationFailure as exc:
        return _incomplete_internal_arm(
            config=config, cell_id=cell_id, ledger=ledger,
            baseline_score=baseline_score, ceiling_official=ceiling_official,
            baseline_before_ref=baseline_before_ref, failure_detail=str(exc),
            concurrency_halved=concurrency_halved,
            spend_before=spend_before, credits_fetcher=credits_fetcher,
            is_openrouter=is_openrouter, wall_s=clock() - start,
            resumed=resumed, restarted=restarted,
        )

    # --- 3. Best-candidate official eval on the SAME official Eval Config. ---
    best = evaluate_split(
        experiment,
        candidate=opt.best_candidate,
        instances=official,
        split_role="official",
        transport=config.rollout_transport,
        execution_policy=config.execution_policy,
        policy=experiment.completeness_policy,
        repeats=official_repeats,
        store=backing,
        execution_mode=config.execution_mode,
        fanout=_fanout(),
        partial_log=partial_log,
    )
    _observe(best)
    best_score = best.score
    best_per_task = best.per_task_scores
    best_counts = best.per_task_counts

    # --- Incomplete-official-arm guard (before ANY determination). ---
    # An official arm whose aggregate never resolved (``score is None`` under
    # PROPAGATE -- some rollouts failed after FIX 2's bounded re-drive) has an
    # UNTRUSTWORTHY scalar even though its per-task vector still has one entry
    # per task (a failed task's absent repeats count 0, so the vector LOOKS
    # complete). Emitting headroom / no-demonstrable-headroom / a terminal
    # statistical status off that partial vector produces certified-looking
    # output from an incomplete measurement (the c18:a1 defect: naive=None yet
    # headroom + no-improvement emitted). So when either official arm is
    # incomplete we STOP here: no headroom, no gate flag, no statistical
    # status -- the cell finalizes as ``incomplete-arm`` naming the failed
    # arm(s)/rows, keeps its partials for a resume, and reports real spend.
    incomplete_note = _incomplete_arm_note(
        baseline_score=baseline_score,
        best_score=best_score,
        baseline_eval=baseline_eval,
        best_eval=best,
    )

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
    # BUT an incomplete official arm must NEVER emit a headroom / no-headroom
    # determination: those would be certified-looking verdicts from a partial
    # measurement (the c18:a1 defect). When incomplete, they stay None.
    if incomplete_note is None:
        headroom_ci = _paired_or_none(naive_per_task, ceiling_per_task, seed)
        headroom_delta = (
            headroom_ci.point if headroom_ci is not None else None
        )
        no_headroom: bool | None = None
        if is_eval_row and headroom_ci is not None:
            no_headroom = not headroom_ci.excludes_zero()
    else:
        headroom_ci = None
        headroom_delta = None
        no_headroom = None

    pooled_counts = {
        "naive": sum(naive_counts),
        "best": sum(best_counts),
    }

    escalated = False
    escalation_note = ""
    # An incomplete official arm finalizes as ``incomplete-arm`` -- NOT a
    # terminal statistical status. ``_status_from(None, None)`` would return
    # ``no-improvement`` (a certified-looking no-improvement verdict), so we
    # override it here and record which arm/rows failed.
    if incomplete_note is not None:
        status = "incomplete-arm"
    else:
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
            # Escalation ADDS repeats (pooled), so it must NOT restore/skip the
            # already-recorded observations: pass no partial_log here.
            extra_naive = evaluate_split(
                experiment, candidate=naive, instances=official,
                split_role="official", transport=config.rollout_transport,
                execution_policy=config.execution_policy,
                policy=experiment.completeness_policy,
                repeats=official_repeats, store=backing,
                execution_mode=config.execution_mode, fanout=_fanout(),
            )
            _observe(extra_naive)
            extra_best = evaluate_split(
                experiment, candidate=opt.best_candidate, instances=official,
                split_role="official", transport=config.rollout_transport,
                execution_policy=config.execution_policy,
                policy=experiment.completeness_policy,
                repeats=official_repeats, store=backing,
                execution_mode=config.execution_mode, fanout=_fanout(),
            )
            _observe(extra_best)
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

    wall_s = clock() - start

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
    # FIX 8: the cell's final spend_usd sums the credits deltas across ALL
    # attempts of this cell_id (crashed attempts included) from the spend.jsonl
    # before/after pairs -- not merely this attempt's before/after delta. A
    # crashed prior attempt recorded a `before` with no `after`; its burned
    # credits are captured by pairing that `before` with the next snapshot.
    # Falls back to this attempt's own delta when no credits were recorded.
    if is_openrouter and credits_fetcher is not None:
        spend_usd, _spend_gaps = ledger.spend_for_cell(cell_id)
    else:
        spend_usd = _spend_between(spend_before, spend_after)

    # --- Whole-cell wall deadline halts ONLY when work was cut short. ---
    # A cell is ``halted`` only when a phase actually STOPPED dispatching work
    # because the deadline was reached (``halt_reason`` was set by ``_observe``
    # when a phase reported ``deadline_reached``: planned observations were
    # never driven). A cell that COMPLETED every planned phase -- even if the
    # total elapsed happens to exceed ``max_wall_seconds`` because the final
    # phase finished just past the line -- is NOT halted: all its work landed,
    # so its statistical status stands (this was the c11 defect: 2000/2000
    # observations + stats done, then wrongly stamped ``halted``). The elapsed
    # overrun is recorded in the note for transparency, without forcing halt.
    elapsed = clock() - cell_start
    overran = elapsed >= config.max_wall_seconds

    # Stop-loss overrides the statistical status.
    if budget.would_halt(spend_usd):
        status = "halted"
    if halt_reason:
        # Work was actually cut short (dispatch stopped mid-phase).
        status = "halted"

    # Fold the halt reason + any rate-limit concurrency halving into the note.
    notes: list[str] = []
    if escalation_note:
        notes.append(escalation_note)
    if incomplete_note is not None and status == "incomplete-arm":
        notes.append(incomplete_note)
    # A tolerant SKIP anchor that certified with some rows skipped records the
    # explicit skipped-row counts here so the anchor is honest about what it
    # tolerated (never a silently-dropped row).
    tolerated_note = _tolerated_skip_note(
        ("naive", baseline_eval), ("best", best)
    )
    if tolerated_note is not None:
        notes.append(tolerated_note)
    if concurrency_halved:
        notes.append(
            "concurrency halved after a rate-limit failure (all lanes one key)"
        )
    if halt_reason:
        notes.append(f"halted: {halt_reason}")
    elif overran:
        # Completed all phases but ran past the wall budget: NOT a halt, just a
        # transparency note (no work was cut short).
        notes.append(
            f"completed all phases; elapsed {elapsed:.0f}s exceeded wall "
            f"budget {config.max_wall_seconds:.0f}s (not halted -- all "
            "planned observations landed)"
        )
    combined_note = "; ".join(notes)

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
        escalation_note=combined_note,
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
        sampling_overrides=config.record_overrides(),
    )
    ledger.append_cell(record)
    # A cleanly-completed (non-halted) cell has its authoritative result in the
    # ledger, so the per-call partial log is no longer needed -- drop it. A
    # halted OR incomplete-arm cell KEEPS its partial so a resume can pick up
    # the recorded calls (an incomplete arm re-drives only the failed rows).
    if status not in ("halted", "incomplete-arm"):
        partial_log.delete()
    # The Eval row establishes the per-env official cache (naive + ceiling
    # scalars AND per-task vectors) so later optimizer cells reuse it. NEVER
    # cache an incomplete arm: caching None/partial vectors would poison every
    # downstream optimizer cell for this (env, task-model).
    if is_eval_row and cache is None and incomplete_note is None:
        ledger.append_env_cache(
            EnvOfficialCache(
                env=config.env,
                naive_official=baseline_score,
                ceiling_official=ceiling_official,
                naive_per_task=naive_per_task,
                ceiling_per_task=ceiling_per_task,
                official_repeats_used=official_repeats,
                task_model=config.task_model,
                eval_config_hash=eval_config_hash,
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
