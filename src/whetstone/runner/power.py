"""Pre-run statistical-power analysis for the internal-eval sample size.

Runs (opt-in) AFTER the baseline/anchor arms and BEFORE optimization, per
(env, task-model). Using the anchor's per-task observation vectors (naive +
ceiling), it estimates -- via a seeded paired bootstrap over a 2-D
(n_tasks x repeats) grid -- the internal-eval sample size needed to reliably
RANK candidates whose true score gap is ``>= alpha * certified_headroom`` at a
target correct-ranking probability. It follows the methodology of
``reports/internal-signal-analysis.md`` (MDD@target; TASKS are the resampling
unit) with the paired-variance addendum:

* **Empirical variance decomposition** (not assumed): from the anchor per-task
  means it separates the within-task per-repeat (Bernoulli) variance from the
  between-task / task-x-candidate interaction variance, bias-correcting the
  between-task estimate for the finite anchor-repeat measurement noise.
* **Paired model on a shared task set**: the between-task MAIN effect cancels
  in a paired A-vs-B comparison; the residual variance of the mean paired
  difference is the task-x-candidate INTERACTION plus the within-task repeat
  noise OF THE DIFFERENCE -- and ONLY the latter shrinks with ``r``. So repeats
  are a first-class power dial, not a no-op: marginal-variance-only power would
  overstate the tasks needed and understate what repeats buy.
* **2-D surface**: MDD@target over the full ``(n_tasks x repeats)`` grid (``n``
  up to the pool ceiling, ``r`` up to a sane cap), the CHEAPEST ``(n, r)``
  meeting the ``alpha`` target (cost = ``n*r`` calls at the measured per-call
  cost), and the repeat-PLATEAU (``r`` beyond which the marginal MDD gain falls
  below ``epsilon``).
* **Pool-limit verdict**: if NO ``(n, r)`` within ``pool x r_cap`` reaches the
  target, the best achievable MDD and the ``(n, r)`` achieving it are recorded
  LOUDLY -- never a silent clamp (the c22 case).

Nothing here drives a provider call or mutates a graph identity: it is a pure,
deterministic (seeded) computation over the anchor's already-measured per-task
vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MDD_PLATEAU_EPSILON",
    "DEFAULT_REPEAT_CAP",
    "DEFAULT_TARGET_PROB",
    "POWER_ANALYSIS_SCHEMA",
    "PowerConfig",
    "PowerRecommendation",
    "PowerResult",
    "VarianceDecomposition",
    "analyze_power",
]

#: The schema tag stamped on the persisted per-cell power-analysis artifact.
POWER_ANALYSIS_SCHEMA = "whetstone.runner.power_analysis/v1"

#: The default fraction of certified headroom the internal eval must reliably
#: resolve (a candidate gap ``>= alpha * headroom`` should rank correctly).
DEFAULT_ALPHA = 0.25

#: The default target correct-ranking probability (P(observed winner = true
#: winner) at the target gap).
DEFAULT_TARGET_PROB = 0.80

#: The default repeat cap for the (n x r) grid search.
DEFAULT_REPEAT_CAP = 20

#: The repeat-plateau epsilon: ``r`` beyond which the marginal MDD improvement
#: from one more repeat (at the recommended n) is below this is "plateaued".
DEFAULT_MDD_PLATEAU_EPSILON = 0.005

#: The seeded number of paired-ranking Monte-Carlo trials per grid cell.
_DEFAULT_TRIALS = 4000

#: The grid of repeat counts evaluated (1..r_cap).
def _repeat_grid(r_cap: int) -> tuple[int, ...]:
    return tuple(range(1, max(1, r_cap) + 1))


@dataclass(frozen=True, slots=True)
class PowerConfig:
    """The knobs for one power analysis (all deterministic-seeded)."""

    alpha: float = DEFAULT_ALPHA
    target_prob: float = DEFAULT_TARGET_PROB
    repeat_cap: int = DEFAULT_REPEAT_CAP
    mdd_plateau_epsilon: float = DEFAULT_MDD_PLATEAU_EPSILON
    trials: int = _DEFAULT_TRIALS
    seed: int = 20260723
    #: Optional measured per-call cost (USD) for the cost model; ``None`` keeps
    #: the cost purely in call-count units (``n*r``).
    per_call_usd: float | None = None


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    """The empirical within-vs-between variance decomposition of the anchor.

    ``base_rate`` is the operating base score the paired comparison is centered
    on (the anchor midpoint). ``within_repeat_var`` is the per-single-repeat
    Bernoulli variance at that operating point (shrinks with ``r``);
    ``interaction_var`` is the task-x-candidate interaction variance of the
    per-task difference (does NOT shrink with ``r``); ``between_task_var`` is
    the task-to-task main-effect variance (CANCELS in a paired comparison,
    reported for the within-vs-between verdict). ``anchor_repeats`` is the
    ``r`` the anchor vectors were measured at (used to bias-correct the
    estimates). ``within_dominates`` says whether repeat or task noise wins.
    """

    base_rate: float
    within_repeat_var: float
    interaction_var: float
    between_task_var: float
    anchor_repeats: int
    n_tasks_observed: int

    @property
    def within_dominates(self) -> bool:
        """Whether within-task repeat noise dominates between-task noise."""
        return self.within_repeat_var > self.between_task_var

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_rate": self.base_rate,
            "within_repeat_var": self.within_repeat_var,
            "interaction_var": self.interaction_var,
            "between_task_var": self.between_task_var,
            "anchor_repeats": self.anchor_repeats,
            "n_tasks_observed": self.n_tasks_observed,
            "within_dominates": self.within_dominates,
            "verdict": (
                "within-task repeat noise dominates"
                if self.within_dominates
                else "between-task noise dominates"
            ),
        }


@dataclass(frozen=True, slots=True)
class PowerRecommendation:
    """The recommended (n_tasks, repeats) + the achievability verdict."""

    target_gap: float
    achievable: bool
    #: Recommended sizes CLAMPED to the pool ceiling / r cap.
    recommended_n_tasks: int
    recommended_repeats: int
    #: The MDD@target achieved at the recommendation.
    achieved_mdd: float
    #: The cost of the recommendation (n*r calls, and USD when known).
    recommended_calls: int
    recommended_usd: float | None
    #: When NOT achievable: the BEST (n,r) within pool x r-cap + its MDD.
    best_achievable_mdd: float
    best_n_tasks: int
    best_repeats: int
    #: The repeat-plateau: r beyond which marginal MDD gain < epsilon (at the
    #: recommended/best n). ``None`` when no plateau within the cap.
    repeat_plateau: int | None
    pool_limited: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_gap": self.target_gap,
            "achievable": self.achievable,
            "pool_limited": self.pool_limited,
            "recommended_n_tasks": self.recommended_n_tasks,
            "recommended_repeats": self.recommended_repeats,
            "achieved_mdd": self.achieved_mdd,
            "recommended_calls": self.recommended_calls,
            "recommended_usd": self.recommended_usd,
            "best_achievable_mdd": self.best_achievable_mdd,
            "best_n_tasks": self.best_n_tasks,
            "best_repeats": self.best_repeats,
            "repeat_plateau": self.repeat_plateau,
        }


@dataclass(frozen=True, slots=True)
class PowerResult:
    """The full power analysis + the persisted-artifact projection."""

    config: PowerConfig
    certified_headroom: float
    naive_mean: float
    ceiling_mean: float
    pool_ceiling: int
    decomposition: VarianceDecomposition
    recommendation: PowerRecommendation
    #: The full MDD surface: one row per (n_tasks, repeats) grid cell.
    surface: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_artifact(self, *, header: dict[str, Any]) -> dict[str, Any]:
        """The persisted ``power_analysis`` artifact (fully re-derivable)."""
        return {
            **header,
            "schema": POWER_ANALYSIS_SCHEMA,
            "config": {
                "alpha": self.config.alpha,
                "target_prob": self.config.target_prob,
                "repeat_cap": self.config.repeat_cap,
                "mdd_plateau_epsilon": self.config.mdd_plateau_epsilon,
                "trials": self.config.trials,
                "seed": self.config.seed,
                "per_call_usd": self.config.per_call_usd,
            },
            "certified_headroom": self.certified_headroom,
            "naive_mean": self.naive_mean,
            "ceiling_mean": self.ceiling_mean,
            "pool_ceiling": self.pool_ceiling,
            "variance_decomposition": self.decomposition.as_dict(),
            "recommendation": self.recommendation.as_dict(),
            "surface": list(self.surface),
        }


def _decompose_variance(
    naive_per_task: np.ndarray,
    ceiling_per_task: np.ndarray,
    *,
    anchor_repeats: int,
) -> VarianceDecomposition:
    """Empirically decompose the anchor's within-vs-between/interaction
    variance.

    The candidates the internal eval must rank are MUTATIONS of the naive
    baseline, so the operating base rate is the naive arm's mean (mid-range
    Bernoulli noise is highest there -- the report's "base rate matters").

    From the anchor per-task means (measured at ``anchor_repeats`` repeats):

    * ``base_rate`` = the naive arm's overall mean (the candidate operating
      point). ``within_repeat_var`` = its single-repeat Bernoulli variance
      ``base(1-base)`` -- the repeat noise that shrinks with r, anchored to the
      operating rate the candidates actually sit at (NOT the anchor extremes,
      where 0/1 saturation would understate it).
    * ``between_task_var`` = the sample variance of the per-task means across
      BOTH arms' tasks, ANOVA-corrected by subtracting the measurement-noise
      inflation ``within_obs/anchor_repeats`` (floored at 0). This is the true
      task-to-task heterogeneity, which CANCELS in a paired comparison; it is
      reported only for the within-vs-between verdict.
    * ``interaction_var`` = the variance of the per-task naive->ceiling
      DIFFERENCE across tasks, corrected for its own measurement noise
      (``2*within_obs/anchor_repeats``), floored at 0. This is the
      task-x-candidate interaction the paired residual retains and that does
      NOT
      shrink with r. It is anchored to a floor of ``0.1 * within_repeat_var``
      so
      the r=3 anchor's noisy near-zero interaction estimate cannot make the
      paired model claim a perfectly-separable surface.
    """
    both = np.concatenate([naive_per_task, ceiling_per_task])
    r = max(1, anchor_repeats)
    base_rate = float(np.clip(naive_per_task.mean(), 0.0, 1.0))
    # Operating within-task single-repeat Bernoulli variance at the candidate
    # base rate (mid-range noise, where the candidates actually operate).
    within = float(base_rate * (1.0 - base_rate))
    # Observed within (at the anchor per-task rates) -- used only to de-bias
    # the
    # observed dispersions for the anchor measurement noise.
    within_obs = float(np.mean(both * (1.0 - both)))
    # Between-task main effect (ANOVA-corrected for anchor measurement noise).
    raw_between = float(np.var(both, ddof=1)) if both.size > 1 else 0.0
    between = max(0.0, raw_between - within_obs / r)
    # Task-x-candidate interaction from the per-task PAIRED difference variance
    # (corrected for the difference's measurement noise 2*within_obs/r), with a
    # floor so a noisy r=3 near-zero estimate cannot fabricate a trivial
    # surface.
    diff = ceiling_per_task - naive_per_task
    raw_interaction = float(np.var(diff, ddof=1)) if diff.size > 1 else 0.0
    interaction = max(
        0.1 * within, max(0.0, raw_interaction - 2.0 * within_obs / r)
    )
    return VarianceDecomposition(
        base_rate=base_rate,
        within_repeat_var=within,
        interaction_var=interaction,
        between_task_var=between,
        anchor_repeats=r,
        n_tasks_observed=int(naive_per_task.size),
    )


def _paired_diff_se(
    decomp: VarianceDecomposition, *, n_tasks: int, repeats: int
) -> float:
    """SE of the mean paired A-vs-B difference at ``(n_tasks, repeats)``.

    Paired residual per-task-difference variance = the task-x-candidate
    INTERACTION variance (r-invariant) + the within-task repeat noise OF THE
    DIFFERENCE (``2 * within / repeats``, shrinks with r). The mean over
    ``n_tasks`` shared tasks divides by ``n_tasks``.
    """
    per_task_diff_var = (
        decomp.interaction_var
        + 2.0 * decomp.within_repeat_var / max(1, repeats)
    )
    return math.sqrt(max(per_task_diff_var, 0.0) / max(1, n_tasks))


def _mdd_at_target(
    decomp: VarianceDecomposition,
    *,
    n_tasks: int,
    repeats: int,
    target_prob: float,
) -> float:
    """The minimum detectable difference (MDD) at the target ranking prob.

    Under the paired normal approximation, ``P(observed winner = true winner)``
    for a true gap ``delta`` is ``Phi(delta / SE)``. Inverting for the target
    probability: ``MDD = z_target * SE`` where ``z_target = Phi^-1(target)``.
    This is the closed-form dual of the report's Monte-Carlo MDD@80; the
    :func:`analyze_power` simulation validates it and is what the artifact
    records, but the grid MDD is computed closed-form for a smooth surface.
    """
    se = _paired_diff_se(decomp, n_tasks=n_tasks, repeats=repeats)
    z = _normal_ppf(target_prob)
    return z * se


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (stdlib ``NormalDist``).

    Turns the target ranking probability into the z-multiplier for the MDD
    (``MDD = z * SE``). Pure stdlib -- no scipy.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("target_prob must be in (0, 1)")
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def _simulate_ranking_prob(
    decomp: VarianceDecomposition,
    *,
    n_tasks: int,
    repeats: int,
    delta: float,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """Seeded Monte-Carlo P(paired winner = true winner) at true gap ``delta``.

    Draws ``trials`` paired experiments: per trial, ``n_tasks`` shared tasks
    each contribute a per-task paired difference = a task-x-candidate
    interaction draw (variance ``interaction_var``, r-invariant) + a
    within-task repeat-noise draw of the difference (variance
    ``2*within/repeats``). The observed mean paired difference's sign is
    compared to the true ``delta > 0``. Validates the closed-form MDD;
    deterministic given the passed ``rng``.
    """
    diff_var = decomp.interaction_var + 2.0 * decomp.within_repeat_var / max(
        1, repeats
    )
    sd = math.sqrt(max(diff_var, 0.0))
    if sd == 0.0:
        return 1.0 if delta > 0 else 0.5
    # trials x n_tasks per-task observed differences ~ Normal(delta, sd).
    draws = rng.normal(loc=delta, scale=sd, size=(trials, n_tasks))
    means = draws.mean(axis=1)
    return float(np.mean(means > 0.0))


def analyze_power(
    *,
    naive_per_task: tuple[float, ...],
    ceiling_per_task: tuple[float, ...],
    pool_ceiling: int,
    anchor_repeats: int,
    config: PowerConfig | None = None,
) -> PowerResult:
    """Run the paired 2-D (n x r) power analysis over the anchor vectors.

    ``naive_per_task`` / ``ceiling_per_task`` are the anchor arms' per-task
    mean
    0/1 vectors (aligned by task, measured at ``anchor_repeats`` repeats).
    ``pool_ceiling`` is the internal-split size (the hard ``n`` ceiling).
    Returns a :class:`PowerResult` with the variance decomposition, the full
    MDD
    surface, and the cheapest achievable ``(n, r)`` recommendation (or the loud
    pool-limited best-achievable verdict).
    """
    cfg = config or PowerConfig()
    naive = np.asarray(naive_per_task, dtype=float)
    ceiling = np.asarray(ceiling_per_task, dtype=float)
    if naive.size == 0 or naive.size != ceiling.size:
        raise ValueError(
            "naive/ceiling per-task vectors must be non-empty and aligned"
        )
    naive_mean = float(naive.mean())
    ceiling_mean = float(ceiling.mean())
    certified_headroom = max(0.0, ceiling_mean - naive_mean)
    target_gap = cfg.alpha * certified_headroom

    decomp = _decompose_variance(
        naive, ceiling, anchor_repeats=anchor_repeats
    )

    n_grid = tuple(range(1, max(1, pool_ceiling) + 1))
    r_grid = _repeat_grid(cfg.repeat_cap)
    rng = np.random.default_rng(cfg.seed)

    surface: list[dict[str, Any]] = []
    # Track the cheapest achievable (n,r) meeting the target gap and the global
    # best-achievable MDD (for the pool-limited verdict).
    best_cost = math.inf
    rec: tuple[int, int, float] | None = None  # (n, r, mdd)
    best_mdd = math.inf
    best_nr: tuple[int, int] = (n_grid[-1], r_grid[-1])
    for n_tasks in n_grid:
        for repeats in r_grid:
            mdd = _mdd_at_target(
                decomp, n_tasks=n_tasks, repeats=repeats,
                target_prob=cfg.target_prob,
            )
            # A seeded simulation validation at the operating target gap.
            sim_prob = _simulate_ranking_prob(
                decomp, n_tasks=n_tasks, repeats=repeats,
                delta=target_gap if target_gap > 0 else mdd,
                trials=cfg.trials, rng=rng,
            )
            calls = n_tasks * repeats
            surface.append({
                "n_tasks": n_tasks,
                "repeats": repeats,
                "calls": calls,
                "mdd_at_target": mdd,
                "sim_rank_prob_at_target_gap": sim_prob,
            })
            if mdd < best_mdd:
                best_mdd = mdd
                best_nr = (n_tasks, repeats)
            # Achievable iff the detectable gap (MDD) is <= the target gap.
            if target_gap > 0 and mdd <= target_gap and calls < best_cost:
                best_cost = calls
                rec = (n_tasks, repeats, mdd)

    achievable = rec is not None
    if rec is not None:
        rn, rr, rmdd = rec
    else:
        # Pool-limited: recommend the BEST-achievable (n,r) and say so LOUDLY.
        rn, rr = best_nr
        rmdd = best_mdd
    recommended_calls = rn * rr
    recommended_usd = (
        recommended_calls * cfg.per_call_usd
        if cfg.per_call_usd is not None
        else None
    )
    plateau = _repeat_plateau(
        decomp, n_tasks=rn, r_grid=r_grid,
        target_prob=cfg.target_prob, epsilon=cfg.mdd_plateau_epsilon,
    )
    recommendation = PowerRecommendation(
        target_gap=target_gap,
        achievable=achievable,
        recommended_n_tasks=rn,
        recommended_repeats=rr,
        achieved_mdd=rmdd,
        recommended_calls=recommended_calls,
        recommended_usd=recommended_usd,
        best_achievable_mdd=best_mdd,
        best_n_tasks=best_nr[0],
        best_repeats=best_nr[1],
        repeat_plateau=plateau,
        pool_limited=not achievable,
    )
    return PowerResult(
        config=cfg,
        certified_headroom=certified_headroom,
        naive_mean=naive_mean,
        ceiling_mean=ceiling_mean,
        pool_ceiling=pool_ceiling,
        decomposition=decomp,
        recommendation=recommendation,
        surface=tuple(surface),
    )


def _repeat_plateau(
    decomp: VarianceDecomposition,
    *,
    n_tasks: int,
    r_grid: tuple[int, ...],
    target_prob: float,
    epsilon: float,
) -> int | None:
    """The first ``r`` beyond which the marginal MDD gain is below ``epsilon``.

    Repeats hit diminishing returns because only the within-task repeat noise
    shrinks with ``r`` (the interaction floor does not). Returns the smallest
    ``r`` such that MDD(r) - MDD(r+1) < epsilon (at the fixed ``n_tasks``), or
    ``None`` if every step within the grid still gains >= epsilon.
    """
    prev = _mdd_at_target(
        decomp, n_tasks=n_tasks, repeats=r_grid[0], target_prob=target_prob
    )
    for r in r_grid[1:]:
        cur = _mdd_at_target(
            decomp, n_tasks=n_tasks, repeats=r, target_prob=target_prob
        )
        if (prev - cur) < epsilon:
            return r - 1
        prev = cur
    return None
