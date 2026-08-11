from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MDD_PLATEAU_EPSILON",
    "DEFAULT_REPEAT_CAP",
    "DEFAULT_TARGET_PROB",
    "PowerConfig",
    "PowerRecommendation",
    "PowerResult",
    "PowerSurfacePoint",
    "VarianceDecomposition",
    "analyze_power",
]

DEFAULT_ALPHA = 0.25

DEFAULT_TARGET_PROB = 0.80

DEFAULT_REPEAT_CAP = 20

DEFAULT_MDD_PLATEAU_EPSILON = 0.005

_DEFAULT_TRIALS = 4000


def _repeat_grid(r_cap: int) -> tuple[int, ...]:
    return tuple(range(1, max(1, r_cap) + 1))


@dataclass(frozen=True, slots=True)
class PowerConfig:
    alpha: float = DEFAULT_ALPHA
    target_prob: float = DEFAULT_TARGET_PROB
    repeat_cap: int = DEFAULT_REPEAT_CAP
    mdd_plateau_epsilon: float = DEFAULT_MDD_PLATEAU_EPSILON
    trials: int = _DEFAULT_TRIALS
    seed: int = 20260723
    per_call_usd: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if (
            not math.isfinite(self.target_prob)
            or not 0.5 < self.target_prob < 1.0
        ):
            raise ValueError("target_prob must be in (0.5, 1)")
        if self.repeat_cap < 1:
            raise ValueError("repeat_cap must be at least 1")
        if (
            not math.isfinite(self.mdd_plateau_epsilon)
            or self.mdd_plateau_epsilon < 0.0
        ):
            raise ValueError(
                "mdd_plateau_epsilon must be finite and non-negative"
            )
        if self.trials < 1:
            raise ValueError("trials must be at least 1")
        if self.per_call_usd is not None and (
            not math.isfinite(self.per_call_usd) or self.per_call_usd < 0.0
        ):
            raise ValueError("per_call_usd must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    """Empirical variance terms; paired comparisons cancel the between-task
    main effect.
    """

    base_rate: float
    within_repeat_var: float
    interaction_var: float
    between_task_var: float
    anchor_repeats: int
    n_tasks_observed: int

    @property
    def within_dominates(self) -> bool:
        return self.within_repeat_var > self.between_task_var

    @property
    def noise_verdict(self) -> str:
        if self.within_dominates:
            return "within-task repeat noise dominates"
        return "between-task noise dominates"


@dataclass(frozen=True, slots=True)
class PowerRecommendation:
    target_gap: float
    achievable: bool
    recommended_n_tasks: int
    recommended_repeats: int
    achieved_mdd: float
    recommended_calls: int
    recommended_usd: float | None
    best_achievable_mdd: float
    best_n_tasks: int
    best_repeats: int
    repeat_plateau: int | None
    pool_limited: bool


@dataclass(frozen=True, slots=True)
class PowerSurfacePoint:
    n_tasks: int
    num_samples: int
    calls: int
    mdd_at_target: float
    simulated_rank_probability: float


@dataclass(frozen=True, slots=True)
class PowerResult:
    config: PowerConfig
    certified_headroom: float
    naive_mean: float
    ceiling_mean: float
    pool_ceiling: int
    decomposition: VarianceDecomposition
    recommendation: PowerRecommendation
    surface: tuple[PowerSurfacePoint, ...] = ()


def _decompose_variance(
    naive_per_task: np.ndarray,
    ceiling_per_task: np.ndarray,
    *,
    anchor_repeats: int,
) -> VarianceDecomposition:
    """Estimate the paired comparison's empirical variance terms."""
    both = np.concatenate([naive_per_task, ceiling_per_task])
    r = max(1, anchor_repeats)
    base_rate = float(np.clip(naive_per_task.mean(), 0.0, 1.0))
    within = float(base_rate * (1.0 - base_rate))
    within_obs = float(np.mean(both * (1.0 - both)))
    midpoints = (naive_per_task + ceiling_per_task) / 2.0
    raw_between = (
        float(np.var(midpoints, ddof=1)) if midpoints.size > 1 else 0.0
    )
    between = max(0.0, raw_between - within_obs / (2.0 * r))
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
    decomp: VarianceDecomposition, *, n_tasks: int, num_samples: int
) -> float:
    """SE of the mean paired A-vs-B difference at ``(n_tasks, repeats)``.

    Paired residual per-task-difference variance = the task-x-candidate
    INTERACTION variance (r-invariant) + the within-task repeat noise OF THE
    DIFFERENCE (``2 * within / repeats``, shrinks with r). The mean over
    ``n_tasks`` shared tasks divides by ``n_tasks``.
    """
    per_task_diff_var = (
        decomp.interaction_var
        + 2.0 * decomp.within_repeat_var / max(1, num_samples)
    )
    return math.sqrt(max(per_task_diff_var, 0.0) / max(1, n_tasks))


def _mdd_at_target(
    decomp: VarianceDecomposition,
    *,
    n_tasks: int,
    num_samples: int,
    target_prob: float,
) -> float:
    """The minimum detectable difference (MDD) at the target ranking prob.

    Under the paired normal approximation, ``P(observed winner = true winner)``
    for a true gap ``delta`` is ``Phi(delta / SE)``. Inverting for the target
    probability: ``MDD = z_target * SE`` where ``z_target = Phi^-1(target)``.
    The seeded simulation in :func:`analyze_power` validates the closed-form
    grid MDD while keeping the surface smooth.
    """
    se = _paired_diff_se(decomp, n_tasks=n_tasks, num_samples=num_samples)
    z = _normal_ppf(target_prob)
    return z * se


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (stdlib ``NormalDist``).

    Turns the target ranking probability into the z-multiplier for the MDD
    (``MDD = z * SE``). Pure stdlib -- no scipy.
    """
    if not math.isfinite(p) or not 0.5 < p < 1.0:
        raise ValueError("target_prob must be in (0.5, 1)")
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def _simulate_ranking_prob(
    decomp: VarianceDecomposition,
    *,
    n_tasks: int,
    num_samples: int,
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
        1, num_samples
    )
    sd = math.sqrt(max(diff_var, 0.0))
    if sd == 0.0:
        return 1.0 if delta > 0 else 0.5
    # The mean of n_tasks independent normal differences is itself normal.
    draws = rng.normal(
        loc=delta,
        scale=sd / math.sqrt(n_tasks),
        size=trials,
    )
    return float(np.mean(draws > 0.0))


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
    ``pool_ceiling`` is the available task-pool size (the hard ``n`` ceiling).
    Returns a :class:`PowerResult` with the variance decomposition, the full
    MDD
    surface, and the cheapest achievable ``(n, r)`` recommendation (or the loud
    pool-limited best-achievable verdict).
    """
    cfg = config or PowerConfig()
    if pool_ceiling < 1:
        raise ValueError("pool_ceiling must be at least 1")
    if anchor_repeats < 1:
        raise ValueError("anchor_repeats must be at least 1")
    naive = np.asarray(naive_per_task, dtype=float)
    ceiling = np.asarray(ceiling_per_task, dtype=float)
    if (
        naive.ndim != 1
        or ceiling.ndim != 1
        or naive.size == 0
        or naive.shape != ceiling.shape
        or not np.all(np.isfinite(naive))
        or not np.all(np.isfinite(ceiling))
        or not np.all((0.0 <= naive) & (naive <= 1.0))
        or not np.all((0.0 <= ceiling) & (ceiling <= 1.0))
    ):
        raise ValueError(
            "naive/ceiling per-task scores must be one-dimensional, "
            "finite probabilities in [0, 1]"
        )
    naive_mean = float(naive.mean())
    ceiling_mean = float(ceiling.mean())
    certified_headroom = max(0.0, ceiling_mean - naive_mean)
    target_gap = cfg.alpha * certified_headroom

    decomp = _decompose_variance(naive, ceiling, anchor_repeats=anchor_repeats)

    n_grid = tuple(range(1, max(1, pool_ceiling) + 1))
    r_grid = _repeat_grid(cfg.repeat_cap)
    rng = np.random.default_rng(cfg.seed)

    surface: list[PowerSurfacePoint] = []
    # Track the cheapest achievable (n,r) meeting the target gap and the global
    # best-achievable MDD (for the pool-limited verdict).
    best_cost = math.inf
    rec: tuple[int, int, float] | None = None  # (n, r, mdd)
    best_mdd = math.inf
    best_nr: tuple[int, int] = (n_grid[-1], r_grid[-1])
    for n_tasks in n_grid:
        for num_samples in r_grid:
            mdd = _mdd_at_target(
                decomp,
                n_tasks=n_tasks,
                num_samples=num_samples,
                target_prob=cfg.target_prob,
            )
            # A seeded simulation validation at the operating target gap.
            sim_prob = _simulate_ranking_prob(
                decomp,
                n_tasks=n_tasks,
                num_samples=num_samples,
                delta=target_gap if target_gap > 0 else mdd,
                trials=cfg.trials,
                rng=rng,
            )
            calls = n_tasks * num_samples
            surface.append(
                PowerSurfacePoint(
                    n_tasks=n_tasks,
                    num_samples=num_samples,
                    calls=calls,
                    mdd_at_target=mdd,
                    simulated_rank_probability=sim_prob,
                )
            )
            if mdd < best_mdd:
                best_mdd = mdd
                best_nr = (n_tasks, num_samples)
            # Achievable iff the detectable gap (MDD) is <= the target gap.
            if target_gap > 0 and mdd <= target_gap and calls < best_cost:
                best_cost = calls
                rec = (n_tasks, num_samples, mdd)

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
        decomp,
        n_tasks=rn,
        r_grid=r_grid,
        target_prob=cfg.target_prob,
        epsilon=cfg.mdd_plateau_epsilon,
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

    Samples hit diminishing returns because only the within-task repeat noise
    shrinks with ``r`` (the interaction floor does not). Returns the smallest
    ``r`` such that MDD(r) - MDD(r+1) < epsilon (at the fixed ``n_tasks``), or
    ``None`` if every step within the grid still gains >= epsilon.
    """
    prev = _mdd_at_target(
        decomp, n_tasks=n_tasks, num_samples=r_grid[0], target_prob=target_prob
    )
    for r in r_grid[1:]:
        cur = _mdd_at_target(
            decomp, n_tasks=n_tasks, num_samples=r, target_prob=target_prob
        )
        if (prev - cur) < epsilon:
            return r - 1
        prev = cur
    return None
