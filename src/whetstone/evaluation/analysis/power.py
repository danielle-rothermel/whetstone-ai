from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_SAMPLE_CAP",
    "DEFAULT_TARGET_PROB",
    "PowerConfig",
    "PowerResult",
    "PowerSurfacePoint",
    "VarianceDecomposition",
    "analyze_power",
]

DEFAULT_ALPHA = 0.25

DEFAULT_TARGET_PROB = 0.80

DEFAULT_SAMPLE_CAP = 16


def _sample_grid(sample_cap: int) -> tuple[int, ...]:
    return tuple(range(1, max(1, sample_cap) + 1))


@dataclass(frozen=True, slots=True)
class PowerConfig:
    alpha: float = DEFAULT_ALPHA
    target_prob: float = DEFAULT_TARGET_PROB
    sample_cap: int = DEFAULT_SAMPLE_CAP

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if (
            not math.isfinite(self.target_prob)
            or not 0.5 < self.target_prob < 1.0
        ):
            raise ValueError("target_prob must be in (0.5, 1)")
        if self.sample_cap < 1:
            raise ValueError("sample_cap must be at least 1")


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    base_rate: float
    within_sample_var: float
    interaction_var: float
    between_task_var: float
    anchor_samples: int
    n_tasks_observed: int

    @property
    def within_dominates(self) -> bool:
        return self.within_sample_var > self.between_task_var

    @property
    def noise_verdict(self) -> str:
        if self.within_dominates:
            return "within-task sample noise dominates"
        return "between-task noise dominates"


@dataclass(frozen=True, slots=True)
class PowerSurfacePoint:
    n_tasks: int
    num_samples: int
    calls: int
    mdd_at_target: float


@dataclass(frozen=True, slots=True)
class PowerResult:
    config: PowerConfig
    certified_headroom: float
    target_gap: float
    naive_mean: float
    ceiling_mean: float
    pool_ceiling: int
    decomposition: VarianceDecomposition
    surface: tuple[PowerSurfacePoint, ...] = ()


def _decompose_variance(
    naive_per_task: np.ndarray,
    ceiling_per_task: np.ndarray,
    *,
    anchor_samples: int,
) -> VarianceDecomposition:
    both = np.concatenate([naive_per_task, ceiling_per_task])
    r = max(1, anchor_samples)
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
        within_sample_var=within,
        interaction_var=interaction,
        between_task_var=between,
        anchor_samples=r,
        n_tasks_observed=int(naive_per_task.size),
    )


def _paired_diff_se(
    decomp: VarianceDecomposition, *, n_tasks: int, num_samples: int
) -> float:
    per_task_diff_var = (
        decomp.interaction_var
        + 2.0 * decomp.within_sample_var / max(1, num_samples)
    )
    return math.sqrt(max(per_task_diff_var, 0.0) / max(1, n_tasks))


def _mdd_at_target(
    decomp: VarianceDecomposition,
    *,
    n_tasks: int,
    num_samples: int,
    target_prob: float,
) -> float:
    se = _paired_diff_se(decomp, n_tasks=n_tasks, num_samples=num_samples)
    z = _normal_ppf(target_prob)
    return z * se


def _normal_ppf(p: float) -> float:
    if not math.isfinite(p) or not 0.5 < p < 1.0:
        raise ValueError("target_prob must be in (0.5, 1)")
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def analyze_power(
    *,
    naive_per_task: tuple[float, ...],
    ceiling_per_task: tuple[float, ...],
    pool_ceiling: int,
    anchor_samples: int,
    config: PowerConfig | None = None,
) -> PowerResult:
    cfg = config or PowerConfig()
    if pool_ceiling < 1:
        raise ValueError("pool_ceiling must be at least 1")
    if anchor_samples < 1:
        raise ValueError("anchor_samples must be at least 1")
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

    decomp = _decompose_variance(naive, ceiling, anchor_samples=anchor_samples)

    n_grid = tuple(range(1, max(1, pool_ceiling) + 1))
    s_grid = _sample_grid(cfg.sample_cap)

    surface: list[PowerSurfacePoint] = []
    for n_tasks in n_grid:
        for num_samples in s_grid:
            mdd = _mdd_at_target(
                decomp,
                n_tasks=n_tasks,
                num_samples=num_samples,
                target_prob=cfg.target_prob,
            )
            surface.append(
                PowerSurfacePoint(
                    n_tasks=n_tasks,
                    num_samples=num_samples,
                    calls=n_tasks * num_samples,
                    mdd_at_target=mdd,
                )
            )
    return PowerResult(
        config=cfg,
        certified_headroom=certified_headroom,
        target_gap=target_gap,
        naive_mean=naive_mean,
        ceiling_mean=ceiling_mean,
        pool_ceiling=pool_ceiling,
        decomposition=decomp,
        surface=tuple(surface),
    )
