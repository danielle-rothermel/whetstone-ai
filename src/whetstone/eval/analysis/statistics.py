from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = [
    "DEFAULT_RESAMPLES",
    "BootstrapCI",
    "bootstrap_delta_ci",
    "bootstrap_mean_ci",
    "bootstrap_paired_delta_ci",
    "mean",
    "resample_indices",
]


DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    level: float
    resamples: int

    @property
    def delta(self) -> float:
        return self.point

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)


def mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(values) / len(values)


def resample_indices(
    n: int, *, resamples: int, seed: int
) -> list[tuple[int, ...]]:
    if n <= 0:
        raise ValueError("bootstrap requires at least one task")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    rng = random.Random(seed)
    return [
        tuple(rng.randrange(n) for _ in range(n)) for _ in range(resamples)
    ]


def _percentile_bounds(
    values: list[float], level: float, resamples: int
) -> tuple[float, float]:
    values.sort()
    tail = (1.0 - level) / 2.0
    low_i = max(0, int(tail * resamples))
    high_i = min(resamples - 1, int((1.0 - tail) * resamples))
    return values[low_i], values[high_i]


def _validate_interval(*, level: float, resamples: int) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")


def bootstrap_mean_ci(
    per_task: tuple[float, ...],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    _validate_interval(level=level, resamples=resamples)
    if not per_task:
        raise ValueError("bootstrap requires at least one task")
    n = len(per_task)
    point = mean(per_task)
    if n == 1:
        return BootstrapCI(point, point, point, level, resamples)
    draws = resample_indices(n, resamples=resamples, seed=seed)
    means = [sum(per_task[i] for i in idx) / n for idx in draws]
    low, high = _percentile_bounds(means, level, resamples)
    return BootstrapCI(point, low, high, level, resamples)


def bootstrap_paired_delta_ci(
    a_per_task: tuple[float, ...],
    b_per_task: tuple[float, ...],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    _validate_interval(level=level, resamples=resamples)
    if len(a_per_task) != len(b_per_task):
        raise ValueError(
            "paired bootstrap requires aligned per-task score vectors"
        )
    if not a_per_task:
        raise ValueError("bootstrap requires at least one task")
    n = len(a_per_task)
    point = mean(b_per_task) - mean(a_per_task)
    if n == 1:
        return BootstrapCI(point, point, point, level, resamples)
    draws = resample_indices(n, resamples=resamples, seed=seed)
    deltas: list[float] = []
    for idx in draws:
        b = sum(b_per_task[i] for i in idx) / n
        a = sum(a_per_task[i] for i in idx) / n
        deltas.append(b - a)
    low, high = _percentile_bounds(deltas, level, resamples)
    return BootstrapCI(point, low, high, level, resamples)


def bootstrap_delta_ci(
    naive_per_task: tuple[float, ...],
    best_per_task: tuple[float, ...],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    return bootstrap_paired_delta_ci(
        naive_per_task,
        best_per_task,
        level=level,
        resamples=resamples,
        seed=seed,
    )
