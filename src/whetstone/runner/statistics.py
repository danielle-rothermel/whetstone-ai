"""Cheap, deterministic bootstrap CI over tasks (no extra LLM calls).

The validation plan requires each cell to report ``delta + bootstrap CI over
tasks`` for the official before/after comparison. The delta is
``official(best) - official(naive)``; the CI is a paired bootstrap over the
per-task official scores (resampling tasks with replacement). It is a pure
function of already-collected per-task scores (no provider call is made) and
seeded so the interval is reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = [
    "BootstrapCI",
    "bootstrap_delta_ci",
    "mean",
]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A paired-bootstrap confidence interval for a mean delta."""

    delta: float
    low: float
    high: float
    level: float
    resamples: int

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "delta": self.delta,
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "resamples": self.resamples,
        }


def mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(values) / len(values)


def bootstrap_delta_ci(
    naive_per_task: tuple[float, ...],
    best_per_task: tuple[float, ...],
    *,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Paired bootstrap CI for ``mean(best) - mean(naive)`` over tasks.

    ``naive_per_task`` and ``best_per_task`` are aligned per-task official
    scores (task ``i`` compared before/after). Each resample draws task indices
    with replacement and recomputes the mean delta; the CI is the empirical
    ``level`` interval of those resampled deltas. Deterministic given ``seed``.
    """
    if len(naive_per_task) != len(best_per_task):
        raise ValueError(
            "paired bootstrap requires aligned per-task score vectors"
        )
    if not naive_per_task:
        raise ValueError("bootstrap requires at least one task")
    n = len(naive_per_task)
    point_delta = mean(best_per_task) - mean(naive_per_task)
    if n == 1:
        # A single task cannot form an interval; the point estimate is the CI.
        return BootstrapCI(
            delta=point_delta,
            low=point_delta,
            high=point_delta,
            level=level,
            resamples=resamples,
        )
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(best_per_task[i] for i in idx) / n
        a = sum(naive_per_task[i] for i in idx) / n
        deltas.append(b - a)
    deltas.sort()
    tail = (1.0 - level) / 2.0
    low_i = max(0, int(tail * resamples))
    high_i = min(resamples - 1, int((1.0 - tail) * resamples))
    return BootstrapCI(
        delta=point_delta,
        low=deltas[low_i],
        high=deltas[high_i],
        level=level,
        resamples=resamples,
    )
