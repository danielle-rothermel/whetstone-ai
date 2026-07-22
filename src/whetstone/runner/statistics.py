"""Percentile bootstrap CIs over TASKS (no extra LLM calls).

The validation plan's "Statistical confidence" directive requires that every
claim carry a bootstrap CI computed over the exchangeable unit -- the **task**
(repeats within a task are correlated; the task is the resampling unit). Each
interval is a percentile bootstrap, 10k resamples, 95%, seeded per cell so the
interval is reproducible.

Two interval shapes are produced:

* **marginal** (:func:`bootstrap_mean_ci`): the CI of a single arm's mean over
  per-task mean scores (``naive_ci95``, ``ceiling_ci95``).
* **paired** (:func:`bootstrap_paired_delta_ci`): the CI of ``mean(b) -
  mean(a)`` where each resample draws the SAME task indices for both arms
  (``delta_ci95`` = best-naive, ``headroom_ci95`` = ceiling-naive). Pairing
  cancels shared per-task variance, so a paired interval is generally tighter
  than the unpaired one on the same data.

Every function is a pure function of already-collected per-task scores (no
provider call is made) and deterministic given ``seed``: the same fixed seed
draws the same resample index sets, so paired variants that share a seed also
share their resample indices across both arms.
"""

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

#: The plan-mandated resample count (percentile bootstrap, 10k, 95%).
DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A percentile-bootstrap confidence interval for a scalar statistic.

    ``point`` is the observed statistic (a mean, or a paired mean-delta);
    ``low``/``high`` are the empirical ``level`` percentile bounds over the
    resampled statistic.
    """

    point: float
    low: float
    high: float
    level: float
    resamples: int

    @property
    def delta(self) -> float:
        """Alias for ``point`` (paired-delta call sites read ``delta``)."""
        return self.point

    def excludes_zero(self) -> bool:
        """True when the whole interval lies strictly on one side of 0."""
        return self.low > 0.0 or self.high < 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "resamples": self.resamples,
        }


def mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(values) / len(values)


def resample_indices(
    n: int, *, resamples: int, seed: int
) -> list[tuple[int, ...]]:
    """The ``resamples`` task-index draws (with replacement) for ``n`` tasks.

    Deterministic given ``seed``. Paired intervals reuse ONE such index set
    across both arms so the same tasks are resampled together; that is what
    makes a paired CI a paired CI (shared per-task variance cancels).
    """
    if n <= 0:
        raise ValueError("bootstrap requires at least one task")
    rng = random.Random(seed)
    return [
        tuple(rng.randrange(n) for _ in range(n))
        for _ in range(resamples)
    ]


def _percentile_bounds(
    values: list[float], level: float, resamples: int
) -> tuple[float, float]:
    values.sort()
    tail = (1.0 - level) / 2.0
    low_i = max(0, int(tail * resamples))
    high_i = min(resamples - 1, int((1.0 - tail) * resamples))
    return values[low_i], values[high_i]


def bootstrap_mean_ci(
    per_task: tuple[float, ...],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    """Marginal percentile-bootstrap CI for ``mean(per_task)`` over tasks."""
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
    """Paired percentile-bootstrap CI for ``mean(b) - mean(a)`` over tasks.

    ``a_per_task`` and ``b_per_task`` are aligned per-task means (task ``i`` in
    both arms). Each resample draws ONE task-index set (via
    :func:`resample_indices`) and applies it to BOTH arms, so the shared
    per-task variance cancels -- a genuine paired bootstrap. Deterministic
    given ``seed``.
    """
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
    """Paired CI for ``mean(best) - mean(naive)`` (the cell delta).

    A thin alias of :func:`bootstrap_paired_delta_ci` in ``(naive, best)``
    order, kept as the delta call site's name.
    """
    return bootstrap_paired_delta_ci(
        naive_per_task,
        best_per_task,
        level=level,
        resamples=resamples,
        seed=seed,
    )
