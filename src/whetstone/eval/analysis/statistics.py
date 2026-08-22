from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = [
    "DEFAULT_RESAMPLES",
    "BootstrapCI",
    "bootstrap_delta_ci",
    "bootstrap_mean_ci",
    "bootstrap_paired_delta_ci",
    "holm_adjust",
    "mean",
    "resample_indices",
]


DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A percentile bootstrap interval and its companion two-sided p-value.

    ``p_value`` is computed from the same resample vector the interval's
    order statistics come from, as
    ``2 * min(P(stat* <= 0), P(stat* >= 0))`` clamped into
    ``[1 / resamples, 1]``. The lower clamp matters: an all-positive bootstrap
    would otherwise report an exact zero, which a Holm correction propagates
    as an exact zero rather than as "smaller than this bootstrap can resolve".

    ``degenerate`` marks an interval built from fewer than two paired
    observations. With one observation every resample is that same point, so
    the "interval" is the point estimate and the resample vector carries no
    information about sampling uncertainty at all. A degenerate interval
    reports ``p_value == 1.0``: not because the effect is known to be absent,
    but because a single observation cannot support a significance claim. The
    unclamped alternative is worse -- one task with a nonzero delta lands the
    ``1 / resamples`` floor, the most significant value the estimator can
    emit, and that floor then survives a Holm correction as a false positive.
    Only ``n < 2`` is degenerate: at ``n >= 2`` the resampler genuinely varies
    its draws, so a vector with a single nonzero delta still produces a
    real -- and appropriately unimpressive -- p-value.

    The p-value is *percentile-based*, not null-centered: the resamples are
    drawn from the observed data, so they describe the estimator's confidence
    distribution rather than a null distribution built by centering or sign
    flipping. It is therefore exactly the interval read as a tail area, and
    it carries no evidence the interval does not already carry. In
    particular, when every paired delta shares a sign, every resample mean
    shares that sign too, so ``raw`` is 0 and the reported value is the
    clamp floor ``1 / resamples`` -- at any ``n``, including ``n == 2``,
    where no sign-flip test could approach that significance. Step 10 must
    read a p-value at or near the floor as "smaller than this bootstrap can
    resolve" and must not treat it as evidence beyond what the interval
    shows.
    """

    point: float
    low: float
    high: float
    level: float
    resamples: int
    p_value: float
    degenerate: bool = False

    @property
    def delta(self) -> float:
        return self.point

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)


def _two_sided_p(values: list[float], *, resamples: int) -> float:
    """``2 * min(P(stat* <= 0), P(stat* >= 0))`` clamped to [1/resamples, 1]."""
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not values:
        raise ValueError("a bootstrap p-value needs at least one resample")
    n = len(values)
    at_or_below = sum(1 for value in values if value <= 0.0)
    at_or_above = sum(1 for value in values if value >= 0.0)
    raw = 2.0 * min(at_or_below, at_or_above) / n
    return min(1.0, max(1.0 / resamples, raw))


def _degenerate_ci(
    point: float, *, level: float, resamples: int
) -> BootstrapCI:
    """The interval for a sample too small to resample: no significance.

    A one-observation bootstrap resamples that observation and nothing else,
    so the interval collapses onto the point estimate and the resample vector
    is constant. There is no sampling distribution to read a p-value off, so
    the p-value is 1.0 and the interval is flagged ``degenerate``.
    """
    return BootstrapCI(
        point,
        point,
        point,
        level,
        resamples,
        1.0,
        degenerate=True,
    )


def holm_adjust(pvalues: tuple[float, ...]) -> tuple[float, ...]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    With ``m`` hypotheses sorted ascending, the ``i``-th smallest raw p-value
    is scaled by ``m - i`` and then made monotone non-decreasing by a running
    maximum; every value is capped at 1. Rejecting each hypothesis whose
    adjusted p-value is at or below ``alpha`` controls the family-wise error
    rate at ``alpha``. Ties keep their input order, so the mapping back is
    stable.
    """
    for value in pvalues:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p-values must be in [0, 1]; got {value!r}")
    m = len(pvalues)
    if m == 0:
        return ()
    order = sorted(range(m), key=lambda index: pvalues[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        scaled = min(1.0, (m - rank) * pvalues[index])
        running = max(running, scaled)
        adjusted[index] = running
    return tuple(adjusted)


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
        return _degenerate_ci(point, level=level, resamples=resamples)
    draws = resample_indices(n, resamples=resamples, seed=seed)
    means = [sum(per_task[i] for i in idx) / n for idx in draws]
    low, high = _percentile_bounds(means, level, resamples)
    return BootstrapCI(
        point,
        low,
        high,
        level,
        resamples,
        _two_sided_p(means, resamples=resamples),
    )


def bootstrap_paired_delta_ci(
    a_per_task: tuple[float, ...],
    b_per_task: tuple[float, ...],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI and p-value for the paired ``b - a`` delta.

    With a single task there is nothing to resample: the interval collapses to
    the point estimate and the result is marked ``degenerate`` with
    ``p_value == 1.0``, so a one-task delta never claims significance.
    """
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
        return _degenerate_ci(point, level=level, resamples=resamples)
    draws = resample_indices(n, resamples=resamples, seed=seed)
    deltas: list[float] = []
    for idx in draws:
        b = sum(b_per_task[i] for i in idx) / n
        a = sum(a_per_task[i] for i in idx) / n
        deltas.append(b - a)
    low, high = _percentile_bounds(deltas, level, resamples)
    return BootstrapCI(
        point,
        low,
        high,
        level,
        resamples,
        _two_sided_p(deltas, resamples=resamples),
    )


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
