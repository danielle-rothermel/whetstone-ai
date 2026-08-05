"""Bootstrap CI tests (pure, deterministic, no LLM calls)."""

from __future__ import annotations

import random

import pytest

from whetstone.code_eval.statistics import (
    DEFAULT_RESAMPLES,
    bootstrap_delta_ci,
    bootstrap_mean_ci,
    bootstrap_paired_delta_ci,
    mean,
    resample_indices,
)


def _unpaired_delta_ci(
    a: tuple[float, ...], b: tuple[float, ...], *, seed: int
) -> tuple[float, float]:
    """An UNPAIRED delta CI: each arm resampled with independent indices.

    This is the wrong thing to do for a paired comparison; it exists only so a
    test can prove the paired CI (shared indices) differs from -- and is
    tighter than -- the unpaired one on correlated data.
    """
    n = len(a)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(DEFAULT_RESAMPLES):
        ia = [rng.randrange(n) for _ in range(n)]
        ib = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(b[i] for i in ib) / n - sum(a[i] for i in ia) / n)
    deltas.sort()
    tail = 0.025
    lo = deltas[int(tail * DEFAULT_RESAMPLES)]
    hi = deltas[int((1.0 - tail) * DEFAULT_RESAMPLES)]
    return lo, hi


def test_point_delta_is_mean_difference() -> None:
    naive = (0.0, 0.0, 1.0, 0.0)
    best = (1.0, 1.0, 1.0, 0.0)
    ci = bootstrap_delta_ci(naive, best, seed=0)
    assert ci.delta == pytest.approx(mean(best) - mean(naive))


def test_ci_brackets_the_delta() -> None:
    naive = tuple([0.0] * 10)
    best = tuple([1.0] * 10)
    ci = bootstrap_delta_ci(naive, best, seed=1)
    # Every task improves by 1.0, so the whole interval sits at 1.0.
    assert ci.low == pytest.approx(1.0)
    assert ci.high == pytest.approx(1.0)
    assert ci.delta == pytest.approx(1.0)


def test_no_improvement_delta_zero() -> None:
    scores = (1.0, 0.0, 1.0, 0.0, 1.0)
    ci = bootstrap_delta_ci(scores, scores, seed=2)
    assert ci.delta == pytest.approx(0.0)
    assert ci.low == pytest.approx(0.0)
    assert ci.high == pytest.approx(0.0)


def test_deterministic_given_seed() -> None:
    naive = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    best = (1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
    a = bootstrap_delta_ci(naive, best, seed=7)
    b = bootstrap_delta_ci(naive, best, seed=7)
    assert a == b


def test_single_task_returns_point_interval() -> None:
    ci = bootstrap_delta_ci((0.0,), (1.0,), seed=0)
    assert ci.low == ci.high == ci.delta == pytest.approx(1.0)


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        bootstrap_delta_ci((0.0, 1.0), (1.0,), seed=0)


def test_empty_rejected() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        bootstrap_delta_ci((), (), seed=0)


def test_default_resamples_is_ten_thousand() -> None:
    assert DEFAULT_RESAMPLES == 10_000


def test_reproducible_intervals_fixed_seed() -> None:
    # Fixed seed -> identical intervals across independent calls (marginal +
    # paired). Reproducibility is the crux of the statistical-confidence work.
    naive = (0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.25)
    best = (1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0, 0.75)
    m1 = bootstrap_mean_ci(best, seed=1234)
    m2 = bootstrap_mean_ci(best, seed=1234)
    assert m1.as_tuple() == m2.as_tuple()
    p1 = bootstrap_paired_delta_ci(naive, best, seed=1234)
    p2 = bootstrap_paired_delta_ci(naive, best, seed=1234)
    assert p1.as_tuple() == p2.as_tuple()
    # A different seed generally yields a different interval (not degenerate).
    p3 = bootstrap_paired_delta_ci(naive, best, seed=9999)
    assert p3.point == pytest.approx(p1.point)  # point delta is seed-free


def test_resample_indices_shared_across_paired_arms() -> None:
    # The paired bootstrap draws ONE index set per resample and applies it to
    # both arms: resample_indices with the same seed reproduces those draws.
    a = resample_indices(5, resamples=10, seed=7)
    b = resample_indices(5, resamples=10, seed=7)
    assert a == b
    assert all(0 <= i < 5 for draw in a for i in draw)


def test_paired_ci_differs_from_unpaired_on_correlated_data() -> None:
    # Construct correlated data: best = naive + a constant per-task lift, so
    # per-task deltas are (nearly) constant. The PAIRED CI (shared indices)
    # collapses tightly around the true lift; the UNPAIRED CI (independent
    # indices) is much wider because it double-counts the shared variance.
    naive = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.7, 0.1, 0.5)
    best = tuple(min(1.0, v + 0.05) for v in naive)
    paired = bootstrap_paired_delta_ci(naive, best, seed=3)
    u_lo, u_hi = _unpaired_delta_ci(naive, best, seed=3)
    paired_width = paired.high - paired.low
    unpaired_width = u_hi - u_lo
    # The paired interval is materially tighter (shared variance cancels).
    assert paired_width < unpaired_width
    # And the paired interval is the correct one to report: it brackets the
    # true per-task lift tightly (all deltas ~0.05).
    assert paired.low <= 0.05 <= paired.high
    assert paired_width < 0.02


def test_paired_delta_ci_excludes_zero_helper() -> None:
    naive = tuple([0.0] * 12)
    best = tuple([1.0] * 12)
    ci = bootstrap_paired_delta_ci(naive, best, seed=5)
    assert ci.excludes_zero() is True
    # A zero-delta comparison never excludes zero.
    same = bootstrap_paired_delta_ci(naive, naive, seed=5)
    assert same.excludes_zero() is False


def test_marginal_mean_ci_brackets_mean() -> None:
    scores = (0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.5, 0.5)
    ci = bootstrap_mean_ci(scores, seed=11)
    assert ci.low <= mean(scores) <= ci.high
    assert ci.point == pytest.approx(mean(scores))
