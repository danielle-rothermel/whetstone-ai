"""Closed-form and property tests for bootstrap interval estimation.

Expected values here are computed independently of the implementation:
either analytically (a degenerate sample has a degenerate interval, the
paired delta of ``(x, x + c)`` is exactly ``c``) or from the definition of
percentile-bootstrap coverage. Every draw is seeded, so each assertion is
deterministic; tolerances bound resampling error, never wall-clock timing.
"""

from __future__ import annotations

import random

import pytest

from whetstone.eval.analysis.statistics import (
    BootstrapCI,
    bootstrap_delta_ci,
    bootstrap_mean_ci,
    bootstrap_paired_delta_ci,
    mean,
    resample_indices,
)


def test_mean_of_known_values_is_the_arithmetic_mean() -> None:
    assert mean((1.0,)) == 1.0
    assert mean((0.0, 1.0)) == 0.5
    assert mean((0.25, 0.25, 0.25, 0.25)) == 0.25
    assert mean((-1.0, 1.0)) == 0.0


def test_mean_of_empty_sample_is_undefined() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        mean(())


def test_resample_indices_is_seeded_shaped_and_in_range() -> None:
    draws = resample_indices(5, resamples=7, seed=11)

    assert len(draws) == 7
    assert all(len(draw) == 5 for draw in draws)
    assert all(0 <= index < 5 for draw in draws for index in draw)


def test_resample_indices_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    same = resample_indices(6, resamples=20, seed=3)

    assert same == resample_indices(6, resamples=20, seed=3)
    assert same != resample_indices(6, resamples=20, seed=4)


@pytest.mark.parametrize(
    ("n", "resamples"),
    [(0, 10), (-1, 10), (5, 0)],
)
def test_resample_indices_rejects_degenerate_shapes(n: int, resamples: int) -> None:
    with pytest.raises(ValueError):
        resample_indices(n, resamples=resamples, seed=0)


def test_bootstrap_mean_ci_on_a_constant_sample_is_degenerate() -> None:
    # Every resample of a constant sample has the same mean, so the
    # percentile interval collapses onto that constant exactly.
    ci = bootstrap_mean_ci((0.4,) * 6, resamples=200, seed=1)

    assert ci.point == pytest.approx(0.4)
    assert ci.low == pytest.approx(0.4)
    assert ci.high == pytest.approx(0.4)
    assert ci.level == 0.95
    assert ci.resamples == 200


def test_bootstrap_mean_ci_of_a_single_task_is_the_point_estimate() -> None:
    ci = bootstrap_mean_ci((0.7,), resamples=50, seed=0)

    assert (ci.point, ci.low, ci.high) == (0.7, 0.7, 0.7)


def test_bootstrap_mean_ci_is_symmetric_for_a_symmetric_sample() -> None:
    # A balanced two-point sample has a mean-resampling distribution that is
    # symmetric about 0.5, so both tails must be equidistant from the point
    # estimate. Tolerance covers percentile-index granularity at 4000 draws.
    ci = bootstrap_mean_ci((0.0, 1.0) * 8, resamples=4000, seed=3)

    assert ci.point == pytest.approx(0.5)
    upper_width = ci.high - ci.point
    lower_width = ci.point - ci.low
    assert upper_width == pytest.approx(lower_width, abs=0.02)


def test_bootstrap_mean_ci_brackets_the_point_estimate() -> None:
    ci = bootstrap_mean_ci((0.1, 0.35, 0.6, 0.85), resamples=500, seed=9)

    assert ci.low <= ci.point <= ci.high


def test_bootstrap_mean_ci_covers_the_true_mean_at_about_the_nominal_level() -> None:
    # Draw 200 seeded samples from a fair coin (true mean 0.5) and count how
    # often the nominal 95% interval covers it. The check is deterministic:
    # the sampler is seeded once and every interval uses a fixed seed. The
    # bound is deliberately loose (>=0.85) because percentile bootstrap
    # under-covers for small n and the Monte-Carlo standard error over 200
    # trials is about 1.5 points; it still fails a genuinely broken interval.
    rng = random.Random(20260821)
    true_mean = 0.5
    covered = 0
    trials = 200

    for trial in range(trials):
        sample = tuple(rng.choice([0.0, 1.0]) for _ in range(40))
        ci = bootstrap_mean_ci(sample, level=0.95, resamples=400, seed=trial)
        if ci.low <= true_mean <= ci.high:
            covered += 1

    assert covered / trials >= 0.85


def test_wider_level_yields_a_wider_interval() -> None:
    sample = (0.1, 0.3, 0.45, 0.62, 0.8, 0.95)
    narrow = bootstrap_mean_ci(sample, level=0.80, resamples=2000, seed=17)
    wide = bootstrap_mean_ci(sample, level=0.99, resamples=2000, seed=17)

    assert (wide.high - wide.low) > (narrow.high - narrow.low)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_mean_ci_rejects_levels_outside_the_unit_interval(
    level: float,
) -> None:
    with pytest.raises(ValueError, match="level must be in"):
        bootstrap_mean_ci((0.1, 0.2), level=level, resamples=10, seed=0)


def test_bootstrap_mean_ci_rejects_nonpositive_resamples() -> None:
    with pytest.raises(ValueError, match="resamples must be at least 1"):
        bootstrap_mean_ci((0.1, 0.2), resamples=0, seed=0)


def test_bootstrap_mean_ci_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        bootstrap_mean_ci((), resamples=10, seed=0)


def test_paired_delta_of_a_constant_shift_is_exactly_that_shift() -> None:
    # b == a + c on every task, so every paired resample has delta exactly c
    # and the interval has zero width. This is an exact analytic result.
    a = tuple(index / 20 for index in range(20))
    b = tuple(value + 0.1 for value in a)

    ci = bootstrap_paired_delta_ci(a, b, resamples=2000, seed=5)

    assert ci.point == pytest.approx(0.1)
    assert ci.low == pytest.approx(0.1)
    assert ci.high == pytest.approx(0.1)
    assert (ci.high - ci.low) == pytest.approx(0.0, abs=1e-12)


def test_paired_delta_is_oriented_second_argument_minus_first() -> None:
    a = (0.1, 0.5, 0.9, 0.3)
    b = (0.2, 0.6, 1.0, 0.4)

    forward = bootstrap_paired_delta_ci(a, b, resamples=100, seed=0)
    reverse = bootstrap_paired_delta_ci(b, a, resamples=100, seed=0)

    assert forward.point == pytest.approx(mean(b) - mean(a))
    assert reverse.point == pytest.approx(-forward.point)


def test_paired_delta_of_identical_samples_straddles_zero() -> None:
    sample = tuple(index / 20 for index in range(20))

    ci = bootstrap_paired_delta_ci(sample, sample, resamples=2000, seed=5)

    assert ci.point == pytest.approx(0.0)
    assert ci.low <= 0.0 <= ci.high
    assert not ci.excludes_zero()


def test_pairing_is_narrower_than_resampling_the_arms_independently() -> None:
    # Correlated arms: pairing cancels the shared between-task variance, so
    # the paired interval must be strictly narrower than one built by
    # resampling each arm with independent index draws.
    a = tuple(index / 20 for index in range(20))
    b = tuple(value + 0.1 for value in a)
    n = len(a)

    paired = bootstrap_paired_delta_ci(a, b, resamples=2000, seed=5)

    a_draws = resample_indices(n, resamples=2000, seed=5)
    b_draws = resample_indices(n, resamples=2000, seed=6)
    unpaired = sorted(
        sum(b[i] for i in b_idx) / n - sum(a[i] for i in a_idx) / n
        for a_idx, b_idx in zip(a_draws, b_draws, strict=True)
    )
    tail = int(0.025 * 2000)
    unpaired_width = unpaired[2000 - tail - 1] - unpaired[tail]

    assert (paired.high - paired.low) < unpaired_width


def test_bootstrap_delta_ci_is_the_paired_estimator() -> None:
    # bootstrap_delta_ci delegates to the paired estimator; anchor that
    # contract so a future unpaired rewrite cannot land silently.
    a = (0.1, 0.5, 0.9, 0.3)
    b = (0.2, 0.6, 1.0, 0.4)

    assert bootstrap_delta_ci(
        a, b, resamples=500, seed=2
    ) == bootstrap_paired_delta_ci(a, b, resamples=500, seed=2)


def test_paired_delta_ci_rejects_misaligned_score_vectors() -> None:
    with pytest.raises(ValueError, match="aligned per-task score vectors"):
        bootstrap_paired_delta_ci((0.1, 0.2), (0.1,), resamples=10, seed=0)


def test_paired_delta_ci_rejects_empty_score_vectors() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        bootstrap_paired_delta_ci((), (), resamples=10, seed=0)


def test_bootstrap_ci_reports_direction_and_tuple_projection() -> None:
    positive = BootstrapCI(0.2, 0.1, 0.3, 0.95, 100)
    negative = BootstrapCI(-0.2, -0.3, -0.1, 0.95, 100)
    straddling = BootstrapCI(0.0, -0.1, 0.1, 0.95, 100)

    assert positive.excludes_zero()
    assert negative.excludes_zero()
    assert not straddling.excludes_zero()
    assert positive.delta == 0.2
    assert positive.as_tuple() == (0.1, 0.3)
