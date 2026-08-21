"""Closed-form tests for the variance decomposition and power surface.

Expected values are derived analytically from the formulas the module
documents, or from published normal quantiles, rather than pinned from a
prior run. Tolerances are stated per assertion and bound floating-point or
quantile-table error only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from whetstone.eval.analysis.power import (
    DEFAULT_ALPHA,
    DEFAULT_SAMPLE_CAP,
    DEFAULT_TARGET_PROB,
    PowerConfig,
    VarianceDecomposition,
    _decompose_variance,
    _mdd_at_target,
    _normal_ppf,
    _paired_diff_se,
    analyze_power,
)


def _decomposition(
    *,
    within_sample_var: float,
    interaction_var: float,
    between_task_var: float = 0.05,
) -> VarianceDecomposition:
    return VarianceDecomposition(
        base_rate=0.5,
        within_sample_var=within_sample_var,
        interaction_var=interaction_var,
        between_task_var=between_task_var,
        anchor_samples=1,
        n_tasks_observed=10,
    )


@pytest.mark.parametrize(
    ("probability", "quantile"),
    [
        (0.8, 0.8416212335729143),
        (0.9, 1.2815515655446004),
        (0.95, 1.6448536269514722),
        (0.975, 1.9599639845400545),
        (0.99, 2.3263478740408408),
    ],
)
def test_normal_ppf_matches_published_quantiles(
    probability: float, quantile: float
) -> None:
    # Standard normal quantiles from published tables; 1e-3 is far looser
    # than the underlying implementation's accuracy but immune to table
    # rounding in the reference values.
    assert _normal_ppf(probability) == pytest.approx(quantile, abs=1e-3)


@pytest.mark.parametrize("probability", [0.5, 0.4, 0.0, 1.0, 1.5, math.nan])
def test_normal_ppf_rejects_probabilities_outside_its_contract(
    probability: float,
) -> None:
    # The helper is a target-power quantile, documented as (0.5, 1); the
    # median and both tails are outside that contract.
    with pytest.raises(ValueError, match="target_prob must be in"):
        _normal_ppf(probability)


def test_normal_ppf_is_increasing_in_the_target_probability() -> None:
    assert _normal_ppf(0.6) < _normal_ppf(0.8) < _normal_ppf(0.95)


def test_decompose_variance_recovers_known_components() -> None:
    # Two identical arms with per-task rates 0.2/0.4/0.6/0.8.
    #   base_rate = mean(naive) = 0.5, within = 0.5 * 0.5 = 0.25
    #   within_obs = mean(p(1-p)) over both arms = 0.20
    #   midpoints equal the rates, so raw between = var(ddof=1) = 1/15
    #   between = 1/15 - 0.20 / (2 * 4) = 0.0416666...
    #   diff is identically zero, so raw interaction = 0 and the floor
    #   0.1 * within = 0.025 applies.
    naive = np.array([0.2, 0.4, 0.6, 0.8])
    ceiling = naive.copy()

    decomposition = _decompose_variance(naive, ceiling, anchor_samples=4)

    assert decomposition.base_rate == pytest.approx(0.5)
    assert decomposition.within_sample_var == pytest.approx(0.25)
    assert decomposition.between_task_var == pytest.approx(1 / 15 - 0.20 / 8)
    assert decomposition.interaction_var == pytest.approx(0.025)
    assert decomposition.anchor_samples == 4
    assert decomposition.n_tasks_observed == 4


def test_decompose_variance_reports_zero_between_variance_for_flat_tasks() -> None:
    # Identical per-task rates leave no between-task spread to recover, and
    # the noise-floor subtraction cannot push the estimate below zero.
    flat = np.full(5, 0.5)

    decomposition = _decompose_variance(flat, flat, anchor_samples=2)

    assert decomposition.between_task_var == 0.0
    assert decomposition.within_sample_var == pytest.approx(0.25)


def test_decompose_variance_reports_zero_between_variance_for_one_task() -> None:
    single = np.array([0.4])

    decomposition = _decompose_variance(single, single, anchor_samples=1)

    assert decomposition.between_task_var == 0.0
    assert decomposition.n_tasks_observed == 1


def test_decompose_variance_clamps_the_base_rate_into_the_unit_interval() -> None:
    # within = p(1-p) is only a Bernoulli variance for p in [0, 1]; the
    # clamp keeps the component nonnegative.
    decomposition = _decompose_variance(
        np.array([1.0, 1.0]), np.array([1.0, 1.0]), anchor_samples=1
    )

    assert decomposition.base_rate == 1.0
    assert decomposition.within_sample_var == pytest.approx(0.0)


def test_noise_verdict_names_the_dominant_component() -> None:
    within_heavy = _decomposition(
        within_sample_var=0.25, interaction_var=0.1, between_task_var=0.05
    )
    between_heavy = _decomposition(
        within_sample_var=0.01, interaction_var=0.1, between_task_var=0.2
    )

    assert within_heavy.within_dominates
    assert within_heavy.noise_verdict == "within-task sample noise dominates"
    assert not between_heavy.within_dominates
    assert between_heavy.noise_verdict == "between-task noise dominates"


def test_paired_diff_se_matches_its_analytic_expression() -> None:
    # SE = sqrt((interaction + 2 * within / seeds) / tasks)
    decomposition = _decomposition(within_sample_var=0.25, interaction_var=0.1)

    observed = _paired_diff_se(decomposition, n_tasks=10, num_seeds=2)

    assert observed == pytest.approx(math.sqrt((0.1 + 2 * 0.25 / 2) / 10))


def test_paired_diff_se_shrinks_with_more_tasks_and_more_repeats() -> None:
    decomposition = _decomposition(within_sample_var=0.25, interaction_var=0.1)
    base = _paired_diff_se(decomposition, n_tasks=10, num_seeds=2)

    assert _paired_diff_se(decomposition, n_tasks=20, num_seeds=2) < base
    assert _paired_diff_se(decomposition, n_tasks=10, num_seeds=4) < base


def test_repeats_cannot_reduce_the_irreducible_interaction_floor() -> None:
    # Repeats only average away within-sample noise; the interaction term
    # survives, so the SE has a positive limit as seeds grow.
    decomposition = _decomposition(within_sample_var=0.25, interaction_var=0.1)
    floor = math.sqrt(0.1 / 10)

    assert _paired_diff_se(decomposition, n_tasks=10, num_seeds=10**6) > floor
    assert _paired_diff_se(
        decomposition, n_tasks=10, num_seeds=10**6
    ) == pytest.approx(floor, abs=1e-5)


def test_mdd_is_the_target_quantile_times_the_standard_error() -> None:
    decomposition = _decomposition(within_sample_var=0.25, interaction_var=0.1)

    mdd = _mdd_at_target(
        decomposition, n_tasks=10, num_seeds=2, target_prob=0.8
    )
    expected = _normal_ppf(0.8) * _paired_diff_se(
        decomposition, n_tasks=10, num_seeds=2
    )

    assert mdd == pytest.approx(expected)


def test_mdd_scales_as_one_over_sqrt_task_count() -> None:
    # MDD is proportional to 1/sqrt(n_tasks), so doubling the task count
    # divides it by sqrt(2). Exact up to floating-point error.
    decomposition = _decomposition(within_sample_var=0.25, interaction_var=0.1)

    small = _mdd_at_target(
        decomposition, n_tasks=10, num_seeds=2, target_prob=0.8
    )
    large = _mdd_at_target(
        decomposition, n_tasks=20, num_seeds=2, target_prob=0.8
    )

    assert small / large == pytest.approx(math.sqrt(2.0), rel=1e-9)


def test_mdd_grows_with_the_required_power() -> None:
    decomposition = _decomposition(within_sample_var=0.25, interaction_var=0.1)

    at_80 = _mdd_at_target(
        decomposition, n_tasks=10, num_seeds=2, target_prob=0.80
    )
    at_95 = _mdd_at_target(
        decomposition, n_tasks=10, num_seeds=2, target_prob=0.95
    )

    assert at_95 > at_80
    assert at_95 / at_80 == pytest.approx(_normal_ppf(0.95) / _normal_ppf(0.80))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"alpha": math.nan},
        {"target_prob": 0.5},
        {"target_prob": 1.0},
        {"target_prob": math.nan},
        {"sample_cap": 0},
    ],
)
def test_power_config_rejects_out_of_contract_values(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        PowerConfig(**kwargs)


def test_power_config_defaults_match_the_published_constants() -> None:
    config = PowerConfig()

    assert config.alpha == DEFAULT_ALPHA
    assert config.target_prob == DEFAULT_TARGET_PROB
    assert config.sample_cap == DEFAULT_SAMPLE_CAP


def test_analyze_power_reports_headroom_and_the_alpha_scaled_target_gap() -> None:
    result = analyze_power(
        naive_per_task=(0.2, 0.4, 0.6),
        ceiling_per_task=(0.5, 0.7, 0.9),
        pool_ceiling=3,
        anchor_samples=2,
        config=PowerConfig(sample_cap=2),
    )

    assert result.naive_mean == pytest.approx(0.4)
    assert result.ceiling_mean == pytest.approx(0.7)
    assert result.certified_headroom == pytest.approx(0.3)
    assert result.target_gap == pytest.approx(DEFAULT_ALPHA * 0.3)
    assert result.pool_ceiling == 3


def test_analyze_power_floors_headroom_at_zero_when_the_ceiling_is_worse() -> None:
    result = analyze_power(
        naive_per_task=(0.8, 0.9),
        ceiling_per_task=(0.2, 0.3),
        pool_ceiling=2,
        anchor_samples=1,
    )

    assert result.certified_headroom == 0.0
    assert result.target_gap == 0.0


def test_power_surface_covers_the_full_task_by_repeat_grid() -> None:
    result = analyze_power(
        naive_per_task=(0.2, 0.4, 0.6),
        ceiling_per_task=(0.5, 0.7, 0.9),
        pool_ceiling=3,
        anchor_samples=2,
        config=PowerConfig(sample_cap=2),
    )

    assert {(point.n_tasks, point.num_seeds) for point in result.surface} == {
        (tasks, seeds) for tasks in (1, 2, 3) for seeds in (1, 2)
    }
    assert all(
        point.calls == point.n_tasks * point.num_seeds
        for point in result.surface
    )


def test_power_surface_respects_the_sample_cap_and_pool_ceiling() -> None:
    result = analyze_power(
        naive_per_task=(0.2, 0.4),
        ceiling_per_task=(0.5, 0.7),
        pool_ceiling=5,
        anchor_samples=1,
        config=PowerConfig(sample_cap=3),
    )

    assert max(point.num_seeds for point in result.surface) == 3
    assert max(point.n_tasks for point in result.surface) == 5
    assert min(point.num_seeds for point in result.surface) == 1
    assert min(point.n_tasks for point in result.surface) == 1


def test_power_surface_is_monotone_in_tasks_and_repeats() -> None:
    # More tasks or more repeats can only shrink the detectable difference.
    result = analyze_power(
        naive_per_task=(0.2, 0.4, 0.6, 0.8),
        ceiling_per_task=(0.5, 0.7, 0.9, 0.95),
        pool_ceiling=4,
        anchor_samples=2,
        config=PowerConfig(sample_cap=3),
    )
    mdd = {
        (point.n_tasks, point.num_seeds): point.mdd_at_target
        for point in result.surface
    }

    for tasks in (1, 2, 3, 4):
        for seeds in (1, 2, 3):
            if tasks < 4:
                assert mdd[(tasks + 1, seeds)] < mdd[(tasks, seeds)]
            if seeds < 3:
                assert mdd[(tasks, seeds + 1)] < mdd[(tasks, seeds)]


def test_analyze_power_accepts_a_single_calibrated_task() -> None:
    # A one-task design is degenerate but supported: between-task variance
    # is unidentifiable and reported as zero rather than rejected.
    result = analyze_power(
        naive_per_task=(0.4,),
        ceiling_per_task=(0.9,),
        pool_ceiling=1,
        anchor_samples=1,
    )

    assert result.decomposition.n_tasks_observed == 1
    assert result.decomposition.between_task_var == 0.0
    assert len(result.surface) == DEFAULT_SAMPLE_CAP


@pytest.mark.parametrize(
    ("naive", "ceiling"),
    [
        ((), ()),
        ((0.2, 0.4), (0.5,)),
        ((0.2, 1.5), (0.5, 0.7)),
        ((0.2, -0.1), (0.5, 0.7)),
        ((0.2, math.nan), (0.5, 0.7)),
        ((0.2, 0.4), (0.5, math.inf)),
    ],
)
def test_analyze_power_rejects_malformed_per_task_scores(
    naive: tuple[float, ...], ceiling: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="per-task scores must be"):
        analyze_power(
            naive_per_task=naive,
            ceiling_per_task=ceiling,
            pool_ceiling=2,
            anchor_samples=1,
        )


@pytest.mark.parametrize(
    ("pool_ceiling", "anchor_samples", "message"),
    [
        (0, 1, "pool_ceiling must be at least 1"),
        (2, 0, "anchor_samples must be at least 1"),
    ],
)
def test_analyze_power_rejects_degenerate_budget_inputs(
    pool_ceiling: int, anchor_samples: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_power(
            naive_per_task=(0.2, 0.4),
            ceiling_per_task=(0.5, 0.7),
            pool_ceiling=pool_ceiling,
            anchor_samples=anchor_samples,
        )
