from __future__ import annotations

import math
import re
from statistics import NormalDist
from typing import cast

import pytest

from whetstone.evaluation.analysis.power import (
    DEFAULT_SAMPLE_CAP,
    PowerConfig,
    PowerResult,
    PowerSurfacePoint,
    analyze_power,
)

_C11_NAIVE = tuple([0.0] * 10)
_C11_CEILING = (1.0, 1.0, 0.667, 1.0, 1.0, 1.0, 0.667, 1.0, 1.0, 1.0)

_C22_NAIVE = (
    0.667,
    0.667,
    0.667,
    0.333,
    0.667,
    1.0,
    0.667,
    0.667,
    0.333,
    0.667,
    0.667,
    0.667,
)
_C22_CEILING = (
    1.0,
    0.667,
    1.0,
    1.0,
    1.0,
    1.0,
    0.667,
    1.0,
    0.667,
    1.0,
    1.0,
    0.667,
)

_PROBABILITY_ERROR = (
    "naive/ceiling per-task scores must be one-dimensional, "
    "finite probabilities in [0, 1]"
)


def _surface_mdd(result: PowerResult) -> dict[tuple[int, int], float]:
    return {
        (point.n_tasks, point.num_samples): point.mdd_at_target
        for point in result.surface
    }


def test_variance_decomposition_is_empirical_and_signed() -> None:
    res = analyze_power(
        naive_per_task=_C22_NAIVE,
        ceiling_per_task=_C22_CEILING,
        pool_ceiling=12,
        anchor_samples=3,
    )
    d = res.decomposition
    assert 0.2 < d.within_sample_var <= 0.25
    assert d.interaction_var >= 0.0
    assert d.between_task_var >= 0.0
    assert d.within_dominates is True
    assert d.noise_verdict == "within-task sample noise dominates"


def test_c11_shape_between_task_dominates() -> None:
    res = analyze_power(
        naive_per_task=_C11_NAIVE,
        ceiling_per_task=_C11_CEILING,
        pool_ceiling=10,
        anchor_samples=3,
    )
    d = res.decomposition
    assert d.within_dominates is False
    assert d.noise_verdict == "between-task noise dominates"
    assert res.certified_headroom > 0.0
    assert res.target_gap == pytest.approx(0.25 * res.certified_headroom)


def test_mdd_golden_flat_anchors_hit_the_interaction_floor() -> None:
    """Golden closed-form MDDs for the fully-flat degenerate anchors.

    naive = ceiling = (0.5,) * 4 at 1 anchor sample gives, by hand:
    base_rate = 0.5, within = 0.25, between = max(0, 0 - 0.25/2) = 0, and
    interaction = max(0.1 * 0.25, max(0, 0 - 2 * 0.25)) = 0.025 (the 10%
    floor). MDD(n, s) = Phi^-1(0.8) * sqrt((0.025 + 0.5/s) / n) with
    z = 0.8416212335729143.
    """
    flat = (0.5, 0.5, 0.5, 0.5)
    res = analyze_power(
        naive_per_task=flat,
        ceiling_per_task=flat,
        pool_ceiling=4,
        anchor_samples=1,
        config=PowerConfig(sample_cap=16),
    )
    d = res.decomposition
    assert d.base_rate == 0.5
    assert d.within_sample_var == 0.25
    assert d.between_task_var == 0.0
    assert d.interaction_var == pytest.approx(0.025)
    mdd = _surface_mdd(res)
    assert mdd[(4, 1)] == pytest.approx(0.3049062593324451)
    assert mdd[(4, 4)] == pytest.approx(0.162979251072122)
    assert mdd[(4, 16)] == pytest.approx(0.09980400094691178)


def test_mdd_golden_shifted_anchors_and_target_gap() -> None:
    """Golden case with real headroom: naive (0.2, 0.4) vs ceiling (0.6, 0.8).

    At 2 anchor samples, by hand: base_rate = 0.3, within = 0.21,
    within_obs = 0.20, midpoints (0.4, 0.6) give raw_between = 0.02 so
    between = max(0, 0.02 - 0.20/4) = 0; constant diff (0.4, 0.4) gives
    raw_interaction = 0 so interaction = 0.1 * 0.21 = 0.021. Headroom is
    0.7 - 0.3 = 0.4 and target_gap = 0.25 * 0.4 = 0.1.
    MDD(n, s) = Phi^-1(0.8) * sqrt((0.021 + 0.42/s) / n).
    """
    res = analyze_power(
        naive_per_task=(0.2, 0.4),
        ceiling_per_task=(0.6, 0.8),
        pool_ceiling=2,
        anchor_samples=2,
        config=PowerConfig(sample_cap=16),
    )
    assert res.certified_headroom == pytest.approx(0.4)
    assert res.target_gap == pytest.approx(0.1)
    d = res.decomposition
    assert d.within_sample_var == pytest.approx(0.21)
    assert d.interaction_var == pytest.approx(0.021)
    assert d.between_task_var == 0.0
    mdd = _surface_mdd(res)
    assert mdd[(2, 1)] == pytest.approx(0.3952036808110156)
    assert mdd[(2, 2)] == pytest.approx(0.286027424808342)
    assert mdd[(1, 16)] == pytest.approx(0.18294375559946702)


def test_mdd_golden_interaction_dominated_anchors() -> None:
    """Golden case where the task-x-candidate interaction survives.

    naive (0.0, 0.5) vs ceiling (1.0, 0.5) at 1 anchor sample, by hand:
    base_rate = 0.25, within = 0.1875, within_obs = 0.125; the paired diff
    (1.0, 0.0) gives raw_interaction = 0.5, so interaction =
    max(0.01875, 0.5 - 0.25) = 0.25.
    MDD(n, s) = z * sqrt((0.25 + 0.375/s) / n).
    """
    naive = (0.0, 0.5)
    ceiling = (1.0, 0.5)
    at_default = analyze_power(
        naive_per_task=naive,
        ceiling_per_task=ceiling,
        pool_ceiling=2,
        anchor_samples=1,
    )
    assert at_default.decomposition.interaction_var == pytest.approx(0.25)
    mdd = _surface_mdd(at_default)
    assert mdd[(2, 1)] == pytest.approx(0.4704805723940662)
    assert mdd[(2, 16)] == pytest.approx(0.3111936478105049)

    at_stricter_target = analyze_power(
        naive_per_task=naive,
        ceiling_per_task=ceiling,
        pool_ceiling=2,
        anchor_samples=1,
        config=PowerConfig(target_prob=0.9),
    )
    stricter = _surface_mdd(at_stricter_target)
    assert stricter[(2, 1)] == pytest.approx(0.7164091043072511)


def test_mdd_scaling_laws_hold_exactly_on_the_surface() -> None:
    res = analyze_power(
        naive_per_task=_C22_NAIVE,
        ceiling_per_task=_C22_CEILING,
        pool_ceiling=12,
        anchor_samples=3,
        config=PowerConfig(sample_cap=8),
    )
    mdd = _surface_mdd(res)
    # Quadrupling the task count exactly halves every closed-form MDD.
    for num_samples in (1, 4, 8):
        assert mdd[(12, num_samples)] == pytest.approx(
            mdd[(3, num_samples)] / 2.0
        )
    # More samples shrink the MDD monotonically toward the interaction floor.
    d = res.decomposition
    z = NormalDist().inv_cdf(res.config.target_prob)
    floor = z * math.sqrt(d.interaction_var / 12.0)
    at_n = [mdd[(12, s)] for s in range(1, 9)]
    assert at_n == sorted(at_n, reverse=True)
    assert all(value > floor for value in at_n)


def test_sample_cap_changes_the_grid_and_defaults_to_sixteen() -> None:
    low = analyze_power(
        naive_per_task=_C22_NAIVE,
        ceiling_per_task=_C22_CEILING,
        pool_ceiling=12,
        anchor_samples=3,
        config=PowerConfig(sample_cap=6),
    )
    default = analyze_power(
        naive_per_task=_C22_NAIVE,
        ceiling_per_task=_C22_CEILING,
        pool_ceiling=12,
        anchor_samples=3,
    )
    assert max(point.num_samples for point in low.surface) == 6
    assert DEFAULT_SAMPLE_CAP == 16
    assert max(point.num_samples for point in default.surface) == 16


def test_surface_covers_full_grid_and_cost_model() -> None:
    res = analyze_power(
        naive_per_task=_C22_NAIVE,
        ceiling_per_task=_C22_CEILING,
        pool_ceiling=12,
        anchor_samples=3,
        config=PowerConfig(sample_cap=10),
    )
    assert len(res.surface) == 12 * 10
    for row in res.surface:
        assert isinstance(row, PowerSurfacePoint)
        assert row.calls == row.n_tasks * row.num_samples
        assert row.mdd_at_target > 0.0


def test_result_contains_every_input_needed_to_rederive_surface() -> None:
    res = analyze_power(
        naive_per_task=_C22_NAIVE,
        ceiling_per_task=_C22_CEILING,
        pool_ceiling=12,
        anchor_samples=3,
    )
    rerun = analyze_power(
        naive_per_task=_C22_NAIVE,
        ceiling_per_task=_C22_CEILING,
        pool_ceiling=res.pool_ceiling,
        anchor_samples=res.decomposition.anchor_samples,
        config=res.config,
    )
    assert rerun == res


def test_zero_headroom_gives_zero_target_gap() -> None:
    flat = (0.5, 0.5, 0.5, 0.5)
    res = analyze_power(
        naive_per_task=flat,
        ceiling_per_task=flat,
        pool_ceiling=4,
        anchor_samples=3,
    )
    assert res.certified_headroom == 0.0
    assert res.target_gap == 0.0


@pytest.mark.parametrize("arm", ["naive", "ceiling"])
@pytest.mark.parametrize(
    "invalid_probability",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_invalid_probability_rejected_in_either_arm(
    arm: str, invalid_probability: float
) -> None:
    valid = (0.25, 0.75)
    invalid = (0.25, invalid_probability)
    naive = invalid if arm == "naive" else valid
    ceiling = invalid if arm == "ceiling" else valid

    with pytest.raises(ValueError, match=r"finite probabilities in \[0, 1\]"):
        analyze_power(
            naive_per_task=naive,
            ceiling_per_task=ceiling,
            pool_ceiling=2,
            anchor_samples=3,
        )


def test_non_vector_probability_arrays_rejected() -> None:
    matrix = cast(tuple[float, ...], ((0.2, 0.3), (0.4, 0.5)))
    vector = (0.2, 0.3)

    with pytest.raises(ValueError, match="one-dimensional"):
        analyze_power(
            naive_per_task=matrix,
            ceiling_per_task=vector,
            pool_ceiling=2,
            anchor_samples=3,
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        analyze_power(
            naive_per_task=vector,
            ceiling_per_task=matrix,
            pool_ceiling=2,
            anchor_samples=3,
        )


def test_mismatched_or_empty_probability_arrays_rejected() -> None:
    with pytest.raises(ValueError, match=re.escape(_PROBABILITY_ERROR)):
        analyze_power(
            naive_per_task=(),
            ceiling_per_task=(),
            pool_ceiling=4,
            anchor_samples=3,
        )
    with pytest.raises(ValueError, match=re.escape(_PROBABILITY_ERROR)):
        analyze_power(
            naive_per_task=(0.5, 0.5),
            ceiling_per_task=(0.5,),
            pool_ceiling=4,
            anchor_samples=3,
        )


def test_pool_ceiling_and_anchor_samples_must_be_positive() -> None:
    with pytest.raises(ValueError, match="pool_ceiling must be at least 1"):
        analyze_power(
            naive_per_task=(0.5,),
            ceiling_per_task=(0.5,),
            pool_ceiling=0,
            anchor_samples=3,
        )
    with pytest.raises(ValueError, match="anchor_samples must be at least 1"):
        analyze_power(
            naive_per_task=(0.5,),
            ceiling_per_task=(0.5,),
            pool_ceiling=1,
            anchor_samples=0,
        )


@pytest.mark.parametrize(
    "target_prob",
    [float("-inf"), 0.0, 0.5, 1.0, float("inf"), float("nan")],
)
def test_target_probability_must_be_finite_and_above_chance(
    target_prob: float,
) -> None:
    with pytest.raises(ValueError, match=r"target_prob must be in \(0.5, 1\)"):
        PowerConfig(target_prob=target_prob)


@pytest.mark.parametrize("alpha", [float("-inf"), float("inf"), float("nan")])
def test_alpha_must_be_finite(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        PowerConfig(alpha=alpha)


def test_sample_cap_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sample_cap must be at least 1"):
        PowerConfig(sample_cap=0)


def test_constant_arm_shift_is_not_between_task_variance() -> None:
    result = analyze_power(
        naive_per_task=(0.2, 0.2, 0.2, 0.2),
        ceiling_per_task=(0.8, 0.8, 0.8, 0.8),
        pool_ceiling=1,
        anchor_samples=3,
        config=PowerConfig(sample_cap=1),
    )

    assert result.decomposition.between_task_var == 0.0


def test_midpoint_and_interaction_measurement_noise_corrections() -> None:
    result = analyze_power(
        naive_per_task=(0.1, 0.2, 0.7, 0.8),
        ceiling_per_task=(0.1, 0.8, 0.7, 1.0),
        pool_ceiling=1,
        anchor_samples=100,
        config=PowerConfig(sample_cap=1),
    )

    assert result.decomposition.between_task_var == pytest.approx(
        0.11599166666666669
    )
    assert result.decomposition.interaction_var == pytest.approx(0.0773)
