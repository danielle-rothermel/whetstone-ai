"""Unit tests for the pre-run statistical-power analysis (deterministic seed).

No network, no cell -- pure computation over anchor per-task vectors. Covers
the paired-variance addendum: the empirical within-vs-between decomposition,
the 2-D (n x r) surface, the cheapest-achievable recommendation, the
repeat-plateau, and the LOUD pool-limited verdict (never a silent clamp).
"""

from __future__ import annotations

import pytest

from whetstone.runner.power import (
    POWER_ANALYSIS_SCHEMA,
    PowerConfig,
    analyze_power,
)

# c11-like: naive is a perfect 0 (candidates start at the floor), ceiling high.
_C11_NAIVE = tuple([0.0] * 10)
_C11_CEILING = (1.0, 1.0, 0.667, 1.0, 1.0, 1.0, 0.667, 1.0, 1.0, 1.0)

# c22-like: naive mid-range (~0.6, high within-task Bernoulli noise), tight
# headroom -> repeat noise DOMINATES, repeats are the power dial.
_C22_NAIVE = (
    0.667, 0.667, 0.667, 0.333, 0.667, 1.0, 0.667, 0.667, 0.333, 0.667,
    0.667, 0.667,
)
_C22_CEILING = (
    1.0, 0.667, 1.0, 1.0, 1.0, 1.0, 0.667, 1.0, 0.667, 1.0, 1.0, 0.667,
)


def test_determinism_same_seed_same_result() -> None:
    cfg = PowerConfig(seed=123)
    a = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3, config=cfg,
    )
    b = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3, config=cfg,
    )
    assert a.to_artifact(header={}) == b.to_artifact(header={})


def test_variance_decomposition_is_empirical_and_signed() -> None:
    # The decomposition separates within-repeat from between-task/interaction,
    # anchored to the naive operating base rate (NOT the anchor extremes).
    res = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3,
    )
    d = res.decomposition
    # naive mean ~0.63 -> high mid-range within-task Bernoulli variance.
    assert 0.2 < d.within_repeat_var <= 0.25
    assert d.interaction_var >= 0.0
    assert d.between_task_var >= 0.0
    # On c22 repeat noise dominates (the addendum's headline for this shape).
    assert d.within_dominates is True
    assert d.as_dict()["verdict"] == "within-task repeat noise dominates"


def test_c11_shape_between_task_dominates_and_easy() -> None:
    # c11 naive is a perfect 0 -> no within-task Bernoulli noise at the floor;
    # task-to-task heterogeneity dominates and ranking is trivially achievable.
    res = analyze_power(
        naive_per_task=_C11_NAIVE, ceiling_per_task=_C11_CEILING,
        pool_ceiling=10, anchor_repeats=3,
    )
    d = res.decomposition
    assert d.within_dominates is False
    assert res.recommendation.achievable is True
    assert res.recommendation.pool_limited is False


def test_repeats_shrink_mdd_and_a_plateau_is_detected() -> None:
    # Only the within-task repeat noise shrinks with r, so MDD decreases with r
    # at fixed n, then plateaus (the interaction floor does not shrink).
    res = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3, config=PowerConfig(repeat_cap=20),
    )
    # Extract the MDD-vs-r curve at the full pool n.
    at_n = {
        row["repeats"]: row["mdd_at_target"]
        for row in res.surface
        if row["n_tasks"] == 12
    }
    assert at_n[1] > at_n[2] > at_n[3]  # strictly shrinking early
    assert at_n[20] < at_n[1]
    # A plateau is detected within the cap (diminishing returns from repeats).
    assert res.recommendation.repeat_plateau is not None
    assert 1 <= res.recommendation.repeat_plateau <= 20


def test_cheapest_recommendation_meets_target_at_min_cost() -> None:
    res = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3, config=PowerConfig(repeat_cap=20),
    )
    rec = res.recommendation
    if rec.achievable:
        # The recommended cell meets the target (MDD <= target gap) ...
        assert rec.achieved_mdd <= rec.target_gap + 1e-9
        # ... and no CHEAPER (n*r) surface cell also meets it.
        rec_cost = rec.recommended_n_tasks * rec.recommended_repeats
        cheaper_meeting = [
            row for row in res.surface
            if row["mdd_at_target"] <= rec.target_gap
            and row["calls"] < rec_cost
        ]
        assert not cheaper_meeting


def test_pool_limited_verdict_is_loud_not_silently_clamped() -> None:
    # Tiny headroom + small pool + low repeat cap: NO (n,r) reaches the target.
    naive = (0.5, 0.6, 0.55, 0.45, 0.5, 0.6)
    ceiling = (0.6, 0.65, 0.6, 0.5, 0.58, 0.68)  # headroom ~0.06
    res = analyze_power(
        naive_per_task=naive, ceiling_per_task=ceiling,
        pool_ceiling=6, anchor_repeats=3,
        config=PowerConfig(alpha=0.25, repeat_cap=5),
    )
    rec = res.recommendation
    assert rec.achievable is False
    assert rec.pool_limited is True
    # The BEST achievable (n,r) + its MDD are recorded (never a silent clamp).
    assert rec.best_achievable_mdd > rec.target_gap
    assert rec.best_n_tasks == 6  # spends the whole pool
    assert rec.best_repeats == 5  # spends the whole repeat cap
    assert rec.achieved_mdd == pytest.approx(rec.best_achievable_mdd)


def test_repeat_cap_changes_the_grid_and_the_c22_verdict() -> None:
    # The repeat cap bounds the (n x r) grid AND can flip the verdict: c22
    # (repeat-noise-dominated) is POOL-LIMITED at a low cap but ACHIEVABLE once
    # the cap admits enough repeats (~r=11) -- the exact coordinator concern.
    low = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3, config=PowerConfig(repeat_cap=6),
    )
    high = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3, config=PowerConfig(repeat_cap=20),
    )
    # The grid's repeat extent follows the cap.
    assert max(r["repeats"] for r in low.surface) == 6
    assert max(r["repeats"] for r in high.surface) == 20
    # Low cap: pool-limited (best repeats == the cap). High cap: achievable
    # with repeats the low cap could not reach.
    assert low.recommendation.pool_limited is True
    assert low.recommendation.best_repeats == 6
    assert high.recommendation.achievable is True
    assert high.recommendation.recommended_repeats > 6


def test_surface_covers_full_grid_and_cost_model() -> None:
    res = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3,
        config=PowerConfig(repeat_cap=10, per_call_usd=0.0004),
    )
    # Full (n=1..12) x (r=1..10) grid.
    assert len(res.surface) == 12 * 10
    for row in res.surface:
        assert row["calls"] == row["n_tasks"] * row["repeats"]
        assert 0.0 <= row["sim_rank_prob_at_target_gap"] <= 1.0
    # USD cost is populated when a per-call cost is supplied.
    assert res.recommendation.recommended_usd is not None
    assert res.recommendation.recommended_usd == pytest.approx(
        res.recommendation.recommended_calls * 0.0004
    )


def test_artifact_is_fully_rederivable() -> None:
    res = analyze_power(
        naive_per_task=_C22_NAIVE, ceiling_per_task=_C22_CEILING,
        pool_ceiling=12, anchor_repeats=3,
    )
    art = res.to_artifact(header={"cell_id": "copro:c22:a0"})
    assert art["schema"] == POWER_ANALYSIS_SCHEMA
    assert art["cell_id"] == "copro:c22:a0"
    # Everything needed to re-derive: config (incl. seed), headroom, pool
    # ceiling, decomposition, recommendation, and the full surface.
    assert art["config"]["seed"] == res.config.seed
    assert art["config"]["alpha"] == res.config.alpha
    assert art["certified_headroom"] == res.certified_headroom
    assert art["pool_ceiling"] == 12
    assert "variance_decomposition" in art
    assert "recommendation" in art
    assert len(art["surface"]) == len(res.surface)


def test_zero_headroom_gives_zero_target_gap() -> None:
    # A naive == ceiling anchor has no certified headroom -> target gap 0; the
    # analysis still runs (no achievable positive gap), no divide-by-zero.
    flat = (0.5, 0.5, 0.5, 0.5)
    res = analyze_power(
        naive_per_task=flat, ceiling_per_task=flat,
        pool_ceiling=4, anchor_repeats=3,
    )
    assert res.certified_headroom == 0.0
    assert res.recommendation.target_gap == 0.0
    # No positive target gap is achievable; pool-limited is recorded.
    assert res.recommendation.achievable is False


def test_mismatched_or_empty_vectors_rejected() -> None:
    with pytest.raises(ValueError):
        analyze_power(
            naive_per_task=(), ceiling_per_task=(),
            pool_ceiling=4, anchor_repeats=3,
        )
    with pytest.raises(ValueError):
        analyze_power(
            naive_per_task=(0.5, 0.5), ceiling_per_task=(0.5,),
            pool_ceiling=4, anchor_repeats=3,
        )
