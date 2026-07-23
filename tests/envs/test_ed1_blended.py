"""Tests for the ed1 weighted-blend reward (task 22): formula properties,
per-task composition, identity fold, retro-compute utility. No network.
"""

from __future__ import annotations

import pytest

from whetstone.envs.ed1_blended import (
    BLENDED_METRIC_ID,
    DEFAULT_COMPRESSION_WEIGHT,
    BoundedCompressionMetricConfig,
    blend_per_task,
    blended_reward,
    blended_reward_from_components,
    compression_score,
)


def _cfg(weight: float = 0.10, lo: float = 0.01, hi: float = 4.0):
    return BoundedCompressionMetricConfig(
        weight=weight, min_compression_ratio=lo, max_compression_ratio=hi
    )


# --- formula properties ------------------------------------------------------


def test_compression_score_bounds_and_direction() -> None:
    cfg = _cfg()
    # A ratio AT max -> score 0 (worst); AT min -> score 1 (best).
    assert compression_score(4.0, cfg) == pytest.approx(0.0)
    assert compression_score(0.01, cfg) == pytest.approx(1.0)
    # Clamped beyond the bounds.
    assert compression_score(10.0, cfg) == pytest.approx(0.0)
    assert compression_score(0.0, cfg) == pytest.approx(1.0)
    # Monotonic: a LOWER ratio (tighter) -> a HIGHER score.
    assert compression_score(0.5, cfg) > compression_score(2.0, cfg)


def test_reward_output_always_in_unit_interval() -> None:
    cfg = _cfg(weight=0.5)
    for pr in (0.0, 0.25, 0.5, 0.75, 1.0):
        for cr in (None, 0.0, 0.5, 1.0, 2.0, 4.0, 100.0):
            r = blended_reward(
                pass_rate=pr, compression_ratio=cr, config=cfg
            )
            assert 0.0 <= r <= 1.0


def test_pass_zero_gives_zero_regardless_of_compression() -> None:
    cfg = _cfg(weight=0.9)
    for cr in (None, 0.0, 0.5, 4.0, 100.0):
        assert blended_reward(
            pass_rate=0.0, compression_ratio=cr, config=cfg
        ) == pytest.approx(0.0)


def test_pass_one_lands_in_one_minus_w_to_one() -> None:
    w = 0.30
    cfg = _cfg(weight=w)
    # Worst compression (ratio at max) -> reward == 1 - w.
    assert blended_reward(
        pass_rate=1.0, compression_ratio=4.0, config=cfg
    ) == pytest.approx(1.0 - w)
    # Best compression (ratio at min) -> reward == 1.
    assert blended_reward(
        pass_rate=1.0, compression_ratio=0.01, config=cfg
    ) == pytest.approx(1.0)
    # A mid ratio lands strictly inside (1-w, 1).
    mid = blended_reward(pass_rate=1.0, compression_ratio=1.0, config=cfg)
    assert (1.0 - w) < mid < 1.0


def test_weight_zero_degenerates_to_pure_pass_rate() -> None:
    cfg = _cfg(weight=0.0)
    for pr in (0.0, 0.4, 1.0):
        for cr in (None, 0.0, 2.0, 4.0):
            assert blended_reward(
                pass_rate=pr, compression_ratio=cr, config=cfg
            ) == pytest.approx(pr)


def test_missing_compression_falls_back_to_pass_only() -> None:
    cfg = _cfg(weight=0.5)
    # No compression sample -> pass-only (no fabricated credit, no zeroing).
    assert blended_reward(
        pass_rate=0.8, compression_ratio=None, config=cfg
    ) == pytest.approx(0.8)


def test_degenerate_bounds_give_no_compression_credit() -> None:
    cfg = _cfg(weight=0.5, lo=2.0, hi=2.0)  # max == min
    assert compression_score(1.0, cfg) == pytest.approx(0.0)
    # reward = pass * ((1-w) + w*0) = pass * (1-w)
    assert blended_reward(
        pass_rate=1.0, compression_ratio=1.0, config=cfg
    ) == pytest.approx(0.5)


# --- per-task composition + pairing ------------------------------------------


def test_blend_per_task_composes_per_task() -> None:
    cfg = _cfg(weight=0.5)
    per_task_pass = (1.0, 0.5, 0.0)
    per_task_comp = (0.01, 2.0, 0.5)  # best, mid, mid
    blended = blend_per_task(per_task_pass, per_task_comp, cfg)
    assert len(blended) == 3
    # Task 0: pass 1.0, best compression -> 1.0.
    assert blended[0] == pytest.approx(1.0)
    # Task 2: pass 0.0 -> 0.0 regardless.
    assert blended[2] == pytest.approx(0.0)
    # The mean over tasks is the aggregate blended reward.
    assert sum(blended) / len(blended) == pytest.approx(
        (blended[0] + blended[1] + blended[2]) / 3
    )


def test_blend_per_task_missing_compression_is_pass_only() -> None:
    cfg = _cfg(weight=0.5)
    blended = blend_per_task((1.0, 0.6), (None, 0.01), cfg)
    assert blended[0] == pytest.approx(1.0)   # pass-only fallback
    assert blended[1] == pytest.approx(0.6)   # pass 0.6, best compression


def test_blend_per_task_misaligned_vectors_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        blend_per_task((1.0,), (0.5, 0.5), _cfg())


# --- identity fold -----------------------------------------------------------


def test_metric_identity_folds_metric_id_weight_bounds() -> None:
    base = _cfg(weight=0.10)
    diff_w = _cfg(weight=0.05)
    diff_lo = _cfg(weight=0.10, lo=0.02)
    diff_hi = _cfg(weight=0.10, hi=5.0)
    keys = {
        base.identity_key(), diff_w.identity_key(),
        diff_lo.identity_key(), diff_hi.identity_key(),
    }
    assert len(keys) == 4  # all distinct
    assert BLENDED_METRIC_ID in base.identity_key()
    assert "w=0.1" in base.identity_key()
    # Same config -> same key (stable).
    assert base.identity_key() == _cfg(weight=0.10).identity_key()


def test_default_weight_is_the_named_start() -> None:
    assert DEFAULT_COMPRESSION_WEIGHT == 0.10
    assert BoundedCompressionMetricConfig().weight == 0.10
    assert BoundedCompressionMetricConfig().metric_id == BLENDED_METRIC_ID


def test_weight_out_of_range_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BoundedCompressionMetricConfig(weight=1.5)
    with pytest.raises(ValidationError):
        BoundedCompressionMetricConfig(weight=-0.1)


# --- retro-compute (analysis-side, derived) ----------------------------------


def test_retro_compute_matches_live_blend() -> None:
    # The DERIVED retro-compute reproduces the live blend from recorded
    # components under any weight -- so past rows are comparable without
    # re-driving.
    cfg = _cfg(weight=0.2, lo=0.01, hi=4.0)
    live = blended_reward(pass_rate=0.9, compression_ratio=1.3, config=cfg)
    derived = blended_reward_from_components(
        pass_rate=0.9, compression_ratio=1.3, weight=0.2,
        min_compression_ratio=0.01, max_compression_ratio=4.0,
    )
    assert derived == pytest.approx(live)


def test_retro_compute_varies_with_weight() -> None:
    # A recorded (pass, compression) row scores differently under different
    # weights -- the point of the retro-compute (comparable under any weight).
    row = dict(pass_rate=0.8, compression_ratio=2.0)
    w0 = blended_reward_from_components(**row, weight=0.0)
    w1 = blended_reward_from_components(**row, weight=0.5)
    assert w0 == pytest.approx(0.8)  # weight 0 -> pure pass
    assert w1 != pytest.approx(0.8)  # weight 0.5 -> compression matters


# --- retro-compute over recorded rows (derived) ------------------------------


def test_retro_blend_recorded_rows() -> None:
    from whetstone.envs.ed1_blended import retro_blend_recorded_rows

    rows: list[dict[str, object]] = [
        {"pass_rate": 1.0, "compression_ratio": 0.01},  # best comp
        {"pass_rate": 0.0, "compression_ratio": 0.5},   # pass 0 -> 0
        {"pass_rate": 0.5, "compression_ratio": None},  # pass-only fallback
        {"pass_rate": None, "compression_ratio": 1.0},  # skipped (no pass)
    ]
    out = retro_blend_recorded_rows(rows, weight=0.5)
    assert out["derived"] is True
    assert out["rows_used"] == 3
    assert out["rows_skipped"] == 1
    blends = out["per_row_blended"]
    assert isinstance(blends, list)
    assert blends[0] == pytest.approx(1.0)   # pass 1, best comp
    assert blends[1] == pytest.approx(0.0)   # pass 0
    assert blends[2] == pytest.approx(0.5)   # pass-only
    assert out["mean_blended"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_retro_blend_varies_with_weight() -> None:
    from whetstone.envs.ed1_blended import retro_blend_recorded_rows

    rows: list[dict[str, object]] = [
        {"pass_rate": 1.0, "compression_ratio": 2.0}
    ]
    w0 = retro_blend_recorded_rows(rows, weight=0.0)["mean_blended"]
    w5 = retro_blend_recorded_rows(rows, weight=0.5)["mean_blended"]
    assert w0 == pytest.approx(1.0)   # weight 0 -> pure pass
    assert isinstance(w5, float) and w5 < 1.0  # weight 0.5 penalizes ratio 2.0
