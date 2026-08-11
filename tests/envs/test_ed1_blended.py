from __future__ import annotations

import pytest
from pydantic import ValidationError

from whetstone.envs.ed1_blended import (
    BLENDED_METRIC_ID,
    DEFAULT_COMPRESSION_WEIGHT,
    BoundedCompressionMetricConfig,
    blended_reward,
)
from whetstone.evaluation.metrics.blended import blended_reward_from_components


def _cfg(weight: float = 0.10, lo: float = 0.01, hi: float = 4.0):
    return BoundedCompressionMetricConfig(
        weight=weight, min_compression_ratio=lo, max_compression_ratio=hi
    )


def test_metric_identity_folds_metric_id_weight_bounds() -> None:
    base = _cfg(weight=0.10)
    diff_w = _cfg(weight=0.05)
    diff_lo = _cfg(weight=0.10, lo=0.02)
    diff_hi = _cfg(weight=0.10, hi=5.0)
    keys = {
        base.identity_key(),
        diff_w.identity_key(),
        diff_lo.identity_key(),
        diff_hi.identity_key(),
    }
    assert len(keys) == 4
    assert BLENDED_METRIC_ID in base.identity_key()
    assert "w=0.1" in base.identity_key()
    assert base.identity_key() == _cfg(weight=0.10).identity_key()


def test_default_ed1_blend_config() -> None:
    assert DEFAULT_COMPRESSION_WEIGHT == 0.10
    assert BoundedCompressionMetricConfig().weight == 0.10
    assert BoundedCompressionMetricConfig().metric_id == BLENDED_METRIC_ID


def test_ed1_config_accepts_shared_blend_math() -> None:
    cfg = _cfg(weight=0.2)
    live = blended_reward(primary_score=0.9, compression_ratio=1.3, config=cfg)
    derived = blended_reward_from_components(
        primary_score=0.9,
        compression_ratio=1.3,
        weight=0.2,
    )
    assert live == pytest.approx(derived)


def test_ed1_config_rejects_invalid_blend_parameters() -> None:
    with pytest.raises(ValidationError):
        BoundedCompressionMetricConfig(weight=1.5)
