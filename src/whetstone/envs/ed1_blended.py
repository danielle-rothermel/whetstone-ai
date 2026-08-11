from __future__ import annotations

from typing import Literal

from whetstone.evaluation.metrics.blended import (
    DEFAULT_COMPRESSION_WEIGHT,
    BoundedCompressionBlendConfig,
    blend_per_task,
    blended_reward,
    blended_reward_from_components,
    compression_score,
    retro_blend_recorded_rows,
)

BLENDED_METRIC_ID = "primary_score_with_bounded_compression_penalty"


class BoundedCompressionMetricConfig(BoundedCompressionBlendConfig):
    """ED1 identity-bearing blended-reward configuration."""

    metric_id: Literal["primary_score_with_bounded_compression_penalty"] = (
        BLENDED_METRIC_ID
    )

    def identity_key(self) -> str:
        """Fold ED1 metric identity with the shared blend parameters."""
        return f"{self.metric_id}|{self.blend_identity_key()}"


__all__ = [
    "BLENDED_METRIC_ID",
    "DEFAULT_COMPRESSION_WEIGHT",
    "BoundedCompressionMetricConfig",
    "blend_per_task",
    "blended_reward",
    "blended_reward_from_components",
    "compression_score",
    "retro_blend_recorded_rows",
]
