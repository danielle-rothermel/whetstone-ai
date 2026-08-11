from __future__ import annotations

from typing import Literal

from dr_store import ObjectStore

from whetstone.evaluation.metrics.blended import (
    DEFAULT_COMPRESSION_WEIGHT,
    BoundedCompressionBlendConfig,
    blend_per_task,
    blended_reward,
    blended_reward_from_components,
    compression_score,
    retro_blend_recorded_rows,
)
from whetstone.evaluation.preview.persisted import load_aggregate_value
from whetstone.experiment.reward import RewardRef

BLENDED_METRIC_ID = "primary_score_with_bounded_compression_penalty"


class BoundedCompressionMetricConfig(BoundedCompressionBlendConfig):
    """ED1 identity-bearing blended-reward configuration."""

    metric_id: Literal["primary_score_with_bounded_compression_penalty"] = (
        BLENDED_METRIC_ID
    )

    def identity_key(self) -> str:
        """Fold ED1 metric identity with the shared blend parameters."""
        return f"{self.metric_id}|{self.blend_identity_key()}"


def ed1_blended_aggregate_values(
    store: ObjectStore,
    reward_ref: RewardRef,
) -> tuple[float | None, float | None]:
    """Load primary and compression aggregate values from a blended reward."""
    if len(reward_ref.record.evidence_refs) != 2:
        raise RuntimeError(
            "ED1 blended Reward must cite primary and compression aggregates"
        )
    return (
        load_aggregate_value(store, reward_ref.record.evidence_refs[0]),
        load_aggregate_value(store, reward_ref.record.evidence_refs[1]),
    )


__all__ = [
    "BLENDED_METRIC_ID",
    "DEFAULT_COMPRESSION_WEIGHT",
    "BoundedCompressionMetricConfig",
    "blend_per_task",
    "blended_reward",
    "blended_reward_from_components",
    "compression_score",
    "ed1_blended_aggregate_values",
    "retro_blend_recorded_rows",
]
