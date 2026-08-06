from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BLENDED_METRIC_ID = "primary_score_with_bounded_compression_penalty"

DEFAULT_COMPRESSION_WEIGHT = 0.10


class BoundedCompressionMetricConfig(BaseModel):
    """Identity-bearing ed1 blended-reward configuration.

    ``metric_id`` + ``weight`` + clamp bounds fold into the eval/reward config
    identity via :meth:`identity_key`, so a different weight is a distinct
    comparable-or-not config (visible in traces/cells). ``weight`` in [0, 1];
    ``weight=0`` degenerates to the primary score.
    """

    #: ``allow_inf_nan=False``: a NaN or infinite weight/bound survives every
    #: clamp in :func:`compression_score` and poisons EVERY blended reward
    #: (NaN in, NaN out), silently breaking the documented [0, 1] output
    #: contract. Non-finite values are rejected at construction instead.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    metric_id: Literal["primary_score_with_bounded_compression_penalty"] = (
        BLENDED_METRIC_ID
    )
    weight: float = Field(ge=0.0, le=1.0, default=DEFAULT_COMPRESSION_WEIGHT)
    #: Constrained (not bare floats) so a NaN or negative bound cannot enter:
    #: NaN survives every clamp in :func:`compression_score` (all NaN
    #: comparisons are False) and would poison EVERY blended reward with NaN,
    #: silently breaking this module's documented [0, 1] output contract.
    #: ``ge=0.0`` rejects NaN as a side effect -- every NaN comparison fails --
    #: which is exactly the intent; a compression RATIO is non-negative anyway.
    min_compression_ratio: float = Field(ge=0.0, default=0.01)
    max_compression_ratio: float = Field(ge=0.0, default=4.0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> BoundedCompressionMetricConfig:
        """Reject INVERTED bounds (max < min), which flip the score's sign.

        ``max == min`` stays legal: it is the documented degenerate case
        :func:`compression_score` guards, returning the neutral 0.0 (no
        compression credit). Only an inverted pair is incoherent.
        """
        if self.max_compression_ratio < self.min_compression_ratio:
            raise ValueError(
                "max_compression_ratio must not be below "
                f"min_compression_ratio (got "
                f"min={self.min_compression_ratio}, "
                f"max={self.max_compression_ratio})"
            )
        return self

    def identity_key(self) -> str:
        """A stable identity string folding metric_id + weight + bounds.

        Folded into the ed1 eval/reward config identity so a distinct weight
        (or bounds) is a distinct, visibly-comparable config.

        Identity is pinned to 6 SIGNIFICANT FIGURES (``:.6g``): configs that
        differ only below that resolution are deliberately THE SAME comparable
        config. No production caller varies a weight or bound at sub-1e-7
        resolution, and the pinned format is what every recorded policy
        identity hash was computed under -- changing the precision would be a
        versioned identity change, not a formatting tweak.
        """
        return (
            f"{self.metric_id}"
            f"|w={self.weight:.6g}"
            f"|min={self.min_compression_ratio:.6g}"
            f"|max={self.max_compression_ratio:.6g}"
        )


def compression_score(
    compression_ratio: float, config: BoundedCompressionMetricConfig
) -> float:
    """The bounded compression score in [0, 1] (1 = maximally compressed).

    ``clamped = clamp(compression_ratio, min, max)``;
    ``score = (max - clamped) / (max - min)``. A lower ratio (tighter
    compression) -> a higher score. Guards a degenerate ``max == min`` (returns
    the neutral 0.0, i.e. no compression credit).
    """
    lo = config.min_compression_ratio
    hi = config.max_compression_ratio
    if hi <= lo:
        return 0.0
    clamped = min(max(compression_ratio, lo), hi)
    return (hi - clamped) / (hi - lo)


def blended_reward(
    *,
    primary_score: float,
    compression_ratio: float | None,
    config: BoundedCompressionMetricConfig,
) -> float:
    """The blended reward for one unit (task or arm), in [0, 1].

    ``reward = primary_score * ((1 - w) + w * compression_score)``. A ``None``
    compression (no sample survived) uses the primary score for that unit --
    a missing channel never fabricates compression credit or erases a measured
    value. ``primary_score=0 -> 0`` regardless of compression;
    ``weight=0 -> primary_score`` exactly.
    """
    w = config.weight
    if compression_ratio is None or w == 0.0:
        return primary_score
    cs = compression_score(compression_ratio, config)
    return primary_score * ((1.0 - w) + w * cs)


def blend_per_task(
    per_task_primary: tuple[float, ...],
    per_task_compression: tuple[float | None, ...],
    config: BoundedCompressionMetricConfig,
) -> tuple[float, ...]:
    """Compute aligned per-task blended rewards.

    Each task's blend is its repeats-mean primary score times its compression
    score, so the paired bootstrap operates on per-task blended rewards. The
    two input vectors are aligned by task (same order the eval produced). A
    task with no compression sample falls back to its primary score.
    """
    if len(per_task_primary) != len(per_task_compression):
        raise ValueError(
            "per-task primary and compression vectors must be aligned"
        )
    return tuple(
        blended_reward(primary_score=p, compression_ratio=c, config=config)
        for p, c in zip(per_task_primary, per_task_compression, strict=True)
    )


def blended_reward_from_components(
    *,
    primary_score: float,
    compression_ratio: float | None,
    weight: float = DEFAULT_COMPRESSION_WEIGHT,
    min_compression_ratio: float = 0.01,
    max_compression_ratio: float = 4.0,
) -> float:
    """Derive an analysis-side blend from already-recorded components.

    Recomputes the blended reward from a recorded (pass, compression) row under
    any weight and bounds without re-driving. It reads recorded measurements
    and never drives a call.
    """
    config = BoundedCompressionMetricConfig(
        weight=weight,
        min_compression_ratio=min_compression_ratio,
        max_compression_ratio=max_compression_ratio,
    )
    return blended_reward(
        primary_score=primary_score,
        compression_ratio=compression_ratio,
        config=config,
    )


def retro_blend_recorded_rows(
    rows: list[dict[str, object]],
    *,
    weight: float = DEFAULT_COMPRESSION_WEIGHT,
    min_compression_ratio: float = 0.01,
    max_compression_ratio: float = 4.0,
    primary_key: str = "primary_score",
    compression_key: str = "compression_ratio",
) -> dict[str, object]:
    """Derive blends over already-recorded rows without re-driving calls.

    Reads recorded per-task/per-arm rows -- each a mapping carrying the two
    recorded components (``primary_score`` + ``compression_ratio``, from any
    /anchor artifact) -- and recomputes the blended reward per row under the
    given weight/bounds, WITHOUT re-driving. Returns the per-row blends + their
    mean. It never drives a call; a row missing a pass value is skipped
    (reported in ``skipped``), and missing compression uses the primary score.
    """
    config = BoundedCompressionMetricConfig(
        weight=weight,
        min_compression_ratio=min_compression_ratio,
        max_compression_ratio=max_compression_ratio,
    )
    blends: list[float] = []
    skipped = 0
    for row in rows:
        pr = row.get(primary_key)
        if pr is None or not isinstance(pr, int | float):
            skipped += 1
            continue
        cr = row.get(compression_key)
        cr_val = float(cr) if isinstance(cr, int | float) else None
        blends.append(
            blended_reward(
                primary_score=float(pr),
                compression_ratio=cr_val,
                config=config,
            )
        )
    return {
        "derived": True,
        "weight": weight,
        "identity_key": config.identity_key(),
        "per_row_blended": blends,
        "mean_blended": (sum(blends) / len(blends)) if blends else None,
        "rows_used": len(blends),
        "rows_skipped": skipped,
    }


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
