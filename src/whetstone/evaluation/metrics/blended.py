from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_COMPRESSION_WEIGHT = 0.10


class BoundedCompressionBlendConfig(BaseModel):
    """Identity-bearing bounded compression blend parameters.

    ``weight`` in [0, 1]; ``weight=0`` degenerates to the primary score.
    """

    #: ``allow_inf_nan=False``: a NaN or infinite weight/bound survives every
    #: clamp in :func:`compression_score` and poisons EVERY blended reward
    #: (NaN in, NaN out), silently breaking the documented [0, 1] output
    #: contract. Non-finite values are rejected at construction instead.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    weight: float = Field(ge=0.0, le=1.0, default=DEFAULT_COMPRESSION_WEIGHT)
    min_compression_ratio: float = Field(ge=0.0, default=0.01)
    max_compression_ratio: float = Field(ge=0.0, default=4.0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> BoundedCompressionBlendConfig:
        if self.max_compression_ratio < self.min_compression_ratio:
            raise ValueError(
                "max_compression_ratio must not be below "
                f"min_compression_ratio (got "
                f"min={self.min_compression_ratio}, "
                f"max={self.max_compression_ratio})"
            )
        return self

    def blend_identity_key(self) -> str:
        """Stable identity for weight and bounds at six significant figures."""
        return (
            f"w={self.weight:.6g}"
            f"|min={self.min_compression_ratio:.6g}"
            f"|max={self.max_compression_ratio:.6g}"
        )


def compression_score(
    compression_ratio: float, config: BoundedCompressionBlendConfig
) -> float:
    """The bounded compression score in [0, 1] (1 = maximally compressed)."""
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
    config: BoundedCompressionBlendConfig,
) -> float:
    """The blended reward for one unit (task or arm), in [0, 1]."""
    w = config.weight
    if compression_ratio is None or w == 0.0:
        return primary_score
    cs = compression_score(compression_ratio, config)
    return primary_score * ((1.0 - w) + w * cs)


def blend_per_task(
    per_task_primary: tuple[float, ...],
    per_task_compression: tuple[float | None, ...],
    config: BoundedCompressionBlendConfig,
) -> tuple[float, ...]:
    """Compute aligned per-task blended rewards."""
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
    """Derive an analysis-side blend from already-recorded components."""
    config = BoundedCompressionBlendConfig(
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
    """Derive blends over already-recorded rows without re-driving calls."""
    config = BoundedCompressionBlendConfig(
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
        "identity_key": config.blend_identity_key(),
        "per_row_blended": blends,
        "mean_blended": (sum(blends) / len(blends)) if blends else None,
        "rows_used": len(blends),
        "rows_skipped": skipped,
    }


__all__ = [
    "DEFAULT_COMPRESSION_WEIGHT",
    "BoundedCompressionBlendConfig",
    "blend_per_task",
    "blended_reward",
    "blended_reward_from_components",
    "compression_score",
    "retro_blend_recorded_rows",
]
