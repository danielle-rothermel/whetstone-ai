"""The ed1 weighted-blend reward: pass rate with a bounded compression penalty.

The USER's standing rule (task 22) for ALL enc-dec optimization: the ed1/ed1m
optimizer's reward is a weighted blend of the Binary Test Pass rate and a
bounded compression score -- NOT pass rate alone. Optimizer cells REFUSE to run
on pass-only reward (see :mod:`whetstone.runner.optimizers`); eval/identity
anchor cells compute + record the blend for both probes so anchors pair with
optimizer cells on the SAME metric.

SPEC (verbatim semantics)::

    clamped = clamp(compression_ratio, min_ratio, max_ratio)
    compression_score = (max_ratio - clamped) / (max_ratio - min_ratio)
    reward = test_pass_rate * ((1 - weight) + weight * compression_score)

``compression_ratio`` is the EXISTING recorded metric: the whetstone zstd-19
Compression Ratio (compressed encoder-output bytes / ``gt_code_wo_comments``
bytes). Definitional continuity with everything already recorded -- her spec
said "or similar", and reusing the recorded metric keeps every past screen/
anchor row retro-computable (:func:`blended_reward_from_components`).

PROPERTIES (tested): output always in [0, 1]; ``pass=0 -> 0`` regardless of
compression; ``pass=1 -> [1-weight, 1]``; ``weight=0`` degenerates to pure pass
rate. A LOWER compression_ratio (fewer bytes) -> a HIGHER compression_score ->
a higher reward, so the blend rewards tighter compression.

COMPOSITION (task 22.1): the blend is computed PER TASK -- a task's
repeats-mean pass rate times that task's compression score -- then averaged
over tasks. So the paired-bootstrap machinery (internal selection + official
CIs) operates on per-task blended rewards exactly as ``env_exact_match`` does
for QA. A task with NO compression sample (every row failed) falls back to
PASS-ONLY for that task (:func:`blend_per_task`): a missing channel never
fabricates compression credit and never zeroes a passing task.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BLENDED_METRIC_ID = "pass_rate_with_bounded_compression_penalty"

#: The starting weight the user named (start at 0.05 or 0.10); the CLI default.
DEFAULT_COMPRESSION_WEIGHT = 0.10


class BoundedCompressionMetricConfig(BaseModel):
    """The ed1 blended-reward config (identity-bearing, task 22.2).

    ``metric_id`` + ``weight`` + clamp bounds fold into the eval/reward config
    identity via :meth:`identity_key`, so a different weight is a distinct
    comparable-or-not config (visible in traces/cells). ``weight`` in [0, 1];
    ``weight=0`` degenerates to pure pass rate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: Literal[
        "pass_rate_with_bounded_compression_penalty"
    ] = BLENDED_METRIC_ID
    weight: float = Field(ge=0.0, le=1.0, default=DEFAULT_COMPRESSION_WEIGHT)
    min_compression_ratio: float = 0.01
    max_compression_ratio: float = 4.0

    def identity_key(self) -> str:
        """A stable identity string folding metric_id + weight + bounds.

        Folded into the ed1 eval/reward config identity so a distinct weight
        (or bounds) is a distinct, visibly-comparable config.
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
    pass_rate: float,
    compression_ratio: float | None,
    config: BoundedCompressionMetricConfig,
) -> float:
    """The blended reward for one unit (task or arm), in [0, 1].

    ``reward = pass_rate * ((1 - w) + w * compression_score)``. A ``None``
    compression (no sample survived) falls back to PASS-ONLY for that unit --
    a missing channel never fabricates compression credit and never zeroes a
    passing unit. ``pass=0 -> 0`` regardless of compression;
    ``weight=0 -> pass_rate`` exactly.
    """
    w = config.weight
    if compression_ratio is None or w == 0.0:
        return pass_rate
    cs = compression_score(compression_ratio, config)
    return pass_rate * ((1.0 - w) + w * cs)


def blend_per_task(
    per_task_pass: tuple[float, ...],
    per_task_compression: tuple[float | None, ...],
    config: BoundedCompressionMetricConfig,
) -> tuple[float, ...]:
    """Per-task blended rewards (task 22.1 composition).

    Each task's blend = its repeats-mean pass rate times its own compression
    score, so the paired bootstrap operates on per-task blended rewards. The
    two input vectors are aligned by task (same order the eval produced). A
    task with no compression sample falls back to pass-only (documented).
    """
    if len(per_task_pass) != len(per_task_compression):
        raise ValueError(
            "per-task pass and compression vectors must be aligned"
        )
    return tuple(
        blended_reward(
            pass_rate=p, compression_ratio=c, config=config
        )
        for p, c in zip(per_task_pass, per_task_compression, strict=True)
    )


def blended_reward_from_components(
    *,
    pass_rate: float,
    compression_ratio: float | None,
    weight: float = DEFAULT_COMPRESSION_WEIGHT,
    min_compression_ratio: float = 0.01,
    max_compression_ratio: float = 4.0,
) -> float:
    """DERIVED (analysis-side) blend from already-recorded components (22.5).

    Recomputes the blended reward from a recorded (pass, compression) row
    under ANY weight/bounds WITHOUT re-driving -- so every past screen/
    anchor row (which carries both components) is comparable under the blend.
    CLEARLY LABELED DERIVED: reads recorded measurements; never drives a
    call. Use for retro-analysis only.
    """
    config = BoundedCompressionMetricConfig(
        weight=weight,
        min_compression_ratio=min_compression_ratio,
        max_compression_ratio=max_compression_ratio,
    )
    return blended_reward(
        pass_rate=pass_rate,
        compression_ratio=compression_ratio,
        config=config,
    )


def retro_blend_recorded_rows(
    rows: list[dict[str, object]],
    *,
    weight: float = DEFAULT_COMPRESSION_WEIGHT,
    min_compression_ratio: float = 0.01,
    max_compression_ratio: float = 4.0,
    pass_key: str = "pass_rate",
    compression_key: str = "compression_ratio",
) -> dict[str, object]:
    """DERIVED retro-compute of the blend over already-recorded rows (22.5).

    Reads recorded per-task/per-arm rows -- each a mapping carrying the two
    RECORDED components (``pass_rate`` + ``compression_ratio``, from any screen
    /anchor artifact) -- and recomputes the blended reward per row under the
    given weight/bounds, WITHOUT re-driving. Returns the per-row blends + their
    mean, so a past measurement is comparable under any weight. CLEARLY LABELED
    DERIVED: it never drives a call; a row missing a pass value is skipped
    (reported in ``skipped``), a missing compression falls back to pass-only.
    """
    config = BoundedCompressionMetricConfig(
        weight=weight,
        min_compression_ratio=min_compression_ratio,
        max_compression_ratio=max_compression_ratio,
    )
    blends: list[float] = []
    skipped = 0
    for row in rows:
        pr = row.get(pass_key)
        if pr is None or not isinstance(pr, int | float):
            skipped += 1
            continue
        cr = row.get(compression_key)
        cr_val = float(cr) if isinstance(cr, int | float) else None
        blends.append(
            blended_reward(
                pass_rate=float(pr), compression_ratio=cr_val, config=config
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
