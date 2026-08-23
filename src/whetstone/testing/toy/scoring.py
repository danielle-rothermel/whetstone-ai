from __future__ import annotations

import hashlib
import json

__all__ = [
    "score_generation",
    "stable_unit_score",
]


def stable_unit_score(*parts: object) -> float:
    """Map arbitrary parts to a deterministic score in [0, 1]."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def score_generation(
    *,
    generation: str,
    gold: str,
    task_id: str,
    seed_index: int | None = None,
) -> float:
    """Deterministic fake eval: higher when generation overlaps gold.

    ``seed_index`` is opt-in. By default a task scores the same on every
    repeat, which keeps every existing toy golden and pinned control hash
    at ``num_seeds=1`` byte-identical. Pass a ``seed_index`` when a test needs
    the repeats of one task to *differ*, so that a repeat-mean is
    distinguishable from repeat 0, the max, or the sum. The two callers that
    opt in are ``RepeatVaryingEvalProcedureRunner`` (the graph rollout path
    the optimizers actually run through) and
    ``FakeEvalDriver(vary_score_by_repeat=True)``.

    A gold-matching generation stays 1.0 at every repeat: it is the toy
    scorer's one exact-match anchor, and repeat noise there would make a
    perfect score unreachable.
    """
    if not generation.strip():
        return 0.0
    if gold and gold in generation:
        return 1.0
    if seed_index is None:
        return stable_unit_score(task_id, generation, gold)
    return stable_unit_score(task_id, generation, gold, seed_index)
