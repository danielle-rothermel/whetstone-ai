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
) -> float:
    """Deterministic fake eval: higher when generation overlaps gold."""
    if not generation.strip():
        return 0.0
    if gold and gold in generation:
        return 1.0
    return stable_unit_score(task_id, generation, gold)
