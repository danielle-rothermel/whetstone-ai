from __future__ import annotations

import hashlib
import json
from typing import Any

from dr_dspy.records.models import DimensionsPayload

TEXT_ENCODING = "utf-8"
PREDICTION_ID_DIGEST_LENGTH = 24
DIMENSIONS_DIGEST_LENGTH = 16
FAIR_ORDER_DIGEST_LENGTH = 32


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def dimensions_digest(
    dimensions: DimensionsPayload,
    *,
    length: int = DIMENSIONS_DIGEST_LENGTH,
) -> str:
    return _sha256_digest(
        {"dimensions": dimensions.model_dump(mode="json")},
        length=length,
    )


def stable_prediction_id(
    *,
    experiment_name: str,
    task_id: str,
    graph_digest: str,
    dimensions_digest: str,
    repetition_seed: int,
    length: int = PREDICTION_ID_DIGEST_LENGTH,
) -> str:
    return _sha256_digest(
        {
            "experiment_name": experiment_name,
            "task_id": task_id,
            "graph_digest": graph_digest,
            "dimensions_digest": dimensions_digest,
            "repetition_seed": repetition_seed,
        },
        length=length,
    )


def fair_order_key(
    *,
    experiment_seed: str,
    prediction_id: str,
    provider: str,
    model: str,
    graph_layout: str,
    task_id: str,
    repetition_seed: int,
    length: int = FAIR_ORDER_DIGEST_LENGTH,
) -> str:
    return _sha256_digest(
        {
            "experiment_seed": experiment_seed,
            "prediction_id": prediction_id,
            "provider": provider,
            "model": model,
            "graph_layout": graph_layout,
            "task_id": task_id,
            "repetition_seed": repetition_seed,
        },
        length=length,
    )


def _sha256_digest(value: Any, *, length: int) -> str:
    raw = canonical_json(value)
    digest = hashlib.sha256(raw.encode(TEXT_ENCODING)).hexdigest()
    return digest[:length]
