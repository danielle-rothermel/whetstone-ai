from typing import Any

from dr_dspy.hashing import canonical_json as canonical_json
from dr_dspy.hashing import sha256_json_digest
from dr_dspy.records.models import DimensionsPayload

PREDICTION_ID_DIGEST_LENGTH = 24
DIMENSIONS_DIGEST_LENGTH = 16
FAIR_ORDER_DIGEST_LENGTH = 32


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
    provider_kind: str,
    endpoint_kind: str,
    model: str,
    throttle_key: str,
    length: int = PREDICTION_ID_DIGEST_LENGTH,
) -> str:
    """Return the v1 graph-aware prediction id.

    This is intentionally not compatible with legacy v0 prediction ids from
    ``dr_dspy.harness.flow``. Backfills should store v0 ids as source metadata
    and compute fresh v1 ids from graph, dimensions, and provider axes.
    """

    return _sha256_digest(
        {
            "experiment_name": experiment_name,
            "task_id": task_id,
            "graph_digest": graph_digest,
            "dimensions_digest": dimensions_digest,
            "repetition_seed": repetition_seed,
            "provider_kind": provider_kind,
            "endpoint_kind": endpoint_kind,
            "model": model,
            "throttle_key": throttle_key,
        },
        length=length,
    )


def fair_order_key(
    *,
    experiment_seed: str,
    prediction_id: str,
    provider: str,
    endpoint_kind: str,
    model: str,
    throttle_key: str,
    graph_layout: str,
    task_id: str,
    repetition_seed: int,
    config_axis: str,
    length: int = FAIR_ORDER_DIGEST_LENGTH,
) -> str:
    return _sha256_digest(
        {
            "experiment_seed": experiment_seed,
            "prediction_id": prediction_id,
            "provider": provider,
            "endpoint_kind": endpoint_kind,
            "model": model,
            "throttle_key": throttle_key,
            "graph_layout": graph_layout,
            "task_id": task_id,
            "repetition_seed": repetition_seed,
            "config_axis": config_axis,
        },
        length=length,
    )


def _sha256_digest(value: Any, *, length: int) -> str:
    return sha256_json_digest(value, length=length)
