from __future__ import annotations

from enum import UNIQUE, StrEnum, verify
from typing import Any

from whetstone.core.identity import IdentityHash, compute_identity_hash

__all__ = [
    "EXPERIMENT_CONFIG_LAYER_SCHEMA",
    "EXPERIMENT_CONFIG_LAYER_SCHEMA_VERSION",
    "ExperimentConfigLayer",
    "experiment_config_layer_hash",
    "layered_experiment_config_payload",
]


EXPERIMENT_CONFIG_LAYER_SCHEMA = "whetstone.experiment_config_layer"
EXPERIMENT_CONFIG_LAYER_SCHEMA_VERSION = 1


@verify(UNIQUE)
class ExperimentConfigLayer(StrEnum):
    GRAPH = "graph"
    CANDIDATE_BINDING = "candidate_binding"
    COMP = "comp"
    PROFILES = "profiles"
    PROVIDER_POLICY = "provider_policy"
    SPLITS = "splits"


def experiment_config_layer_hash(
    layer: ExperimentConfigLayer, payload: Any
) -> IdentityHash:
    return compute_identity_hash(
        schema=EXPERIMENT_CONFIG_LAYER_SCHEMA,
        schema_version=EXPERIMENT_CONFIG_LAYER_SCHEMA_VERSION,
        payload={"layer": layer.value, "payload": payload},
    )


def layered_experiment_config_payload(
    layer_payloads: dict[ExperimentConfigLayer, Any],
) -> dict[str, Any]:
    if set(layer_payloads) != set(ExperimentConfigLayer):
        missing = sorted(
            layer.value
            for layer in ExperimentConfigLayer
            if layer not in layer_payloads
        )
        extra = sorted(
            str(getattr(layer, "value", layer))
            for layer in layer_payloads
            if layer not in set(ExperimentConfigLayer)
        )
        raise ValueError(
            "experiment config identity requires every layer exactly once; "
            f"missing {missing}, extra {extra}"
        )

    def layer_hash(layer: ExperimentConfigLayer) -> str:
        return str(experiment_config_layer_hash(layer, layer_payloads[layer]))

    return {
        "layers": {
            "graph": layer_hash(ExperimentConfigLayer.GRAPH),
            "candidate_binding": layer_hash(
                ExperimentConfigLayer.CANDIDATE_BINDING
            ),
            "comp": layer_hash(ExperimentConfigLayer.COMP),
            "profiles": layer_hash(ExperimentConfigLayer.PROFILES),
            "provider_policy": layer_hash(
                ExperimentConfigLayer.PROVIDER_POLICY
            ),
            "splits": layer_hash(ExperimentConfigLayer.SPLITS),
        }
    }
