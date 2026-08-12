"""Canonical layered experiment-config identity.

An experiment config's Identity Hash closes over everything that can change
an evaluation output (the G1 closure invariant). The closure is organized as
named layers: each layer is a named, hashed sub-payload, and the config-level
identity payload maps layer names to layer hashes. The layer set names the
previously uncoordinated sibling-hash families in one place:

- ``graph`` — the graph_hash root: the mode and mode settings from which the
  generation graph and evaluation procedure are built.
- ``candidate_binding`` — the initial and ceiling candidate identities bound
  to the experiment.
- ``comp`` — the compression operator configuration.
- ``profiles`` — the sampling and completeness profile.
- ``provider_policy`` — the encoder/decoder provider routes.
- ``splits`` — task pool selection and internal/official split semantics.

Layer payload shapes are persisted format: every layer's exact payload and
hash is pinned by a per-layer golden test.
"""

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

# Persisted-format contract: schema, version, and payload keys are pinned by
# golden tests. Never derive these payload keys from model fields.
EXPERIMENT_CONFIG_LAYER_SCHEMA = "whetstone.experiment_config_layer"
EXPERIMENT_CONFIG_LAYER_SCHEMA_VERSION = 1


@verify(UNIQUE)
class ExperimentConfigLayer(StrEnum):
    """Closed set of named experiment-config identity layers.

    These values are persisted contract literals (they appear inside identity
    payloads). Never iterate over this enum to construct a persisted payload.
    """

    GRAPH = "graph"
    CANDIDATE_BINDING = "candidate_binding"
    COMP = "comp"
    PROFILES = "profiles"
    PROVIDER_POLICY = "provider_policy"
    SPLITS = "splits"


def experiment_config_layer_hash(
    layer: ExperimentConfigLayer, payload: Any
) -> IdentityHash:
    """Hash one named layer's exact sub-payload."""
    return compute_identity_hash(
        schema=EXPERIMENT_CONFIG_LAYER_SCHEMA,
        schema_version=EXPERIMENT_CONFIG_LAYER_SCHEMA_VERSION,
        payload={"layer": layer.value, "payload": payload},
    )


def layered_experiment_config_payload(
    layer_payloads: dict[ExperimentConfigLayer, Any],
) -> dict[str, Any]:
    """Build the config-level identity payload from all named layers.

    Requires every :class:`ExperimentConfigLayer` exactly once so the config
    hash always closes over the complete layer set.
    """
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

    # Persisted-format contract: these payload keys are explicit literals
    # pinned by golden tests, never derived from enum iteration.
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
