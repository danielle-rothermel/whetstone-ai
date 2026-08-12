"""Pinned generation work-key derivation.

``work_key = f(experiment_config_hash, GenerationIndex)``: one durable key
per planned generation slot, derived through the canonical
payload -> Identity Hash -> prefixed-key path
(:func:`whetstone.core.identity.compute_prefixed_identity_key`). The
``experiment_config_hash`` is the layered canonical experiment-config
identity, so a work key changes exactly when the closed experiment identity
or the planned slot changes.

Slot-key adoption of dr-code's ``derive_work_key`` (the pinned
``dr-code/generation-work-key-v1`` wire schema) rides the later dependency
repin; this module stays whetstone-internal until then and must not import
``dr_code``.
"""

from __future__ import annotations

from whetstone.core.identity import (
    IdentityHash,
    OpaqueKey,
    compute_prefixed_identity_key,
)
from whetstone.evaluation.generation import GenerationIndex

__all__ = [
    "GENERATION_WORK_KEY_PREFIX",
    "GENERATION_WORK_KEY_SCHEMA",
    "GENERATION_WORK_KEY_SCHEMA_VERSION",
    "generation_work_key",
]

# Persisted-format contract: schema, version, prefix, and payload keys are
# pinned by golden tests. Never derive these payload keys from model fields.
GENERATION_WORK_KEY_SCHEMA = "whetstone.generation_work_key"
GENERATION_WORK_KEY_SCHEMA_VERSION = 1
GENERATION_WORK_KEY_PREFIX = "whetstone.generation_work:"


def generation_work_key(
    *,
    experiment_config_hash: IdentityHash,
    index: GenerationIndex,
) -> OpaqueKey:
    """Derive the one work key for a planned generation slot."""
    return compute_prefixed_identity_key(
        schema=GENERATION_WORK_KEY_SCHEMA,
        schema_version=GENERATION_WORK_KEY_SCHEMA_VERSION,
        prefix=GENERATION_WORK_KEY_PREFIX,
        payload={
            "experiment_config_hash": str(experiment_config_hash),
            "task_index": index.task_index,
            "sample_index": index.sample_index,
        },
    )
