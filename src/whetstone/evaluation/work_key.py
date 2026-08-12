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


GENERATION_WORK_KEY_SCHEMA = "whetstone.generation_work_key"
GENERATION_WORK_KEY_SCHEMA_VERSION = 1
GENERATION_WORK_KEY_PREFIX = "whetstone.generation_work:"


def generation_work_key(
    *,
    experiment_config_hash: IdentityHash,
    index: GenerationIndex,
) -> OpaqueKey:
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
