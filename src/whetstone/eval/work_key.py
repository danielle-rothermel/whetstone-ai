from __future__ import annotations

from whetstone.core.identity import (
    IdentityHash,
    OpaqueKey,
    compute_prefixed_identity_key,
)
from whetstone.eval.task_trial import TaskTrialKey

__all__ = [
    "TASK_TRIAL_WORK_KEY_PREFIX",
    "TASK_TRIAL_WORK_KEY_SCHEMA",
    "TASK_TRIAL_WORK_KEY_SCHEMA_VERSION",
    "task_trial_work_key",
]


TASK_TRIAL_WORK_KEY_SCHEMA = "whetstone.task_trial_work_key"
TASK_TRIAL_WORK_KEY_SCHEMA_VERSION = 1
TASK_TRIAL_WORK_KEY_PREFIX = "whetstone.task_trial_work:"


def task_trial_work_key(
    *,
    experiment_config_hash: IdentityHash,
    key: TaskTrialKey,
) -> OpaqueKey:
    return compute_prefixed_identity_key(
        schema=TASK_TRIAL_WORK_KEY_SCHEMA,
        schema_version=TASK_TRIAL_WORK_KEY_SCHEMA_VERSION,
        prefix=TASK_TRIAL_WORK_KEY_PREFIX,
        payload={
            "experiment_config_hash": str(experiment_config_hash),
            "task_index": key.task_index,
            "seed_index": key.seed_index,
        },
    )
