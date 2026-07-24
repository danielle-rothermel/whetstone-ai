"""Whetstone orchestration: binding execution to dr-platform + DBOS.

This package is Workstream 3 · Platform Integration, Orchestration, Retry, and
Concurrency. It binds Whetstone rollout execution to dr-platform's generic
Stage mechanics and DBOS durability without pushing any provider/node policy
into the platform:

* :mod:`whetstone.orchestration.pipeline` — the concrete versioned linear
  Orchestration Pipeline (one rollout-execution Stage) and the Whetstone→
  platform key mapping (Evaluation Campaign→Platform Campaign, Submission
  Batch→Submission Run, Rollout Work Item→one Rollout Execution Key).
* :mod:`whetstone.orchestration.work_request` — the immutable Rollout Work
  Request and its opaque typed object-reference transport (no Materialization
  Record ref).
* :mod:`whetstone.orchestration.executor` — the durable rollout-execution stage
  body: one ``@DBOS.step(retries_allowed=False)`` per Provider Call Attempt,
  durable backoff sleep, deterministic replay, terminal Rollout Result
  assembly, immutable put, and authoritative Result Store binding.
* :mod:`whetstone.orchestration.retry_gate` — the Operator Retry gate and the
  retried-executor binding recheck.
* :mod:`whetstone.orchestration.labels` — collision-free Provider Quota
  Identity label derivation.
* :mod:`whetstone.orchestration.concurrency` — Provider Concurrency Control:
  the mandatory default empty-selector capacity plus every exact-label
  capacity.
"""

from __future__ import annotations

from whetstone.orchestration.concurrency import (
    ConcurrencyConfiguration,
    QuotaCapacity,
    configure_provider_concurrency,
)
from whetstone.orchestration.executor import (
    ExecutorContext,
    RolloutExecutionOutcome,
    TerminalBindError,
)
from whetstone.orchestration.labels import (
    QUOTA_LABEL_KEY,
    quota_label,
    quota_label_value,
    quota_labels_for,
    quota_selector,
)
from whetstone.orchestration.pipeline import (
    ORCHESTRATION_PIPELINE_KEY,
    ORCHESTRATION_PIPELINE_VERSION,
    ROLLOUT_EXECUTION_STAGE_KEY,
    ROLLOUT_EXECUTION_STAGE_QUEUE,
    orchestration_pipeline,
    orchestration_pipeline_identity,
    rollout_work_input,
    work_key_for_execution_key,
)
from whetstone.orchestration.retry_gate import (
    OperatorRetryRefused,
    RetriedExecutorStop,
    RetryRefusalReason,
    assert_unbound_before_effect,
    operator_retry,
)
from whetstone.orchestration.work_request import (
    ROLLOUT_WORK_REQUEST_SCHEMA,
    ExpectedSchemaIdentities,
    RepeatData,
    RolloutWorkRequest,
    decode_object_reference,
    encode_object_reference,
    encode_work_request_ref,
    work_request_reference,
)

__all__ = [
    "ORCHESTRATION_PIPELINE_KEY",
    "ORCHESTRATION_PIPELINE_VERSION",
    "QUOTA_LABEL_KEY",
    "ROLLOUT_EXECUTION_STAGE_KEY",
    "ROLLOUT_EXECUTION_STAGE_QUEUE",
    "ROLLOUT_WORK_REQUEST_SCHEMA",
    "ConcurrencyConfiguration",
    "ExecutorContext",
    "ExpectedSchemaIdentities",
    "OperatorRetryRefused",
    "QuotaCapacity",
    "RepeatData",
    "RetriedExecutorStop",
    "RetryRefusalReason",
    "RolloutExecutionOutcome",
    "RolloutWorkRequest",
    "TerminalBindError",
    "assert_unbound_before_effect",
    "configure_provider_concurrency",
    "decode_object_reference",
    "encode_object_reference",
    "encode_work_request_ref",
    "operator_retry",
    "orchestration_pipeline",
    "orchestration_pipeline_identity",
    "quota_label",
    "quota_label_value",
    "quota_labels_for",
    "quota_selector",
    "rollout_work_input",
    "work_key_for_execution_key",
    "work_request_reference",
]
