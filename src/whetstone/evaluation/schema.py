"""Durable records produced by the canonical evaluation engine."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
)

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.schema import CandidateRef, EvalConfigRef

EVALUATION_EVIDENCE_SCHEMA = "whetstone.evaluation_evidence"
EVALUATION_OUTPUTS_SCHEMA = "whetstone.evaluation_outputs"
ROLLOUT_AGGREGATE_SCHEMA = "whetstone.rollout_aggregate"
REWARD_SCHEMA = "whetstone.reward"
EVALUATION_FAILURE_SCHEMA = "whetstone.evaluation_failure"
EVALUATION_INTENT_CLAIM_SCHEMA = "whetstone.evaluation_intent_claim"
INTENT_RESOLUTION_SCHEMA = "whetstone.intent_resolution"


class RowAccounting(BaseModel):
    """Complete accounting for the exact task-by-repeat matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planned: StrictInt
    present: StrictInt
    missing: StrictInt
    failed: StrictInt
    invalid: StrictInt


class CacheEvidence(BaseModel):
    """Cache and partial-log provenance observed by one evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partial_row_count: StrictInt = 0
    cache_hit_count: StrictInt = 0
    source_call_ids: tuple[str, ...] = ()


class EvaluationEvidence(BaseModel):
    """Exact, content-addressed evidence for one candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef
    eval_config: EvalConfigRef
    graph_hash: StrictStr
    graph_config_ref: StrictStr
    evaluation_role: EvaluationRole
    evaluation_context_id: StrictStr
    purpose: StrictStr
    task_identities: tuple[str, ...]
    repeat_count: StrictInt
    per_task_values: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    row_accounting: RowAccounting
    outputs_ref: TypedRef
    aggregate_ref: TypedRef
    aggregate_name: StrictStr
    aggregate_value: float | None
    aggregate_status: StrictStr
    reward_ref: TypedRef | None = None
    cache: CacheEvidence = Field(default_factory=CacheEvidence)
    concurrency_halved: bool = False
    deadline_reached: bool = False
    guard_timeouts: StrictInt = 0

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationFailureEvidence(BaseModel):
    """Typed terminal evidence when execution started but did not score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef
    eval_config: EvalConfigRef
    purpose: StrictStr
    exception_type: StrictStr
    message: StrictStr

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationIntentClaim(BaseModel):
    """One event in an intent's globally ordered lease stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_ref: TypedRef
    owner_id: StrictStr
    event_ordinal: StrictInt
    generation: StrictInt
    heartbeat_ordinal: StrictInt
    expires_at: StrictFloat


__all__ = [
    "EVALUATION_EVIDENCE_SCHEMA",
    "EVALUATION_FAILURE_SCHEMA",
    "EVALUATION_INTENT_CLAIM_SCHEMA",
    "EVALUATION_OUTPUTS_SCHEMA",
    "INTENT_RESOLUTION_SCHEMA",
    "REWARD_SCHEMA",
    "ROLLOUT_AGGREGATE_SCHEMA",
    "CacheEvidence",
    "EvaluationEvidence",
    "EvaluationFailureEvidence",
    "EvaluationIntentClaim",
    "RowAccounting",
]
