"""Durable records produced by the canonical evaluation engine."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.identity import TypedRef, reject_non_json
from whetstone.optimization.schema import CandidateRef, EvalConfigRef

EVALUATION_EVIDENCE_SCHEMA = "whetstone.evaluation_evidence"
#: Persisted-format contract for EvaluationOutputsRecord. Exact wire fields
#: are pinned by a golden test; never derive them from internal dataclass
#: names.
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


class EvaluationOutputComponentTraceStep(BaseModel):
    """One ordered native component trace persisted with an output row."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    component_id: StrictStr
    inputs: dict[StrictStr, Any]
    outputs: dict[StrictStr, Any]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationOutputComponentTraceStep:
        if not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        reject_non_json(self.inputs, field="inputs")
        reject_non_json(self.outputs, field="outputs")
        return self


class EvaluationOutputRow(BaseModel):
    """Stable serialized projection of one driven evaluation output row."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    candidate_id: StrictStr
    instance_id: StrictStr
    task_identity: StrictStr
    repeat: StrictInt
    rendered_prompt: StrictStr
    output_text: StrictStr | None
    score: StrictFloat | None
    failure_code: StrictStr
    component_trace_steps: tuple[EvaluationOutputComponentTraceStep, ...]
    finish_reason: StrictStr | None
    provider_error: dict[StrictStr, Any] | None
    max_budget: StrictInt | None
    over_budget: StrictBool | None

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationOutputRow:
        for field_name in ("candidate_id", "instance_id", "task_identity"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.repeat < 0:
            raise ValueError("repeat must be non-negative")
        if self.max_budget is not None and self.max_budget < 0:
            raise ValueError("max_budget must be non-negative")
        if self.provider_error is not None:
            reject_non_json(self.provider_error, field="provider_error")
        return self


class EvaluationOutputsRecord(BaseModel):
    """Exact ordered output rows persisted at EVALUATION_OUTPUTS_SCHEMA."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    candidate_id: StrictStr
    outputs: tuple[EvaluationOutputRow, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationOutputsRecord:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")

        task_to_instance: dict[str, str] = {}
        instance_to_task: dict[str, str] = {}
        seen_keys: set[tuple[str, int]] = set()
        closed_tasks: set[str] = set()
        current_task: str | None = None
        prior_repeat = -1
        for row in self.outputs:
            if row.candidate_id != self.candidate_id:
                raise ValueError(
                    "every output row candidate_id must match the record"
                )
            if (
                task_to_instance.setdefault(row.task_identity, row.instance_id)
                != row.instance_id
            ):
                raise ValueError(
                    "one task_identity cannot name multiple instance_ids"
                )
            if (
                instance_to_task.setdefault(row.instance_id, row.task_identity)
                != row.task_identity
            ):
                raise ValueError(
                    "one instance_id cannot name multiple task_identities"
                )
            key = (row.task_identity, row.repeat)
            if key in seen_keys:
                raise ValueError(
                    "output rows must have unique task_identity/repeat keys"
                )
            seen_keys.add(key)

            if row.task_identity != current_task:
                if row.task_identity in closed_tasks:
                    raise ValueError(
                        "output rows for one task_identity must be contiguous"
                    )
                if current_task is not None:
                    closed_tasks.add(current_task)
                current_task = row.task_identity
                prior_repeat = -1
            if row.repeat <= prior_repeat:
                raise ValueError(
                    "output repeats must be strictly increasing within a task"
                )
            prior_repeat = row.repeat
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


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
    #: Source dataset revision/manifest identity. The ordered TaskSet identity
    #: is a separate sampling/config identity and must not be substituted here.
    dataset_identity: StrictStr
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

    @model_validator(mode="after")
    def _validate_dataset_identity(self) -> EvaluationEvidence:
        if not self.dataset_identity.strip():
            raise ValueError("dataset_identity must be non-empty")
        return self

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
    "EvaluationOutputComponentTraceStep",
    "EvaluationOutputRow",
    "EvaluationOutputsRecord",
    "RowAccounting",
]
