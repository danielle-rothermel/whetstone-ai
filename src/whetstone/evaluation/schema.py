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

from whetstone.code_eval.aggregate import ROLLOUT_AGGREGATE_SCHEMA
from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.identity import (
    IdentityHash,
    ImmutableJsonObject,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.optimization.reward import REWARD_SCHEMA, RewardRef
from whetstone.optimization.schema import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
    CandidateRef,
    EvaluationBinding,
    IntentOutcome,
    IntentResolution,
)

#: Persisted-format contract for EvaluationOutputsRecord. Exact wire fields
#: are pinned by a golden test; never derive them from internal dataclass
#: names.
EVALUATION_OUTPUTS_SCHEMA = "whetstone.evaluation_outputs"
EVALUATION_RESULT_ATTESTATION_SCHEMA = (
    "whetstone.evaluation_result_attestation"
)
EVALUATION_INTENT_CLAIM_SCHEMA = "whetstone.evaluation_intent_claim"


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
    failed: StrictBool
    missing: StrictBool
    invalid: StrictBool
    failure_code: StrictStr
    finish_reason: StrictStr | None
    provider_error: ImmutableJsonObject | None
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
        state_count = sum((self.failed, self.missing, self.invalid))
        if self.score is not None and state_count:
            raise ValueError("a scored output row must be present")
        if self.score is None and state_count != 1:
            raise ValueError(
                "an unscored output row requires exactly one absent state"
            )
        return self


class EvaluationOutputsRecord(BaseModel):
    """Exact ordered output rows persisted at EVALUATION_OUTPUTS_SCHEMA."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    candidate: CandidateRef
    evaluation_binding: EvaluationBinding
    evaluation_role: EvaluationRole
    graph_hash: IdentityHash
    purpose: StrictStr
    split_role: StrictStr
    task_identities: tuple[StrictStr, ...]
    repeat_count: StrictInt
    outputs: tuple[EvaluationOutputRow, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationOutputsRecord:
        if self.evaluation_role is not self.evaluation_binding.role:
            raise ValueError(
                "evaluation_role must match the exact Evaluation Binding"
            )
        if not self.purpose.strip():
            raise ValueError("purpose must be non-empty")
        if not self.split_role.strip():
            raise ValueError("split_role must be non-empty")
        if self.repeat_count < 1:
            raise ValueError("repeat_count must be at least 1")
        if not self.task_identities:
            raise ValueError("task_identities must be non-empty")
        if any(not task.strip() for task in self.task_identities):
            raise ValueError("task_identities must be non-empty")
        if len(set(self.task_identities)) != len(self.task_identities):
            raise ValueError("task_identities must be unique")

        task_to_instance: dict[str, str] = {}
        instance_to_task: dict[str, str] = {}
        seen_keys: set[tuple[str, int]] = set()
        planned_ordinal = {
            (task_identity, repeat): task_index * self.repeat_count + repeat
            for task_index, task_identity in enumerate(self.task_identities)
            for repeat in range(self.repeat_count)
        }
        prior_ordinal = -1
        for row in self.outputs:
            if row.candidate_id != self.candidate.record.candidate_id:
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
            ordinal = planned_ordinal.get(key)
            if ordinal is None:
                raise ValueError(
                    "output row is outside the exact task/repeat plan"
                )
            if ordinal <= prior_ordinal:
                raise ValueError(
                    "output rows must follow exact task/repeat order"
                )
            prior_ordinal = ordinal
        if seen_keys != set(planned_ordinal):
            raise ValueError(
                "output rows must cover the exact task/repeat plan"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationEvidence(BaseModel):
    """Exact, content-addressed evidence for one candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef
    evaluation_binding: EvaluationBinding
    graph_hash: StrictStr
    graph_config_ref: StrictStr
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
    reward_ref: RewardRef | None = None
    cache: CacheEvidence = Field(default_factory=CacheEvidence)
    concurrency_halved: StrictBool = False
    deadline_reached: StrictBool = False
    guard_timeouts: StrictInt = 0

    @model_validator(mode="after")
    def _validate_dataset_identity(self) -> EvaluationEvidence:
        if not self.dataset_identity.strip():
            raise ValueError("dataset_identity must be non-empty")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationEvidenceRef(BaseModel):
    """An exact persisted Evaluation Evidence record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvaluationEvidence
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> EvaluationEvidenceRef:
        expected = typed_ref_for_record(
            EVALUATION_EVIDENCE_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected:
            raise ValueError(
                "Evaluation Evidence record_ref must address the exact record"
            )
        return self


class EvaluationFailureEvidence(BaseModel):
    """Typed terminal evidence when execution started but did not score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef
    evaluation_binding: EvaluationBinding
    purpose: StrictStr
    exception_type: StrictStr
    message: StrictStr

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationFailureEvidenceRef(BaseModel):
    """An exact persisted Evaluation Failure record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvaluationFailureEvidence
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> EvaluationFailureEvidenceRef:
        expected = typed_ref_for_record(
            EVALUATION_FAILURE_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected:
            raise ValueError(
                "Evaluation Failure record_ref must address the exact record"
            )
        return self


class EvaluationIntentClaim(BaseModel):
    """One event in an intent's globally ordered lease stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_ref: TypedRef
    owner_id: StrictStr
    event_ordinal: StrictInt
    generation: StrictInt
    heartbeat_ordinal: StrictInt
    expires_at: StrictFloat
    result_attestation_ref: TypedRef | None = None


class EvaluationResultAttestation(BaseModel):
    """The exact terminal evaluator result won through claim arbitration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    graph_hash: IdentityHash
    resolution: IntentResolution

    @model_validator(mode="after")
    def _validate(self) -> EvaluationResultAttestation:
        if self.resolution.outcome not in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            raise ValueError(
                "an Evaluation Result Attestation requires a terminal "
                "executed outcome"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = [
    "EVALUATION_EVIDENCE_SCHEMA",
    "EVALUATION_FAILURE_SCHEMA",
    "EVALUATION_INTENT_CLAIM_SCHEMA",
    "EVALUATION_OUTPUTS_SCHEMA",
    "EVALUATION_RESULT_ATTESTATION_SCHEMA",
    "REWARD_SCHEMA",
    "ROLLOUT_AGGREGATE_SCHEMA",
    "CacheEvidence",
    "EvaluationEvidence",
    "EvaluationEvidenceRef",
    "EvaluationFailureEvidence",
    "EvaluationFailureEvidenceRef",
    "EvaluationIntentClaim",
    "EvaluationOutputRow",
    "EvaluationOutputsRecord",
    "EvaluationResultAttestation",
    "RowAccounting",
]
