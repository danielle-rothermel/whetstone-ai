from __future__ import annotations

from typing import Any, Literal

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

from whetstone.core.identity import (
    IdentityHash,
    IdentityRef,
    ImmutableJsonObject,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.attribution import (
    require_exclusive_row_state,
    require_exhaustive_row_accounting,
)
from whetstone.evaluation.generation import GenerationIndex
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA as _EVALUATION_EVIDENCE_SCHEMA,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_FAILURE_SCHEMA as _EVALUATION_FAILURE_SCHEMA,
)
from whetstone.evaluation.traces import ExecutedComponentTracePayload
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import CandidateRef
from whetstone.experiment.reward import RewardRef

#: Persisted-format contracts. Exact wire fields and versions are pinned by
#: golden tests; never derive them from internal dataclass names.
EVALUATION_COMPONENT_TRACES_SCHEMA = "whetstone.evaluation_component_traces"
EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION = 2
EVALUATION_OUTPUTS_SCHEMA = "whetstone.evaluation_outputs"
EVALUATION_OUTPUTS_SCHEMA_VERSION = 4
EVALUATION_EVIDENCE_SCHEMA_VERSION = 3


class RowAccounting(BaseModel):
    """Complete accounting for the exact task-by-sample matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planned: StrictInt
    present: StrictInt
    missing: StrictInt
    failed: StrictInt
    invalid: StrictInt

    @model_validator(mode="after")
    def _validate_exhaustive(self) -> RowAccounting:
        require_exhaustive_row_accounting(
            planned=self.planned,
            present=self.present,
            missing=self.missing,
            failed=self.failed,
            invalid=self.invalid,
        )
        return self


class CacheEvidence(BaseModel):
    """Cache and partial-log provenance observed by one evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partial_row_count: StrictInt = 0
    cache_hit_count: StrictInt = 0
    source_call_ids: tuple[str, ...] = ()


class SubmissionScoreRecord(BaseModel):
    """Persisted reward-facing scalar for one submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: StrictBool
    infrastructure_unknown: StrictBool = False
    outcome: StrictStr = ""


class SubmissionResultRecord(BaseModel):
    """Opaque persisted submission result with caller-defined detail payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: StrictStr
    score: SubmissionScoreRecord
    outcome: StrictStr = ""
    details: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )


class EvaluationOutputRow(BaseModel):
    """Stable serialized projection of one driven evaluation output row."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    candidate_id: StrictStr
    task_id: StrictStr
    task_hash: StrictStr
    task_index: StrictInt
    sample_index: StrictInt
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
    submission_result: SubmissionResultRecord | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationOutputRow:
        for field_name in ("candidate_id", "task_id", "task_hash"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        if self.max_budget is not None and self.max_budget < 0:
            raise ValueError("max_budget must be non-negative")
        require_exclusive_row_state(
            scored=self.score is not None,
            failed=self.failed,
            missing=self.missing,
            invalid=self.invalid,
        )
        return self

    def generation_index(self) -> GenerationIndex:
        return GenerationIndex(
            task_index=self.task_index,
            sample_index=self.sample_index,
        )


class EvaluationComponentTraceRow(BaseModel):
    """Exact executed-component observation for one planned row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr
    task_hash: StrictStr
    task_index: StrictInt
    sample_index: StrictInt
    executed_component_trace: ExecutedComponentTracePayload

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationComponentTraceRow:
        for field_name in ("task_id", "task_hash"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        return self

    def generation_index(self) -> GenerationIndex:
        return GenerationIndex(
            task_index=self.task_index,
            sample_index=self.sample_index,
        )


class EvaluationComponentTraces(BaseModel):
    """Ordered authoritative component traces for one evaluation matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvaluationRole
    provider_execution_policy_ref: IdentityRef | None = None
    graph_hash: IdentityHash
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    split_role: StrictStr
    task_hashes: tuple[StrictStr, ...]
    num_samples: StrictInt
    rows: tuple[EvaluationComponentTraceRow, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationComponentTraces:
        if not self.split_role.strip():
            raise ValueError("split_role must be non-empty")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if not self.task_hashes:
            raise ValueError("task_hashes must be non-empty")
        if any(not task.strip() for task in self.task_hashes):
            raise ValueError("task_hashes must be non-empty")
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError("task_hashes must be unique")

        task_hash_to_id: dict[str, str] = {}
        task_id_to_hash: dict[str, str] = {}
        planned_ordinal = {
            GenerationIndex(
                task_index=task_index, sample_index=sample_index
            ): task_index * self.num_samples + sample_index
            for task_index, _task_hash in enumerate(self.task_hashes)
            for sample_index in range(self.num_samples)
        }
        seen_keys: set[GenerationIndex] = set()
        prior_ordinal = -1
        for row in self.rows:
            if (
                task_hash_to_id.setdefault(row.task_hash, row.task_id)
                != row.task_id
            ):
                raise ValueError("one task_hash cannot name multiple task_ids")
            if (
                task_id_to_hash.setdefault(row.task_id, row.task_hash)
                != row.task_hash
            ):
                raise ValueError(
                    "one task_id cannot name multiple task_hashes"
                )
            key = row.generation_index()
            if key in seen_keys:
                raise ValueError(
                    "trace rows must have unique generation index keys"
                )
            seen_keys.add(key)
            ordinal = planned_ordinal.get(key)
            if ordinal is None:
                raise ValueError(
                    "trace row is outside the exact task/sample plan"
                )
            if ordinal <= prior_ordinal:
                raise ValueError(
                    "trace rows must follow exact task/sample order"
                )
            prior_ordinal = ordinal
        if seen_keys != set(planned_ordinal):
            raise ValueError(
                "trace rows must cover the exact task/sample plan"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationComponentTracesRef(BaseModel):
    """An exact persisted executed-component trace artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvaluationComponentTraces
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> EvaluationComponentTracesRef:
        if self.record_ref.schema_name != EVALUATION_COMPONENT_TRACES_SCHEMA:
            raise ValueError(
                "component trace record_ref must use the exact trace schema"
            )
        expected = typed_ref_for_record(
            EVALUATION_COMPONENT_TRACES_SCHEMA,
            self.record.record_content(),
        )
        if self.record_ref != expected:
            raise ValueError(
                "component trace record_ref must address the exact record"
            )
        return self


class EvaluationOutputsRecord(BaseModel):
    """Exact ordered output rows persisted at EVALUATION_OUTPUTS_SCHEMA."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    schema_version: Literal[4]
    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvaluationRole
    provider_execution_policy_ref: IdentityRef | None = None
    graph_hash: IdentityHash
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    split_role: StrictStr
    task_hashes: tuple[StrictStr, ...]
    num_samples: StrictInt
    component_traces_ref: TypedRef
    outputs: tuple[EvaluationOutputRow, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationOutputsRecord:
        if (
            self.component_traces_ref.schema_name
            != EVALUATION_COMPONENT_TRACES_SCHEMA
        ):
            raise ValueError(
                "component_traces_ref must use the exact trace schema"
            )
        if not self.split_role.strip():
            raise ValueError("split_role must be non-empty")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if not self.task_hashes:
            raise ValueError("task_hashes must be non-empty")
        if any(not task.strip() for task in self.task_hashes):
            raise ValueError("task_hashes must be non-empty")
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError("task_hashes must be unique")

        task_hash_to_id: dict[str, str] = {}
        task_id_to_hash: dict[str, str] = {}
        seen_keys: set[GenerationIndex] = set()
        planned_ordinal = {
            GenerationIndex(
                task_index=task_index, sample_index=sample_index
            ): task_index * self.num_samples + sample_index
            for task_index, _task_hash in enumerate(self.task_hashes)
            for sample_index in range(self.num_samples)
        }
        prior_ordinal = -1
        for row in self.outputs:
            if row.candidate_id != self.candidate.record.candidate_id:
                raise ValueError(
                    "every output row candidate_id must match the record"
                )
            if (
                task_hash_to_id.setdefault(row.task_hash, row.task_id)
                != row.task_id
            ):
                raise ValueError("one task_hash cannot name multiple task_ids")
            if (
                task_id_to_hash.setdefault(row.task_id, row.task_hash)
                != row.task_hash
            ):
                raise ValueError(
                    "one task_id cannot name multiple task_hashes"
                )
            key = row.generation_index()
            if key in seen_keys:
                raise ValueError(
                    "output rows must have unique generation index keys"
                )
            seen_keys.add(key)
            ordinal = planned_ordinal.get(key)
            if ordinal is None:
                raise ValueError(
                    "output row is outside the exact task/sample plan"
                )
            if ordinal <= prior_ordinal:
                raise ValueError(
                    "output rows must follow exact task/sample order"
                )
            prior_ordinal = ordinal
        if seen_keys != set(planned_ordinal):
            raise ValueError(
                "output rows must cover the exact task/sample plan"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationEvidence(BaseModel):
    """Exact, content-addressed evidence for one candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[3]
    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvaluationRole
    provider_execution_policy_ref: IdentityRef | None = None
    graph_hash: StrictStr
    graph_config_ref: StrictStr
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    #: Source dataset revision/manifest identity. The ordered TaskSet identity
    #: is a separate sampling/config identity and must not be substituted here.
    dataset_hash: StrictStr
    task_hashes: tuple[str, ...]
    num_samples: StrictInt
    per_task_values: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    row_accounting: RowAccounting
    component_traces_ref: TypedRef
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
    def _validate_dataset_hash(self) -> EvaluationEvidence:
        if not self.dataset_hash.strip():
            raise ValueError("dataset_hash must be non-empty")
        if (
            self.component_traces_ref.schema_name
            != EVALUATION_COMPONENT_TRACES_SCHEMA
        ):
            raise ValueError(
                "component_traces_ref must use the exact trace schema"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvaluationFailureEvidence(BaseModel):
    """Typed terminal evidence when execution started but did not score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvaluationRole
    provider_execution_policy_ref: IdentityRef | None = None
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    exception_type: StrictStr
    message: StrictStr

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = [
    "EVALUATION_COMPONENT_TRACES_SCHEMA",
    "EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION",
    "EVALUATION_EVIDENCE_SCHEMA_VERSION",
    "EVALUATION_OUTPUTS_SCHEMA",
    "EVALUATION_OUTPUTS_SCHEMA_VERSION",
    "CacheEvidence",
    "EvaluationComponentTraceRow",
    "EvaluationComponentTraces",
    "EvaluationComponentTracesRef",
    "EvaluationEvidence",
    "EvaluationFailureEvidence",
    "EvaluationOutputRow",
    "EvaluationOutputsRecord",
    "RowAccounting",
    "SubmissionResultRecord",
    "SubmissionScoreRecord",
]
