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
    ImmutableJsonObject,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA as _EVALUATION_EVIDENCE_SCHEMA,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_FAILURE_SCHEMA as _EVALUATION_FAILURE_SCHEMA,
)
from whetstone.evaluation.traces import ExecutedComponentTracePayload
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import CandidateRef
from whetstone.experiment.reward import RewardRef

#: Persisted-format contracts. Exact wire fields and versions are pinned by
#: golden tests; never derive them from internal dataclass names.
EVALUATION_COMPONENT_TRACES_SCHEMA = "whetstone.evaluation_component_traces"
EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION = 2
EVALUATION_OUTPUTS_SCHEMA = "whetstone.evaluation_outputs"
EVALUATION_OUTPUTS_SCHEMA_VERSION = 3
EVALUATION_EVIDENCE_SCHEMA_VERSION = 3


class RowAccounting(BaseModel):
    """Complete accounting for the exact task-by-sample matrix."""

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
        state_count = sum((self.failed, self.missing, self.invalid))
        if self.score is not None and state_count:
            raise ValueError("a scored output row must be present")
        if self.score is None and state_count != 1:
            raise ValueError(
                "an unscored output row requires exactly one absent state"
            )
        return self


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


class EvaluationComponentTraces(BaseModel):
    """Ordered authoritative component traces for one evaluation matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    candidate: CandidateRef
    evaluation_binding: EvaluationBinding
    evaluation_role: EvaluationRole
    graph_hash: IdentityHash
    purpose: StrictStr
    split_role: StrictStr
    task_hashes: tuple[StrictStr, ...]
    num_samples: StrictInt
    rows: tuple[EvaluationComponentTraceRow, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvaluationComponentTraces:
        if self.evaluation_role is not self.evaluation_binding.role:
            raise ValueError(
                "evaluation_role must match the exact Evaluation Binding"
            )
        if not self.purpose.strip():
            raise ValueError("purpose must be non-empty")
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
            (task_index, sample_index): task_index * self.num_samples
            + sample_index
            for task_index, _task_hash in enumerate(self.task_hashes)
            for sample_index in range(self.num_samples)
        }
        seen_keys: set[tuple[int, int]] = set()
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
            key = (row.task_index, row.sample_index)
            if key in seen_keys:
                raise ValueError(
                    "trace rows must have unique task_hash/sample_index keys"
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

    schema_version: Literal[3]
    candidate: CandidateRef
    evaluation_binding: EvaluationBinding
    evaluation_role: EvaluationRole
    graph_hash: IdentityHash
    purpose: StrictStr
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
        if self.evaluation_role is not self.evaluation_binding.role:
            raise ValueError(
                "evaluation_role must match the exact Evaluation Binding"
            )
        if not self.purpose.strip():
            raise ValueError("purpose must be non-empty")
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
        seen_keys: set[tuple[int, int]] = set()
        planned_ordinal = {
            (task_index, sample_index): task_index * self.num_samples
            + sample_index
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
            key = (row.task_index, row.sample_index)
            if key in seen_keys:
                raise ValueError(
                    "output rows must have unique task_hash/sample_index keys"
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
    evaluation_binding: EvaluationBinding
    graph_hash: StrictStr
    graph_config_ref: StrictStr
    purpose: StrictStr
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


class EvaluationEvidenceRef(BaseModel):
    """An exact persisted Evaluation Evidence record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvaluationEvidence
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> EvaluationEvidenceRef:
        expected = typed_ref_for_record(
            _EVALUATION_EVIDENCE_SCHEMA, self.record.record_content()
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
            _EVALUATION_FAILURE_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected:
            raise ValueError(
                "Evaluation Failure record_ref must address the exact record"
            )
        return self


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
    "EvaluationEvidenceRef",
    "EvaluationFailureEvidence",
    "EvaluationFailureEvidenceRef",
    "EvaluationOutputRow",
    "EvaluationOutputsRecord",
    "RowAccounting",
]
