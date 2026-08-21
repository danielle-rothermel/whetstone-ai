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
from whetstone.core.roles import EvalRole
from whetstone.eval.attribution import (
    require_exclusive_row_state,
    require_exhaustive_row_accounting,
)
from whetstone.eval.task_trial import TaskTrialKey
from whetstone.eval.schema_names import (
    EVAL_EVIDENCE_SCHEMA as _EVAL_EVIDENCE_SCHEMA,
)
from whetstone.eval.schema_names import (
    EVAL_FAILURE_SCHEMA as _EVAL_FAILURE_SCHEMA,
)
from whetstone.eval.traces import ExecutedComponentTracePayload
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import CandidateRef
from whetstone.experiment.reward import RewardRef

#: Persisted-format contracts. Exact wire fields and versions are pinned by
#: golden tests; never derive them from internal dataclass names.
EVAL_TRACES_SCHEMA = "whetstone.eval_component_traces"
EVAL_TRACES_SCHEMA_VERSION = 2
EVAL_OUTPUTS_SCHEMA = "whetstone.eval_outputs"
EVAL_OUTPUTS_SCHEMA_VERSION = 4
EVAL_EVIDENCE_SCHEMA_VERSION = 4


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


class EvalOutputRow(BaseModel):
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
    seed_index: StrictInt
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
    def _validate_contract(self) -> EvalOutputRow:
        for field_name in ("candidate_id", "task_id", "task_hash"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative")
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

    def task_trial_key(self) -> TaskTrialKey:
        return TaskTrialKey(
            task_index=self.task_index,
            seed_index=self.seed_index,
        )


class EvalTraceRow(BaseModel):
    """Exact executed-component observation for one planned row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr
    task_hash: StrictStr
    task_index: StrictInt
    seed_index: StrictInt
    trace: ExecutedComponentTracePayload

    @model_validator(mode="after")
    def _validate_contract(self) -> EvalTraceRow:
        for field_name in ("task_id", "task_hash"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.seed_index < 0:
            raise ValueError("seed_index must be non-negative")
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        return self

    def task_trial_key(self) -> TaskTrialKey:
        return TaskTrialKey(
            task_index=self.task_index,
            seed_index=self.seed_index,
        )


class EvalTraces(BaseModel):
    """Ordered authoritative component traces for one evaluation matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
    provider_execution_policy_ref: IdentityRef | None = None
    graph_hash: IdentityHash
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    split_role: StrictStr
    task_hashes: tuple[StrictStr, ...]
    num_seeds: StrictInt
    rows: tuple[EvalTraceRow, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvalTraces:
        if not self.split_role.strip():
            raise ValueError("split_role must be non-empty")
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be at least 1")
        if not self.task_hashes:
            raise ValueError("task_hashes must be non-empty")
        if any(not task.strip() for task in self.task_hashes):
            raise ValueError("task_hashes must be non-empty")
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError("task_hashes must be unique")

        task_hash_to_id: dict[str, str] = {}
        task_id_to_hash: dict[str, str] = {}
        planned_ordinal = {
            TaskTrialKey(
                task_index=task_index, seed_index=seed_index
            ): task_index * self.num_seeds + seed_index
            for task_index, _task_hash in enumerate(self.task_hashes)
            for seed_index in range(self.num_seeds)
        }
        seen_keys: set[TaskTrialKey] = set()
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
            key = row.task_trial_key()
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


class EvalTracesRef(BaseModel):
    """An exact persisted executed-component trace artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvalTraces
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> EvalTracesRef:
        if self.record_ref.schema_name != EVAL_TRACES_SCHEMA:
            raise ValueError(
                "component trace record_ref must use the exact trace schema"
            )
        expected = typed_ref_for_record(
            EVAL_TRACES_SCHEMA,
            self.record.record_content(),
        )
        if self.record_ref != expected:
            raise ValueError(
                "component trace record_ref must address the exact record"
            )
        return self


class EvalOutputsRecord(BaseModel):
    """Exact ordered output rows persisted at EVAL_OUTPUTS_SCHEMA."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    schema_version: Literal[4]
    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
    provider_execution_policy_ref: IdentityRef | None = None
    graph_hash: IdentityHash
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    split_role: StrictStr
    task_hashes: tuple[StrictStr, ...]
    num_seeds: StrictInt
    traces_ref: TypedRef
    outputs: tuple[EvalOutputRow, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> EvalOutputsRecord:
        if (
            self.traces_ref.schema_name
            != EVAL_TRACES_SCHEMA
        ):
            raise ValueError(
                "traces_ref must use the exact trace schema"
            )
        if not self.split_role.strip():
            raise ValueError("split_role must be non-empty")
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be at least 1")
        if not self.task_hashes:
            raise ValueError("task_hashes must be non-empty")
        if any(not task.strip() for task in self.task_hashes):
            raise ValueError("task_hashes must be non-empty")
        if len(set(self.task_hashes)) != len(self.task_hashes):
            raise ValueError("task_hashes must be unique")

        task_hash_to_id: dict[str, str] = {}
        task_id_to_hash: dict[str, str] = {}
        seen_keys: set[TaskTrialKey] = set()
        planned_ordinal = {
            TaskTrialKey(
                task_index=task_index, seed_index=seed_index
            ): task_index * self.num_seeds + seed_index
            for task_index, _task_hash in enumerate(self.task_hashes)
            for seed_index in range(self.num_seeds)
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
            key = row.task_trial_key()
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


class EvalEvidence(BaseModel):
    """Exact, content-addressed evidence for one candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[4]
    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
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
    num_seeds: StrictInt
    per_task_values: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    row_accounting: RowAccounting
    traces_ref: TypedRef
    outputs_ref: TypedRef
    aggregate_ref: TypedRef
    aggregate_name: StrictStr
    aggregate_value: float | None
    aggregate_status: StrictStr
    reward_ref: RewardRef | None = None
    cache: CacheEvidence = Field(default_factory=CacheEvidence)
    deadline_reached: StrictBool = False

    @model_validator(mode="after")
    def _validate_dataset_hash(self) -> EvalEvidence:
        if not self.dataset_hash.strip():
            raise ValueError("dataset_hash must be non-empty")
        if (
            self.traces_ref.schema_name
            != EVAL_TRACES_SCHEMA
        ):
            raise ValueError(
                "traces_ref must use the exact trace schema"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EvalFailureEvidence(BaseModel):
    """Typed terminal evidence when execution started but did not score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
    provider_execution_policy_ref: IdentityRef | None = None
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    exception_type: StrictStr
    message: StrictStr

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = [
    "EVAL_TRACES_SCHEMA",
    "EVAL_TRACES_SCHEMA_VERSION",
    "EVAL_EVIDENCE_SCHEMA_VERSION",
    "EVAL_OUTPUTS_SCHEMA",
    "EVAL_OUTPUTS_SCHEMA_VERSION",
    "CacheEvidence",
    "EvalTraceRow",
    "EvalTraces",
    "EvalTracesRef",
    "EvalEvidence",
    "EvalFailureEvidence",
    "EvalOutputRow",
    "EvalOutputsRecord",
    "RowAccounting",
    "SubmissionResultRecord",
    "SubmissionScoreRecord",
]
