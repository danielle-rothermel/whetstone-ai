from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
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

from dr_dspy.eval_failures.types import FailureClass
from dr_dspy.graph import GraphSpec
from dr_dspy.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.humaneval.task import EvaluationCaseStatus, EvaluationCaseSummary
from dr_dspy.lm.boundary import EndpointKind, ProviderConfig, ProviderKind


class NodeAttemptStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class ScoreAttemptStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class GenerationRunStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"
    PARTIAL = "partial"


class BatchSubmitOperationStatus(StrEnum):
    PREPARED = "prepared"
    COMPLETED = "completed"
    PARTIAL = "partial"
    ERROR = "error"


class BatchSubmitItemStatus(StrEnum):
    INSERTED = "inserted"
    ALREADY_PRESENT = "already_present"
    ENQUEUED = "enqueued"
    FAILED = "failed"


class TaskInputsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[StrictStr, Any]


class TaskSnapshotPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr
    inputs: TaskInputsPayload
    source: StrictStr | None = None
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class DimensionsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[StrictStr, Any]


class GraphSnapshotPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: GraphSpec
    graph_digest: StrictStr
    layout: StrictStr

    @model_validator(mode="after")
    def validate_graph_digest(self) -> GraphSnapshotPayload:
        from dr_dspy.graph import graph_digest

        if self.graph_digest != graph_digest(self.graph):
            raise ValueError("graph_digest must match graph")
        return self


class ProviderConfigRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_kind: ProviderKind
    endpoint_kind: EndpointKind
    model: StrictStr
    config_id: StrictStr | None = None
    throttle_key: StrictStr
    parameters: dict[StrictStr, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: ProviderConfig,
        *,
        config_id: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> ProviderConfigRef:
        return cls(
            provider_kind=config.provider_kind,
            endpoint_kind=config.endpoint_kind,
            model=config.model,
            config_id=config_id,
            throttle_key=config.throttle_identity,
            parameters=dict(parameters or {}),
        )


class UsageCostPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usage_metadata: dict[StrictStr, Any] = Field(default_factory=dict)
    provider_cost: StrictFloat | None = None


class ResponseMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class FailureMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_class: FailureClass | None = None
    error_type: StrictStr
    message: StrictStr
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class NodeOutputPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[StrictStr, Any]
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class GenerationTerminalErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    status: GenerationRunStatus
    failure: FailureMetadataPayload | None = None
    blocked_by: tuple[StrictStr, ...] = ()


class GenerationRunSummaryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_order: tuple[StrictStr, ...]
    terminal_node_id: StrictStr
    terminal_output: Any | None = None
    terminal_error: GenerationTerminalErrorPayload | None = None
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class ExtractedCodePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_generation: StrictStr | None = None
    extracted_code: StrictStr | None = None
    extraction_method: StrictStr | None = None
    parser_profile_id: StrictStr
    parser_version: StrictStr
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class MetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: StrictStr
    profile_version: StrictStr
    text: TextMetricsPayload | None = None
    python_leakage: PythonLeakageMetricsPayload | None = None
    ast: AstMetricsPayload | None = None
    compression: dict[StrictStr, Any] = Field(default_factory=dict)
    custom: dict[StrictStr, Any] = Field(default_factory=dict)


class TextMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_count: StrictInt
    byte_count: StrictInt
    line_count: StrictInt
    nonempty_line_count: StrictInt
    word_count: StrictInt
    average_word_length: StrictFloat | None = None
    punctuation_count: StrictInt | None = None
    symbol_count: StrictInt | None = None


class PythonLeakageMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword_count: StrictInt
    code_marker_count: StrictInt
    fenced_code_block_count: StrictInt
    code_like_line_count: StrictInt
    operator_count: StrictInt
    punctuation_density: StrictFloat | None = None
    task_name_hit_count: StrictInt | None = None


class AstMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse_ok: StrictBool
    parse_error: StrictStr | None = None
    top_level_function_count: StrictInt = 0
    class_count: StrictInt = 0
    import_count: StrictInt = 0
    ast_node_count: StrictInt = 0
    statement_count: StrictInt = 0
    branch_count: StrictInt = 0


class PerTestResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr
    test_id: StrictStr
    function_name: StrictStr
    status: EvaluationCaseStatus
    message: StrictStr = ""
    test_type: HumanEvalTestCaseKind
    input_repr: StrictStr = ""
    expected_output_repr: StrictStr = ""
    actual_output_repr: StrictStr = ""

    @classmethod
    def from_evaluation_case(
        cls,
        case: EvaluationCaseSummary,
    ) -> PerTestResultPayload:
        return cls(
            task_id=case.task_id,
            test_id=case.case_id,
            function_name=case.function_name,
            status=case.status,
            message=case.message,
            test_type=case.test_type,
            input_repr=case.input_repr,
            expected_output_repr=case.expected_output_repr,
            actual_output_repr=case.actual_output_repr,
        )


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: StrictStr
    description: StrictStr | None = None
    config_metadata: dict[StrictStr, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PredictionSpecRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    task_id: StrictStr
    repetition_seed: StrictInt
    graph: GraphSnapshotPayload
    dimensions: DimensionsPayload
    dimensions_digest: StrictStr
    task: TaskSnapshotPayload
    provider_configs: tuple[ProviderConfigRef, ...]
    provider_axis: ProviderConfigRef
    fair_order_seed: StrictStr
    fair_order_key: StrictStr
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_spec_shape(self) -> PredictionSpecRecord:
        if self.repetition_seed < 0:
            raise ValueError("repetition_seed must be non-negative")
        if self.task.task_id != self.task_id:
            raise ValueError("task snapshot task_id must match spec task_id")
        if self.provider_axis not in self.provider_configs:
            raise ValueError("provider_axis must be one of provider_configs")
        from dr_dspy.records.hashing import (
            dimensions_digest,
            fair_order_key,
            stable_prediction_id,
        )

        if self.dimensions_digest != dimensions_digest(self.dimensions):
            raise ValueError("dimensions_digest must match dimensions")
        expected_prediction_id = stable_prediction_id(
            experiment_name=self.experiment_name,
            task_id=self.task_id,
            graph_digest=self.graph.graph_digest,
            dimensions_digest=self.dimensions_digest,
            repetition_seed=self.repetition_seed,
            provider_kind=self.provider_axis.provider_kind.value,
            endpoint_kind=self.provider_axis.endpoint_kind.value,
            model=self.provider_axis.model,
            throttle_key=self.provider_axis.throttle_key,
        )
        if self.prediction_id != expected_prediction_id:
            raise ValueError("prediction_id must match stable prediction id")
        expected_fair_order_key = fair_order_key(
            experiment_seed=self.fair_order_seed,
            prediction_id=self.prediction_id,
            provider=self.provider_axis.provider_kind.value,
            endpoint_kind=self.provider_axis.endpoint_kind.value,
            model=self.provider_axis.model,
            throttle_key=self.provider_axis.throttle_key,
            graph_layout=self.graph.layout,
            task_id=self.task_id,
            repetition_seed=self.repetition_seed,
            config_axis=self.dimensions_digest,
        )
        if self.fair_order_key != expected_fair_order_key:
            raise ValueError("fair_order_key must match spec axes")
        return self


class GenerationRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_run_id: StrictStr
    prediction_id: StrictStr
    attempt_index: StrictInt
    status: GenerationRunStatus
    terminal_node_id: StrictStr
    terminal_output_node_id: StrictStr | None = None
    summary: GenerationRunSummaryPayload
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_run_shape(self) -> GenerationRunRecord:
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if self.summary.terminal_node_id != self.terminal_node_id:
            raise ValueError("summary terminal_node_id must match run")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.status is GenerationRunStatus.SUCCESS:
            if self.summary.terminal_error is not None:
                raise ValueError(
                    "successful generation runs cannot have terminal_error"
                )
        if self.status in {
            GenerationRunStatus.ERROR,
            GenerationRunStatus.BLOCKED,
        }:
            if self.summary.terminal_error is None:
                raise ValueError(
                    "error and blocked generation runs require terminal_error"
                )
        return self


class NodeAttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_attempt_id: StrictStr
    generation_run_id: StrictStr
    prediction_id: StrictStr
    node_id: StrictStr
    attempt_index: StrictInt
    status: NodeAttemptStatus
    provider_config: ProviderConfigRef | None = None
    output: NodeOutputPayload | None = None
    usage_cost: UsageCostPayload = Field(default_factory=UsageCostPayload)
    response_metadata: ResponseMetadataPayload = Field(
        default_factory=ResponseMetadataPayload
    )
    failure: FailureMetadataPayload | None = None
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_attempt_shape(self) -> NodeAttemptRecord:
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.status is NodeAttemptStatus.SUCCESS:
            if self.output is None:
                raise ValueError("successful node attempts require output")
            if self.failure is not None:
                raise ValueError(
                    "successful node attempts cannot have failure"
                )
        if self.status is NodeAttemptStatus.ERROR:
            if self.failure is None:
                raise ValueError("error node attempts require failure")
            if self.output is not None:
                raise ValueError("error node attempts cannot have output")
        return self


class ScoreAttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_attempt_id: StrictStr
    prediction_id: StrictStr
    generation_run_id: StrictStr
    attempt_index: StrictInt
    scoring_profile_id: StrictStr
    scoring_profile_version: StrictStr
    parser_profile_id: StrictStr
    parser_version: StrictStr
    status: ScoreAttemptStatus
    generated_code_outcome: GeneratedCodeOutcome | None = None
    score: StrictFloat | None = None
    extracted_code: ExtractedCodePayload | None = None
    metrics: MetricsPayload | None = None
    per_test_results: tuple[PerTestResultPayload, ...] = ()
    failure: FailureMetadataPayload | None = None
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_attempt_shape(self) -> ScoreAttemptRecord:
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1 inclusive")
        if self.status is ScoreAttemptStatus.SUCCESS:
            if self.score is None:
                raise ValueError("successful score attempts require score")
            if self.failure is not None:
                raise ValueError(
                    "successful score attempts cannot have failure"
                )
        if self.status is ScoreAttemptStatus.ERROR:
            if self.failure is None:
                raise ValueError("error score attempts require failure")
            if self.score is not None:
                raise ValueError("error score attempts cannot have score")
            if self.per_test_results:
                raise ValueError(
                    "error score attempts cannot have per_test_results"
                )
        return self


class PredictionProjectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    generation_run_id: StrictStr | None = None
    score_attempt_id: StrictStr | None = None
    projection_profile_id: StrictStr
    projection_version: StrictStr
    selected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    selection_reason: StrictStr | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> PredictionProjectionRecord:
        if self.generation_run_id is None and self.score_attempt_id is None:
            raise ValueError(
                "projection requires generation_run_id or score_attempt_id"
            )
        return self


class BatchSubmitOperationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_key: StrictStr
    experiment_name: StrictStr
    status: BatchSubmitOperationStatus
    requested_count: StrictInt
    inserted_count: StrictInt = 0
    already_present_count: StrictInt = 0
    enqueued_count: StrictInt = 0
    failed_count: StrictInt = 0
    spec: dict[StrictStr, Any] = Field(default_factory=dict)
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> BatchSubmitOperationRecord:
        counts = (
            self.requested_count,
            self.inserted_count,
            self.already_present_count,
            self.enqueued_count,
            self.failed_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("batch submit counts must be non-negative")
        if (
            self.completed_at is not None
            and self.completed_at < self.created_at
        ):
            raise ValueError("completed_at must not precede created_at")
        return self


class BatchSubmitItemRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_submit_item_id: StrictStr
    operation_key: StrictStr
    item_index: StrictInt
    prediction_id: StrictStr
    fair_order_key: StrictStr
    status: BatchSubmitItemStatus
    enqueue_metadata: dict[StrictStr, Any] = Field(default_factory=dict)
    failure: FailureMetadataPayload | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_item_shape(self) -> BatchSubmitItemRecord:
        if self.item_index < 0:
            raise ValueError("item_index must be non-negative")
        if (
            self.status is BatchSubmitItemStatus.FAILED
            and self.failure is None
        ):
            raise ValueError("failed batch submit items require failure")
        return self
