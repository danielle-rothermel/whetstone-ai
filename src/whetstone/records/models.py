from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from dr_code.humaneval import (
    EvaluationCaseStatus,
    EvaluationCaseSummary,
    HumanEvalTestCaseKind,
    SubmissionOutcome,
    metric_models,
)
from dr_graph import GraphSpec, validate_external_bindings
from dr_providers import (
    EndpointKind,
    FailureClass,
    ProviderConfig,
    ProviderKind,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.records.limits import (
    GRAPH_SNAPSHOT_MAX_BYTES,
    METRICS_MAX_BYTES,
    METRICS_STAGES_MAX_COUNT,
    NODE_OUTPUT_MAX_BYTES,
    PER_TEST_RESULTS_MAX_BYTES,
    PROVIDER_TELEMETRY_MAX_BYTES,
    TASK_INPUTS_MAX_BYTES,
    validate_payload_size,
)

AstMetricsPayload = metric_models.AstMetricsPayload
HumanEvalTaskTestMetricsPayload = metric_models.HumanEvalTaskTestMetricsPayload
MetricsPayload = metric_models.MetricsPayload
MetricsStagePayload = metric_models.MetricsStagePayload
PythonLeakageMetricsPayload = metric_models.PythonLeakageMetricsPayload
TextMetricsPayload = metric_models.TextMetricsPayload


class NodeAttemptStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class ScoreAttemptStatus(StrEnum):
    SUCCESS = "success"


class GenerationRunStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"
    PARTIAL = "partial"


class TaskInputsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[StrictStr, Any]

    @model_validator(mode="after")
    def validate_values_size(self) -> TaskInputsPayload:
        validate_payload_size(
            self.values,
            max_bytes=TASK_INPUTS_MAX_BYTES,
            label="task inputs",
        )
        return self


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
        from dr_graph import graph_digest

        if self.graph_digest != graph_digest(self.graph):
            raise ValueError("graph_digest must match graph")
        validate_payload_size(
            self.model_dump(mode="json"),
            max_bytes=GRAPH_SNAPSHOT_MAX_BYTES,
            label="graph snapshot",
        )
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

    @model_validator(mode="after")
    def validate_usage_metadata_size(self) -> UsageCostPayload:
        validate_payload_size(
            self.usage_metadata,
            max_bytes=PROVIDER_TELEMETRY_MAX_BYTES,
            label="usage metadata",
        )
        return self


class ResponseMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_response_metadata_size(self) -> ResponseMetadataPayload:
        validate_payload_size(
            self.response_metadata,
            max_bytes=PROVIDER_TELEMETRY_MAX_BYTES,
            label="response metadata",
        )
        return self


class FailureMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_class: FailureClass | None = None
    error_type: StrictStr
    underlying_exception_type: StrictStr | None = None
    message: StrictStr
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class NodeOutputPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[StrictStr, Any]
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_output_size(self) -> NodeOutputPayload:
        validate_payload_size(
            {"values": self.values, "metadata": self.metadata},
            max_bytes=NODE_OUTPUT_MAX_BYTES,
            label="node output",
        )
        return self


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
    terminal_submission_text: StrictStr
    terminal_error: GenerationTerminalErrorPayload | None = None
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class ExtractedSubmissionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_submission: StrictStr | None = None
    extracted_code: StrictStr | None = None
    extraction_method: StrictStr | None = None
    parser_profile_id: StrictStr
    parser_version: StrictStr
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


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

    # dr-platform SubmittableItem protocol view: the frozen domain axes
    # under the library's neutral names.
    @property
    def item_id(self) -> str:
        return self.prediction_id

    @property
    def order_key(self) -> str:
        return self.fair_order_key

    @property
    def group_key(self) -> str:
        return self.experiment_name

    @model_validator(mode="after")
    def validate_spec_shape(self) -> PredictionSpecRecord:
        if self.repetition_seed < 0:
            raise ValueError("repetition_seed must be non-negative")
        if self.task.task_id != self.task_id:
            raise ValueError("task snapshot task_id must match spec task_id")
        validate_external_bindings(
            self.graph.graph,
            allowed_fields=self.task.inputs.values.keys(),
        )
        if self.provider_axis not in self.provider_configs:
            raise ValueError("provider_axis must be one of provider_configs")
        from whetstone.records.providers import (
            validate_provider_configs_identity,
        )

        validate_provider_configs_identity(self.provider_configs)
        from whetstone.records.hashing import (
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
        if self.status in (
            GenerationRunStatus.SUCCESS,
            GenerationRunStatus.PARTIAL,
        ):
            if self.summary.terminal_error is not None:
                raise ValueError(
                    f"{self.status.value} generation runs cannot have "
                    "terminal_error"
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
        if self.provider_config is not None:
            from whetstone.records.providers import (
                validate_provider_configs_identity,
            )

            validate_provider_configs_identity((self.provider_config,))
        if self.status is NodeAttemptStatus.SUCCESS:
            if self.output is None:
                raise ValueError("successful node attempts require output")
            if self.failure is not None:
                raise ValueError(
                    "successful node attempts cannot have failure"
                )
            if self.provider_config is None:
                raise ValueError(
                    "successful node attempts require provider_config"
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
    dataset_name: StrictStr
    dataset_split: StrictStr
    status: ScoreAttemptStatus
    submission_outcome: SubmissionOutcome
    score: StrictFloat | None = None
    extracted_submission: ExtractedSubmissionPayload
    metrics: MetricsPayload | None = None
    per_test_results: tuple[PerTestResultPayload, ...] = ()
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
        if self.per_test_results:
            per_test_payload = [
                case.model_dump(mode="json") for case in self.per_test_results
            ]
            validate_payload_size(
                per_test_payload,
                max_bytes=PER_TEST_RESULTS_MAX_BYTES,
                label="per_test_results",
            )
        if self.metrics is not None:
            if len(self.metrics.stages) > METRICS_STAGES_MAX_COUNT:
                raise ValueError(
                    f"metrics.stages cannot exceed {METRICS_STAGES_MAX_COUNT} "
                    "entries"
                )
            validate_payload_size(
                self.metrics.model_dump(mode="json"),
                max_bytes=METRICS_MAX_BYTES,
                label="metrics",
            )
        if (
            self.extracted_submission.parser_profile_id
            != self.parser_profile_id
        ):
            raise ValueError(
                "extracted_submission parser_profile_id must match "
                "parser_profile_id"
            )
        if self.extracted_submission.parser_version != self.parser_version:
            raise ValueError(
                "extracted_submission parser_version must match parser_version"
            )
        return self


class HarnessFailureCausePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exception_type: StrictStr
    message: StrictStr


class ScoreHarnessFailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_attempt_id: StrictStr
    prediction_id: StrictStr
    generation_run_id: StrictStr
    attempt_index: StrictInt
    scoring_profile_id: StrictStr
    scoring_profile_version: StrictStr
    parser_profile_id: StrictStr
    parser_version: StrictStr
    dataset_name: StrictStr
    dataset_split: StrictStr
    kind: StrictStr
    raw_submission: StrictStr
    extracted_submission: ExtractedSubmissionPayload | None = None
    cause: HarnessFailureCausePayload
    failure_class: FailureClass
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_failure_shape(self) -> ScoreHarnessFailureRecord:
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.kind != "harness_failure":
            raise ValueError("kind must be harness_failure")
        if (
            self.extracted_submission is not None
            and self.extracted_submission.parser_profile_id
            != self.parser_profile_id
        ):
            raise ValueError(
                "extracted_submission parser_profile_id must match failure"
            )
        if (
            self.extracted_submission is not None
            and self.extracted_submission.parser_version
            != self.parser_version
        ):
            raise ValueError(
                "extracted_submission parser_version must match failure"
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
