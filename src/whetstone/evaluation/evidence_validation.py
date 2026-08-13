from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from whetstone.core.identity import TypedRef, typed_ref_for_record
from whetstone.evaluation import AggregationOutput
from whetstone.evaluation.aggregate import (
    AGGREGATE_SCHEMA,
    Aggregate,
    RowValue,
    TaskRows,
)
from whetstone.evaluation.attribution import attribute_published_row
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    EvaluationComponentTraces,
    EvaluationComponentTracesRef,
    EvaluationEvidence,
    EvaluationEvidenceRef,
    EvaluationFailureEvidence,
    EvaluationFailureEvidenceRef,
    EvaluationOutputsRecord,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
)
from whetstone.evaluation.protocol import EvalRequest
from whetstone.experiment.binding import EVAL_CONFIG_RECORD_SCHEMA
from whetstone.experiment.candidate import CANDIDATE_RECORD_SCHEMA
from whetstone.experiment.reward import REWARD_SCHEMA, Reward, RewardRef
from whetstone.experiment.candidate import candidate_reference
from whetstone.optimization.contracts import (
    IntentOutcome,
    IntentResolution,
    OptimEvalRequest,
    ResolutionClass,
)
from whetstone.experiment.sampling import evaluation_role_for_split
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)


class EvaluationEvidenceValidation:
    _store: Any
    _engine: EvaluationEngine
    _attested_resolution: Any

    if TYPE_CHECKING:

        @staticmethod
        def _typed_ref(reference: Any) -> TypedRef: ...

    def _plan(self) -> Any:
        return self._engine.plan_snapshot

    def _load_exact(
        self,
        reference: Any,
        *,
        expected_schema: str,
    ) -> tuple[TypedRef, dict[str, Any]]:
        record_ref = self._typed_ref(reference)
        if record_ref.schema_name != expected_schema:
            raise ValueError(
                f"durable record must use schema {expected_schema!r}"
            )
        content = self._store.get(record_ref.reference)
        if not isinstance(content, dict):
            raise ValueError("durable record content must be a JSON object")
        if typed_ref_for_record(expected_schema, content) != record_ref:
            raise ValueError(
                "durable record reference does not address its exact content"
            )
        return record_ref, content

    def _load(
        self,
        reference: Any,
        *,
        expected_schema: str,
    ) -> tuple[TypedRef, dict[str, Any]]:
        return self._load_exact(reference, expected_schema=expected_schema)

    def _persist_intent_targets(
        self, optim_eval_request: OptimEvalRequest
    ) -> None:
        candidate, _ = self._store.put(
            CANDIDATE_RECORD_SCHEMA,
            optim_eval_request.eval_request.candidate.record_content(),
        )
        candidate_ref = candidate_reference(
            optim_eval_request.eval_request.candidate
        )
        if self._typed_ref(candidate) != candidate_ref.record_ref:
            raise ValueError("persisted candidate reference diverged")
        eval_config, _ = self._store.put(
            EVAL_CONFIG_RECORD_SCHEMA,
            self._engine.eval_config_ref.record.model_dump(mode="json"),
        )
        if self._typed_ref(eval_config) != self._engine.eval_config_ref.record_ref:
            raise ValueError("persisted Eval Config reference diverged")
        policy, _ = self._store.put(
            PROVIDER_EXECUTION_POLICY_SCHEMA,
            self._engine.provider_execution_policy_record,
        )
        if (
            self._typed_ref(policy)
            != self._engine.provider_execution_policy_ref.record_ref
        ):
            raise ValueError(
                "persisted Provider Execution Policy reference diverged"
            )

    def _validate_target_objects(
        self, optim_eval_request: OptimEvalRequest
    ) -> None:
        candidate_ref = candidate_reference(
            optim_eval_request.eval_request.candidate
        )
        _candidate_ref, candidate_content = self._load_exact(
            candidate_ref.record_ref,
            expected_schema=CANDIDATE_RECORD_SCHEMA,
        )
        if candidate_content != candidate_ref.record.record_content():
            raise ValueError(
                "durable candidate does not equal the Optim Eval Request "
                "candidate"
            )
        _eval_config_ref, eval_config_content = self._load_exact(
            self._engine.eval_config_ref.record_ref,
            expected_schema=EVAL_CONFIG_RECORD_SCHEMA,
        )
        if eval_config_content != (
            self._engine.eval_config_ref.record.model_dump(mode="json")
        ):
            raise ValueError(
                "durable Eval Config does not equal the engine's exact config"
            )

    def _expected_eval_role(self):
        return evaluation_role_for_split(self._plan().split_role)

    def _validate_engine_eval_context(self, *, eval_config_ref, eval_role, provider_execution_policy_ref) -> None:
        if eval_config_ref != self._engine.eval_config_ref:
            raise ValueError("evidence uses another Eval Config")
        if eval_role is not self._expected_eval_role():
            raise ValueError("evidence uses another Evaluation Role")
        if provider_execution_policy_ref != self._engine.provider_execution_policy_ref:
            raise ValueError(
                "evidence uses another Provider Execution Policy"
            )

    def _validate_execution_contract(
        self, optim_eval_request: OptimEvalRequest
    ) -> None:
        request = EvalRequest(
            request_id=optim_eval_request.eval_request.request_id,
            candidate=optim_eval_request.eval_request.candidate,
            metadata=optim_eval_request.eval_request.metadata,
        )
        self._engine.validate_request(request)
        policy_ref = self._engine.provider_execution_policy_ref
        _record_ref, content = self._load_exact(
            policy_ref.record_ref,
            expected_schema=PROVIDER_EXECUTION_POLICY_SCHEMA,
        )
        policy = ProviderExecutionPolicy.model_validate(content)
        if policy.identity_payload() != content:
            raise ValueError(
                "Provider Execution Policy content is not canonical"
            )
        if policy.identity_hash != policy_ref.record_hash:
            raise ValueError(
                "Provider Execution Policy identity hash disagrees with "
                "its exact record"
            )

    def _load_outputs(
        self,
        evidence: EvaluationEvidence,
        intent: OptimEvalRequest,
    ) -> EvaluationOutputsRecord:
        plan = self._plan()
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        _outputs_ref, content = self._load_exact(
            evidence.outputs_ref,
            expected_schema=EVALUATION_OUTPUTS_SCHEMA,
        )
        outputs = EvaluationOutputsRecord.model_validate(content)
        if outputs.component_traces_ref != evidence.component_traces_ref:
            raise ValueError(
                "evaluation outputs and evidence disagree on component traces"
            )
        if outputs.candidate != candidate_ref:
            raise ValueError("evaluation outputs belong to another candidate")
        self._validate_engine_eval_context(
            eval_config_ref=outputs.eval_config_ref,
            eval_role=outputs.eval_role,
            provider_execution_policy_ref=outputs.provider_execution_policy_ref,
        )
        if outputs.graph_hash != plan.graph_hash:
            raise ValueError("evaluation outputs use another generation graph")
        if outputs.metadata != intent.eval_request.metadata:
            raise ValueError("evaluation outputs use another purpose")
        if outputs.split_role != plan.split_role:
            raise ValueError("evaluation outputs use another sampling split")
        if outputs.task_hashes != plan.task_hashes:
            raise ValueError("evaluation outputs use another ordered Task Set")
        if outputs.num_samples != plan.num_samples:
            raise ValueError("evaluation outputs use another Sample Plan")
        expected_rows = len(plan.task_hashes) * plan.num_samples
        if len(outputs.outputs) != expected_rows:
            raise ValueError(
                "evaluation outputs do not cover the planned row matrix"
            )
        return outputs

    def _load_component_traces(
        self,
        evidence: EvaluationEvidence,
        intent: OptimEvalRequest,
        outputs: EvaluationOutputsRecord,
    ) -> EvaluationComponentTraces:
        plan = self._plan()
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        traces_ref, content = self._load_exact(
            evidence.component_traces_ref,
            expected_schema=EVALUATION_COMPONENT_TRACES_SCHEMA,
        )
        traces = EvaluationComponentTraces.model_validate_json(
            json.dumps(content)
        )
        if traces.record_content() != content:
            raise ValueError("component trace content is not canonical")
        EvaluationComponentTracesRef(record=traces, record_ref=traces_ref)
        if traces.candidate != candidate_ref:
            raise ValueError("component traces belong to another candidate")
        self._validate_engine_eval_context(
            eval_config_ref=traces.eval_config_ref,
            eval_role=traces.eval_role,
            provider_execution_policy_ref=traces.provider_execution_policy_ref,
        )
        if traces.graph_hash != plan.graph_hash:
            raise ValueError("component traces use another generation graph")
        if traces.metadata != intent.eval_request.metadata:
            raise ValueError("component traces use another purpose")
        if traces.split_role != plan.split_role:
            raise ValueError("component traces use another sampling split")
        if traces.task_hashes != plan.task_hashes:
            raise ValueError("component traces use another ordered Task Set")
        if traces.num_samples != plan.num_samples:
            raise ValueError("component traces use another Sample Plan")
        if len(traces.rows) != len(outputs.outputs):
            raise ValueError(
                "component traces and outputs must cover the same rows"
            )
        for trace_row, output_row in zip(
            traces.rows, outputs.outputs, strict=True
        ):
            if (
                trace_row.task_id != output_row.task_id
                or trace_row.task_hash != output_row.task_hash
                or trace_row.sample_index != output_row.sample_index
            ):
                raise ValueError(
                    "component trace row identity/order disagrees with outputs"
                )
        return traces

    def _load_aggregate(
        self,
        evidence: EvaluationEvidence,
        intent: OptimEvalRequest,
    ) -> Aggregate:
        plan = self._plan()
        _aggregate_ref, content = self._load_exact(
            evidence.aggregate_ref,
            expected_schema=AGGREGATE_SCHEMA,
        )
        expected_fields = {
            "name",
            "graph_hash",
            "eval_config_hash",
            "task_count",
            "num_samples",
            "aggregation_output",
            "rows_present",
            "rows_missing",
            "rows_failed",
            "rows_invalid",
        }
        if set(content) != expected_fields:
            raise ValueError("Aggregate wire fields are not exact")
        aggregate = Aggregate(
            name=content["name"],
            graph_hash=content["graph_hash"],
            eval_config_hash=content["eval_config_hash"],
            task_count=content["task_count"],
            num_samples=content["num_samples"],
            aggregation_output=AggregationOutput.model_validate(
                content["aggregation_output"]
            ),
            rows_present=content["rows_present"],
            rows_missing=content["rows_missing"],
            rows_failed=content["rows_failed"],
            rows_invalid=content["rows_invalid"],
        )
        if aggregate.record_content() != content:
            raise ValueError("Aggregate content is not canonical")
        if aggregate.graph_hash != evidence.graph_hash:
            raise ValueError("Evaluation Evidence graph hash is inconsistent")
        if aggregate.graph_hash != plan.graph_hash:
            raise ValueError("Aggregate uses another generation graph")
        if evidence.graph_config_ref != aggregate.graph_hash:
            raise ValueError(
                "Evaluation Evidence graph config is inconsistent"
            )
        if aggregate.eval_config_hash != self._engine.eval_config_ref.config_hash:
            raise ValueError("Aggregate uses another Eval Config")
        if aggregate.task_count != len(evidence.task_hashes):
            raise ValueError("Aggregate task count is inconsistent")
        if aggregate.num_samples != evidence.num_samples:
            raise ValueError("Aggregate repeat count is inconsistent")
        if aggregate.name != evidence.aggregate_name:
            raise ValueError("Aggregate name is inconsistent")
        if aggregate.aggregation_output.value != evidence.aggregate_value:
            raise ValueError("Aggregate value is inconsistent")
        if (
            aggregate.aggregation_output.status.value
            != evidence.aggregate_status
        ):
            raise ValueError("Aggregate status is inconsistent")
        row_accounting = evidence.row_accounting
        if (
            row_accounting.planned
            != aggregate.task_count * aggregate.num_samples
            or row_accounting.present != aggregate.rows_present
            or row_accounting.missing != aggregate.rows_missing
            or row_accounting.failed != aggregate.rows_failed
            or row_accounting.invalid != aggregate.rows_invalid
        ):
            raise ValueError(
                "Evaluation Evidence row accounting is inconsistent"
            )
        return aggregate

    def _validate_row_derivation(
        self,
        *,
        outputs: EvaluationOutputsRecord,
        evidence: EvaluationEvidence,
        aggregate: Aggregate,
        intent: OptimEvalRequest,
    ) -> None:
        rows_by_task: dict[str, list[RowValue]] = {
            task_hash: [] for task_hash in outputs.task_hashes
        }
        for row in outputs.outputs:
            value = attribute_published_row(
                score=row.score,
                failed=row.failed,
                missing=row.missing,
                invalid=row.invalid,
            )
            rows_by_task[row.task_hash].append(value)
        task_rows = tuple(
            TaskRows(
                task_hash=task_hash,
                rows=tuple(rows_by_task[task_hash]),
            )
            for task_hash in outputs.task_hashes
        )
        expected_per_task_values = tuple(
            sum(
                float(row.value or 0.0) if row.is_present else 0.0
                for row in task.rows
            )
            / outputs.num_samples
            for task in task_rows
        )
        expected_per_task_counts = tuple(
            outputs.num_samples for _task in task_rows
        )
        if evidence.per_task_values != expected_per_task_values:
            raise ValueError(
                "Evaluation Evidence per-task values do not match outputs"
            )
        if evidence.per_task_counts != expected_per_task_counts:
            raise ValueError(
                "Evaluation Evidence per-task counts do not match outputs"
            )

    def _load_reward(
        self,
        reward_ref: RewardRef,
        *,
        aggregate_ref: TypedRef,
        aggregate_name: str,
        aggregate_value: float | None,
    ) -> Reward:
        reward_record_ref, reward_content = self._load_exact(
            reward_ref.record_ref,
            expected_schema=REWARD_SCHEMA,
        )
        reward = Reward.model_validate(reward_content)
        RewardRef(record=reward, record_ref=reward_record_ref)
        if reward.evidence_refs != (aggregate_ref,):
            raise ValueError("Reward must cite the primary aggregate only")
        if reward.input_citations[0].name != aggregate_name:
            raise ValueError("Reward citation name is inconsistent")
        if reward.input_citations[0].value != aggregate_value:
            raise ValueError("Reward citation value is inconsistent")
        return reward

    def _validate_completed_graph(
        self,
        resolution: IntentResolution,
    ) -> None:
        intent = resolution.optim_eval_request
        plan = self._plan()
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        assert resolution.evaluation_result_ref is not None
        evidence_ref, content = self._load_exact(
            resolution.evaluation_result_ref,
            expected_schema=EVALUATION_EVIDENCE_SCHEMA,
        )
        evidence = EvaluationEvidence.model_validate(content)
        EvaluationEvidenceRef(record=evidence, record_ref=evidence_ref)
        if evidence.candidate != candidate_ref:
            raise ValueError(
                "Evaluation Evidence belongs to another candidate"
            )
        self._validate_engine_eval_context(
            eval_config_ref=evidence.eval_config_ref,
            eval_role=evidence.eval_role,
            provider_execution_policy_ref=evidence.provider_execution_policy_ref,
        )
        if evidence.metadata != intent.eval_request.metadata:
            raise ValueError("Evaluation Evidence uses another purpose")
        if evidence.dataset_hash != plan.dataset_hash:
            raise ValueError("Evaluation Evidence uses another dataset")
        if evidence.task_hashes != plan.task_hashes:
            raise ValueError(
                "Evaluation Evidence uses another ordered Task Set"
            )
        if evidence.num_samples != plan.num_samples:
            raise ValueError("Evaluation Evidence uses another Sample Plan")
        if len(evidence.per_task_values) != len(evidence.task_hashes):
            raise ValueError(
                "Evaluation Evidence per-task values are incomplete"
            )
        if len(evidence.per_task_counts) != len(evidence.task_hashes):
            raise ValueError(
                "Evaluation Evidence per-task counts are incomplete"
            )
        if any(
            count < 0 or count > evidence.num_samples
            for count in evidence.per_task_counts
        ):
            raise ValueError("Evaluation Evidence per-task count is invalid")
        outputs = self._load_outputs(evidence, intent)
        self._load_component_traces(evidence, intent, outputs)
        aggregate = self._load_aggregate(evidence, intent)
        self._validate_row_derivation(
            outputs=outputs,
            evidence=evidence,
            aggregate=aggregate,
            intent=intent,
        )
        if evidence.reward_ref != resolution.reward_ref:
            raise ValueError(
                "Evaluation Evidence and Intent Resolution disagree on Reward"
            )
        expected_reward_evidence: tuple[TypedRef, ...] = ()
        if evidence.reward_ref is not None:
            reward = self._load_reward(
                evidence.reward_ref,
                aggregate_ref=evidence.aggregate_ref,
                aggregate_name=evidence.aggregate_name,
                aggregate_value=evidence.aggregate_value,
            )
            if reward.evidence_role is not self._expected_eval_role():
                raise ValueError("Reward uses another Evaluation Role")
            expected_reward_evidence = (evidence.aggregate_ref,)
        if resolution.reward_evidence_refs != expected_reward_evidence:
            raise ValueError(
                "Intent Resolution Reward citations are not aggregate-only"
            )

    def _validate_failed_graph(self, resolution: IntentResolution) -> None:
        intent = resolution.optim_eval_request
        assert resolution.evaluation_result_ref is not None
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        failure_ref, content = self._load_exact(
            resolution.evaluation_result_ref,
            expected_schema=EVALUATION_FAILURE_SCHEMA,
        )
        failure = EvaluationFailureEvidence.model_validate(content)
        EvaluationFailureEvidenceRef(record=failure, record_ref=failure_ref)
        if failure.candidate != candidate_ref:
            raise ValueError("Evaluation Failure belongs to another candidate")
        self._validate_engine_eval_context(
            eval_config_ref=failure.eval_config_ref,
            eval_role=failure.eval_role,
            provider_execution_policy_ref=failure.provider_execution_policy_ref,
        )
        if failure.metadata != intent.eval_request.metadata:
            raise ValueError("Evaluation Failure uses another purpose")
        if (
            resolution.detail.classification
            is not ResolutionClass.INFRASTRUCTURE
            or resolution.detail.message != failure.message
        ):
            raise ValueError(
                "failed resolution detail disagrees with failure evidence"
            )
        terminal = resolution.terminal_failure
        assert terminal is not None
        expected_details = {
            "evidence_schema": failure_ref.schema_name,
            "evidence_content_hash": failure_ref.content_hash,
        }
        if terminal.code != f"evaluation_{failure.exception_type}":
            raise ValueError("terminal failure code disagrees with evidence")
        if terminal.message != failure.message:
            raise ValueError(
                "terminal failure message disagrees with evidence"
            )
        if dict(terminal.details) != expected_details:
            raise ValueError("terminal failure details disagree with evidence")

    def _validate_result_graph(
        self,
        resolution: IntentResolution,
        *,
        expected_optim_eval_request: OptimEvalRequest,
        require_attestation: bool = True,
    ) -> None:
        if resolution.optim_eval_request != expected_optim_eval_request:
            raise ValueError(
                "durable Intent Resolution belongs to another Optim Eval Request"
            )
        self._validate_target_objects(expected_optim_eval_request)
        if resolution.outcome in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            self._validate_execution_contract(expected_optim_eval_request)
        if require_attestation and resolution.outcome in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            attested = self._attested_resolution(expected_optim_eval_request)
            if attested != resolution:
                raise ValueError(
                    "Intent Resolution does not equal the exact terminal "
                    "Evaluation Result Attestation"
                )
        if resolution.outcome is IntentOutcome.COMPLETED:
            self._validate_completed_graph(resolution)
        elif resolution.outcome is IntentOutcome.FAILED:
            self._validate_failed_graph(resolution)


__all__ = ["EvaluationEvidenceValidation"]
