from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from whetstone.core.identity import TypedRef, typed_ref_for_record
from whetstone.eval import AggregationOutput
from whetstone.eval.aggregate import (
    AGGREGATE_SCHEMA,
    Aggregate,
    RowValue,
    TaskRows,
)
from whetstone.eval.attribution import attribute_published_row
from whetstone.eval.schema import (
    EVAL_TRACES_SCHEMA,
    EVAL_OUTPUTS_SCHEMA,
    EvalTraces,
    EvalTracesRef,
    EvalEvidence,
    EvalFailureEvidence,
    EvalOutputsRecord,
)
from whetstone.eval.schema_names import (
    EVAL_EVIDENCE_SCHEMA,
    EVAL_FAILURE_SCHEMA,
)
from whetstone.eval.config_ref import EVAL_CONFIG_RECORD_SCHEMA
from whetstone.experiment.candidate import CANDIDATE_RECORD_SCHEMA
from whetstone.experiment.reward import REWARD_SCHEMA, Reward, RewardRef
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import (
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


if TYPE_CHECKING:
    from whetstone.eval.protocol import EvalEngine


class EvalEvidenceValidation:
    _store: Any
    _engine: EvalEngine
    _attested_resolution: Any

    if TYPE_CHECKING:

        @staticmethod
        def _typed_ref(reference: Any) -> TypedRef: ...

    def _engine_for_intent(self, intent: OptimEvalRequest) -> EvalEngine:
        """The engine whose sampling this intent's evidence must match.

        An intent that declares a task subset was evaluated by the engine
        narrowed to that subset, and its evidence names that subset's Eval
        Config and Task Set. Validating it against the un-narrowed engine
        would reject honest evidence, so the narrowing is reproduced here.
        """

        task_hashes = intent.task_hashes
        if task_hashes is None:
            return self._engine
        from whetstone.optim.miprov2.engine_binding import (
            engine_for_task_hashes,
        )

        return engine_for_task_hashes(self._engine, task_hashes)

    def _plan(self, intent: OptimEvalRequest) -> Any:
        return self._engine_for_intent(intent).plan_snapshot

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
        engine = self._engine_for_intent(optim_eval_request)
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
            engine.eval_config_ref.record.model_dump(mode="json"),
        )
        if self._typed_ref(eval_config) != engine.eval_config_ref.record_ref:
            raise ValueError("persisted Eval Config reference diverged")
        policy, _ = self._store.put(
            PROVIDER_EXECUTION_POLICY_SCHEMA,
            engine.provider_execution_policy_record,
        )
        if (
            self._typed_ref(policy)
            != engine.provider_execution_policy_ref.record_ref
        ):
            raise ValueError(
                "persisted Provider Execution Policy reference diverged"
            )

    def _validate_target_objects(
        self, optim_eval_request: OptimEvalRequest
    ) -> None:
        engine = self._engine_for_intent(optim_eval_request)
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
            engine.eval_config_ref.record_ref,
            expected_schema=EVAL_CONFIG_RECORD_SCHEMA,
        )
        if eval_config_content != (
            engine.eval_config_ref.record.model_dump(mode="json")
        ):
            raise ValueError(
                "durable Eval Config does not equal the engine's exact config"
            )

    def _expected_eval_role(self, intent: OptimEvalRequest):
        return evaluation_role_for_split(self._plan(intent).split_role)

    def _validate_engine_eval_context(
        self,
        *,
        intent: OptimEvalRequest,
        eval_config_ref,
        eval_role,
        provider_execution_policy_ref,
    ) -> None:
        engine = self._engine_for_intent(intent)
        if eval_config_ref != engine.eval_config_ref:
            raise ValueError("evidence uses another Eval Config")
        if eval_role is not self._expected_eval_role(intent):
            raise ValueError("evidence uses another Evaluation Role")
        if provider_execution_policy_ref != engine.provider_execution_policy_ref:
            raise ValueError(
                "evidence uses another Provider Execution Policy"
            )

    def _validate_execution_contract(
        self, optim_eval_request: OptimEvalRequest
    ) -> None:
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
        evidence: EvalEvidence,
        intent: OptimEvalRequest,
    ) -> EvalOutputsRecord:
        plan = self._plan(intent)
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        _outputs_ref, content = self._load_exact(
            evidence.outputs_ref,
            expected_schema=EVAL_OUTPUTS_SCHEMA,
        )
        outputs = EvalOutputsRecord.model_validate(content)
        if outputs.traces_ref != evidence.traces_ref:
            raise ValueError(
                "evaluation outputs and evidence disagree on component traces"
            )
        if outputs.candidate != candidate_ref:
            raise ValueError("evaluation outputs belong to another candidate")
        self._validate_engine_eval_context(
            intent=intent,
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
        if outputs.num_seeds != plan.num_seeds:
            raise ValueError("evaluation outputs use another TaskTrial Plan")
        expected_rows = len(plan.task_hashes) * plan.num_seeds
        if len(outputs.outputs) != expected_rows:
            raise ValueError(
                "evaluation outputs do not cover the planned row matrix"
            )
        return outputs

    def _load_component_traces(
        self,
        evidence: EvalEvidence,
        intent: OptimEvalRequest,
        outputs: EvalOutputsRecord,
    ) -> EvalTraces:
        plan = self._plan(intent)
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        traces_ref, content = self._load_exact(
            evidence.traces_ref,
            expected_schema=EVAL_TRACES_SCHEMA,
        )
        traces = EvalTraces.model_validate_json(
            json.dumps(content)
        )
        if traces.record_content() != content:
            raise ValueError("component trace content is not canonical")
        EvalTracesRef(record=traces, record_ref=traces_ref)
        if traces.candidate != candidate_ref:
            raise ValueError("component traces belong to another candidate")
        self._validate_engine_eval_context(
            intent=intent,
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
        if traces.num_seeds != plan.num_seeds:
            raise ValueError("component traces use another TaskTrial Plan")
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
                or trace_row.seed_index != output_row.seed_index
            ):
                raise ValueError(
                    "component trace row identity/order disagrees with outputs"
                )
        return traces

    def _load_aggregate(
        self,
        evidence: EvalEvidence,
        intent: OptimEvalRequest,
    ) -> Aggregate:
        plan = self._plan(intent)
        _aggregate_ref, content = self._load_exact(
            evidence.aggregate_ref,
            expected_schema=AGGREGATE_SCHEMA,
        )
        expected_fields = {
            "name",
            "graph_hash",
            "eval_config_hash",
            "task_count",
            "num_seeds",
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
            num_seeds=content["num_seeds"],
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
        if (
            aggregate.eval_config_hash
            != self._engine_for_intent(intent).eval_config_ref.config_hash
        ):
            raise ValueError("Aggregate uses another Eval Config")
        if aggregate.task_count != len(evidence.task_hashes):
            raise ValueError("Aggregate task count is inconsistent")
        if aggregate.num_seeds != evidence.num_seeds:
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
            != aggregate.task_count * aggregate.num_seeds
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
        outputs: EvalOutputsRecord,
        evidence: EvalEvidence,
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
        # The per-task vector and the evaluation-level aggregate must read the
        # same rows the same way. A task's count is its number of *present*
        # rows, and its value is the mean over exactly those rows -- never a
        # sum padded with zeros over ``num_seeds``, which would score an
        # unobserved repeat as a real zero and never let a count fall below
        # ``num_seeds``.
        #
        # The producer's aggregation policy decides whether a partially
        # observed task keeps its present-row mean (skip) or withholds a value
        # (propagate). This validator does not hold that config, so it checks
        # the two claims every mean policy must satisfy: a value that is
        # present must be exactly the present-row mean, and a value may be
        # withheld only when the policy could withhold it -- which requires at
        # least one non-present row.
        expected_per_task_counts = tuple(
            sum(1 for row in task.completed_rows(outputs.num_seeds) if row.is_present)
            for task in task_rows
        )
        if evidence.per_task_counts != expected_per_task_counts:
            raise ValueError(
                "Evaluation Evidence per-task counts do not match outputs"
            )
        if len(evidence.per_task_values) != len(task_rows):
            raise ValueError(
                "Evaluation Evidence per-task values do not match outputs"
            )
        for task, count, value in zip(
            task_rows, expected_per_task_counts, evidence.per_task_values,
            strict=True,
        ):
            present = tuple(
                float(row.value)
                for row in task.completed_rows(outputs.num_seeds)
                if row.is_present and row.value is not None
            )
            if value is None:
                # Only an incomplete task may withhold a value. A fully present
                # task always reduces to a real mean, so ``None`` there means
                # the producer dropped a score it actually held.
                if count == outputs.num_seeds:
                    raise ValueError(
                        "Evaluation Evidence per-task values do not match "
                        "outputs"
                    )
                continue
            if not present or value != sum(present) / len(present):
                raise ValueError(
                    "Evaluation Evidence per-task values do not match outputs"
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
        plan = self._plan(intent)
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        assert resolution.eval_result_ref is not None
        evidence_ref, content = self._load_exact(
            resolution.eval_result_ref,
            expected_schema=EVAL_EVIDENCE_SCHEMA,
        )
        evidence = EvalEvidence.model_validate(content)
        if evidence.candidate != candidate_ref:
            raise ValueError(
                "Evaluation Evidence belongs to another candidate"
            )
        self._validate_engine_eval_context(
            intent=intent,
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
        if evidence.num_seeds != plan.num_seeds:
            raise ValueError("Evaluation Evidence uses another TaskTrial Plan")
        if len(evidence.per_task_values) != len(evidence.task_hashes):
            raise ValueError(
                "Evaluation Evidence per-task values are incomplete"
            )
        if len(evidence.per_task_counts) != len(evidence.task_hashes):
            raise ValueError(
                "Evaluation Evidence per-task counts are incomplete"
            )
        if any(
            count < 0 or count > evidence.num_seeds
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
            if reward.evidence_role is not self._expected_eval_role(intent):
                raise ValueError("Reward uses another Evaluation Role")
            expected_reward_evidence = (evidence.aggregate_ref,)
        if resolution.reward_evidence_refs != expected_reward_evidence:
            raise ValueError(
                "Intent Resolution Reward citations are not aggregate-only"
            )

    def _validate_failed_graph(self, resolution: IntentResolution) -> None:
        intent = resolution.optim_eval_request
        assert resolution.eval_result_ref is not None
        candidate_ref = candidate_reference(intent.eval_request.candidate)
        failure_ref, content = self._load_exact(
            resolution.eval_result_ref,
            expected_schema=EVAL_FAILURE_SCHEMA,
        )
        failure = EvalFailureEvidence.model_validate(content)
        if failure.candidate != candidate_ref:
            raise ValueError("Evaluation Failure belongs to another candidate")
        self._validate_engine_eval_context(
            intent=intent,
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


__all__ = ["EvalEvidenceValidation"]
