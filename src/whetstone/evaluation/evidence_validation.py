from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from whetstone.core.identity import (
    TypedRef,
    typed_ref_for_record,
)
from whetstone.envs.code_comp.constants import DECODER_TEMPLATE
from whetstone.envs.code_comp.rollout.encdec import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
)
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import LLM_NODE_ID, render_prompt
from whetstone.envs.sampling import validate_evaluation_role_for_split
from whetstone.evaluation import AggregationOutput
from whetstone.evaluation.aggregate import (
    ROLLOUT_AGGREGATE_SCHEMA,
    RolloutAggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.evaluation.engine import EvaluationRequest
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
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.experiment.binding import EVAL_CONFIG_RECORD_SCHEMA
from whetstone.experiment.candidate import CANDIDATE_RECORD_SCHEMA
from whetstone.experiment.reward import REWARD_SCHEMA, Reward, RewardRef
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
)
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)


class EvaluationEvidenceValidation:
    _store: Any
    _engine: Any
    _attested_resolution: Any

    if TYPE_CHECKING:

        @staticmethod
        def _typed_ref(reference: Any) -> TypedRef: ...

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
        expected_intent: EvaluationIntent,
    ) -> IntentResolution:
        _record_ref, content = self._load_exact(
            reference,
            expected_schema=INTENT_RESOLUTION_SCHEMA,
        )
        resolution = IntentResolution.model_validate(content)
        self._validate_result_graph(
            resolution, expected_intent=expected_intent
        )
        return resolution

    def _persist_intent_targets(self, intent: EvaluationIntent) -> None:
        candidate, _ = self._store.put(
            CANDIDATE_RECORD_SCHEMA,
            intent.candidate.record.record_content(),
        )
        if self._typed_ref(candidate) != intent.candidate.record_ref:
            raise ValueError("persisted candidate reference diverged")
        eval_config, _ = self._store.put(
            EVAL_CONFIG_RECORD_SCHEMA,
            intent.target_eval_config.record.model_dump(mode="json"),
        )
        if (
            self._typed_ref(eval_config)
            != intent.target_eval_config.record_ref
        ):
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

    def _validate_target_objects(self, intent: EvaluationIntent) -> None:
        _candidate_ref, candidate_content = self._load_exact(
            intent.candidate.record_ref,
            expected_schema=CANDIDATE_RECORD_SCHEMA,
        )
        if candidate_content != intent.candidate.record.record_content():
            raise ValueError(
                "durable candidate does not equal the Intent candidate"
            )
        _eval_config_ref, eval_config_content = self._load_exact(
            intent.target_eval_config.record_ref,
            expected_schema=EVAL_CONFIG_RECORD_SCHEMA,
        )
        if eval_config_content != intent.target_eval_config.record.model_dump(
            mode="json"
        ):
            raise ValueError(
                "durable Eval Config does not equal the Intent target"
            )

    def _validate_execution_contract(self, intent: EvaluationIntent) -> None:
        request = EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
        self._engine.validate_request(request)
        policy_ref = intent.evaluation_binding.provider_execution_policy_ref
        if policy_ref is None:
            raise ValueError(
                "Evaluation Binding must name a Provider Execution Policy"
            )
        _record_ref, content = self._load_exact(
            policy_ref.record_ref,
            expected_schema=PROVIDER_EXECUTION_POLICY_SCHEMA,
        )
        policy = ProviderExecutionPolicy.model_validate(content)
        if policy.identity_payload() != content:
            raise ValueError(
                "Provider Execution Policy content is not canonical"
            )
        if policy.identity_hash != policy_ref.identity_hash:
            raise ValueError(
                "Provider Execution Policy identity hash disagrees with "
                "its exact record"
            )

    def _load_outputs(
        self,
        evidence: EvaluationEvidence,
        intent: EvaluationIntent,
    ) -> EvaluationOutputsRecord:
        _outputs_ref, content = self._load_exact(
            evidence.outputs_ref,
            expected_schema=EVALUATION_OUTPUTS_SCHEMA,
        )
        outputs = EvaluationOutputsRecord.model_validate(content)
        if outputs.component_traces_ref != evidence.component_traces_ref:
            raise ValueError(
                "evaluation outputs and evidence disagree on component traces"
            )
        if outputs.candidate != intent.candidate:
            raise ValueError("evaluation outputs belong to another candidate")
        if outputs.evaluation_binding != intent.evaluation_binding:
            raise ValueError(
                "evaluation outputs use another Evaluation Binding"
            )
        if outputs.evaluation_role is not intent.evaluation_binding.role:
            raise ValueError("evaluation outputs use another Evaluation Role")
        if (
            outputs.graph_hash
            != self._engine.experiment.rollout_definition.graph_hash
        ):
            raise ValueError("evaluation outputs use another rollout graph")
        if outputs.purpose != intent.purpose:
            raise ValueError("evaluation outputs use another purpose")
        if outputs.split_role != self._engine.sampling.split_role:
            raise ValueError("evaluation outputs use another sampling split")
        validate_evaluation_role_for_split(
            split_role=outputs.split_role,
            evaluation_role=outputs.evaluation_role,
        )
        expected_tasks = self._engine.sampling.task_set.task_identities
        expected_repeats = self._engine.sampling.repeat_plan.repeat_count
        if outputs.task_identities != expected_tasks:
            raise ValueError("evaluation outputs use another ordered Task Set")
        if outputs.repeat_count != expected_repeats:
            raise ValueError("evaluation outputs use another Repeat Plan")

        instances = tuple(self._engine.sampling.instances)
        expected_instance_by_task = {
            task_identity: instance
            for task_identity, instance in zip(
                expected_tasks, instances, strict=True
            )
        }
        spec = env_spec(self._engine.experiment.env_name)
        procedure_config_hash = (
            self._engine.experiment.rollout_definition.procedure_config_hash
        )
        for row in outputs.outputs:
            instance = expected_instance_by_task[row.task_identity]
            if row.instance_id != str(instance.id):
                raise ValueError(
                    "evaluation output task and instance do not align"
                )
            expected_prompt = render_prompt(
                spec,
                intent.candidate.record,
                instance,
            )
            if row.rendered_prompt != expected_prompt:
                raise ValueError(
                    "evaluation output trace does not match the candidate"
                )
            if row.max_budget is not None or row.over_budget is not None:
                raise ValueError(
                    "generic evaluation outputs cannot carry budget accounting"
                )
            if row.invalid:
                raise ValueError(
                    "generic evaluation outputs cannot be invalid rows"
                )
            if row.failed:
                if not row.failure_code:
                    raise ValueError(
                        "failed evaluation output has inconsistent failure "
                        "accounting"
                    )
                continue
            if row.missing:
                if (
                    row.output_text is not None
                    or row.finish_reason is not None
                    or row.provider_error is not None
                    or row.failure_code
                ):
                    raise ValueError(
                        "missing evaluation output has inconsistent absence "
                        "accounting"
                    )
                continue
            if (
                row.output_text is None
                or row.provider_error is not None
                or row.failure_code
            ):
                raise ValueError(
                    "successful evaluation output has inconsistent provider "
                    "accounting"
                )
            expected_score = env_exact_match_score(
                env=spec,
                generation=row.output_text,
                gold=instance.gold,
                evaluation_procedure_config_hash=procedure_config_hash,
            )
            if row.score != float(expected_score.value):
                raise ValueError(
                    "evaluation output score is not derived by the canonical "
                    "local oracle"
                )
        return outputs

    def _load_component_traces(
        self,
        evidence: EvaluationEvidence,
        intent: EvaluationIntent,
        outputs: EvaluationOutputsRecord,
    ) -> EvaluationComponentTraces:
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
        if traces.candidate != intent.candidate:
            raise ValueError("component traces belong to another candidate")
        if traces.evaluation_binding != intent.evaluation_binding:
            raise ValueError("component traces use another Evaluation Binding")
        if traces.evaluation_role is not intent.evaluation_binding.role:
            raise ValueError("component traces use another Evaluation Role")
        if (
            traces.graph_hash
            != self._engine.experiment.rollout_definition.graph_hash
        ):
            raise ValueError("component traces use another rollout graph")
        if traces.purpose != intent.purpose:
            raise ValueError("component traces use another purpose")
        if traces.split_role != self._engine.sampling.split_role:
            raise ValueError("component traces use another sampling split")
        validate_evaluation_role_for_split(
            split_role=traces.split_role,
            evaluation_role=traces.evaluation_role,
        )
        expected_tasks = self._engine.sampling.task_set.task_identities
        expected_repeats = self._engine.sampling.repeat_plan.repeat_count
        if traces.task_identities != expected_tasks:
            raise ValueError("component traces use another ordered Task Set")
        if traces.repeat_count != expected_repeats:
            raise ValueError("component traces use another Repeat Plan")
        if len(traces.rows) != len(outputs.outputs):
            raise ValueError(
                "component traces and outputs must cover the same rows"
            )

        rollout_definition = getattr(
            self._engine.experiment.rollout_definition,
            "definition",
            None,
        )
        if rollout_definition is None:
            raise ValueError(
                "rollout graph must expose its exact Node Definition"
            )
        llm_nodes = tuple(
            node
            for node in rollout_definition.nodes
            if node.node_type == "whetstone.llm-call/v1"
        )
        if not llm_nodes:
            raise ValueError(
                "rollout graph must declare an executed LLM component"
            )
        component_ids = tuple(node.node_id for node in llm_nodes)
        if component_ids not in {
            (LLM_NODE_ID,),
            (ENCODER_NODE_ID, DECODER_NODE_ID),
        }:
            raise ValueError(
                "rollout graph uses an unsupported LLM component transition"
            )

        expected_instance_by_task = {
            task_identity: str(instance.id)
            for task_identity, instance in zip(
                expected_tasks,
                self._engine.sampling.instances,
                strict=True,
            )
        }
        for trace_row, output_row in zip(
            traces.rows, outputs.outputs, strict=True
        ):
            if (
                trace_row.instance_id != output_row.instance_id
                or trace_row.task_identity != output_row.task_identity
                or trace_row.repeat != output_row.repeat
            ):
                raise ValueError(
                    "component trace row identity/order disagrees with outputs"
                )
            if (
                trace_row.instance_id
                != expected_instance_by_task[trace_row.task_identity]
            ):
                raise ValueError(
                    "component trace task and instance do not align"
                )
            trace = trace_row.executed_component_trace
            if trace.row_state is ExecutedRowState.SUCCESS:
                if (
                    output_row.failed
                    or output_row.missing
                    or output_row.invalid
                ):
                    raise ValueError(
                        "successful component trace disagrees with output "
                        "state"
                    )
                if len(trace.executed_component_steps) != len(llm_nodes):
                    raise ValueError(
                        "successful row must trace every declared LLM "
                        "component"
                    )
            elif trace.row_state is ExecutedRowState.FAILED:
                if not output_row.failed:
                    raise ValueError(
                        "failed component trace disagrees with output state"
                    )
            else:
                if not output_row.missing:
                    raise ValueError(
                        "missing component trace disagrees with output state"
                    )

            if len(trace.executed_component_steps) > len(llm_nodes):
                raise ValueError(
                    "component trace exceeds the declared LLM graph"
                )
            for step, node in zip(
                trace.executed_component_steps, llm_nodes, strict=False
            ):
                expected_inputs = tuple(
                    field.name
                    for field in node.fields
                    if field.role.value == "input"
                )
                expected_outputs = tuple(
                    field.name
                    for field in node.fields
                    if field.role.value == "output"
                )
                if step.component_id != node.node_id:
                    raise ValueError(
                        "component trace is not the declared LLM-node prefix"
                    )
                if (
                    step.input_field_names != expected_inputs
                    or step.output_field_names != expected_outputs
                ):
                    raise ValueError(
                        "component trace fields do not match the Node "
                        "Definition"
                    )

            if trace.executed_component_steps:
                first_step = trace.executed_component_steps[0]
                if first_step.inputs[first_step.input_field_names[0]] != (
                    output_row.rendered_prompt
                ):
                    raise ValueError(
                        "component trace input does not match the rendered "
                        "prompt"
                    )
            if (
                component_ids == (ENCODER_NODE_ID, DECODER_NODE_ID)
                and len(trace.executed_component_steps) == 2
            ):
                encode_step, decode_step = trace.executed_component_steps
                encoder_generation = encode_step.outputs["generation"]
                if type(encoder_generation) is not str:
                    raise ValueError(
                        "ED1 encoder generation must be an exact string"
                    )
                expected_decoder_prompt = DECODER_TEMPLATE.format(
                    encoder_output=encoder_generation
                )
                if decode_step.inputs["prompt"] != expected_decoder_prompt:
                    raise ValueError(
                        "ED1 decoder input does not match the canonical "
                        "encoder-output frame"
                    )

            terminal_step_present = len(trace.executed_component_steps) == len(
                llm_nodes
            )
            if terminal_step_present:
                final_step = trace.executed_component_steps[-1]
                output_field = final_step.output_field_names[0]
                if final_step.outputs[output_field] != output_row.output_text:
                    raise ValueError(
                        "component trace output does not match the final "
                        "output"
                    )
            elif output_row.output_text is not None:
                raise ValueError(
                    "a nonterminal component prefix cannot carry final output"
                )
        return traces

    def _load_aggregate(
        self,
        evidence: EvaluationEvidence,
        intent: EvaluationIntent,
    ) -> RolloutAggregate:
        _aggregate_ref, content = self._load_exact(
            evidence.aggregate_ref,
            expected_schema=ROLLOUT_AGGREGATE_SCHEMA,
        )
        expected_fields = {
            "name",
            "graph_hash",
            "eval_config_hash",
            "evaluation_binding_hash",
            "task_count",
            "repeat_count",
            "aggregation_output",
            "rows_present",
            "rows_missing",
            "rows_failed",
            "rows_invalid",
        }
        if set(content) != expected_fields:
            raise ValueError("Rollout Aggregate wire fields are not exact")
        for field in (
            "name",
            "graph_hash",
            "eval_config_hash",
            "evaluation_binding_hash",
        ):
            if type(content[field]) is not str:
                raise ValueError(f"Rollout Aggregate {field} must be a string")
        for field in (
            "task_count",
            "repeat_count",
            "rows_present",
            "rows_missing",
            "rows_failed",
            "rows_invalid",
        ):
            if type(content[field]) is not int:
                raise ValueError(
                    f"Rollout Aggregate {field} must be an integer"
                )
        aggregate = RolloutAggregate(
            name=content["name"],
            graph_hash=content["graph_hash"],
            eval_config_hash=content["eval_config_hash"],
            evaluation_binding_hash=content["evaluation_binding_hash"],
            task_count=content["task_count"],
            repeat_count=content["repeat_count"],
            aggregation_output=AggregationOutput.model_validate(
                content["aggregation_output"]
            ),
            rows_present=content["rows_present"],
            rows_missing=content["rows_missing"],
            rows_failed=content["rows_failed"],
            rows_invalid=content["rows_invalid"],
        )
        if aggregate.record_content() != content:
            raise ValueError("Rollout Aggregate content is not canonical")
        if aggregate.graph_hash != evidence.graph_hash:
            raise ValueError("Evaluation Evidence graph hash is inconsistent")
        if (
            aggregate.graph_hash
            != self._engine.experiment.rollout_definition.graph_hash
        ):
            raise ValueError("Rollout Aggregate uses another rollout graph")
        if evidence.graph_config_ref != aggregate.graph_hash:
            raise ValueError(
                "Evaluation Evidence graph config is inconsistent"
            )
        if (
            aggregate.eval_config_hash
            != intent.target_eval_config.identity_hash
        ):
            raise ValueError("Rollout Aggregate uses another Eval Config")
        if (
            aggregate.evaluation_binding_hash
            != intent.evaluation_binding.identity_hash()
        ):
            raise ValueError(
                "Rollout Aggregate uses another Evaluation Binding"
            )
        if aggregate.task_count != len(evidence.task_identities):
            raise ValueError("Rollout Aggregate task count is inconsistent")
        if aggregate.repeat_count != evidence.repeat_count:
            raise ValueError("Rollout Aggregate repeat count is inconsistent")
        if aggregate.name != evidence.aggregate_name:
            raise ValueError("Rollout Aggregate name is inconsistent")
        if aggregate.aggregation_output.value != evidence.aggregate_value:
            raise ValueError("Rollout Aggregate value is inconsistent")
        if (
            aggregate.aggregation_output.status.value
            != evidence.aggregate_status
        ):
            raise ValueError("Rollout Aggregate status is inconsistent")
        row_accounting = evidence.row_accounting
        if (
            row_accounting.planned
            != aggregate.task_count * aggregate.repeat_count
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
        aggregate: RolloutAggregate,
        intent: EvaluationIntent,
    ) -> None:
        rows_by_task: dict[str, list[RowValue]] = {
            task_identity: [] for task_identity in outputs.task_identities
        }
        for row in outputs.outputs:
            if row.failed:
                value = RowValue(failed=True)
            elif row.missing:
                value = RowValue(missing=True)
            elif row.invalid:
                value = RowValue(invalid=True)
            else:
                assert row.score is not None
                value = RowValue(value=row.score)
            rows_by_task[row.task_identity].append(value)
        task_rows = tuple(
            TaskRows(
                task_identity=task_identity,
                rows=tuple(rows_by_task[task_identity]),
            )
            for task_identity in outputs.task_identities
        )
        expected_per_task_values = tuple(
            sum(
                float(row.value or 0.0) if row.is_present else 0.0
                for row in task.rows
            )
            / outputs.repeat_count
            for task in task_rows
        )
        expected_per_task_counts = tuple(
            outputs.repeat_count for _task in task_rows
        )
        if evidence.per_task_values != expected_per_task_values:
            raise ValueError(
                "Evaluation Evidence per-task values do not match outputs"
            )
        if evidence.per_task_counts != expected_per_task_counts:
            raise ValueError(
                "Evaluation Evidence per-task counts do not match outputs"
            )
        expected_aggregate = unweighted_task_mean(
            aggregate_name=evidence.aggregate_name,
            graph_hash=self._engine.experiment.rollout_definition.graph_hash,
            evaluation_binding_hash=intent.evaluation_binding.identity_hash(),
            task_rows=task_rows,
            plan=self._engine.sampling.evaluation_matrix_plan,
        )
        if aggregate.record_content() != expected_aggregate.record_content():
            raise ValueError(
                "Rollout Aggregate is not derived from the exact output rows"
            )

    def _load_reward(
        self,
        reward_ref: RewardRef,
        *,
        aggregate_ref: TypedRef,
        aggregate_name: str,
        aggregate_value: float | None,
    ) -> Reward:
        _record_ref, content = self._load_exact(
            reward_ref.record_ref,
            expected_schema=REWARD_SCHEMA,
        )
        reward = Reward.model_validate(content)
        loaded_ref = RewardRef(record=reward, record_ref=reward_ref.record_ref)
        if loaded_ref != reward_ref:
            raise ValueError(
                "persisted Reward differs from its embedded record"
            )
        if reward.evidence_refs != (aggregate_ref,):
            raise ValueError(
                "Reward evidence must be the ordered aggregate-only citations"
            )
        if len(reward.input_citations) != 1:
            raise ValueError("evaluation Reward must have one aggregate input")
        citation = reward.input_citations[0]
        if (
            citation.name != aggregate_name
            or citation.value != aggregate_value
        ):
            raise ValueError("Reward citation does not equal its aggregate")
        return reward

    def _validate_completed_graph(
        self,
        resolution: IntentResolution,
    ) -> None:
        intent = resolution.intent
        assert resolution.evaluation_result_ref is not None
        evidence_ref, content = self._load_exact(
            resolution.evaluation_result_ref,
            expected_schema=EVALUATION_EVIDENCE_SCHEMA,
        )
        evidence = EvaluationEvidence.model_validate(content)
        EvaluationEvidenceRef(record=evidence, record_ref=evidence_ref)
        if evidence.candidate != intent.candidate:
            raise ValueError(
                "Evaluation Evidence belongs to another candidate"
            )
        if evidence.evaluation_binding != intent.evaluation_binding:
            raise ValueError(
                "Evaluation Evidence uses another Evaluation Binding"
            )
        if evidence.purpose != intent.purpose:
            raise ValueError("Evaluation Evidence uses another purpose")
        if (
            evidence.dataset_identity
            != self._engine.sampling.task_set.dataset_revision
        ):
            raise ValueError("Evaluation Evidence uses another dataset")
        if (
            evidence.task_identities
            != self._engine.sampling.task_set.task_identities
        ):
            raise ValueError(
                "Evaluation Evidence uses another ordered Task Set"
            )
        if (
            evidence.repeat_count
            != self._engine.sampling.repeat_plan.repeat_count
        ):
            raise ValueError("Evaluation Evidence uses another Repeat Plan")
        if len(evidence.per_task_values) != len(evidence.task_identities):
            raise ValueError(
                "Evaluation Evidence per-task values are incomplete"
            )
        if len(evidence.per_task_counts) != len(evidence.task_identities):
            raise ValueError(
                "Evaluation Evidence per-task counts are incomplete"
            )
        if any(
            count < 0 or count > evidence.repeat_count
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
            if reward.evidence_role is not intent.evaluation_binding.role:
                raise ValueError("Reward uses another Evaluation Role")
            expected_reward_evidence = (evidence.aggregate_ref,)
        if resolution.reward_evidence_refs != expected_reward_evidence:
            raise ValueError(
                "Intent Resolution Reward citations are not aggregate-only"
            )

    def _validate_failed_graph(self, resolution: IntentResolution) -> None:
        intent = resolution.intent
        assert resolution.evaluation_result_ref is not None
        failure_ref, content = self._load_exact(
            resolution.evaluation_result_ref,
            expected_schema=EVALUATION_FAILURE_SCHEMA,
        )
        failure = EvaluationFailureEvidence.model_validate(content)
        EvaluationFailureEvidenceRef(record=failure, record_ref=failure_ref)
        if failure.candidate != intent.candidate:
            raise ValueError("Evaluation Failure belongs to another candidate")
        if failure.evaluation_binding != intent.evaluation_binding:
            raise ValueError(
                "Evaluation Failure uses another Evaluation Binding"
            )
        if failure.purpose != intent.purpose:
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
        expected_intent: EvaluationIntent,
        require_attestation: bool = True,
    ) -> None:
        if resolution.intent != expected_intent:
            raise ValueError(
                "durable Intent Resolution belongs to another intent"
            )
        self._validate_target_objects(expected_intent)
        if resolution.outcome in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            self._validate_execution_contract(expected_intent)
        if require_attestation and resolution.outcome in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            attested = self._attested_resolution(expected_intent)
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
