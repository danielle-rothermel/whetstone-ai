"""Restart-safe Evaluation Service backed by the canonical engine."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dr_code.eval import AggregationOutput
from dr_store import BindingConflictError, BindStatus, ObjectStore

from whetstone.code_eval.aggregate import (
    ROLLOUT_AGGREGATE_SCHEMA,
    RolloutAggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.envs.ed1 import DECODER_TEMPLATE
from whetstone.envs.encdec_rollout import DECODER_NODE_ID, ENCODER_NODE_ID
from whetstone.envs.internal_eval import ExecutedRowState
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import LLM_NODE_ID, render_prompt
from whetstone.envs.sampling import validate_evaluation_role_for_split
from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_INTENT_CLAIM_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    EVALUATION_RESULT_ATTESTATION_SCHEMA,
    EvaluationComponentTraces,
    EvaluationComponentTracesRef,
    EvaluationEvidence,
    EvaluationEvidenceRef,
    EvaluationFailureEvidence,
    EvaluationFailureEvidenceRef,
    EvaluationIntentClaim,
    EvaluationOutputsRecord,
    EvaluationResultAttestation,
)
from whetstone.optimization.effect_authority import ReplayPolicy
from whetstone.optimization.identity import (
    TerminalFailure,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.optimization.reward import REWARD_SCHEMA, Reward, RewardRef
from whetstone.optimization.schema import (
    CANDIDATE_RECORD_SCHEMA,
    EVAL_CONFIG_RECORD_SCHEMA,
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
    INTENT_RESOLUTION_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
)
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)

_EVALUATION_SERVICE_NAMESPACE = "whetstone.evaluation_service.v3"


@dataclass(frozen=True, slots=True)
class _OwnedClaim:
    intent_ref: TypedRef
    generation: int


class _LeaseLostError(RuntimeError):
    pass


def _wait_for_renewal(interval: float, stop: threading.Event) -> bool:
    return stop.wait(interval)


def _ignore_renewal_publication(
    _claim: EvaluationIntentClaim,
) -> None:
    pass


class EngineEvaluationService:
    """Resolve each immutable intent exactly once across process restarts."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        engine: EvaluationEngine,
        claim_lease_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        _renewal_wait: Callable[[float, threading.Event], bool] = (
            _wait_for_renewal
        ),
        _renewal_published: Callable[[EvaluationIntentClaim], None] = (
            _ignore_renewal_publication
        ),
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be positive")
        self._store = store
        self._engine = engine
        self._claim_lease_seconds = claim_lease_seconds
        self._clock = clock
        self._sleep = sleep
        self._renewal_wait = _renewal_wait
        self._renewal_published = _renewal_published
        self._owner_id = uuid.uuid4().hex
        self._resolve_lock = threading.Lock()

    @property
    def replay_policy(self) -> ReplayPolicy:
        """Return the recovery policy of the durable evaluator workflow."""
        return ReplayPolicy.DURABLE_WORKFLOW

    def validate_resolution_graph(self, resolution: IntentResolution) -> None:
        """Validate one exact result graph without mutating durable state."""
        self._validate_result_graph(
            resolution,
            expected_intent=resolution.intent,
        )

    @staticmethod
    def _intent_ref(intent: EvaluationIntent) -> TypedRef:
        return typed_ref_for_record(
            "whetstone.evaluation_intent", intent.model_dump(mode="json")
        )

    @classmethod
    def _key(cls, intent: EvaluationIntent) -> str:
        return (
            f"{_EVALUATION_SERVICE_NAMESPACE}.intent_resolution:"
            f"{cls._intent_ref(intent).content_hash}"
        )

    @classmethod
    def _claim_key(
        cls,
        intent: EvaluationIntent,
        event_ordinal: int,
    ) -> str:
        return (
            f"{_EVALUATION_SERVICE_NAMESPACE}.intent_claim:"
            f"{cls._intent_ref(intent).content_hash}"
            f"#{event_ordinal}"
        )

    @staticmethod
    def _typed_ref(reference: Any) -> TypedRef:
        if isinstance(reference, TypedRef):
            return reference
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

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

    def _bind(
        self, intent: EvaluationIntent, resolution: IntentResolution
    ) -> IntentResolution:
        self._validate_result_graph(resolution, expected_intent=intent)
        content = resolution.model_dump(mode="json")
        reference, _ = self._store.put(INTENT_RESOLUTION_SCHEMA, content)
        try:
            self._store.bind(self._key(intent), reference)
        except BindingConflictError:
            winner = self._store.resolve(self._key(intent))
            assert winner is not None
            loaded = self._load(winner, expected_intent=intent)
            return loaded
        return resolution

    def _load_claim(self, reference: Any) -> EvaluationIntentClaim:
        return EvaluationIntentClaim.model_validate(self._store.get(reference))

    def _load_result_attestation(
        self,
        reference: Any,
        *,
        expected_intent: EvaluationIntent,
    ) -> EvaluationResultAttestation:
        _attestation_ref, content = self._load_exact(
            reference,
            expected_schema=EVALUATION_RESULT_ATTESTATION_SCHEMA,
        )
        attestation = EvaluationResultAttestation.model_validate(content)
        if attestation.resolution.intent != expected_intent:
            raise ValueError(
                "Evaluation Result Attestation belongs to another Intent"
            )
        if (
            attestation.graph_hash
            != self._engine.experiment.rollout_definition.graph_hash
        ):
            raise ValueError(
                "Evaluation Result Attestation uses another rollout graph"
            )
        return attestation

    def _latest_claim(
        self,
        intent: EvaluationIntent,
    ) -> EvaluationIntentClaim | None:
        latest: EvaluationIntentClaim | None = None
        event_ordinal = 0
        intent_ref = self._intent_ref(intent)
        while True:
            bound = self._store.resolve(self._claim_key(intent, event_ordinal))
            if bound is None:
                return latest
            claim = self._load_claim(bound)
            if (
                claim.intent_ref != intent_ref
                or claim.event_ordinal != event_ordinal
            ):
                raise ValueError(
                    "durable evaluation claim has invalid lease identity"
                )
            if latest is None:
                if claim.generation != 0 or claim.heartbeat_ordinal != 0:
                    raise ValueError(
                        "durable evaluation claim stream has invalid origin"
                    )
            elif latest.result_attestation_ref is not None:
                raise ValueError(
                    "durable evaluation claim stream continues after its "
                    "terminal attestation"
                )
            elif claim.owner_id == latest.owner_id:
                if (
                    claim.generation != latest.generation
                    or claim.heartbeat_ordinal != latest.heartbeat_ordinal + 1
                ):
                    raise ValueError(
                        "durable evaluation claim has invalid renewal order"
                    )
            elif (
                claim.generation != latest.generation + 1
                or claim.heartbeat_ordinal != 0
            ):
                raise ValueError(
                    "durable evaluation claim has invalid takeover order"
                )
            latest = claim
            event_ordinal += 1

    def _append_claim_event(
        self,
        *,
        intent: EvaluationIntent,
        intent_ref: TypedRef,
        prior: EvaluationIntentClaim | None,
        generation: int,
        heartbeat_ordinal: int,
        result_attestation_ref: TypedRef | None = None,
    ) -> EvaluationIntentClaim:
        if prior is None:
            event_ordinal = 0
            if generation != 0 or heartbeat_ordinal != 0:
                raise ValueError("initial evaluation claim must start at zero")
        elif generation == prior.generation:
            event_ordinal = prior.event_ordinal + 1
            if (
                prior.owner_id != self._owner_id
                or heartbeat_ordinal != prior.heartbeat_ordinal + 1
            ):
                raise _LeaseLostError(
                    "evaluation lease cannot be renewed by another owner"
                )
            if prior.result_attestation_ref is not None:
                raise _LeaseLostError(
                    "terminal evaluation claim cannot be extended"
                )
        else:
            event_ordinal = prior.event_ordinal + 1
            if generation != prior.generation + 1 or heartbeat_ordinal != 0:
                raise ValueError("evaluation takeover must start a generation")
            if prior.expires_at > self._clock():
                raise _LeaseLostError(
                    "evaluation lease cannot be taken over before expiry"
                )
        claim = EvaluationIntentClaim(
            intent_ref=intent_ref,
            owner_id=self._owner_id,
            event_ordinal=event_ordinal,
            generation=generation,
            heartbeat_ordinal=heartbeat_ordinal,
            expires_at=float(self._clock() + self._claim_lease_seconds),
            result_attestation_ref=result_attestation_ref,
        )
        reference, _ = self._store.put(
            EVALUATION_INTENT_CLAIM_SCHEMA,
            claim.model_dump(mode="json"),
        )
        try:
            status = self._store.bind(
                self._claim_key(intent, event_ordinal),
                reference,
            )
        except BindingConflictError:
            status = None
        if status not in (None, BindStatus.BOUND, BindStatus.IDEMPOTENT):
            raise _LeaseLostError(
                "evaluation claim event was not durably bound"
            )
        bound = self._store.resolve(self._claim_key(intent, event_ordinal))
        assert bound is not None
        persisted = self._load_claim(bound)
        if (
            persisted.intent_ref != intent_ref
            or persisted.event_ordinal != event_ordinal
        ):
            raise ValueError(
                "durable evaluation claim has invalid event identity"
            )
        return persisted

    def _publish_result_attestation(
        self,
        *,
        intent: EvaluationIntent,
        resolution: IntentResolution,
        owned: _OwnedClaim,
    ) -> EvaluationResultAttestation:
        self._validate_result_graph(
            resolution,
            expected_intent=intent,
            require_attestation=False,
        )
        attestation = EvaluationResultAttestation(
            graph_hash=self._engine.experiment.rollout_definition.graph_hash,
            resolution=resolution,
        )
        persisted, _ = self._store.put(
            EVALUATION_RESULT_ATTESTATION_SCHEMA,
            attestation.record_content(),
        )
        attestation_ref = self._typed_ref(persisted)
        while True:
            latest = self._latest_claim(intent)
            if (
                latest is None
                or latest.owner_id != self._owner_id
                or latest.generation != owned.generation
            ):
                raise _LeaseLostError(
                    "evaluation lease is not owned by this resolver"
                )
            if latest.result_attestation_ref is not None:
                existing = self._load_result_attestation(
                    latest.result_attestation_ref,
                    expected_intent=intent,
                )
                if existing != attestation:
                    raise _LeaseLostError(
                        "terminal evaluation claim names another result"
                    )
                return existing
            winner = self._append_claim_event(
                intent=intent,
                intent_ref=owned.intent_ref,
                prior=latest,
                generation=owned.generation,
                heartbeat_ordinal=latest.heartbeat_ordinal + 1,
                result_attestation_ref=attestation_ref,
            )
            if (
                winner.owner_id != self._owner_id
                or winner.generation != owned.generation
            ):
                raise _LeaseLostError(
                    "evaluation result lost claim arbitration"
                )
            if winner.result_attestation_ref == attestation_ref:
                return attestation
            if winner.result_attestation_ref is not None:
                raise _LeaseLostError(
                    "terminal evaluation claim names another result"
                )

    def _attested_resolution(
        self,
        intent: EvaluationIntent,
    ) -> IntentResolution | None:
        latest = self._latest_claim(intent)
        if latest is None or latest.result_attestation_ref is None:
            return None
        return self._load_result_attestation(
            latest.result_attestation_ref,
            expected_intent=intent,
        ).resolution

    def _renew_claim(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> None:
        latest = self._latest_claim(intent)
        if (
            latest is None
            or latest.owner_id != self._owner_id
            or latest.generation != owned.generation
        ):
            raise _LeaseLostError(
                "evaluation lease is not owned by this resolver"
            )
        if latest.result_attestation_ref is not None:
            return
        winner = self._append_claim_event(
            intent=intent,
            intent_ref=owned.intent_ref,
            prior=latest,
            generation=owned.generation,
            heartbeat_ordinal=latest.heartbeat_ordinal + 1,
        )
        if (
            winner.owner_id != self._owner_id
            or winner.generation != owned.generation
        ):
            raise _LeaseLostError(
                "evaluation lease renewal lost claim arbitration"
            )
        self._renewal_published(winner)

    def _assert_generation_current(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> None:
        latest = self._latest_claim(intent)
        if (
            latest is None
            or latest.owner_id != self._owner_id
            or latest.generation != owned.generation
        ):
            raise _LeaseLostError(
                "evaluation lease is not owned by this resolver"
            )

    def _claim(self, intent: EvaluationIntent) -> _OwnedClaim | None:
        """Acquire the current durable lease or await its resolution.

        After a crashed owner's persisted lease expires, a fresh resolver
        claims the next append-only generation and safely retries. Concurrent
        live resolvers observe the winning unexpired claim and wait for its
        terminal resolution.
        """
        intent_ref = self._intent_ref(intent)
        while True:
            if self._store.resolve(self._key(intent)) is not None:
                return
            winner = self._latest_claim(intent)
            if winner is None:
                winner = self._append_claim_event(
                    intent=intent,
                    intent_ref=intent_ref,
                    prior=None,
                    generation=0,
                    heartbeat_ordinal=0,
                )
            if winner.result_attestation_ref is not None:
                return None
            if winner.owner_id == self._owner_id:
                return _OwnedClaim(
                    intent_ref=intent_ref,
                    generation=winner.generation,
                )
            remaining = winner.expires_at - self._clock()
            if remaining <= 0:
                takeover = self._append_claim_event(
                    intent=intent,
                    intent_ref=intent_ref,
                    prior=winner,
                    generation=winner.generation + 1,
                    heartbeat_ordinal=0,
                )
                if takeover.owner_id == self._owner_id:
                    return _OwnedClaim(
                        intent_ref=intent_ref,
                        generation=takeover.generation,
                    )
                continue
            self._sleep(min(0.05, remaining))

    def _evaluate_with_heartbeat(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        """Evaluate under a renewed lease, keeping any durable binding.

        A heartbeat failure only aborts when nothing was bound: once
        :meth:`_evaluate_and_bind` has durably committed a resolution, that
        paid evaluation is the answer and a transient renewal error must not
        discard it.
        """
        stop = threading.Event()
        heartbeat_errors: list[Exception] = []

        def heartbeat() -> None:
            interval = self._claim_lease_seconds / 3
            while True:
                try:
                    if self._renewal_wait(interval, stop):
                        return
                    self._renew_claim(intent, owned)
                except Exception as exc:
                    heartbeat_errors.append(exc)
                    return

        self._renew_claim(intent, owned)
        thread = threading.Thread(
            target=heartbeat,
            name=f"evaluation-heartbeat-{owned.generation}",
            daemon=True,
        )
        thread.start()
        try:
            self._assert_generation_current(intent, owned)
            resolution = self._evaluate_and_bind(intent, owned)
        finally:
            stop.set()
            thread.join()
        if heartbeat_errors and self._store.resolve(self._key(intent)) is None:
            raise RuntimeError("evaluation lease heartbeat failed") from (
                heartbeat_errors[0]
            )
        return resolution

    def _resolve_claimed(self, intent: EvaluationIntent) -> IntentResolution:
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing, expected_intent=intent)
        attested = self._attested_resolution(intent)
        if attested is not None:
            return self._bind(intent, attested)
        owned = self._claim(intent)
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing, expected_intent=intent)
        attested = self._attested_resolution(intent)
        if attested is not None:
            return self._bind(intent, attested)
        if owned is None:
            raise RuntimeError("evaluation claim resolved without a result")
        return self._evaluate_with_heartbeat(intent, owned)

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        with self._resolve_lock:
            return self._resolve_claimed(intent)

    def _bind_if_owned(
        self,
        intent: EvaluationIntent,
        resolution: IntentResolution,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        if resolution.outcome in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            self._publish_result_attestation(
                intent=intent,
                resolution=resolution,
                owned=owned,
            )
        else:
            self._assert_generation_current(intent, owned)
        return self._bind(intent, resolution)

    def _evaluate_and_bind(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        self._persist_intent_targets(intent)
        if intent.target_eval_config != self._engine.eval_config_ref:
            return self._bind_if_owned(
                intent,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    intent=intent,
                    outcome=IntentOutcome.REJECTED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.VALIDATION,
                        message=(
                            "intent target Eval Config is not the engine's "
                            "exact sampling binding"
                        ),
                    ),
                    resolved_eval_config=intent.target_eval_config,
                ),
                owned,
            )
        request = EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
        try:
            self._engine.validate_request(request)
        except (KeyError, TypeError, ValueError) as exc:
            return self._bind_if_owned(
                intent,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    intent=intent,
                    outcome=IntentOutcome.REJECTED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.VALIDATION,
                        message=str(exc) or type(exc).__name__,
                    ),
                    resolved_eval_config=intent.target_eval_config,
                ),
                owned,
            )
        try:
            self._assert_generation_current(intent, owned)
            evaluated = self._engine.evaluate(request)
        except Exception as exc:
            failure = EvaluationFailureEvidence(
                candidate=intent.candidate,
                evaluation_binding=intent.evaluation_binding,
                purpose=intent.purpose,
                exception_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            )
            persisted_ref, _ = self._store.put(
                EVALUATION_FAILURE_SCHEMA, failure.record_content()
            )
            failure_ref = EvaluationFailureEvidenceRef(
                record=failure,
                record_ref=self._typed_ref(persisted_ref),
            )
            terminal_failure = TerminalFailure(
                code=f"evaluation_{failure.exception_type}",
                message=failure.message,
                details={
                    "evidence_schema": failure_ref.record_ref.schema_name,
                    "evidence_content_hash": (
                        failure_ref.record_ref.content_hash
                    ),
                },
            )
            return self._bind_if_owned(
                intent,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    intent=intent,
                    outcome=IntentOutcome.FAILED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.INFRASTRUCTURE,
                        message=failure.message,
                    ),
                    evaluation_result_ref=failure_ref.record_ref,
                    reward_evidence_refs=(),
                    resolved_eval_config=intent.target_eval_config,
                    terminal_failure=terminal_failure,
                ),
                owned,
            )
        reward_ref = evaluated.evidence.reward_ref
        reward_evidence_refs = (
            () if reward_ref is None else reward_ref.record.evidence_refs
        )
        return self._bind_if_owned(
            intent,
            IntentResolution(
                schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                intent=intent,
                outcome=IntentOutcome.COMPLETED,
                detail=ResolutionDetail(
                    classification=ResolutionClass.MEASURED,
                    message="candidate evaluated under exact sampling binding",
                ),
                evaluation_result_ref=evaluated.evidence_ref,
                reward_evidence_refs=reward_evidence_refs,
                resolved_eval_config=intent.target_eval_config,
                reward_ref=reward_ref,
            ),
            owned,
        )


__all__ = ["EngineEvaluationService"]
