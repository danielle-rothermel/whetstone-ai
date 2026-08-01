"""The single injected evaluation engine used by optimization adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dr_store import ObjectStore

from whetstone.envs.factory import EnvExperiment
from whetstone.envs.internal_eval import (
    InternalEvalResult,
    InternalRowJobFactory,
    run_internal_eval,
)
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import (
    render_prompt,
    validate_candidate_prompt,
)
from whetstone.envs.sampling import EnvSplitSampling
from whetstone.evaluation.schema import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    REWARD_SCHEMA,
    ROLLOUT_AGGREGATE_SCHEMA,
    CacheEvidence,
    EvaluationEvidence,
    EvaluationEvidenceRef,
    EvaluationOutputRow,
    EvaluationOutputsRecord,
    RowAccounting,
)
from whetstone.execution.fanout import DEFAULT_CONCURRENCY
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.optimization.identity import (
    IdentityRef,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.optimization.reward import reward_reference
from whetstone.optimization.schema import (
    CANDIDATE_RECORD_SCHEMA,
    EVAL_CONFIG_RECORD_SCHEMA,
    Candidate,
    EvalConfigRef,
    EvaluationBinding,
    candidate_reference,
    eval_config_reference,
)
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Internal value passed to the canonical engine."""

    candidate: Candidate
    evaluation_binding: EvaluationBinding
    purpose: str


@dataclass(frozen=True, slots=True)
class EngineEvaluation:
    """Canonical engine return value and its durable reference."""

    evidence: EvaluationEvidence
    evidence_ref: TypedRef

    def __post_init__(self) -> None:
        EvaluationEvidenceRef(
            record=self.evidence,
            record_ref=self.evidence_ref,
        )

    @property
    def reward_value(self) -> float | None:
        if self.evidence.reward_ref is None:
            return None
        return self.evidence.reward_ref.record.value


class EvaluationEngine:
    """Render, execute, aggregate, and persist one exact sampling binding.

    :func:`run_internal_eval` is the only row-driving loop.
    This engine owns its external contract: exact Config validation, candidate
    preflight, content-addressed evidence, and optimizer-facing references.
    """

    def __init__(
        self,
        *,
        store: ObjectStore,
        experiment: EnvExperiment,
        sampling: EnvSplitSampling,
        execution_policy: ProviderExecutionPolicy,
        row_job_factory: InternalRowJobFactory,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_wall_seconds: float | None = None,
        partial_log: PartialLog | None = None,
        prompt_cache: PromptResultCache | None = None,
    ) -> None:
        self._store = store
        self.experiment = experiment
        self.sampling = sampling
        self._execution_policy = execution_policy
        self._row_job_factory = row_job_factory
        self._concurrency = concurrency
        self._max_wall_seconds = max_wall_seconds
        self._partial_log = partial_log
        self._prompt_cache = prompt_cache
        expected = experiment.eval_configs.eval_config_for(sampling.split_role)
        if expected != sampling.eval_config:
            raise ValueError(
                "engine sampling must be the exact experiment split binding"
            )

    @property
    def eval_config_ref(self) -> EvalConfigRef:
        return eval_config_reference(self.sampling.eval_config)

    @property
    def prompt_cache(self) -> PromptResultCache | None:
        return self._prompt_cache

    @property
    def provider_execution_policy_ref(self) -> IdentityRef:
        return IdentityRef(
            record_ref=typed_ref_for_record(
                PROVIDER_EXECUTION_POLICY_SCHEMA,
                self._execution_policy.identity_payload(),
            ),
            identity_hash=self._execution_policy.identity_hash,
        )

    @property
    def provider_execution_policy_record(self) -> dict[str, Any]:
        """Return the canonical policy record advertised by the engine."""
        return self._execution_policy.identity_payload()

    def preflight(self, candidate: Candidate) -> None:
        """Reject malformed candidates before any provider call."""
        validate_candidate_prompt(
            env_spec(self.experiment.env_name),
            candidate,
            self.sampling.instances,
        )

    def validate_request(self, request: EvaluationRequest) -> None:
        """Validate the complete evaluation request before execution."""
        self._validate_binding(request.evaluation_binding)
        self.preflight(request.candidate)

    def _validate_binding(self, binding: EvaluationBinding) -> None:
        if binding.eval_config != self.eval_config_ref:
            raise ValueError(
                "evaluation binding must name the engine's exact Eval Config"
            )
        if (
            binding.provider_execution_policy_ref
            != self.provider_execution_policy_ref
        ):
            raise ValueError(
                "evaluation binding must name the engine's exact Provider "
                "Execution Policy"
            )

    def _put(self, schema: str, content: dict[str, Any]) -> TypedRef:
        reference, _ = self._store.put(schema, content)
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def _evaluation_outputs_record(
        self,
        request: EvaluationRequest,
        result: InternalEvalResult,
    ) -> EvaluationOutputsRecord:
        instance_ids = tuple(
            str(instance.id) for instance in self.sampling.instances
        )
        task_identities = self.sampling.task_set.task_identities
        if len(instance_ids) != len(task_identities):
            raise ValueError(
                "sampling instances and task identities must align exactly"
            )
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("sampling instance IDs must be unique")
        if len(set(task_identities)) != len(task_identities):
            raise ValueError("sampling task identities must be unique")

        task_identity_by_instance = dict(
            zip(instance_ids, task_identities, strict=True)
        )
        instance_by_id = {
            str(instance.id): instance for instance in self.sampling.instances
        }
        repeat_count = self.sampling.repeat_plan.repeat_count
        planned_ordinal = {
            (instance_id, repeat): instance_index * repeat_count + repeat
            for instance_index, instance_id in enumerate(instance_ids)
            for repeat in range(repeat_count)
        }

        rows: list[EvaluationOutputRow] = []
        prior_ordinal = -1
        for output in result.outputs:
            if output.candidate_id != request.candidate.candidate_id:
                raise ValueError(
                    "evaluation output candidate_id does not match request"
                )
            key = (output.instance_id, output.repeat)
            ordinal = planned_ordinal.get(key)
            if ordinal is None:
                raise ValueError(
                    "evaluation output row is outside the exact sampling plan"
                )
            if ordinal <= prior_ordinal:
                raise ValueError(
                    "evaluation output rows must follow sampling instance/"
                    "repeat order"
                )
            prior_ordinal = ordinal
            rows.append(
                EvaluationOutputRow(
                    candidate_id=output.candidate_id,
                    instance_id=output.instance_id,
                    task_identity=task_identity_by_instance[
                        output.instance_id
                    ],
                    repeat=output.repeat,
                    rendered_prompt=render_prompt(
                        env_spec(self.experiment.env_name),
                        request.candidate,
                        instance_by_id[output.instance_id],
                    ),
                    output_text=output.output_text,
                    score=output.score,
                    failed=output.failed,
                    missing=output.missing,
                    invalid=output.invalid,
                    failure_code=output.failure_code,
                    finish_reason=output.finish_reason,
                    provider_error=output.provider_error,
                    max_budget=output.max_budget,
                    over_budget=output.over_budget,
                )
            )
        return EvaluationOutputsRecord(
            candidate=candidate_reference(request.candidate),
            evaluation_binding=request.evaluation_binding,
            evaluation_role=request.evaluation_binding.role,
            graph_hash=self.experiment.rollout_definition.graph_hash,
            purpose=request.purpose,
            split_role=self.sampling.split_role,
            task_identities=task_identities,
            repeat_count=repeat_count,
            outputs=tuple(rows),
        )

    def evaluate(self, request: EvaluationRequest) -> EngineEvaluation:
        self.validate_request(request)
        result = run_internal_eval(
            self.experiment,
            candidate=request.candidate,
            sampling=self.sampling,
            execution_policy=self._execution_policy,
            row_job_factory=self._row_job_factory,
            evaluation_binding=request.evaluation_binding,
            concurrency=self._concurrency,
            max_wall_seconds=self._max_wall_seconds,
            partial_log=self._partial_log,
            render_guard=True,
            cache=self._prompt_cache,
        )
        return self._persist(request, result)

    def _persist(
        self, request: EvaluationRequest, result: InternalEvalResult
    ) -> EngineEvaluation:
        candidate_ref = candidate_reference(request.candidate)
        persisted_candidate = self._put(
            CANDIDATE_RECORD_SCHEMA, request.candidate.record_content()
        )
        if persisted_candidate != candidate_ref.record_ref:
            raise ValueError("persisted candidate reference diverged")
        eval_ref = self.eval_config_ref
        persisted_eval = self._put(
            EVAL_CONFIG_RECORD_SCHEMA,
            self.sampling.eval_config.model_dump(mode="json"),
        )
        if persisted_eval != eval_ref.record_ref:
            raise ValueError("persisted Eval Config reference diverged")
        aggregate = result.aggregate
        output_record = self._evaluation_outputs_record(request, result)
        outputs_ref = self._put(
            EVALUATION_OUTPUTS_SCHEMA, output_record.record_content()
        )
        aggregation_output = aggregate.aggregation_output
        aggregate_record = aggregate.record_content()
        aggregate_ref = self._put(ROLLOUT_AGGREGATE_SCHEMA, aggregate_record)
        if aggregate_ref != aggregate.record_ref():
            raise ValueError("persisted aggregate reference diverged")
        reward_ref = None
        if result.reward is not None:
            reward_ref = reward_reference(result.reward)
            persisted_reward = self._put(
                REWARD_SCHEMA, result.reward.record_content()
            )
            if persisted_reward != reward_ref.record_ref:
                raise ValueError("persisted Reward reference diverged")
        cache = self._cache_evidence(request.candidate.candidate_id)
        evidence = EvaluationEvidence(
            candidate=candidate_ref,
            evaluation_binding=request.evaluation_binding,
            graph_hash=aggregate.graph_hash,
            graph_config_ref=aggregate.graph_hash,
            purpose=request.purpose,
            dataset_identity=self.sampling.task_set.dataset_revision,
            task_identities=self.sampling.task_set.task_identities,
            repeat_count=self.sampling.repeat_plan.repeat_count,
            per_task_values=result.per_task_scores,
            per_task_counts=result.per_task_counts,
            row_accounting=RowAccounting(
                planned=aggregate.task_count * aggregate.repeat_count,
                present=aggregate.rows_present,
                missing=aggregate.rows_missing,
                failed=aggregate.rows_failed,
                invalid=aggregate.rows_invalid,
            ),
            outputs_ref=outputs_ref,
            aggregate_ref=aggregate_ref,
            aggregate_name=aggregate.name,
            aggregate_value=aggregation_output.value,
            aggregate_status=aggregation_output.status.value,
            reward_ref=reward_ref,
            cache=cache,
            concurrency_halved=result.concurrency_halved,
            deadline_reached=result.deadline_reached,
            guard_timeouts=result.guard_timeouts,
        )
        evidence_ref = self._put(
            EVALUATION_EVIDENCE_SCHEMA, evidence.record_content()
        )
        return EngineEvaluation(evidence=evidence, evidence_ref=evidence_ref)

    def _cache_evidence(self, candidate_id: str) -> CacheEvidence:
        if self._partial_log is None:
            return CacheEvidence()
        rows = [
            row
            for row in self._partial_log.load()
            if row.unit == candidate_id
            and row.phase == self.sampling.split_role
        ]
        hits = [row for row in rows if row.cache_hit]
        return CacheEvidence(
            partial_row_count=len(rows),
            cache_hit_count=len(hits),
            source_call_ids=tuple(
                row.cache_source_call_id
                for row in hits
                if row.cache_source_call_id is not None
            ),
        )


__all__ = ["EngineEvaluation", "EvaluationEngine", "EvaluationRequest"]
