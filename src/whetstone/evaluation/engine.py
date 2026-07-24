"""The single injected evaluation engine used by optimization adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dr_store import ObjectStore

from whetstone.envs.factory import EnvExperiment
from whetstone.envs.internal_eval import InternalEvalResult, run_internal_eval
from whetstone.envs.rollout_definition import validate_candidate_prompt
from whetstone.envs.sampling import EnvSplitSampling
from whetstone.evaluation.schema import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    REWARD_SCHEMA,
    ROLLOUT_AGGREGATE_SCHEMA,
    CacheEvidence,
    EvaluationEvidence,
    RowAccounting,
)
from whetstone.execution.fanout import FanoutConfig
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.schema import (
    CANDIDATE_RECORD_SCHEMA,
    EVAL_CONFIG_RECORD_SCHEMA,
    Candidate,
    EvalConfigRef,
    candidate_reference,
    eval_config_reference,
)
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Internal value passed to the canonical engine."""

    candidate: Candidate
    evaluation_role: EvaluationRole
    evaluation_context_id: str
    purpose: str


@dataclass(frozen=True, slots=True)
class EngineEvaluation:
    """Canonical engine return value and its durable reference."""

    evidence: EvaluationEvidence
    evidence_ref: TypedRef

    @property
    def reward_value(self) -> float | None:
        if self.evidence.reward_ref is None:
            return None
        return self.evidence.aggregate_value


class EvaluationEngine:
    """Render, execute, aggregate, and persist one exact sampling binding.

    The PR6 :func:`run_internal_eval` kernel is the only row-driving loop.
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
        transport: TransportCall,
        fanout: FanoutConfig | None = None,
        partial_log: PartialLog | None = None,
        prompt_cache: PromptResultCache | None = None,
    ) -> None:
        self._store = store
        self.experiment = experiment
        self.sampling = sampling
        self._execution_policy = execution_policy
        self._transport = transport
        self._fanout = fanout
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

    def preflight(self, candidate: Candidate) -> None:
        """Reject malformed candidates before any provider call."""
        from whetstone.envs.registry import env_spec

        validate_candidate_prompt(
            env_spec(self.experiment.env_name),
            candidate,
            self.sampling.instances,
        )

    def _put(self, schema: str, content: dict[str, Any]) -> TypedRef:
        reference, _ = self._store.put(schema, content)
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def evaluate(self, request: EvaluationRequest) -> EngineEvaluation:
        if request.evaluation_role is EvaluationRole.OFFICIAL:
            apply_reward = False
        else:
            apply_reward = True
        self.preflight(request.candidate)
        result = run_internal_eval(
            self.experiment,
            candidate=request.candidate,
            sampling=self.sampling,
            execution_policy=self._execution_policy,
            transport=self._transport,
            fanout=self._fanout,
            partial_log=self._partial_log,
            apply_reward=apply_reward,
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
        output_record = {
            "candidate_id": request.candidate.candidate_id,
            "outputs": [
                {
                    "candidate_id": row.candidate_id,
                    "instance_id": row.instance_id,
                    "repeat": row.repeat,
                    "output_text": row.output_text,
                    "score": row.score,
                    "failure_code": row.failure_code,
                    "finish_reason": row.finish_reason,
                    "provider_error": row.provider_error,
                    "max_budget": row.max_budget,
                    "over_budget": row.over_budget,
                }
                for row in result.outputs
            ],
        }
        outputs_ref = self._put(EVALUATION_OUTPUTS_SCHEMA, output_record)
        aggregation_output = aggregate.aggregation_output
        aggregate_record = {
            "name": aggregate.name,
            "graph_hash": aggregate.graph_hash,
            "eval_config_hash": aggregate.eval_config_hash,
            "evaluation_context_id": request.evaluation_context_id,
            "task_count": aggregate.task_count,
            "repeat_count": aggregate.repeat_count,
            "aggregation_output": aggregation_output.model_dump(mode="json"),
            "rows_present": aggregate.rows_present,
            "rows_missing": aggregate.rows_missing,
            "rows_failed": aggregate.rows_failed,
            "rows_invalid": aggregate.rows_invalid,
        }
        aggregate_ref = self._put(ROLLOUT_AGGREGATE_SCHEMA, aggregate_record)
        reward = (
            result.reward.model_copy(
                update={
                    "evidence_ref_content_hash": aggregate_ref.content_hash
                }
            )
            if result.reward is not None
            else None
        )
        reward_ref = (
            self._put(REWARD_SCHEMA, reward.record_content())
            if reward is not None
            else None
        )
        cache = self._cache_evidence(request.candidate.candidate_id)
        evidence = EvaluationEvidence(
            candidate=candidate_ref,
            eval_config=eval_ref,
            graph_hash=aggregate.graph_hash,
            graph_config_ref=aggregate.graph_hash,
            evaluation_role=request.evaluation_role,
            evaluation_context_id=request.evaluation_context_id,
            purpose=request.purpose,
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
