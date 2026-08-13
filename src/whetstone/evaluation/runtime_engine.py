from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dr_store import ObjectStore

from whetstone.core.identity import IdentityRef, TypedRef, typed_ref_for_record
from whetstone.evaluation.aggregate import AGGREGATE_SCHEMA
from whetstone.evaluation.driver import EvaluationDriver
from whetstone.evaluation.drivers.eval_result import InternalEvalResult
from whetstone.evaluation.generation import GenerationIndex
from whetstone.evaluation.protocol import (
    EngineEvaluation,
    EvaluationPlanSnapshot,
    EvalRequest,
    EvaluationSamplingView,
)
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION,
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EVALUATION_OUTPUTS_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA_VERSION,
    CacheEvidence,
    EvaluationComponentTraceRow,
    EvaluationComponentTraces,
    EvaluationComponentTracesRef,
    EvaluationEvidence,
    EvaluationOutputRow,
    EvaluationOutputsRecord,
    RowAccounting,
)
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.evaluation.traces import ExecutedComponentTracePayload
from whetstone.execution.fanout import DEFAULT_CONCURRENCY
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.binding import (
    EVAL_CONFIG_RECORD_SCHEMA,
    EvalConfigRef,
    eval_config_reference,
)
from whetstone.experiment.candidate import (
    CANDIDATE_RECORD_SCHEMA,
    Candidate,
    candidate_reference,
)
from whetstone.experiment.env import Experiment
from whetstone.experiment.reward import REWARD_SCHEMA, reward_reference
from whetstone.experiment.sampling import SplitSampling, derive_split_sampling, evaluation_role_for_split
from whetstone.core.roles import EvaluationRole
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)

__all__ = ["RuntimeEvaluationEngine", "SamplingTaskView"]


@runtime_checkable
class SamplingTaskView(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def prompt_inputs(self) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class _EngineTaskView:
    task_id: str
    task_hash: str
    prompt_inputs: dict[str, str]


@dataclass(frozen=True, slots=True)
class _EngineSamplingView:
    task_hashes: tuple[str, ...]
    num_samples: int
    split_role: str
    tasks: tuple[_EngineTaskView, ...]


class RuntimeEvaluationEngine:
    """Generic evaluation engine with env-specific flow injected via driver."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        experiment: Experiment,
        sampling: SplitSampling,
        execution_policy: ProviderExecutionPolicy,
        driver: EvaluationDriver,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_wall_seconds: float | None = None,
        partial_log: PartialLog | None = None,
        prompt_cache: PromptResultCache | None = None,
    ) -> None:
        self._store = store
        self._experiment = experiment
        self._sampling = sampling
        self._execution_policy = execution_policy
        self._driver = driver
        self._concurrency = concurrency
        self._max_wall_seconds = max_wall_seconds
        self._partial_log = partial_log
        self._prompt_cache = prompt_cache
        self._validate_sampling_contract()

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def eval_config_ref(self) -> EvalConfigRef:
        return eval_config_reference(self._sampling.eval_config)

    @property
    def sampling(self) -> EvaluationSamplingView:
        return self._sampling_view()

    @property
    def plan_snapshot(self) -> EvaluationPlanSnapshot:
        return EvaluationPlanSnapshot(
            graph_hash=self._experiment.generation_graph.graph_hash,
            dataset_hash=self._sampling.task_set.dataset_revision,
            task_hashes=self._sampling.task_set.task_hashes,
            num_samples=self._sampling.sample_plan.num_samples,
            split_role=self._sampling.split_role,
        )

    @property
    def provider_execution_policy_ref(self) -> IdentityRef:
        return IdentityRef(
            record_ref=typed_ref_for_record(
                PROVIDER_EXECUTION_POLICY_SCHEMA,
                self._execution_policy.identity_payload(),
            ),
            record_hash=self._execution_policy.identity_hash,
        )

    @property
    def provider_execution_policy_record(self) -> dict[str, Any]:
        return self._execution_policy.identity_payload()

    def task_model_identity_hash(self) -> str:
        return self._driver.task_model_identity_hash(self._experiment)

    def execution_policy_identity_hash(self) -> str:
        return self._execution_policy.identity_hash

    def reward_policy_identity_hash(self) -> str:
        return self._experiment.reward_policy.identity_hash()

    def expected_model_route(self) -> str:
        return self._driver.expected_model_route(self._experiment)

    @staticmethod
    def _derive_sampling(
        source: SplitSampling,
        task_ids: tuple[str, ...],
    ) -> SplitSampling:
        if not task_ids:
            raise ValueError("derived sampling requires at least one task")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("derived sampling task IDs must be unique")
        source_by_id = {
            _task_id(task): task
            for task in source.tasks
        }
        unknown = tuple(
            task_id for task_id in task_ids if task_id not in source_by_id
        )
        if unknown:
            raise ValueError(
                f"derived sampling contains unknown task IDs: {unknown!r}"
            )
        selected = tuple(source_by_id[task_id] for task_id in task_ids)
        task_hash_by_task = {
            id(task): task_id for task_id, task in zip(task_ids, selected, strict=True)
        }
        namespace, separator, role = source.task_set.manifest_id.rpartition(".")
        if not separator or role != source.split_role:
            raise ValueError(
                "source sampling manifest does not match its split role"
            )
        return derive_split_sampling(
            namespace=namespace,
            dataset_revision=source.task_set.dataset_revision,
            split_role=source.split_role,
            tasks=selected,
            task_hash_of=lambda task: task_hash_by_task[id(task)],
            procedure=source.procedure_config,
            aggregation=source.aggregation_config,
            num_samples=source.sample_plan.num_samples,
        )

    def for_task_ids(
        self, task_ids: tuple[str, ...]
    ) -> RuntimeEvaluationEngine:
        derived = self._derive_sampling(self._sampling, task_ids)
        return RuntimeEvaluationEngine(
            store=self._store,
            experiment=self._experiment,
            sampling=derived,
            execution_policy=self._execution_policy,
            driver=self._driver,
            concurrency=self._concurrency,
            max_wall_seconds=self._max_wall_seconds,
            partial_log=self._partial_log,
            prompt_cache=self._prompt_cache,
        )

    def preflight(self, candidate: Candidate) -> None:
        self._driver.preflight(candidate)

    def validate_request(self, request: EvalRequest) -> None:
        self.preflight(request.candidate)

    def evaluate(self, request: EvalRequest) -> EngineEvaluation:
        self.validate_request(request)
        result = self._driver.run(
            experiment=self._experiment,
            sampling=self._sampling,
            request=request,
            eval_config_hash=self.eval_config_ref.config_hash,
            execution_policy=self._execution_policy,
            concurrency=self._concurrency,
            max_wall_seconds=self._max_wall_seconds,
            partial_log=self._partial_log,
            prompt_cache=self._prompt_cache,
        )
        return self._persist(request, result)

    def _validate_sampling_contract(self) -> None:
        expected = self._experiment.eval_configs.eval_config_for(
            self._sampling.split_role
        )
        if expected != self._sampling.eval_config:
            canonical = self._experiment.eval_configs.internal
            if self._sampling.split_role != canonical.split_role:
                canonical = self._experiment.eval_configs.official
            expected_subset = self._derive_sampling(
                canonical, self._sampling.task_set.task_hashes
            )
            if expected_subset != self._sampling:
                raise ValueError(
                    "engine sampling must be an exact experiment split "
                    "binding or exact derived subset"
                )

    def _eval_role(self) -> EvaluationRole:
        return evaluation_role_for_split(self._sampling.split_role)

    def _sampling_view(self) -> _EngineSamplingView:
        task_hashes = self._sampling.task_set.task_hashes
        tasks = tuple(
            _EngineTaskView(
                task_id=_task_id(task),
                task_hash=task_hash,
                prompt_inputs=_task_prompt_inputs(task),
            )
            for task, task_hash in zip(
                self._sampling.tasks, task_hashes, strict=True
            )
        )
        return _EngineSamplingView(
            task_hashes=task_hashes,
            num_samples=self._sampling.sample_plan.num_samples,
            split_role=self._sampling.split_role,
            tasks=tasks,
        )

    def _put(self, schema: str, content: dict[str, Any]) -> TypedRef:
        reference, _ = self._store.put(schema, content)
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def _evaluation_outputs_record(
        self,
        request: EvalRequest,
        rows: tuple[EvaluationOutputRow, ...],
        *,
        component_traces_ref: TypedRef,
    ) -> EvaluationOutputsRecord:
        return EvaluationOutputsRecord(
            schema_version=EVALUATION_OUTPUTS_SCHEMA_VERSION,
            candidate=candidate_reference(request.candidate),
            eval_config_ref=self.eval_config_ref,
            eval_role=self._eval_role(),
            provider_execution_policy_ref=self.provider_execution_policy_ref,
            graph_hash=self._experiment.generation_graph.graph_hash,
            metadata=request.metadata,
            split_role=self._sampling.split_role,
            task_hashes=self._sampling.task_set.task_hashes,
            num_samples=self._sampling.sample_plan.num_samples,
            component_traces_ref=component_traces_ref,
            outputs=rows,
        )

    def _evaluation_records(
        self,
        request: EvalRequest,
        result: InternalEvalResult,
    ) -> tuple[EvaluationComponentTraces, tuple[EvaluationOutputRow, ...]]:
        task_ids = tuple(_task_id(task) for task in self._sampling.tasks)
        task_hashes = self._sampling.task_set.task_hashes
        if len(task_ids) != len(task_hashes):
            raise ValueError(
                "sampling tasks and task identities must align exactly"
            )
        task_hash_by_task_id = dict(zip(task_ids, task_hashes, strict=True))
        task_by_id = {_task_id(task): task for task in self._sampling.tasks}
        task_index_by_id = {
            task_id: task_index for task_index, task_id in enumerate(task_ids)
        }
        num_samples = self._sampling.sample_plan.num_samples
        planned_ordinal = {
            GenerationIndex(
                task_index=task_index, sample_index=sample_index
            ): task_index * num_samples + sample_index
            for task_index, task_id in enumerate(task_ids)
            for sample_index in range(num_samples)
        }
        trace_rows: list[EvaluationComponentTraceRow] = []
        output_rows: list[EvaluationOutputRow] = []
        prior_ordinal = -1
        for output in result.outputs:
            if output.candidate_id != request.candidate.candidate_id:
                raise ValueError(
                    "evaluation trace candidate_id does not match request"
                )
            task_index = task_index_by_id[output.task_id]
            generation_index = GenerationIndex(
                task_index=task_index,
                sample_index=output.sample_index,
            )
            ordinal = planned_ordinal.get(generation_index)
            if ordinal is None:
                raise ValueError(
                    "evaluation trace row is outside the exact sampling plan"
                )
            if ordinal <= prior_ordinal:
                raise ValueError(
                    "evaluation trace rows must follow sampling task/"
                    "repeat order"
                )
            prior_ordinal = ordinal
            task_hash = task_hash_by_task_id[output.task_id]
            trace_rows.append(
                EvaluationComponentTraceRow(
                    task_id=output.task_id,
                    task_hash=task_hash,
                    task_index=task_index,
                    sample_index=output.sample_index,
                    executed_component_trace=ExecutedComponentTracePayload(
                        row_state=output.row_state,
                        executed_component_steps=(
                            output.executed_component_steps
                        ),
                    ),
                )
            )
            output_rows.append(
                EvaluationOutputRow(
                    candidate_id=output.candidate_id,
                    task_id=output.task_id,
                    task_hash=task_hash,
                    task_index=task_index,
                    sample_index=output.sample_index,
                    rendered_prompt=self._driver.rendered_prompt(
                        request.candidate,
                        task_by_id[output.task_id],
                        max_budget=output.max_budget,
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
                    submission_result=self._driver.submission_result_record(
                        output.submission_result
                    ),
                )
            )
        return (
            EvaluationComponentTraces(
                schema_version=EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION,
                candidate=candidate_reference(request.candidate),
                eval_config_ref=self.eval_config_ref,
                eval_role=self._eval_role(),
                provider_execution_policy_ref=self.provider_execution_policy_ref,
                graph_hash=self._experiment.generation_graph.graph_hash,
                metadata=request.metadata,
                split_role=self._sampling.split_role,
                task_hashes=task_hashes,
                num_samples=num_samples,
                rows=tuple(trace_rows),
            ),
            tuple(output_rows),
        )

    def _persist(
        self, request: EvalRequest, result: InternalEvalResult
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
            self._sampling.eval_config.model_dump(mode="json"),
        )
        if persisted_eval != eval_ref.record_ref:
            raise ValueError("persisted Eval Config reference diverged")
        aggregate = result.aggregate
        component_traces, output_rows = self._evaluation_records(
            request, result
        )
        component_traces_ref = self._put(
            EVALUATION_COMPONENT_TRACES_SCHEMA,
            component_traces.record_content(),
        )
        EvaluationComponentTracesRef(
            record=component_traces,
            record_ref=component_traces_ref,
        )
        output_record = self._evaluation_outputs_record(
            request,
            output_rows,
            component_traces_ref=component_traces_ref,
        )
        outputs_ref = self._put(
            EVALUATION_OUTPUTS_SCHEMA, output_record.record_content()
        )
        aggregate_record = aggregate.record_content()
        aggregate_ref = self._put(AGGREGATE_SCHEMA, aggregate_record)
        if aggregate_ref != aggregate.record_ref():
            raise ValueError("persisted aggregate reference diverged")
        for supplemental in result.supplemental_aggregates:
            supplemental_ref = self._put(
                AGGREGATE_SCHEMA, supplemental.record_content()
            )
            if supplemental_ref != supplemental.record_ref():
                raise ValueError(
                    "persisted supplemental aggregate reference diverged"
                )
        reward_ref = None
        if result.reward is not None:
            reward_ref = reward_reference(result.reward)
            persisted_reward = self._put(
                REWARD_SCHEMA, result.reward.record_content()
            )
            if persisted_reward != reward_ref.record_ref:
                raise ValueError("persisted Reward reference diverged")
        cache = self._cache_evidence(
            request.candidate.candidate_id, result.request_identities
        )
        evidence = EvaluationEvidence(
            schema_version=EVALUATION_EVIDENCE_SCHEMA_VERSION,
            candidate=candidate_ref,
            eval_config_ref=self.eval_config_ref,
            eval_role=self._eval_role(),
            provider_execution_policy_ref=self.provider_execution_policy_ref,
            graph_hash=aggregate.graph_hash,
            graph_config_ref=aggregate.graph_hash,
            metadata=request.metadata,
            dataset_hash=self._sampling.task_set.dataset_revision,
            task_hashes=self._sampling.task_set.task_hashes,
            num_samples=self._sampling.sample_plan.num_samples,
            per_task_values=result.per_task_scores,
            per_task_counts=result.per_task_counts,
            row_accounting=RowAccounting(
                planned=aggregate.task_count * aggregate.num_samples,
                present=aggregate.rows_present,
                missing=aggregate.rows_missing,
                failed=aggregate.rows_failed,
                invalid=aggregate.rows_invalid,
            ),
            component_traces_ref=component_traces_ref,
            outputs_ref=outputs_ref,
            aggregate_ref=aggregate_ref,
            aggregate_name=aggregate.name,
            aggregate_value=aggregate.aggregation_output.value,
            aggregate_status=aggregate.aggregation_output.status.value,
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

    def _cache_evidence(
        self, candidate_id: str, request_identities: frozenset[str]
    ) -> CacheEvidence:
        if self._partial_log is None:
            return CacheEvidence()
        rows = [
            row
            for row in self._partial_log.load()
            if row.unit == candidate_id
            and row.phase == self._sampling.split_role
            and row.request_hash in request_identities
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


def _task_id(task: object) -> str:
    task_id = getattr(task, "task_id", None)
    if isinstance(task_id, str) and task_id:
        return task_id
    legacy_id = getattr(task, "id", None)
    if isinstance(legacy_id, str) and legacy_id:
        return legacy_id
    raise ValueError("sampling task must expose task_id or id")


def _task_prompt_inputs(task: object) -> dict[str, str]:
    prompt_inputs = getattr(task, "prompt_inputs", None)
    if isinstance(prompt_inputs, dict):
        normalized: dict[str, str] = {}
        for key, value in prompt_inputs.items():
            if type(key) is not str or type(value) is not str:
                raise ValueError("task prompt_inputs must be string pairs")
            normalized[key] = value
        return normalized
    return {}
