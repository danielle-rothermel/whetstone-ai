from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dr_store import ObjectStore

from whetstone.core.identity import IdentityRef, TypedRef, typed_ref_for_record
from whetstone.eval import AggregationOutput
from whetstone.eval.aggregate import (
    AGGREGATE_SCHEMA,
    Aggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.eval.driver import EvalDriver
from whetstone.eval.drivers.eval_result import (
    InternalEvalResult,
    per_task_count,
    per_task_score,
)
from whetstone.eval.row_slice import RowEvalCompletion, RowEvalSlice
from whetstone.eval.plan import TaskTrialProvenanceRow, seed_plan_from_provenance
from whetstone.provider.llm_call import derive_rng_seed
from whetstone.eval.task_trial import TaskTrialKey
from whetstone.eval.protocol import (
    EvalEvidenceWithRef,
    EvalRejected,
    EvalRequest,
    EvalResult,
    EvalPlanSnapshot,
    EvalSplitView,
)
from whetstone.eval.schema import (
    EVAL_TRACES_SCHEMA,
    EVAL_TRACES_SCHEMA_VERSION,
    EVAL_EVIDENCE_SCHEMA_VERSION,
    EVAL_OUTPUTS_SCHEMA,
    EVAL_OUTPUTS_SCHEMA_VERSION,
    CacheEvidence,
    EvalTraceRow,
    EvalTraces,
    EvalEvidence,
    EvalFailureEvidence,
    EvalOutputRow,
    EvalOutputsRecord,
    RowAccounting,
)
from whetstone.eval.schema_names import (
    EVAL_EVIDENCE_SCHEMA,
    EVAL_FAILURE_SCHEMA,
)
from whetstone.optim.contracts import ResolutionClass, ResolutionDetail
from whetstone.eval.traces import ExecutedComponentStep, ExecutedComponentTracePayload, ExecutedRowState
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
from whetstone.experiment.reward import REWARD_SCHEMA, apply_reward_policy, reward_reference
from whetstone.experiment.sampling import EvalSplit, derive_eval_split, evaluation_role_for_split
from whetstone.core.roles import EvalRole
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)

__all__ = ["DEFAULT_CONCURRENCY", "RuntimeEvalEngine", "SamplingTaskView"]


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
    num_seeds: int
    split_role: str
    tasks: tuple[_EngineTaskView, ...]


#: Default number of rollout rows this engine drives at once.
DEFAULT_CONCURRENCY = 5


class RuntimeEvalEngine:
    """Generic evaluation engine with env-specific flow injected via driver."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        experiment: Experiment,
        sampling: EvalSplit,
        execution_policy: ProviderExecutionPolicy,
        driver: EvalDriver,
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
    def sampling(self) -> EvalSplitView:
        return self._sampling_view()

    @property
    def plan_snapshot(self) -> EvalPlanSnapshot:
        return EvalPlanSnapshot(
            graph_hash=self._experiment.rollout_graph.graph_hash,
            dataset_hash=self._sampling.task_set.dataset_revision,
            task_hashes=self._sampling.task_set.task_hashes,
            num_seeds=self._sampling.seed_plan.num_seeds,
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
        source: EvalSplit,
        task_ids: tuple[str, ...],
    ) -> EvalSplit:
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
        task_hash_by_id = dict(
            zip(
                (_task_id(task) for task in source.tasks),
                source.task_set.task_hashes,
                strict=True,
            )
        )
        namespace, separator, role = source.task_set.manifest_id.rpartition(".")
        if not separator or role != source.split_role:
            raise ValueError(
                "source sampling manifest does not match its split role"
            )
        source_rng = dict(source.seed_plan.rng_seeds)
        provenance_rows_list: list[TaskTrialProvenanceRow] = []
        for task_id in task_ids:
            task_hash = task_hash_by_id[task_id]
            for seed_index in range(source.seed_plan.num_seeds):
                key = f"{task_hash}#{seed_index}"
                rng_seed = source_rng.get(key)
                if rng_seed is None:
                    rng_seed = derive_rng_seed(task_hash, seed_index)
                provenance_rows_list.append(
                    TaskTrialProvenanceRow(
                        task_hash=task_hash,
                        seed_index=seed_index,
                        rng_seed=rng_seed,
                    )
                )
        provenance_rows = tuple(provenance_rows_list)
        seed_plan = seed_plan_from_provenance(
            provenance_rows,
            plan_id=f"{namespace}.{role}",
            version=source.seed_plan.version,
        )
        derived = derive_eval_split(
            namespace=namespace,
            dataset_revision=source.task_set.dataset_revision,
            split_role=source.split_role,
            tasks=selected,
            task_hash_of=lambda task: task_hash_by_id[_task_id(task)],
            procedure=source.procedure_config,
            aggregation=source.aggregation_config,
            num_seeds=source.seed_plan.num_seeds,
        )
        from dataclasses import replace

        return replace(derived, seed_plan=seed_plan)

    def for_task_ids(
        self, task_ids: tuple[str, ...]
    ) -> RuntimeEvalEngine:
        derived = self._derive_sampling(self._sampling, task_ids)
        return RuntimeEvalEngine(
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

    def for_task_seed(self, task_id: str, seed_index: int) -> RuntimeEvalEngine:
        if seed_index < 0:
            raise ValueError("seed_index must be non-negative")
        num_seeds = self._sampling.seed_plan.num_seeds
        if seed_index >= num_seeds:
            raise ValueError(
                f"seed_index {seed_index} is outside plan num_seeds {num_seeds}"
            )
        derived = self._derive_task_seed_sampling(
            self._sampling,
            task_id,
            seed_index,
        )
        return RuntimeEvalEngine(
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

    @staticmethod
    def _derive_task_seed_sampling(
        source: EvalSplit,
        task_id: str,
        seed_index: int,
    ) -> EvalSplit:
        from whetstone.eval import EvalDefinition, SamplingDefinition, TaskSet

        source_by_id = {_task_id(task): task for task in source.tasks}
        if task_id not in source_by_id:
            raise ValueError(f"derived sampling contains unknown task ID: {task_id!r}")
        selected = (source_by_id[task_id],)
        task_hash_by_id = dict(
            zip(
                (_task_id(task) for task in source.tasks),
                source.task_set.task_hashes,
                strict=True,
            )
        )
        task_hash = task_hash_by_id[task_id]
        namespace, separator, role = source.task_set.manifest_id.rpartition(".")
        if not separator or role != source.split_role:
            raise ValueError(
                "source sampling manifest does not match its split role"
            )
        rng_seeds = dict(source.seed_plan.rng_seeds)
        rng_seed = rng_seeds.get(f"{task_hash}#{seed_index}")
        if rng_seed is None:
            rng_seed = derive_rng_seed(task_hash, seed_index)
        seed_plan = seed_plan_from_provenance(
            (
                TaskTrialProvenanceRow(
                    task_hash=task_hash,
                    seed_index=0,
                    rng_seed=rng_seed,
                ),
            ),
            plan_id=f"{namespace}.{role}",
            version=source.seed_plan.version,
        )
        task_set = TaskSet(
            manifest_id=f"{namespace}.{role}",
            version=source.task_set.version,
            dataset_revision=source.task_set.dataset_revision,
            task_hashes=(task_hash,),
        )
        sampling = SamplingDefinition(
            definition_id=f"{namespace}.{role}.sampling",
            version=source.sampling_config.definition_ref.version,
        ).materialize(
            {
                "task_set_hash": task_set.identity_hash(),
                "seed_plan_hash": seed_plan.identity_hash(),
            }
        )
        eval_config = EvalDefinition(
            definition_id=f"{namespace}.eval",
            version=source.eval_config.definition_ref.version,
        ).materialize(
            sampling=sampling,
            evaluation_procedure=source.procedure_config,
            aggregation=source.aggregation_config,
        )
        return EvalSplit(
            split_role=source.split_role,
            tasks=selected,
            task_set=task_set,
            seed_plan=seed_plan,
            sampling_config=sampling,
            procedure_config=source.procedure_config,
            aggregation_config=source.aggregation_config,
            eval_config=eval_config,
        )

    def evaluate_row(self, request: EvalRequest) -> RowEvalCompletion:
        rejected = self._preflight(request)
        if rejected is not None:
            return RowEvalCompletion(rejected_detail=rejected.detail)
        try:
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
            evidence_with_ref, supplemental_refs = self._persist_success(
                request,
                result,
            )
            return RowEvalCompletion(
                evidence_ref=evidence_with_ref.evidence_ref,
                supplemental_aggregate_refs=supplemental_refs,
            )
        except Exception as exc:
            failure = self._persist_failure(request, exc)
            return RowEvalCompletion(evidence_ref=failure.evidence_ref)

    def assemble_from_row_slices(
        self,
        request: EvalRequest,
        *,
        row_slices: tuple[RowEvalSlice, ...],
    ) -> EvalResult:
        rejected = self._preflight(request)
        if rejected is not None:
            return rejected
        if not row_slices:
            raise ValueError("row assembly requires at least one slice")
        return self._persist_assembled_success(request, row_slices)

    def _persist_assembled_success(
        self,
        request: EvalRequest,
        row_slices: tuple[RowEvalSlice, ...],
    ) -> EvalEvidenceWithRef:
        template = row_slices[0].evidence
        for row_slice in row_slices[1:]:
            evidence = row_slice.evidence
            if evidence.candidate != template.candidate:
                raise ValueError("row evidence candidates must match")
            if evidence.graph_hash != template.graph_hash:
                raise ValueError("row evidence graph hashes must match")
            if evidence.metadata != template.metadata:
                raise ValueError("row evidence metadata must match")
            if evidence.eval_role != template.eval_role:
                raise ValueError("row evidence eval roles must match")
            if evidence.aggregate_name != template.aggregate_name:
                raise ValueError("row evidence aggregate names must match")

        task_hash_by_id = {
            _task_id(task): task_hash
            for task, task_hash in zip(
                self._sampling.tasks,
                self._sampling.task_set.task_hashes,
                strict=True,
            )
        }
        num_seeds = self._sampling.seed_plan.num_seeds
        rows_by_task_hash: dict[str, dict[int, RowValue]] = {}
        output_rows: list[EvalOutputRow] = []
        trace_rows: list[EvalTraceRow] = []
        supplemental_rows_by_name: dict[str, dict[str, dict[int, RowValue]]] = {}
        supplemental_refs: list[TypedRef] = []
        cache = CacheEvidence()
        deadline_reached = False
        for row_slice in row_slices:
            task_hash = task_hash_by_id.get(row_slice.task_id)
            if task_hash is None:
                raise ValueError(
                    f"row slice task_id is outside sampling plan: {row_slice.task_id!r}"
                )
            outputs_record = EvalOutputsRecord.model_validate(
                self._store.get(row_slice.evidence.outputs_ref.reference)
            )
            output_row = self._load_output_row(row_slice.evidence.outputs_ref)
            trace_row = self._load_trace_row(row_slice.evidence.traces_ref)
            if len(outputs_record.outputs) != 1:
                raise ValueError(
                    "platform row evidence must contain exactly one output row"
                )
            _ = outputs_record
            remapped_output = output_row.model_copy(
                update={
                    "task_index": self._sampling.task_set.task_hashes.index(task_hash),
                    "task_hash": task_hash,
                    "seed_index": row_slice.seed_index,
                }
            )
            remapped_trace = trace_row.model_copy(
                update={
                    "task_index": remapped_output.task_index,
                    "task_hash": task_hash,
                    "seed_index": row_slice.seed_index,
                }
            )
            output_rows.append(remapped_output)
            trace_rows.append(remapped_trace)
            rows_by_task_hash.setdefault(task_hash, {})[row_slice.seed_index] = RowValue(
                value=output_row.score,
                failed=output_row.failed,
                missing=output_row.missing,
                invalid=output_row.invalid,
            )
            aggregate_record = self._load_aggregate_record(
                row_slice.evidence.aggregate_ref
            )
            if aggregate_record.name != template.aggregate_name:
                raise ValueError("row aggregate name must match template")
            partial_aggregate_ref = self._put(
                AGGREGATE_SCHEMA,
                aggregate_record.record_content(),
            )
            if partial_aggregate_ref != row_slice.evidence.aggregate_ref:
                raise ValueError("row aggregate reference diverged")
            for supplemental_ref in row_slice.supplemental_aggregate_refs:
                supplemental_record = self._load_aggregate_record(supplemental_ref)
                supplemental_name = supplemental_record.name
                row_value = RowValue(
                    value=supplemental_record.aggregation_output.value,
                    failed=supplemental_record.rows_failed > 0,
                    missing=supplemental_record.rows_missing > 0,
                    invalid=supplemental_record.rows_invalid > 0,
                )
                supplemental_rows_by_name.setdefault(supplemental_name, {}).setdefault(
                    task_hash, {}
                )[row_slice.seed_index] = row_value
            cache = CacheEvidence(
                partial_row_count=cache.partial_row_count
                + row_slice.evidence.cache.partial_row_count,
                cache_hit_count=cache.cache_hit_count
                + row_slice.evidence.cache.cache_hit_count,
                source_call_ids=cache.source_call_ids
                + row_slice.evidence.cache.source_call_ids,
            )
            deadline_reached = deadline_reached or row_slice.evidence.deadline_reached

        task_rows = tuple(
            TaskRows(
                task_hash=task_hash,
                rows=tuple(
                    rows_by_task_hash.get(task_hash, {}).get(
                        seed_index,
                        RowValue(missing=True),
                    )
                    for seed_index in range(num_seeds)
                ),
            )
            for task_hash in self._sampling.task_set.task_hashes
        )
        matrix_plan = self._sampling.evaluation_matrix_plan
        aggregate_name = template.aggregate_name
        aggregate = unweighted_task_mean(
            aggregate_name=aggregate_name,
            graph_hash=template.graph_hash,
            task_rows=task_rows,
            plan=matrix_plan,
        )
        supplemental_aggregates = tuple(
            unweighted_task_mean(
                aggregate_name=name,
                graph_hash=template.graph_hash,
                task_rows=tuple(
                    TaskRows(
                        task_hash=task_hash,
                        rows=tuple(
                            supplemental_rows_by_name[name]
                            .get(task_hash, {})
                            .get(
                                seed_index,
                                RowValue(missing=True),
                            )
                            for seed_index in range(num_seeds)
                        ),
                    )
                    for task_hash in self._sampling.task_set.task_hashes
                ),
                plan=matrix_plan,
            )
            for name in sorted(supplemental_rows_by_name)
            if name != aggregate_name
        )
        internal_result = InternalEvalResult(
            aggregate=aggregate,
            reward=None,
            per_task_scores=tuple(
                per_task_score(task_row, num_seeds) for task_row in task_rows
            ),
            per_task_counts=tuple(
                per_task_count(task_row, num_seeds) for task_row in task_rows
            ),
            outputs=(),
            supplemental_aggregates=supplemental_aggregates,
            deadline_reached=deadline_reached,
        )
        ordered_outputs = sorted(
            output_rows,
            key=lambda row: (row.task_index, row.seed_index),
        )
        ordered_traces = sorted(
            trace_rows,
            key=lambda row: (row.task_index, row.seed_index),
        )
        traces = EvalTraces(
            schema_version=EVAL_TRACES_SCHEMA_VERSION,
            candidate=template.candidate,
            eval_config_ref=self.eval_config_ref,
            eval_role=template.eval_role,
            provider_execution_policy_ref=template.provider_execution_policy_ref,
            graph_hash=template.graph_hash,
            metadata=template.metadata,
            split_role=self._sampling.split_role,
            task_hashes=self._sampling.task_set.task_hashes,
            num_seeds=num_seeds,
            rows=tuple(ordered_traces),
        )
        traces_ref = self._put(EVAL_TRACES_SCHEMA, traces.record_content())
        output_record = EvalOutputsRecord(
            schema_version=EVAL_OUTPUTS_SCHEMA_VERSION,
            candidate=template.candidate,
            eval_config_ref=self.eval_config_ref,
            eval_role=template.eval_role,
            provider_execution_policy_ref=template.provider_execution_policy_ref,
            graph_hash=template.graph_hash,
            metadata=template.metadata,
            split_role=self._sampling.split_role,
            task_hashes=self._sampling.task_set.task_hashes,
            num_seeds=num_seeds,
            traces_ref=traces_ref,
            outputs=tuple(ordered_outputs),
        )
        outputs_ref = self._put(
            EVAL_OUTPUTS_SCHEMA,
            output_record.record_content(),
        )
        aggregate_ref = self._put(AGGREGATE_SCHEMA, aggregate.record_content())
        if aggregate_ref != aggregate.record_ref():
            raise ValueError("persisted aggregate reference diverged")
        for supplemental in supplemental_aggregates:
            supplemental_ref = self._put(
                AGGREGATE_SCHEMA,
                supplemental.record_content(),
            )
            if supplemental_ref != supplemental.record_ref():
                raise ValueError(
                    "persisted supplemental aggregate reference diverged"
                )
            supplemental_refs.append(supplemental_ref)
        reward = None
        if self._eval_role() is EvalRole.INTERNAL:
            aggregates = {
                aggregate.name: aggregate.aggregation_output.value,
            }
            aggregates.update(
                {
                    supplemental.name: supplemental.aggregation_output.value
                    for supplemental in supplemental_aggregates
                }
            )
            reward = apply_reward_policy(
                self._experiment.reward_policy,
                aggregates=aggregates,
                evidence_role=EvalRole.INTERNAL,
                evidence_refs=(aggregate_ref, *supplemental_refs),
            )
        reward_ref = None
        if reward is not None:
            reward_ref = reward_reference(reward)
            persisted_reward = self._put(
                REWARD_SCHEMA,
                reward.record_content(),
            )
            if persisted_reward != reward_ref.record_ref:
                raise ValueError("persisted Reward reference diverged")
        evidence = EvalEvidence(
            schema_version=EVAL_EVIDENCE_SCHEMA_VERSION,
            candidate=template.candidate,
            eval_config_ref=self.eval_config_ref,
            eval_role=template.eval_role,
            provider_execution_policy_ref=template.provider_execution_policy_ref,
            graph_hash=aggregate.graph_hash,
            graph_config_ref=aggregate.graph_hash,
            metadata=template.metadata,
            dataset_hash=self._sampling.task_set.dataset_revision,
            task_hashes=self._sampling.task_set.task_hashes,
            num_seeds=num_seeds,
            per_task_values=internal_result.per_task_scores,
            per_task_counts=internal_result.per_task_counts,
            row_accounting=RowAccounting(
                planned=aggregate.task_count * aggregate.num_seeds,
                present=aggregate.rows_present,
                missing=aggregate.rows_missing,
                failed=aggregate.rows_failed,
                invalid=aggregate.rows_invalid,
            ),
            traces_ref=traces_ref,
            outputs_ref=outputs_ref,
            aggregate_ref=aggregate_ref,
            aggregate_name=aggregate.name,
            aggregate_value=aggregate.aggregation_output.value,
            aggregate_status=aggregate.aggregation_output.status.value,
            reward_ref=reward_ref,
            cache=cache,
            deadline_reached=deadline_reached,
        )
        evidence_ref = self._put(
            EVAL_EVIDENCE_SCHEMA,
            evidence.record_content(),
        )
        _ = internal_result
        return EvalEvidenceWithRef(evidence=evidence, evidence_ref=evidence_ref)

    def preflight(self, candidate: Candidate) -> None:
        self._driver.preflight(candidate)

    def _preflight(self, request: EvalRequest) -> EvalRejected | None:
        try:
            self.preflight(request.candidate)
        except (KeyError, TypeError, ValueError) as exc:
            return EvalRejected(
                detail=ResolutionDetail(
                    classification=ResolutionClass.VALIDATION,
                    message=str(exc) or type(exc).__name__,
                )
            )
        return None

    def evaluate(self, request: EvalRequest) -> EvalResult:
        rejected = self._preflight(request)
        if rejected is not None:
            return rejected
        try:
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
            evidence_with_ref, _ = self._persist_success(request, result)
            return evidence_with_ref
        except Exception as exc:
            return self._persist_failure(request, exc)

    def _persist_failure(
        self, request: EvalRequest, exc: BaseException
    ) -> EvalEvidenceWithRef:
        failure = EvalFailureEvidence(
            candidate=candidate_reference(request.candidate),
            eval_config_ref=self.eval_config_ref,
            eval_role=self._eval_role(),
            provider_execution_policy_ref=self.provider_execution_policy_ref,
            metadata=request.metadata,
            exception_type=type(exc).__name__,
            message=str(exc) or type(exc).__name__,
        )
        failure_ref = self._put(
            EVAL_FAILURE_SCHEMA, failure.record_content()
        )
        return EvalEvidenceWithRef(evidence=failure, evidence_ref=failure_ref)

    def _validate_sampling_contract(self) -> None:
        expected = self._experiment.eval_configs.eval_config_for(
            self._sampling.split_role
        )
        if expected == self._sampling.eval_config:
            return
        canonical = self._experiment.eval_configs.internal
        if self._sampling.split_role != canonical.split_role:
            canonical = self._experiment.eval_configs.official
        if (
            len(self._sampling.tasks) == 1
            and self._sampling.seed_plan.num_seeds == 1
        ):
            task_id = _task_id(self._sampling.tasks[0])
            for seed_index in range(canonical.seed_plan.num_seeds):
                if (
                    self._derive_task_seed_sampling(
                        canonical,
                        task_id,
                        seed_index,
                    )
                    == self._sampling
                ):
                    return
        expected_subset = self._derive_sampling(
            canonical,
            tuple(_task_id(task) for task in self._sampling.tasks),
        )
        if expected_subset == self._sampling:
            return
        raise ValueError(
            "engine sampling must be an exact experiment split "
            "binding or exact derived subset"
        )

    def _eval_role(self) -> EvalRole:
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
            num_seeds=self._sampling.seed_plan.num_seeds,
            split_role=self._sampling.split_role,
            tasks=tasks,
        )

    def _put(self, schema: str, content: dict[str, Any]) -> TypedRef:
        reference, _ = self._store.put(schema, content)
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def _load_aggregate_record(self, reference: TypedRef) -> Aggregate:
        content = self._store.get(reference.reference)
        if not isinstance(content, dict):
            raise ValueError("aggregate record is not an object")
        return Aggregate(
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

    def _load_trace_row(self, reference: TypedRef) -> EvalTraceRow:
        raw = self._store.get(reference.reference)
        if not isinstance(raw, dict):
            raise ValueError("trace record is not an object")
        rows = raw.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("platform row trace record must contain one row")
        row = rows[0]
        if not isinstance(row, dict):
            raise ValueError("trace row is not an object")
        trace = row.get("trace")
        if not isinstance(trace, dict):
            raise ValueError("trace payload is not an object")
        trace_steps = trace.get("trace_steps", ())
        normalized_steps: list[dict[str, object]] = []
        for step in trace_steps:
            if not isinstance(step, dict):
                raise ValueError("trace step is not an object")
            normalized = dict(step)
            for field_name in ("input_field_names", "output_field_names"):
                values = normalized.get(field_name)
                if isinstance(values, list):
                    normalized[field_name] = tuple(values)
            normalized_steps.append(normalized)
        return EvalTraceRow(
            task_id=row["task_id"],
            task_hash=row["task_hash"],
            task_index=row["task_index"],
            seed_index=row["seed_index"],
            trace=ExecutedComponentTracePayload(
                row_state=ExecutedRowState(trace["row_state"]),
                trace_steps=tuple(
                    ExecutedComponentStep.model_validate(step)
                    for step in normalized_steps
                ),
            ),
        )

    def _load_output_row(self, reference: TypedRef) -> EvalOutputRow:
        raw = self._store.get(reference.reference)
        if not isinstance(raw, dict):
            raise ValueError("outputs record is not an object")
        outputs = raw.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError("platform row outputs record must contain one row")
        return EvalOutputRow.model_validate(outputs[0])

    def _evaluation_outputs_record(
        self,
        request: EvalRequest,
        rows: tuple[EvalOutputRow, ...],
        *,
        traces_ref: TypedRef,
    ) -> EvalOutputsRecord:
        return EvalOutputsRecord(
            schema_version=EVAL_OUTPUTS_SCHEMA_VERSION,
            candidate=candidate_reference(request.candidate),
            eval_config_ref=self.eval_config_ref,
            eval_role=self._eval_role(),
            provider_execution_policy_ref=self.provider_execution_policy_ref,
            graph_hash=self._experiment.rollout_graph.graph_hash,
            metadata=request.metadata,
            split_role=self._sampling.split_role,
            task_hashes=self._sampling.task_set.task_hashes,
            num_seeds=self._sampling.seed_plan.num_seeds,
            traces_ref=traces_ref,
            outputs=rows,
        )

    def _evaluation_records(
        self,
        request: EvalRequest,
        result: InternalEvalResult,
    ) -> tuple[EvalTraces, tuple[EvalOutputRow, ...]]:
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
        num_seeds = self._sampling.seed_plan.num_seeds
        planned_ordinal = {
            TaskTrialKey(
                task_index=task_index, seed_index=seed_index
            ): task_index * num_seeds + seed_index
            for task_index, task_id in enumerate(task_ids)
            for seed_index in range(num_seeds)
        }
        trace_rows: list[EvalTraceRow] = []
        output_rows: list[EvalOutputRow] = []
        prior_ordinal = -1
        for output in result.outputs:
            if output.candidate_id != request.candidate.candidate_id:
                raise ValueError(
                    "evaluation trace candidate_id does not match request"
                )
            task_index = task_index_by_id[output.task_id]
            task_trial_key = TaskTrialKey(
                task_index=task_index,
                seed_index=output.seed_index,
            )
            ordinal = planned_ordinal.get(task_trial_key)
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
                EvalTraceRow(
                    task_id=output.task_id,
                    task_hash=task_hash,
                    task_index=task_index,
                    seed_index=output.seed_index,
                    trace=ExecutedComponentTracePayload(
                        row_state=output.row_state,
                        trace_steps=(
                            output.trace_steps
                        ),
                    ),
                )
            )
            output_rows.append(
                EvalOutputRow(
                    candidate_id=output.candidate_id,
                    task_id=output.task_id,
                    task_hash=task_hash,
                    task_index=task_index,
                    seed_index=output.seed_index,
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
            EvalTraces(
                schema_version=EVAL_TRACES_SCHEMA_VERSION,
                candidate=candidate_reference(request.candidate),
                eval_config_ref=self.eval_config_ref,
                eval_role=self._eval_role(),
                provider_execution_policy_ref=self.provider_execution_policy_ref,
                graph_hash=self._experiment.rollout_graph.graph_hash,
                metadata=request.metadata,
                split_role=self._sampling.split_role,
                task_hashes=task_hashes,
                num_seeds=num_seeds,
                rows=tuple(trace_rows),
            ),
            tuple(output_rows),
        )

    def _persist_success(
        self, request: EvalRequest, result: InternalEvalResult
    ) -> tuple[EvalEvidenceWithRef, tuple[TypedRef, ...]]:
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
        traces_ref = self._put(
            EVAL_TRACES_SCHEMA,
            component_traces.record_content(),
        )
        output_record = self._evaluation_outputs_record(
            request,
            output_rows,
            traces_ref=traces_ref,
        )
        outputs_ref = self._put(
            EVAL_OUTPUTS_SCHEMA, output_record.record_content()
        )
        aggregate_record = aggregate.record_content()
        aggregate_ref = self._put(AGGREGATE_SCHEMA, aggregate_record)
        if aggregate_ref != aggregate.record_ref():
            raise ValueError("persisted aggregate reference diverged")
        supplemental_refs: list[TypedRef] = []
        for supplemental in result.supplemental_aggregates:
            supplemental_ref = self._put(
                AGGREGATE_SCHEMA, supplemental.record_content()
            )
            if supplemental_ref != supplemental.record_ref():
                raise ValueError(
                    "persisted supplemental aggregate reference diverged"
                )
            supplemental_refs.append(supplemental_ref)
        reward = result.reward
        if reward is None and self._eval_role() is EvalRole.INTERNAL:
            aggregates = {
                aggregate.name: aggregate.aggregation_output.value,
            }
            aggregates.update(
                {
                    supplemental.name: supplemental.aggregation_output.value
                    for supplemental in result.supplemental_aggregates
                }
            )
            reward = apply_reward_policy(
                self._experiment.reward_policy,
                aggregates=aggregates,
                evidence_role=EvalRole.INTERNAL,
                evidence_refs=(aggregate_ref, *supplemental_refs),
            )
        reward_ref = None
        if reward is not None:
            reward_ref = reward_reference(reward)
            persisted_reward = self._put(
                REWARD_SCHEMA, reward.record_content()
            )
            if persisted_reward != reward_ref.record_ref:
                raise ValueError("persisted Reward reference diverged")
        cache = self._cache_evidence(
            request.candidate.candidate_id, result.request_identities
        )
        evidence = EvalEvidence(
            schema_version=EVAL_EVIDENCE_SCHEMA_VERSION,
            candidate=candidate_ref,
            eval_config_ref=self.eval_config_ref,
            eval_role=self._eval_role(),
            provider_execution_policy_ref=self.provider_execution_policy_ref,
            graph_hash=aggregate.graph_hash,
            graph_config_ref=aggregate.graph_hash,
            metadata=request.metadata,
            dataset_hash=self._sampling.task_set.dataset_revision,
            task_hashes=self._sampling.task_set.task_hashes,
            num_seeds=self._sampling.seed_plan.num_seeds,
            per_task_values=result.per_task_scores,
            per_task_counts=result.per_task_counts,
            row_accounting=RowAccounting(
                planned=aggregate.task_count * aggregate.num_seeds,
                present=aggregate.rows_present,
                missing=aggregate.rows_missing,
                failed=aggregate.rows_failed,
                invalid=aggregate.rows_invalid,
            ),
            traces_ref=traces_ref,
            outputs_ref=outputs_ref,
            aggregate_ref=aggregate_ref,
            aggregate_name=aggregate.name,
            aggregate_value=aggregate.aggregation_output.value,
            aggregate_status=aggregate.aggregation_output.status.value,
            reward_ref=reward_ref,
            cache=cache,
            deadline_reached=result.deadline_reached,
        )
        evidence_ref = self._put(
            EVAL_EVIDENCE_SCHEMA, evidence.record_content()
        )
        return (
            EvalEvidenceWithRef(evidence=evidence, evidence_ref=evidence_ref),
            tuple(supplemental_refs),
        )

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
