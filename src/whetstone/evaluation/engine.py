from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from dr_store import ObjectStore
from whetstone_envs.core import Instance

from whetstone.core.identity import (
    IdentityRef,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.envs.code_comp.generation_graph.direct import (
    render_direct_frame,
)
from whetstone.envs.code_comp.modes.direct import DirectExperiment
from whetstone.envs.code_comp.mutation_surface import (
    render_encoder_frame,
    validate_instruction_body,
)
from whetstone.envs.code_comp.registry import CodeCompMode, code_comp_mode_for
from whetstone.envs.code_comp.scoring import CodeBatchScorer
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.generation_graph import (
    render_prompt,
    validate_candidate_prompt,
)
from whetstone.envs.registry import env_spec
from whetstone.envs.sampling import EnvSplitSampling, derive_split_sampling
from whetstone.evaluation.aggregate import AGGREGATE_SCHEMA
from whetstone.evaluation.drivers.code_comp.direct import DirectRowJobFactory
from whetstone.evaluation.drivers.code_comp.dispatch import run_code_comp_eval
from whetstone.evaluation.drivers.code_comp.encdec import EncDecRowJobFactory
from whetstone.evaluation.drivers.internal import (
    InternalEvalResult,
    InternalRowJobFactory,
    run_internal_eval,
)
from whetstone.evaluation.generation import GenerationIndex
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
    EvaluationEvidenceRef,
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
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.candidate import (
    CANDIDATE_RECORD_SCHEMA,
    Candidate,
    candidate_reference,
)
from whetstone.experiment.reward import REWARD_SCHEMA, reward_reference
from whetstone.optimization.proposal.mutation import MUTATION_FIELD
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
        row_job_factory: InternalRowJobFactory | EncDecRowJobFactory,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_wall_seconds: float | None = None,
        partial_log: PartialLog | None = None,
        prompt_cache: PromptResultCache | None = None,
        batch_scorer: CodeBatchScorer | None = None,
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
        self._batch_scorer = batch_scorer
        expected = experiment.eval_configs.eval_config_for(sampling.split_role)
        if expected != sampling.eval_config:
            canonical = (
                experiment.eval_configs.internal
                if sampling.split_role
                == experiment.eval_configs.internal.split_role
                else experiment.eval_configs.official
            )
            expected_subset = self._derive_sampling(
                canonical, sampling.task_set.task_hashes
            )
            if expected_subset != sampling:
                raise ValueError(
                    "engine sampling must be an exact experiment split "
                    "binding or exact derived subset"
                )

    @property
    def eval_config_ref(self) -> EvalConfigRef:
        return eval_config_reference(self.sampling.eval_config)

    @property
    def task_model_identity_hash(self) -> str:
        """Identity of the exact task-model Provider Call Config route."""

        provider_config = self.experiment.generation_graph.provider_call_config
        return provider_config.identity_hash

    @property
    def execution_policy_identity_hash(self) -> str:
        """Identity of the retry/backoff policy used for task evaluations."""

        return self._execution_policy.identity_hash

    @property
    def reward_policy_identity_hash(self) -> str:
        """Identity of the Reward Policy applied to internal evaluations."""

        return self.experiment.reward_policy.identity_hash()

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
            record_hash=self._execution_policy.identity_hash,
        )

    @property
    def provider_execution_policy_record(self) -> dict[str, Any]:
        """Return the canonical policy record advertised by the engine."""
        return self._execution_policy.identity_payload()

    @staticmethod
    def _derive_sampling(
        source: EnvSplitSampling,
        task_ids: tuple[str, ...],
    ) -> EnvSplitSampling:
        if not task_ids:
            raise ValueError("derived sampling requires at least one task")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("derived sampling task IDs must be unique")
        source_by_id = dict(
            zip(
                source.task_set.task_hashes,
                source.tasks,
                strict=True,
            )
        )
        unknown = tuple(
            task_id for task_id in task_ids if task_id not in source_by_id
        )
        if unknown:
            raise ValueError(
                f"derived sampling contains unknown task IDs: {unknown!r}"
            )
        selected = tuple(source_by_id[task_id] for task_id in task_ids)
        identity_by_instance = {
            id(instance): task_id
            for task_id, instance in zip(task_ids, selected, strict=True)
        }
        namespace, separator, role = source.task_set.manifest_id.rpartition(
            "."
        )
        if not separator or role != source.split_role:
            raise ValueError(
                "source sampling manifest does not match its split role"
            )
        return derive_split_sampling(
            namespace=namespace,
            dataset_revision=source.task_set.dataset_revision,
            split_role=source.split_role,
            tasks=selected,
            task_hash_of=lambda instance: identity_by_instance[id(instance)],
            procedure=source.procedure_config,
            aggregation=source.aggregation_config,
            num_samples=source.sample_plan.num_samples,
        )

    def for_task_ids(self, task_ids: tuple[str, ...]) -> EvaluationEngine:
        """Return an engine bound to one exact ordered task subset.

        The view derives from this engine's complete sampling contract;
        callers cannot override repeats, role, procedure, aggregation, or
        dataset identity independently.
        """
        derived = self._derive_sampling(self.sampling, task_ids)
        return EvaluationEngine(
            store=self._store,
            experiment=self.experiment,
            sampling=derived,
            execution_policy=self._execution_policy,
            row_job_factory=self._row_job_factory,
            concurrency=self._concurrency,
            max_wall_seconds=self._max_wall_seconds,
            partial_log=self._partial_log,
            prompt_cache=self._prompt_cache,
            batch_scorer=self._batch_scorer,
        )

    def _code_comp_mode(self) -> CodeCompMode | None:
        try:
            return code_comp_mode_for(self.experiment)
        except TypeError:
            return None

    def preflight(self, candidate: Candidate) -> None:
        """Reject malformed candidates before any provider call."""
        mode = self._code_comp_mode()
        if mode is not None:
            body = candidate.payload.get(MUTATION_FIELD)
            if type(body) is not str:
                raise ValueError(
                    "code_comp candidate body must be a strict string"
                )
            validate_instruction_body(body)
            return
        validate_candidate_prompt(
            env_spec(self.experiment.env_name),
            candidate,
            self.sampling.tasks,
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
        rows: tuple[EvaluationOutputRow, ...],
        *,
        component_traces_ref: TypedRef,
    ) -> EvaluationOutputsRecord:
        return EvaluationOutputsRecord(
            schema_version=EVALUATION_OUTPUTS_SCHEMA_VERSION,
            candidate=candidate_reference(request.candidate),
            evaluation_binding=request.evaluation_binding,
            evaluation_role=request.evaluation_binding.role,
            graph_hash=self.experiment.generation_graph.graph_hash,
            purpose=request.purpose,
            split_role=self.sampling.split_role,
            task_hashes=self.sampling.task_set.task_hashes,
            num_samples=self.sampling.sample_plan.num_samples,
            component_traces_ref=component_traces_ref,
            outputs=rows,
        )

    def _evaluation_records(
        self,
        request: EvaluationRequest,
        result: InternalEvalResult,
    ) -> tuple[EvaluationComponentTraces, tuple[EvaluationOutputRow, ...]]:
        task_ids = tuple(str(instance.id) for instance in self.sampling.tasks)
        task_hashes = self.sampling.task_set.task_hashes
        if len(task_ids) != len(task_hashes):
            raise ValueError(
                "sampling instances and task identities must align exactly"
            )
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("sampling instance IDs must be unique")
        if len(set(task_hashes)) != len(task_hashes):
            raise ValueError("sampling task identities must be unique")

        task_hash_by_instance = dict(zip(task_ids, task_hashes, strict=True))
        instance_by_id = {
            str(instance.id): instance for instance in self.sampling.tasks
        }
        task_index_by_id = {
            task_id: task_index for task_index, task_id in enumerate(task_ids)
        }
        num_samples = self.sampling.sample_plan.num_samples
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
                    "evaluation trace rows must follow sampling instance/"
                    "repeat order"
                )
            prior_ordinal = ordinal
            task_hash = task_hash_by_instance[output.task_id]
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
                    rendered_prompt=self._rendered_prompt(
                        request.candidate,
                        instance_by_id[output.task_id],
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
                )
            )
        return (
            EvaluationComponentTraces(
                schema_version=EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION,
                candidate=candidate_reference(request.candidate),
                evaluation_binding=request.evaluation_binding,
                evaluation_role=request.evaluation_binding.role,
                graph_hash=self.experiment.generation_graph.graph_hash,
                purpose=request.purpose,
                split_role=self.sampling.split_role,
                task_hashes=task_hashes,
                num_samples=num_samples,
                rows=tuple(trace_rows),
            ),
            tuple(output_rows),
        )

    def _rendered_prompt(
        self,
        candidate: Candidate,
        instance: Instance,
        *,
        max_budget: int | None,
    ) -> str:
        mode = self._code_comp_mode()
        if mode in {CodeCompMode.ENCDEC, CodeCompMode.ENCDEC_MUTANT}:
            body = candidate.payload[MUTATION_FIELD]
            if type(body) is not str:
                raise ValueError(
                    "code_comp candidate body must be a strict string"
                )
            return render_encoder_frame(
                body,
                input_code=instance.prompt_inputs["input_code"],
                max_budget=max_budget,
            )
        if mode is CodeCompMode.DIRECT:
            body = candidate.payload[MUTATION_FIELD]
            if type(body) is not str:
                raise ValueError(
                    "code_comp candidate body must be a strict string"
                )
            assert isinstance(self.experiment, DirectExperiment)
            return render_direct_frame(
                body,
                input_arm=self.experiment.input_arm,
            )
        return render_prompt(
            env_spec(self.experiment.env_name), candidate, instance
        )

    def evaluate(self, request: EvaluationRequest) -> EngineEvaluation:
        self.validate_request(request)
        result = self._run(request)
        return self._persist(request, result)

    def _run(self, request: EvaluationRequest) -> InternalEvalResult:
        mode = self._code_comp_mode()
        if mode is not None:
            body = request.candidate.payload[MUTATION_FIELD]
            if type(body) is not str:
                raise ValueError(
                    "code_comp candidate body must be a strict string"
                )
            common = {
                "candidate_id": request.candidate.candidate_id,
                "sampling": self.sampling,
                "execution_policy": self._execution_policy,
                "evaluation_binding": request.evaluation_binding,
                "concurrency": self._concurrency,
                "max_wall_seconds": self._max_wall_seconds,
                "partial_log": self._partial_log,
                "cache": self._prompt_cache,
                "batch_scorer": self._batch_scorer,
            }
            if mode is CodeCompMode.DIRECT:
                from whetstone.evaluation.drivers.code_comp.direct import (
                    DirectEvalResult,
                )

                result = run_code_comp_eval(
                    self.experiment,
                    candidate_body=body,
                    row_job_factory=cast(
                        DirectRowJobFactory, self._row_job_factory
                    ),
                    **common,
                )
                assert isinstance(result, DirectEvalResult)
                return InternalEvalResult(
                    aggregate=result.submission_score_aggregate,
                    reward=result.reward,
                    per_task_scores=result.per_task_scores,
                    per_task_counts=result.per_task_counts,
                    outputs=result.outputs,
                    supplemental_aggregates=(),
                )
            from whetstone.evaluation.drivers.code_comp.encdec import (
                EncDecEvalResult,
            )

            result = run_code_comp_eval(
                self.experiment,
                candidate_template=body,
                row_job_factory=cast(
                    EncDecRowJobFactory, self._row_job_factory
                ),
                **common,
            )
            assert isinstance(result, EncDecEvalResult)
            return InternalEvalResult(
                aggregate=result.primary_aggregate,
                reward=result.reward,
                per_task_scores=result.per_task_scores,
                per_task_counts=result.per_task_counts,
                outputs=result.outputs,
                supplemental_aggregates=(result.compression_aggregate,),
                request_identities=result.request_identities,
                concurrency_halved=result.concurrency_halved,
                deadline_reached=result.deadline_reached,
                guard_timeouts=result.guard_timeouts,
            )
        return run_internal_eval(
            self.experiment,
            candidate=request.candidate,
            sampling=self.sampling,
            execution_policy=self._execution_policy,
            row_job_factory=cast(InternalRowJobFactory, self._row_job_factory),
            evaluation_binding=request.evaluation_binding,
            concurrency=self._concurrency,
            max_wall_seconds=self._max_wall_seconds,
            partial_log=self._partial_log,
            render_guard=True,
            cache=self._prompt_cache,
        )

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
        aggregation_output = aggregate.aggregation_output
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
            evaluation_binding=request.evaluation_binding,
            graph_hash=aggregate.graph_hash,
            graph_config_ref=aggregate.graph_hash,
            purpose=request.purpose,
            dataset_hash=self.sampling.task_set.dataset_revision,
            task_hashes=self.sampling.task_set.task_hashes,
            num_samples=self.sampling.sample_plan.num_samples,
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

    def _cache_evidence(
        self, candidate_id: str, request_identities: frozenset[str]
    ) -> CacheEvidence:
        """Summarize only the partial rows this exact binding could restore.

        Restoration is strictly request-identity scoped, so provenance is too:
        a row written by another Evaluation Binding of the same candidate and
        split was never eligible here and is not this evaluation's evidence.
        """
        if self._partial_log is None:
            return CacheEvidence()
        rows = [
            row
            for row in self._partial_log.load()
            if row.unit == candidate_id
            and row.phase == self.sampling.split_role
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


__all__ = ["EngineEvaluation", "EvaluationEngine", "EvaluationRequest"]
