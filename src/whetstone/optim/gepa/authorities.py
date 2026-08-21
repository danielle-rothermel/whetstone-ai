from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

from dr_store import ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    model_validator,
)

from whetstone.coordination.eval_service import EvalEngineService, EvalExecutionContext
from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.core.roles import EvalRole
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import EvalRequest, EvalEngine
from whetstone.eval.schema import (
    EVAL_TRACES_SCHEMA,
    EVAL_OUTPUTS_SCHEMA,
    EvalEvidence,
)
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.optim.contracts import (
    OptimEvalRequest,
    IntentOutcome,
)
from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaComponentTraceProjection,
    GepaDataInstance,
    GepaEvalAuthorityBinding,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaEvaluationRow,
    GepaProposalAuthorityBinding,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
    GepaTrajectoryProjection,
)
from whetstone.optim.gepa.control import GepaControl

from whetstone.optim.gepa.submission_projection import (
    DefaultGepaSubmissionProjector,
    GepaSubmissionProjector,
)
from whetstone.optim.gepa.prompts import GepaPromptServices
from whetstone.optim.proposal.proposer import (
    DurableProposalExecutor,
    ProposalRequest,
    ProposerTransport,
    require_canonical_proposal_executor,
)

GEPA_REFLECTION_BASE_SCHEMA = "whetstone.gepa.reflection_base"
GEPA_CANDIDATE_ASSEMBLER_SCHEMA = "whetstone.gepa.candidate_assembler"
GEPA_CANDIDATE_ASSEMBLER_SCHEMA_VERSION = 1
GEPA_DATA_REGISTRY_SCHEMA = "whetstone.gepa.data_registry"
GEPA_DATA_REGISTRY_SCHEMA_VERSION = 1
GEPA_DATA_RECORD_SCHEMA = "whetstone.gepa.reflection_input"
GEPA_DATA_LOADER_IDENTITY_HASH = compute_identity_hash(
    schema="whetstone.gepa.data_loader",
    schema_version=1,
    payload={
        "source": "EvalEngine.sampling.tasks",
        "projection": "task_hash+task_id+ordered_prompt_inputs/v1",
        "gold_included": False,
    },
)
GEPA_EVALUATION_AUTHORITY_SCHEMA = "whetstone.gepa.evaluation_authority"
GEPA_EVALUATION_AUTHORITY_SCHEMA_VERSION = 1
GEPA_PROPOSAL_AUTHORITY_SCHEMA = "whetstone.gepa.proposal_authority"
GEPA_PROPOSAL_AUTHORITY_SCHEMA_VERSION = 1
GEPA_EVALUATION_REJECTION_SCHEMA = "whetstone.gepa.evaluation_rejection"
GEPA_EVALUATION_ROW_FAILURE_SCHEMA = "whetstone.gepa.evaluation_row_failure"
GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA = (
    "whetstone.gepa.proposal_provider_attempt/v2"
)


GEPA_WHOLE_CALL_EVIDENCE_BOUNDARY = "whole_call"
GEPA_EVALUATION_RESPONSE_PARSER_IDENTITY_HASH = compute_identity_hash(
    schema="whetstone.gepa.evaluation_response_projection",
    schema_version=1,
    payload={
        "evaluation_evidence_schema": EVAL_EVIDENCE_SCHEMA,
        "outputs_schema": EVAL_OUTPUTS_SCHEMA,
        "ordered_rows": True,
        "num_seeds": 1,
        "trace_projection": "ordered_native_component_steps/v1",
    },
)


class GepaCandidateFieldBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_name: StrictStr
    candidate_field: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> GepaCandidateFieldBinding:
        if not self.component_name or not self.candidate_field:
            raise ValueError("GEPA candidate field names must be non-empty")
        return self


class _GepaDataRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_hash: StrictStr
    task_id: StrictStr
    prompt_inputs: dict[StrictStr, StrictStr]


class GepaDataRegistry:
    def __init__(
        self,
        *,
        loader_identity_hash: str,
        entries: tuple[GepaDataInstance, ...],
    ) -> None:
        require_full_hash(
            loader_identity_hash,
            field="data_loader_identity_hash",
        )
        if not entries:
            raise ValueError("GEPA data registry cannot be empty")
        if any(
            entry.loader_identity_hash != loader_identity_hash
            for entry in entries
        ):
            raise ValueError("GEPA data registry loader identity drifted")
        ids = tuple(entry.data_id for entry in entries)
        if len(ids) != len(set(ids)):
            raise ValueError("GEPA data registry identities must be unique")
        if tuple(entry.upstream_position for entry in entries) != tuple(
            range(len(entries))
        ):
            raise ValueError(
                "GEPA data registry positions must be contiguous and ordered"
            )
        self._loader_identity_hash = loader_identity_hash
        self._entries = entries
        self._by_id = {entry.data_id: entry for entry in entries}

    @classmethod
    def from_engine(
        cls,
        *,
        store: ObjectStore,
        engine: EvalEngine,
    ) -> GepaDataRegistry:
        task_ids = engine.sampling.task_hashes
        tasks = engine.sampling.tasks
        if len(task_ids) != len(tasks):
            raise ValueError(
                "GEPA engine task identities and tasks do not align"
            )
        entries: list[GepaDataInstance] = []
        for index, (task_hash, task) in enumerate(
            zip(task_ids, tasks, strict=True)
        ):
            if task.task_hash != task_hash:
                raise ValueError(
                    "GEPA engine task surface disagrees with task hash order"
                )
            prompt_inputs = task.prompt_inputs
            task_id = task.task_id
            prompt_input_items: list[tuple[str, str]] = []
            for name, value in prompt_inputs.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    raise ValueError(
                        "GEPA engine prompt inputs must be string mappings"
                    )
                prompt_input_items.append((name, value))
            ordered_prompt_inputs = dict(sorted(prompt_input_items))
            record = _GepaDataRecord(
                task_hash=task_hash,
                task_id=task_id,
                prompt_inputs=ordered_prompt_inputs,
            )
            ref, _ = store.put(
                GEPA_DATA_RECORD_SCHEMA,
                record.model_dump(mode="json"),
            )
            entries.append(
                GepaDataInstance(
                    upstream_position=index,
                    data_id=task_hash,
                    data_ref=TypedRef(
                        schema_name=ref.schema,
                        content_hash=ref.content_hash,
                    ),
                    loader_identity_hash=GEPA_DATA_LOADER_IDENTITY_HASH,
                )
            )
        return cls(
            loader_identity_hash=GEPA_DATA_LOADER_IDENTITY_HASH,
            entries=tuple(entries),
        )

    @property
    def loader_identity_hash(self) -> str:
        return self._loader_identity_hash

    @property
    def runtime_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_DATA_REGISTRY_SCHEMA,
            schema_version=GEPA_DATA_REGISTRY_SCHEMA_VERSION,
            payload={
                "loader_identity_hash": self._loader_identity_hash,
                "entries": [
                    {
                        "upstream_position": entry.upstream_position,
                        "data_id": entry.data_id,
                        "data_ref": entry.data_ref.model_dump(mode="json"),
                    }
                    for entry in self._entries
                ],
            },
        )

    @property
    def data_ids(self) -> tuple[str, ...]:
        return tuple(entry.data_id for entry in self._entries)

    @property
    def entries(self) -> tuple[GepaDataInstance, ...]:
        return self._entries

    def require_exact(self, item: GepaDataInstance) -> None:
        expected = self._by_id.get(item.data_id)
        if expected is None or item != expected:
            raise ValueError(
                "GEPA data instance conflicts with immutable data registry"
            )


class CanonicalGepaCandidateAssembler:
    def __init__(
        self,
        *,
        base_candidate: CandidateRef,
        fields: tuple[GepaCandidateFieldBinding, ...],
    ) -> None:
        if not fields:
            raise ValueError("GEPA candidate assembler needs a component")
        names = tuple(field.component_name for field in fields)
        payload_fields = tuple(field.candidate_field for field in fields)
        if len(names) != len(set(names)):
            raise ValueError("GEPA component field bindings must be unique")
        if len(payload_fields) != len(set(payload_fields)):
            raise ValueError("GEPA candidate payload fields must be unique")
        self._base_candidate = base_candidate
        self._fields = fields

    @property
    def runtime_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_CANDIDATE_ASSEMBLER_SCHEMA,
            schema_version=GEPA_CANDIDATE_ASSEMBLER_SCHEMA_VERSION,
            payload={
                "base_candidate": self._base_candidate.model_dump(mode="json"),
                "fields": [
                    field.model_dump(mode="json") for field in self._fields
                ],
            },
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(field.component_name for field in self._fields)

    def assemble(
        self,
        components: tuple[GepaCandidateComponent, ...],
    ) -> CandidateRef:
        if tuple(component.name for component in components) != (
            self.component_names
        ):
            raise ValueError(
                "GEPA candidate components conflict with native field binding"
            )
        values = {component.name: component.text for component in components}
        payload = dict(self._base_candidate.record.payload)
        for field in self._fields:
            payload[field.candidate_field] = values[field.component_name]
        candidate_hash = compute_identity_hash(
            schema="whetstone.gepa.native_candidate",
            schema_version=1,
            payload={
                "assembler_identity_hash": self.runtime_hash,
                "components": [
                    component.model_dump(mode="json")
                    for component in components
                ],
            },
        )
        return candidate_reference(
            Candidate(
                candidate_id=f"gepa-{candidate_hash[:24]}",
                base_ref=self._base_candidate.record.base_ref,
                payload=payload,
            )
        )



def _load_component_trace_index(
    store: ObjectStore,
    output_record: dict[str, object],
) -> dict[tuple[str, int], tuple[dict[str, object], ...]]:
    ref_raw = output_record.get("traces_ref")
    if not isinstance(ref_raw, dict):
        return {}
    trace_ref = TypedRef.model_validate(ref_raw)
    if trace_ref.schema_name != EVAL_TRACES_SCHEMA:
        raise ValueError("GEPA component traces ref has the wrong schema")
    trace_content = store.get(trace_ref.reference)
    if not isinstance(trace_content, dict):
        raise ValueError("GEPA component traces record must be an object")
    rows = trace_content.get("rows")
    if not isinstance(rows, list):
        raise ValueError("GEPA component traces must list rows")
    index: dict[tuple[str, int], tuple[dict[str, object], ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GEPA component trace row must be an object")
        task_hash = row.get("task_hash")
        seed_index = row.get("seed_index")
        trace_payload = row.get("trace")
        if (
            type(task_hash) is not str
            or type(seed_index) is not int
            or not isinstance(trace_payload, dict)
        ):
            raise ValueError("GEPA component trace row is malformed")
        steps = trace_payload.get("trace_steps")
        if not isinstance(steps, list):
            raise ValueError("GEPA component trace steps must be a list")
        index[(task_hash, seed_index)] = cast(
            tuple[dict[str, object], ...],
            tuple(step for step in steps if isinstance(step, dict)),
        )
    return index


class CanonicalGepaEvalAuthority:
    def __init__(
        self,
        *,
        store: ObjectStore,
        engine: EvalEngine,
        control: GepaControl,
        candidate_assembler: CanonicalGepaCandidateAssembler,
        data_registry: GepaDataRegistry,
        submission_projector: GepaSubmissionProjector | None = None,
    ) -> None:
        if engine.eval_config_ref != control.metric:
            raise ValueError(
                "GEPA evaluation engine conflicts with the control metric"
            )
        if engine.task_model_identity_hash != control.task_model_identity_hash:
            raise ValueError(
                "GEPA evaluation engine conflicts with the task-model route"
            )
        if (
            engine.execution_policy_identity_hash
            != control.evaluation_execution_policy_hash
        ):
            raise ValueError(
                "GEPA evaluation engine conflicts with execution policy"
            )
        if engine.reward_policy_identity_hash != control.reward_policy_hash:
            raise ValueError(
                "GEPA evaluation engine conflicts with reward policy"
            )

        if engine.sampling.num_seeds != 1:
            raise ValueError(
                "GEPA evaluation engine must use a single-repeat plan"
            )
        expected_data_ids = tuple(
            dict.fromkeys(
                (
                    *control.trainset_task_hashes,
                    *control.valset_task_hashes,
                )
            )
        )
        if (
            data_registry.data_ids != expected_data_ids
            or engine.sampling.task_hashes != expected_data_ids
        ):
            raise ValueError(
                "GEPA data registry/engine sampling conflicts with control"
            )
        canonical_registry = GepaDataRegistry.from_engine(
            store=store,
            engine=engine,
        )
        if data_registry.runtime_hash != canonical_registry.runtime_hash:
            raise ValueError(
                "GEPA data registry refs conflict with engine instances"
            )
        self._store = store
        self._engine = engine
        self._control = control
        self._candidate_assembler = candidate_assembler
        self._data_registry = data_registry
        self._submission_projector = submission_projector or (
            DefaultGepaSubmissionProjector(
                submission_result_field=control.submission_result_field,
            )
        )
        if (
            self._submission_projector.submission_result_field()
            != control.submission_result_field
        ):
            raise ValueError(
                "GEPA submission projector field conflicts with control"
            )
        authority_hash = compute_identity_hash(
            schema=GEPA_EVALUATION_AUTHORITY_SCHEMA,
            schema_version=GEPA_EVALUATION_AUTHORITY_SCHEMA_VERSION,
            payload={
                "control_identity_hash": control.identity_hash(),
                "candidate_assembler_identity_hash": (
                    candidate_assembler.runtime_hash
                ),
                "data_loader_identity_hash": (
                    data_registry.loader_identity_hash
                ),
                "data_registry_identity_hash": (data_registry.runtime_hash),
                "evaluation_config_hash": (control.metric.config_hash),
                "reward_policy_identity_hash": control.reward_policy_hash,
                "provider_route_identity_hash": (
                    control.task_model_identity_hash
                ),
                "execution_policy_identity_hash": (
                    control.evaluation_execution_policy_hash
                ),
                "response_parser_identity_hash": (
                    GEPA_EVALUATION_RESPONSE_PARSER_IDENTITY_HASH
                ),
                "submission_projector_identity_hash": (
                    self._submission_projector.projector_identity_hash()
                ),
            },
        )
        self._binding = GepaEvalAuthorityBinding(
            authority_identity_hash=authority_hash,
            evaluation_config_hash=control.metric.config_hash,
            reward_policy_identity_hash=control.reward_policy_hash,
            provider_route_identity_hash=control.task_model_identity_hash,
            execution_policy_identity_hash=(
                control.evaluation_execution_policy_hash
            ),
            prompt_adapter_identity_hash=(candidate_assembler.runtime_hash),
            response_parser_identity_hash=(
                GEPA_EVALUATION_RESPONSE_PARSER_IDENTITY_HASH
            ),
            data_registry_identity_hash=data_registry.runtime_hash,
            failure_score=control.failure_score,
            add_format_failure_as_feedback=(
                control.add_format_failure_as_feedback
            ),
            warn_on_score_mismatch=control.warn_on_score_mismatch,
            selection_seed=control.seed,
        )

    @property
    def binding(self) -> GepaEvalAuthorityBinding:
        return self._binding

    @property
    def runtime_hash(self) -> str:
        return self._binding.authority_identity_hash

    @property
    def control_identity_hash(self) -> str:
        return self._control.identity_hash()

    @property
    def component_names(self) -> tuple[str, ...]:
        return self._candidate_assembler.component_names

    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult:
        self._require_request_binding(request)
        for item in request.data:
            self._data_registry.require_exact(item)
            self._store.get(item.data_ref.reference)
        candidate = self._candidate_assembler.assemble(request.candidate)
        task_ids = tuple(item.data_id for item in request.data)
        subset_engine = self._engine.for_task_ids(task_ids)
        optim_eval_request = OptimEvalRequest(
            optim_run_id=request.slot.context.run_id,
            optim_step_index=request.slot.invocation_ordinal,
            eval_request=EvalRequest(
                request_id=(
                    f"{request.slot.context.run_id}:gepa:"
                    f"{request.identity_hash()}"
                ),
                candidate=candidate.record,
                metadata=metadata_with_purpose("gepa_metric"),
            ),
            expected_reward_policy_hash=self._control.reward_policy_hash,
        )
        resolution = EvalEngineService(
            store=self._store,
            engine=subset_engine,
        ).resolve_optim_eval_request(
            optim_eval_request,
            context=EvalExecutionContext(),
        )
        if resolution.outcome is not IntentOutcome.COMPLETED:
            return self._failed_result(request, resolution)
        return self._completed_result(request, candidate, resolution)

    def _require_request_binding(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> None:
        if request.authority != self._binding:
            raise ValueError(
                "GEPA evaluation request conflicts with runtime authority"
            )
        if (
            request.slot.context.control_identity_hash
            != self._control.identity_hash()
        ):
            raise ValueError("GEPA evaluation request control drifted")

    def _failed_result(
        self,
        request: GepaEvaluationEffectRequest,
        resolution: Any,
    ) -> GepaEvaluationEffectResult:
        evidence_ref = resolution.eval_result_ref
        refs = () if evidence_ref is None else (evidence_ref,)
        if evidence_ref is not None:
            failure_ref = evidence_ref
        else:
            record = {
                "request_hash": request.identity_hash(),
                "resolution": resolution.model_dump(mode="json"),
            }
            raw_ref, _ = self._store.put(
                GEPA_EVALUATION_REJECTION_SCHEMA,
                record,
            )
            failure_ref = TypedRef(
                schema_name=raw_ref.schema,
                content_hash=raw_ref.content_hash,
            )
        return GepaEvaluationEffectResult(
            request_hash=request.identity_hash(),
            rows=tuple(
                GepaEvaluationRow(
                    data=item,
                    output=None,
                    score=0.0,
                    evidence_refs=refs,
                    failure_ref=failure_ref,
                )
                for item in request.data
            ),
            logical_metric_calls=len(request.data),
        )

    def _completed_result(
        self,
        request: GepaEvaluationEffectRequest,
        candidate: CandidateRef,
        resolution: Any,
    ) -> GepaEvaluationEffectResult:
        evidence_ref = resolution.eval_result_ref
        if evidence_ref is None:
            raise ValueError(
                "GEPA requires one canonical evaluation evidence record"
            )
        if evidence_ref.schema_name != EVAL_EVIDENCE_SCHEMA:
            raise ValueError("GEPA requires canonical evaluation evidence")
        evidence = EvalEvidence.model_validate(
            self._store.get(evidence_ref.reference)
        )
        if (
            evidence.candidate != candidate
            or evidence.task_hashes
            != tuple(item.data_id for item in request.data)
            or evidence.num_seeds != 1
            or len(evidence.per_task_values) != len(request.data)
            or evidence.row_accounting.planned != len(request.data)
        ):
            raise ValueError(
                "canonical evaluation evidence conflicts with GEPA request"
            )
        if evidence.outputs_ref.schema_name != EVAL_OUTPUTS_SCHEMA:
            raise ValueError("GEPA evaluation outputs have the wrong schema")
        output_record = self._store.get(evidence.outputs_ref.reference)
        if not isinstance(output_record, dict):
            raise ValueError("GEPA evaluation outputs must be an object")
        raw_rows = output_record.get("outputs")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(
            request.data
        ):
            raise ValueError(
                "GEPA evaluation outputs do not align with requested data"
            )
        if output_record.get("candidate_id") != candidate.record.candidate_id:
            raise ValueError(
                "GEPA evaluation output record names another candidate"
            )
        common_refs = (
            evidence_ref,
            evidence.outputs_ref,
            evidence.aggregate_ref,
            *(
                (evidence.reward_ref.record_ref,)
                if evidence.reward_ref is not None
                else ()
            ),
        )
        trace_index = _load_component_trace_index(
            self._store,
            cast(dict[str, object], output_record),
        )
        rows = tuple(
            self._project_row(
                request=request,
                data=request.data[index],
                raw=raw,
                score=float(evidence.per_task_values[index]),
                evidence_refs=common_refs,
                candidate_id=candidate.record.candidate_id,
                trace_steps=trace_index.get(
                    (
                        cast(dict[str, object], raw).get("task_hash"),
                        cast(dict[str, object], raw).get("seed_index"),
                    ),
                    (),
                ),
            )
            for index, raw in enumerate(raw_rows)
        )
        return GepaEvaluationEffectResult(
            request_hash=request.identity_hash(),
            rows=rows,
            logical_metric_calls=len(rows),
        )

    def _project_row(
        self,
        *,
        request: GepaEvaluationEffectRequest,
        data: Any,
        raw: Any,
        score: float,
        evidence_refs: tuple[TypedRef, ...],
        candidate_id: str,
        trace_steps: tuple[dict[str, object], ...] = (),
    ) -> GepaEvaluationRow:
        if not isinstance(raw, dict):
            raise ValueError("GEPA evaluation output row must be an object")
        output_text = raw.get("output_text")
        failure_code = raw.get("failure_code")
        data_record = self._store.get(data.data_ref.reference)
        if (
            not isinstance(data_record, dict)
            or data_record.get("task_hash") != data.data_id
            or not isinstance(data_record.get("task_id"), str)
            or not isinstance(data_record.get("prompt_inputs"), dict)
        ):
            raise ValueError(
                "GEPA reflection input record conflicts with data identity"
            )
        if (
            raw.get("candidate_id") != candidate_id
            or raw.get("task_id") != data_record["task_id"]
            or raw.get("seed_index") != 0
        ):
            raise ValueError(
                "GEPA evaluation output row order/identity drifted"
            )
        failed = type(failure_code) is str and bool(failure_code)
        submission_field = self._submission_projector.submission_result_field()
        submission_result = raw.get(submission_field)
        if submission_result is None and submission_field != "submission_result":
            submission_result = raw.get("submission_result")
        prediction_failed = self._submission_projector.prediction_failed(
            failure_code=failure_code,
            submission=submission_result,
        )
        failure_ref = None
        row_evidence_refs = evidence_refs
        if failed:
            failure_record = {
                "data_id": data.data_id,
                "candidate_id": candidate_id,
                "failure_code": failure_code,
                "provider_error": raw.get("provider_error"),
                "finish_reason": raw.get("finish_reason"),
                "source_evidence_refs": [
                    ref.model_dump(mode="json") for ref in evidence_refs
                ],
            }
            failure_raw_ref, _ = self._store.put(
                GEPA_EVALUATION_ROW_FAILURE_SCHEMA,
                failure_record,
            )
            failure_ref = TypedRef(
                schema_name=failure_raw_ref.schema,
                content_hash=failure_raw_ref.content_hash,
            )
            row_evidence_refs = (*evidence_refs, failure_ref)
        component_records: dict[
            str,
            list[GepaComponentTraceProjection],
        ] = defaultdict(list)
        feedback = self._submission_projector.feedback_text(
            score=score, submission=submission_result
        )
        candidate_components = {
            component.name for component in request.candidate
        }
        for trace in trace_steps:
            component_id = trace.get("component_id")
            if not isinstance(component_id, str):
                raise ValueError("GEPA trace names an unknown component")
            if component_id not in candidate_components:
                raise ValueError("GEPA trace names an unknown component")
            component_records[component_id].append(
                GepaComponentTraceProjection(
                    inputs=trace.get("inputs", {}),
                    generated_outputs=trace.get("outputs", {}),
                    feedback=feedback,
                    feedback_score=score,
                    source_refs=row_evidence_refs,
                )
            )
        if failed and failure_code in set(self._control.format_failure_codes):
            data_inputs = data_record["prompt_inputs"]
            for component in request.candidate:
                component_records[component.name].append(
                    GepaComponentTraceProjection(
                        inputs=data_inputs,
                        generated_outputs={
                            "raw_response": output_text,
                            "failure_code": failure_code,
                        },
                        feedback=(
                            "Your output failed native format validation: "
                            f"{failure_code}."
                        ),
                        format_failure=True,
                        source_refs=row_evidence_refs,
                    )
                )
        generated_outputs = {
            "output_text": output_text,
            "failure_code": failure_code,
        }
        projected = self._submission_projector.test_results(
            submission=submission_result
        )
        if projected is not None:
            generated_outputs.update(projected)
        trajectory = (
            GepaTrajectoryProjection(
                data_id=data.data_id,
                inputs=data_record["prompt_inputs"],
                generated_outputs=generated_outputs,
                feedback=feedback,
                component_records={
                    name: tuple(records)
                    for name, records in component_records.items()
                },
                prediction_failed=prediction_failed,
                module_score=score,
                source_refs=row_evidence_refs,
            )
            if request.capture_traces
            else None
        )
        return GepaEvaluationRow(
            data=data,
            output=output_text,
            score=score,
            trajectory=trajectory,
            evidence_refs=row_evidence_refs,
            failure_ref=failure_ref,
        )


class CanonicalGepaProposalAuthority:
    def __init__(
        self,
        *,
        store: ObjectStore,
        control: GepaControl,
        prompt_services: GepaPromptServices,
        transport: ProposerTransport,
        proposal_executor: DurableProposalExecutor,
    ) -> None:
        require_canonical_proposal_executor(
            proposal_executor,
            algorithm="GEPA",
            purpose="paid reflection call",
        )
        if (
            prompt_services.binding.identity_hash()
            != control.prompt_binding_identity_hash
            or prompt_services.descriptor.identity_hash()
            != control.prompt_format_identity_hash
        ):
            raise ValueError("GEPA prompt services conflict with control")
        if (
            transport.execution_policy_hash
            != control.proposal_execution_policy_hash
        ):
            raise ValueError(
                "GEPA proposer transport conflicts with execution policy"
            )
        if (
            transport.prompt_adapter_identity_hash
            != control.proposal_prompt_adapter_identity_hash
            or proposal_executor.policy_identity_hash
            != control.proposal_durability_policy_identity_hash
        ):
            raise ValueError(
                "GEPA proposer transport conflicts with prompt/durability "
                "policy"
            )
        self._store = store
        self._control = control
        self._prompt_services = prompt_services
        self._transport = transport
        self._proposal_executor = proposal_executor
        transport_hash = compute_identity_hash(
            schema="whetstone.gepa.proposer_transport",
            schema_version=1,
            payload={
                "execution_policy_identity_hash": (
                    transport.execution_policy_hash
                ),
                "prompt_adapter_identity_hash": (
                    transport.prompt_adapter_identity_hash
                ),
                "durability_policy_identity_hash": (
                    proposal_executor.policy_identity_hash
                ),
            },
        )
        authority_hash = compute_identity_hash(
            schema=GEPA_PROPOSAL_AUTHORITY_SCHEMA,
            schema_version=GEPA_PROPOSAL_AUTHORITY_SCHEMA_VERSION,
            payload={
                "control_identity_hash": control.identity_hash(),
                "transport_identity_hash": transport_hash,
                "prompt_binding_identity_hash": (
                    control.prompt_binding_identity_hash
                ),
                "proposer_config_hash": (
                    control.reflection_model.identity_hash()
                ),
            },
        )
        self._binding = GepaProposalAuthorityBinding(
            authority_identity_hash=authority_hash,
            proposer_transport_identity_hash=transport_hash,
            prompt_binding_identity_hash=control.prompt_binding_identity_hash,
            execution_policy_identity_hash=(
                control.proposal_execution_policy_hash
            ),
            prompt_adapter_identity_hash=(
                control.proposal_prompt_adapter_identity_hash
            ),
            durability_policy_identity_hash=(
                control.proposal_durability_policy_identity_hash
            ),
            proposer_config=control.reflection_model,
        )

    @property
    def binding(self) -> GepaProposalAuthorityBinding:
        return self._binding

    @property
    def runtime_hash(self) -> str:
        return self._binding.authority_identity_hash

    @property
    def control_identity_hash(self) -> str:
        return self._control.identity_hash()

    @property
    def transport(self) -> ProposerTransport:
        return self._transport

    @property
    def proposal_executor(self) -> DurableProposalExecutor:
        return self._proposal_executor

    def _reflection_base_candidate(
        self,
        request: GepaProposalEffectRequest,
        current: str,
    ) -> CandidateRef:

        request_hash = request.identity_hash()
        field = self._control.mutation_field
        return candidate_reference(
            Candidate(
                candidate_id=f"gepa-reflection-{request_hash[:24]}",
                base_ref=typed_ref_for_record(
                    GEPA_REFLECTION_BASE_SCHEMA,
                    {
                        "component_name": request.component_name,
                        "request_hash": request_hash,
                    },
                ),
                payload={field: current},
            )
        )

    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult:
        if (
            request.authority != self._binding
            or request.slot.context.control_identity_hash
            != self._control.identity_hash()
        ):
            raise ValueError(
                "GEPA proposal request conflicts with runtime authority"
            )
        current = next(
            component.text
            for component in request.candidate
            if component.name == request.component_name
        )
        generic = ProposalRequest(
            proposal_mode="gepa_reflection",
            request_ordinal=request.slot.invocation_ordinal,
            proposal_authority_identity_hash=(
                request.authority.authority_identity_hash
            ),
            base_candidate=self._reflection_base_candidate(request, current),
            mutation_field=self._control.mutation_field,
            context={
                "proposal_prompt": request.rendered_prompt.text,
                "proposal_messages": (
                    list(request.rendered_prompt.messages)
                    if request.rendered_prompt.messages is not None
                    else None
                ),
                "component_name": request.component_name,
                "components_to_update": list(request.components_to_update),
                "prompt_binding_identity_hash": (
                    self._binding.prompt_binding_identity_hash
                ),
            },
        )
        drafts = self._proposal_executor.execute(
            config=self._binding.proposer_config,
            request=generic,
            transport=self._transport,
            count=1,
        )
        if len(drafts) != 1:
            raise ValueError(
                "GEPA reflection proposer must return exactly one draft"
            )
        draft = drafts[0]
        request_evidence = draft.request_evidence.to_json()
        response_evidence = draft.response_evidence.to_json()
        usage = draft.usage.to_json()
        attempt_refs = self._persist_attempt_evidence(response_evidence)
        if draft.failed:
            return GepaProposalEffectResult(
                request_hash=request.identity_hash(),
                request_evidence=request_evidence,
                response_evidence=response_evidence,
                provider_attempt_refs=attempt_refs,
                usage=usage,
                cost=draft.cost,
                failed=True,
                failure_detail=(
                    draft.terminal_failure.message
                    if draft.terminal_failure is not None
                    else None
                ),
            )
        raw = draft.template
        try:
            parsed = self._prompt_services.parse_replacement(
                request.component_name,
                raw,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return GepaProposalEffectResult(
                request_hash=request.identity_hash(),
                raw_response=raw,
                request_evidence=request_evidence,
                response_evidence=response_evidence,
                provider_attempt_refs=attempt_refs,
                usage=usage,
                cost=draft.cost,
                failed=True,
                failure_detail=str(exc) or type(exc).__name__,
            )
        return GepaProposalEffectResult(
            request_hash=request.identity_hash(),
            raw_response=raw,
            parsed_components=(
                GepaCandidateComponent(
                    name=request.component_name,
                    text=parsed,
                ),
            ),
            request_evidence=request_evidence,
            response_evidence=response_evidence,
            provider_attempt_refs=attempt_refs,
            usage=usage,
            cost=draft.cost,
        )

    def _persist_attempt_evidence(
        self,
        response_evidence: dict[str, Any],
    ) -> tuple[TypedRef, ...]:

        result = response_evidence.get("provider_call_result")
        if not isinstance(result, dict):
            if not response_evidence:
                return ()
            ref, _ = self._store.put(
                GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA,
                {
                    "boundary": GEPA_WHOLE_CALL_EVIDENCE_BOUNDARY,
                    "response_evidence": response_evidence,
                },
            )
            return (
                TypedRef(
                    schema_name=ref.schema,
                    content_hash=ref.content_hash,
                ),
            )
        attempts = result.get("attempts")
        if not isinstance(attempts, list):
            return ()
        refs: list[TypedRef] = []
        for attempt in attempts:
            ref, _ = self._store.put(
                GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA,
                attempt,
            )
            refs.append(
                TypedRef(
                    schema_name=ref.schema,
                    content_hash=ref.content_hash,
                )
            )
        return tuple(refs)


__all__ = [
    "GEPA_CANDIDATE_ASSEMBLER_SCHEMA",
    "GEPA_CANDIDATE_ASSEMBLER_SCHEMA_VERSION",
    "GEPA_DATA_LOADER_IDENTITY_HASH",
    "GEPA_DATA_RECORD_SCHEMA",
    "GEPA_DATA_REGISTRY_SCHEMA",
    "GEPA_DATA_REGISTRY_SCHEMA_VERSION",
    "GEPA_EVALUATION_AUTHORITY_SCHEMA",
    "GEPA_EVALUATION_AUTHORITY_SCHEMA_VERSION",
    "GEPA_EVALUATION_RESPONSE_PARSER_IDENTITY_HASH",
    "GEPA_PROPOSAL_ATTEMPT_EVIDENCE_SCHEMA",
    "GEPA_PROPOSAL_AUTHORITY_SCHEMA",
    "GEPA_PROPOSAL_AUTHORITY_SCHEMA_VERSION",
    "GEPA_WHOLE_CALL_EVIDENCE_BOUNDARY",
    "CanonicalGepaCandidateAssembler",
    "CanonicalGepaEvalAuthority",
    "CanonicalGepaProposalAuthority",
    "GepaCandidateFieldBinding",
    "GepaDataRegistry",
]
