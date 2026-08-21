from __future__ import annotations

from typing import Any, Literal, Protocol

from dr_store import BindingConflictError, ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
    reject_non_json,
    require_full_hash,
)
from whetstone.optim.gepa.prompts import GepaRenderedPrompt
from whetstone.optim.proposal.proposer import ProposerConfig

GEPA_EFFECT_CONTEXT_SCHEMA = "whetstone.gepa.effect_context"
GEPA_EFFECT_CONTEXT_SCHEMA_VERSION = 1
GEPA_EVALUATION_EFFECT_SCHEMA = "whetstone.gepa.evaluation_effect"
GEPA_EVALUATION_EFFECT_SCHEMA_VERSION = 1
GEPA_PROPOSAL_EFFECT_SCHEMA = "whetstone.gepa.proposal_effect"
GEPA_PROPOSAL_EFFECT_SCHEMA_VERSION = 1
GEPA_EFFECT_SLOT_SCHEMA = "whetstone.gepa.effect_slot"
GEPA_EFFECT_SLOT_SCHEMA_VERSION = 1
GEPA_EVALUATION_REQUEST_RECORD_SCHEMA = (
    "whetstone.gepa.evaluation_effect_request"
)
GEPA_EVALUATION_RESULT_RECORD_SCHEMA = (
    "whetstone.gepa.evaluation_effect_result"
)
GEPA_PROPOSAL_REQUEST_RECORD_SCHEMA = "whetstone.gepa.proposal_effect_request"
GEPA_PROPOSAL_RESULT_RECORD_SCHEMA = "whetstone.gepa.proposal_effect_result"
GEPA_EFFECT_TRANSCRIPT_SCHEMA = "whetstone.gepa.effect_transcript"
GEPA_EFFECT_TRANSCRIPT_SCHEMA_VERSION = 1


class GepaEffectConflictError(RuntimeError):
    pass


class GepaEffectContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    control_identity_hash: StrictStr
    source_manifest_identity_hash: StrictStr
    adapter_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> GepaEffectContext:
        if not self.run_id:
            raise ValueError("GEPA effect run_id must be non-empty")
        for field_name in (
            "control_identity_hash",
            "source_manifest_identity_hash",
            "adapter_identity_hash",
        ):
            require_full_hash(getattr(self, field_name), field=field_name)
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_EFFECT_CONTEXT_SCHEMA,
            schema_version=GEPA_EFFECT_CONTEXT_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class GepaEffectSlot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context: GepaEffectContext
    invocation_ordinal: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> GepaEffectSlot:
        if self.invocation_ordinal < 0:
            raise ValueError(
                "GEPA effect invocation_ordinal cannot be negative"
            )
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_EFFECT_SLOT_SCHEMA,
            schema_version=GEPA_EFFECT_SLOT_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class GepaCandidateComponent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    text: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> GepaCandidateComponent:
        if not self.name:
            raise ValueError("GEPA candidate component name must be non-empty")
        return self


class GepaDataInstance(BaseModel):
    """One GEPA training instance, keyed by the engine-resolvable task id.

    ``data_id`` is the canonical task identity the evaluation engine resolves
    through ``EvalEngine.for_task_ids``. ``task_hash`` carries the content
    identity of the same task so identity checks stay hash-based without
    forcing a translation at the engine seam.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    upstream_position: StrictInt
    data_id: StrictStr
    task_hash: StrictStr
    data_ref: TypedRef
    loader_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> GepaDataInstance:
        if self.upstream_position < 0:
            raise ValueError("GEPA upstream_position cannot be negative")
        if not self.data_id:
            raise ValueError("GEPA data_id must be non-empty")
        require_full_hash(self.task_hash, field="task_hash")
        require_full_hash(
            self.loader_identity_hash,
            field="loader_identity_hash",
        )
        return self


class GepaEvalAuthorityBinding(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    authority_identity_hash: StrictStr
    evaluation_config_hash: StrictStr
    reward_policy_identity_hash: StrictStr
    provider_route_identity_hash: StrictStr
    execution_policy_identity_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    response_parser_identity_hash: StrictStr
    data_registry_identity_hash: StrictStr
    failure_score: float = 0.0
    add_format_failure_as_feedback: StrictBool = False
    warn_on_score_mismatch: StrictBool = True
    selection_seed: StrictInt = 0

    @model_validator(mode="after")
    def _validate(self) -> GepaEvalAuthorityBinding:
        for field_name in (
            "authority_identity_hash",
            "evaluation_config_hash",
            "reward_policy_identity_hash",
            "provider_route_identity_hash",
            "execution_policy_identity_hash",
            "prompt_adapter_identity_hash",
            "response_parser_identity_hash",
            "data_registry_identity_hash",
        ):
            require_full_hash(getattr(self, field_name), field=field_name)
        return self


class GepaProposalAuthorityBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    authority_identity_hash: StrictStr
    proposer_transport_identity_hash: StrictStr
    prompt_binding_identity_hash: StrictStr
    execution_policy_identity_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    durability_policy_identity_hash: StrictStr
    proposer_config: ProposerConfig

    @model_validator(mode="after")
    def _validate(self) -> GepaProposalAuthorityBinding:
        for field_name in (
            "authority_identity_hash",
            "proposer_transport_identity_hash",
            "prompt_binding_identity_hash",
            "execution_policy_identity_hash",
            "prompt_adapter_identity_hash",
            "durability_policy_identity_hash",
        ):
            require_full_hash(getattr(self, field_name), field=field_name)
        return self


class GepaEvaluationEffectRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: GepaEffectSlot
    candidate: tuple[GepaCandidateComponent, ...]
    upstream_candidate_index: StrictInt | None = None
    data: tuple[GepaDataInstance, ...]
    capture_traces: StrictBool = False
    authority: GepaEvalAuthorityBinding

    @model_validator(mode="after")
    def _validate(self) -> GepaEvaluationEffectRequest:
        if not self.candidate:
            raise ValueError("GEPA evaluation candidate cannot be empty")
        names = [item.name for item in self.candidate]
        if len(names) != len(set(names)):
            raise ValueError("GEPA candidate component names must be unique")
        if self.upstream_candidate_index is not None:
            if self.upstream_candidate_index < 0:
                raise ValueError(
                    "GEPA upstream_candidate_index cannot be negative"
                )
        if not self.data:
            raise ValueError("GEPA evaluation batch cannot be empty")
        positions = [item.upstream_position for item in self.data]
        data_ids = [item.data_id for item in self.data]
        if len(positions) != len(set(positions)):
            raise ValueError("GEPA evaluation positions must be unique")
        if len(data_ids) != len(set(data_ids)):
            raise ValueError("GEPA evaluation data_ids must be unique")
        loaders = {item.loader_identity_hash for item in self.data}
        if len(loaders) != 1:
            raise ValueError(
                "GEPA evaluation batch must use one loader identity"
            )
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_EVALUATION_EFFECT_SCHEMA,
            schema_version=GEPA_EVALUATION_EFFECT_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class GepaTrajectoryProjection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_id: StrictStr
    inputs: Any
    generated_outputs: Any
    feedback: StrictStr
    component_records: dict[
        StrictStr,
        tuple[GepaComponentTraceProjection, ...],
    ] = Field(default_factory=dict)
    prediction_failed: StrictBool = False
    module_score: float | None = None
    source_refs: tuple[TypedRef, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> GepaTrajectoryProjection:
        if not self.data_id:
            raise ValueError("GEPA trajectory data_id must be non-empty")
        reject_non_json(self.inputs, field="GEPA trajectory inputs")
        reject_non_json(
            self.generated_outputs,
            field="GEPA trajectory generated_outputs",
        )
        reject_non_json(
            {
                name: [record.model_dump(mode="json") for record in records]
                for name, records in self.component_records.items()
            },
            field="GEPA trajectory component records",
        )
        return self

    def reflective_record(self, component_name: str) -> dict[str, Any]:
        selected = self.component_records.get(component_name, ())
        if selected:
            return selected[0].reflective_record()
        return {
            "Inputs": self.inputs,
            "Generated Outputs": self.generated_outputs,
            "Feedback": self.feedback,
        }


class GepaComponentTraceProjection(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    inputs: Any
    generated_outputs: Any
    feedback: StrictStr
    format_failure: StrictBool = False
    feedback_score: float | None = None
    source_refs: tuple[TypedRef, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> GepaComponentTraceProjection:
        reject_non_json(self.inputs, field="GEPA component trace inputs")
        reject_non_json(
            self.generated_outputs,
            field="GEPA component trace generated_outputs",
        )
        return self

    def reflective_record(self) -> dict[str, Any]:
        return {
            "Inputs": self.inputs,
            "Generated Outputs": self.generated_outputs,
            "Feedback": self.feedback,
        }


class GepaScoreMismatchEvidence(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    data_id: StrictStr
    component_name: StrictStr
    feedback_score: float
    module_score: float
    source_refs: tuple[TypedRef, ...] = ()


GepaTrajectoryProjection.model_rebuild()


class GepaEvaluationRow(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    data: GepaDataInstance
    output: Any
    score: StrictFloat
    objective_scores: dict[StrictStr, StrictFloat] | None = None
    trajectory: GepaTrajectoryProjection | None = None
    evidence_refs: tuple[TypedRef, ...] = ()
    provider_attempt_refs: tuple[TypedRef, ...] = ()
    failure_ref: TypedRef | None = None

    @model_validator(mode="after")
    def _validate(self) -> GepaEvaluationRow:
        reject_non_json(self.output, field="GEPA evaluation output")
        if self.failure_ref is None and not self.evidence_refs:
            raise ValueError(
                "successful GEPA evaluation row requires canonical evidence"
            )
        if self.failure_ref is not None and self.score != 0.0:
            raise ValueError(
                "failed GEPA evaluation row score must be zero before the "
                "bound failure_score projection"
            )
        if self.objective_scores is not None:
            if not self.objective_scores:
                raise ValueError("GEPA objective_scores cannot be empty")
            if any(not name for name in self.objective_scores):
                raise ValueError(
                    "GEPA objective score names must be non-empty"
                )
        return self


class GepaEvaluationEffectResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_hash: StrictStr
    rows: tuple[GepaEvaluationRow, ...]
    logical_metric_calls: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> GepaEvaluationEffectResult:
        require_full_hash(
            self.request_hash,
            field="request_hash",
        )
        if not self.rows:
            raise ValueError("GEPA evaluation result cannot be empty")
        if self.logical_metric_calls != len(self.rows):
            raise ValueError(
                "GEPA logical_metric_calls must equal returned row count"
            )
        return self


class GepaProposalEffectRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: GepaEffectSlot
    candidate: tuple[GepaCandidateComponent, ...]
    upstream_candidate_index: StrictInt | None = None
    components_to_update: tuple[StrictStr, ...]
    component_name: StrictStr
    rendered_prompt: GepaRenderedPrompt
    authority: GepaProposalAuthorityBinding

    @model_validator(mode="after")
    def _validate(self) -> GepaProposalEffectRequest:
        names = [item.name for item in self.candidate]
        if not names or len(names) != len(set(names)):
            raise ValueError(
                "GEPA proposal candidate components must be non-empty and "
                "unique"
            )
        if self.upstream_candidate_index is not None:
            if self.upstream_candidate_index < 0:
                raise ValueError(
                    "GEPA upstream_candidate_index cannot be negative"
                )
        if (
            not self.component_name
            or self.component_name not in self.components_to_update
            or self.component_name not in names
        ):
            raise ValueError(
                "GEPA proposal component must be selected and present"
            )
        if len(self.components_to_update) != len(
            set(self.components_to_update)
        ):
            raise ValueError(
                "GEPA components_to_update must preserve unique order"
            )
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_PROPOSAL_EFFECT_SCHEMA,
            schema_version=GEPA_PROPOSAL_EFFECT_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class GepaProposalEffectResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    request_hash: StrictStr
    raw_response: StrictStr = ""
    parsed_components: tuple[GepaCandidateComponent, ...] = ()
    request_evidence: dict[str, Any] = Field(default_factory=dict)
    response_evidence: dict[str, Any] = Field(default_factory=dict)
    provider_attempt_refs: tuple[TypedRef, ...] = ()
    usage: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    failed: StrictBool = False
    failure_detail: StrictStr | None = None
    #: The provider answered, but the reflection parser or the component
    #: format contract rejected its response. Retryable, unlike a transport
    #: or provider failure.
    rejected_by_parser: StrictBool = False

    @model_validator(mode="after")
    def _validate(self) -> GepaProposalEffectResult:
        require_full_hash(
            self.request_hash,
            field="request_hash",
        )
        reject_non_json(
            self.request_evidence,
            field="GEPA proposal request evidence",
        )
        reject_non_json(
            self.response_evidence,
            field="GEPA proposal response evidence",
        )
        reject_non_json(self.usage, field="GEPA proposal usage")
        if self.failed:
            if self.parsed_components or not self.failure_detail:
                raise ValueError(
                    "failed GEPA proposal requires detail and no parsed "
                    "components"
                )
        elif self.rejected_by_parser:
            raise ValueError(
                "only a failed GEPA proposal may be rejected by the parser"
            )
        elif (
            not self.raw_response
            or not self.parsed_components
            or self.failure_detail is not None
        ):
            raise ValueError(
                "successful GEPA proposal requires raw and parsed output"
            )
        elif (
            not self.request_evidence
            or not self.response_evidence
            or not self.provider_attempt_refs
        ):
            raise ValueError(
                "successful GEPA proposal requires request, response, and "
                "provider-attempt evidence"
            )
        names = [item.name for item in self.parsed_components]
        if len(names) != len(set(names)):
            raise ValueError(
                "GEPA parsed proposal component names must be unique"
            )
        return self


class GepaEffectBroker(Protocol):
    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult: ...

    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult: ...


class GepaEvaluationEffectAuthority(Protocol):
    @property
    def runtime_hash(self) -> str: ...

    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult: ...


class GepaProposalEffectAuthority(Protocol):
    @property
    def runtime_hash(self) -> str: ...

    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult: ...


GepaEffectRequest = GepaEvaluationEffectRequest | GepaProposalEffectRequest
GepaEffectResult = GepaEvaluationEffectResult | GepaProposalEffectResult


class GepaSkippedMutation(BaseModel):
    """One reflection mutation dropped because its response was rejected.

    Recorded so a rejected reflection is visible in step evidence rather
    than silently narrowing the search.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_name: StrictStr
    attempt_ordinal: StrictInt
    rejection_detail: StrictStr
    raw_response: StrictStr = ""
    provider_attempt_refs: tuple[TypedRef, ...] = ()
    #: True when this rejection consumed the last permitted attempt, so the
    #: component was left unchanged.
    exhausted: StrictBool = False

    @model_validator(mode="after")
    def _validate(self) -> GepaSkippedMutation:
        if not self.component_name:
            raise ValueError("skipped mutation component_name is required")
        if not self.rejection_detail:
            raise ValueError("skipped mutation rejection_detail is required")
        if self.attempt_ordinal < 0:
            raise ValueError("skipped mutation attempt_ordinal cannot be negative")
        return self


class GepaEffectTranscriptEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    invocation_ordinal: StrictInt
    effect_kind: Literal["evaluate", "propose"]
    semantic_candidate_identity_hash: StrictStr
    upstream_candidate_index: StrictInt | None = None
    request_ref: TypedRef
    result_ref: TypedRef
    data_ids: tuple[StrictStr, ...] = ()
    component_names: tuple[StrictStr, ...]
    evidence_refs: tuple[TypedRef, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> GepaEffectTranscriptEntry:
        require_full_hash(
            self.semantic_candidate_identity_hash,
            field="semantic_candidate_identity_hash",
        )
        if self.invocation_ordinal < 0:
            raise ValueError("GEPA transcript ordinal cannot be negative")
        if not self.component_names:
            raise ValueError("GEPA transcript entry requires components")
        return self


class GepaEffectTranscript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context: GepaEffectContext
    entries: tuple[GepaEffectTranscriptEntry, ...]
    score_mismatch_evidence: tuple[GepaScoreMismatchEvidence, ...] = ()
    #: Reflection mutations rejected by the parser or format contract.
    #: Present so a dropped mutation is visible, never silent.
    skipped_mutations: tuple[GepaSkippedMutation, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> GepaEffectTranscript:
        ordinals = tuple(entry.invocation_ordinal for entry in self.entries)
        if ordinals != tuple(range(len(self.entries))):
            raise ValueError(
                "GEPA transcript entries must be contiguous from ordinal zero"
            )
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_EFFECT_TRANSCRIPT_SCHEMA,
            schema_version=GEPA_EFFECT_TRANSCRIPT_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class GepaEffectRecorder:
    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    @staticmethod
    def _slot_key(request: GepaEffectRequest) -> str:
        return f"whetstone.gepa.effect_slot:{request.slot.identity_hash()}"

    @staticmethod
    def _result_key(request: GepaEffectRequest) -> str:
        return f"whetstone.gepa.effect_result:{request.identity_hash()}"

    def record_request(self, request: GepaEffectRequest) -> None:
        schema = (
            GEPA_EVALUATION_REQUEST_RECORD_SCHEMA
            if isinstance(request, GepaEvaluationEffectRequest)
            else GEPA_PROPOSAL_REQUEST_RECORD_SCHEMA
        )
        reference, _ = self._store.put(
            schema,
            request.model_dump(mode="json"),
        )
        key = self._slot_key(request)
        try:
            self._store.bind(key, reference)
        except BindingConflictError:
            pass
        bound = self._store.resolve(key)
        if bound != reference:
            prior = (
                f"{bound.schema}:{bound.content_hash}"
                if bound is not None
                else "missing"
            )
            raise GepaEffectConflictError(
                "GEPA replay changed the semantic effect at invocation "
                f"ordinal {request.slot.invocation_ordinal}; prior={prior}, "
                f"requested={reference.schema}:{reference.content_hash}"
            )

    def load_evaluation_result(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult | None:
        bound = self._store.resolve(self._result_key(request))
        if bound is None:
            return None
        if bound.schema != GEPA_EVALUATION_RESULT_RECORD_SCHEMA:
            raise GepaEffectConflictError(
                "GEPA evaluation result binding names another effect kind"
            )
        result = GepaEvaluationEffectResult.model_validate(
            self._store.get(bound)
        )
        if result.request_hash != request.identity_hash():
            raise GepaEffectConflictError(
                "GEPA evaluation result belongs to another request"
            )
        return result

    def load_proposal_result(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult | None:
        bound = self._store.resolve(self._result_key(request))
        if bound is None:
            return None
        if bound.schema != GEPA_PROPOSAL_RESULT_RECORD_SCHEMA:
            raise GepaEffectConflictError(
                "GEPA proposal result binding names another effect kind"
            )
        result = GepaProposalEffectResult.model_validate(
            self._store.get(bound)
        )
        if result.request_hash != request.identity_hash():
            raise GepaEffectConflictError(
                "GEPA proposal result belongs to another request"
            )
        return result

    def record_evaluation_result(
        self,
        request: GepaEvaluationEffectRequest,
        result: GepaEvaluationEffectResult,
    ) -> GepaEvaluationEffectResult:
        if result.request_hash != request.identity_hash():
            raise ValueError(
                "GEPA evaluation authority returned another request's result"
            )
        reference, _ = self._store.put(
            GEPA_EVALUATION_RESULT_RECORD_SCHEMA,
            result.model_dump(mode="json"),
        )
        key = self._result_key(request)
        try:
            self._store.bind(key, reference)
        except BindingConflictError:
            pass
        loaded = self.load_evaluation_result(request)
        if loaded is None:
            raise AssertionError("recorded GEPA evaluation result is missing")
        if loaded != result:
            raise GepaEffectConflictError(
                "GEPA evaluation result binding conflicts with completed "
                "evidence"
            )
        return loaded

    def record_proposal_result(
        self,
        request: GepaProposalEffectRequest,
        result: GepaProposalEffectResult,
    ) -> GepaProposalEffectResult:
        if result.request_hash != request.identity_hash():
            raise ValueError(
                "GEPA proposal authority returned another request's result"
            )
        reference, _ = self._store.put(
            GEPA_PROPOSAL_RESULT_RECORD_SCHEMA,
            result.model_dump(mode="json"),
        )
        key = self._result_key(request)
        try:
            self._store.bind(key, reference)
        except BindingConflictError:
            pass
        loaded = self.load_proposal_result(request)
        if loaded is None:
            raise AssertionError("recorded GEPA proposal result is missing")
        if loaded != result:
            raise GepaEffectConflictError(
                "GEPA proposal result binding conflicts with completed "
                "evidence"
            )
        return loaded

    @staticmethod
    def _typed_ref(reference: Any) -> TypedRef:
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def build_transcript(
        self,
        *,
        context: GepaEffectContext,
        effect_count: int,
        score_mismatch_evidence: tuple[
            GepaScoreMismatchEvidence,
            ...,
        ] = (),
        skipped_mutations: tuple[GepaSkippedMutation, ...] = (),
    ) -> GepaEffectTranscript:
        if effect_count < 0:
            raise ValueError("GEPA effect_count cannot be negative")
        entries: list[GepaEffectTranscriptEntry] = []
        for ordinal in range(effect_count):
            slot = GepaEffectSlot(
                context=context,
                invocation_ordinal=ordinal,
            )
            request_ref = self._store.resolve(
                f"whetstone.gepa.effect_slot:{slot.identity_hash()}"
            )
            if request_ref is None:
                raise ValueError(
                    f"GEPA transcript is missing request ordinal {ordinal}"
                )
            raw_request = self._store.get(request_ref)
            if request_ref.schema == GEPA_EVALUATION_REQUEST_RECORD_SCHEMA:
                request: GepaEffectRequest = (
                    GepaEvaluationEffectRequest.model_validate(raw_request)
                )
                result = self.load_evaluation_result(request)
                kind: Literal["evaluate", "propose"] = "evaluate"
                data_ids = tuple(item.data_id for item in request.data)
                component_names = tuple(
                    component.name for component in request.candidate
                )
            elif request_ref.schema == GEPA_PROPOSAL_REQUEST_RECORD_SCHEMA:
                request = GepaProposalEffectRequest.model_validate(raw_request)
                result = self.load_proposal_result(request)
                kind = "propose"
                data_ids = ()
                component_names = (request.component_name,)
            else:
                raise ValueError(
                    "GEPA transcript request has an unknown record schema"
                )
            if request.slot != slot:
                raise ValueError("GEPA transcript request slot drifted")
            if result is None:
                raise ValueError(
                    f"GEPA transcript is missing result ordinal {ordinal}"
                )
            result_ref = self._store.resolve(self._result_key(request))
            if result_ref is None:
                raise AssertionError("loaded GEPA result has no binding")
            evidence_refs = self._effect_evidence_refs(result)
            candidate_hash = compute_identity_hash(
                schema="whetstone.gepa.semantic_candidate",
                schema_version=1,
                payload=[
                    component.model_dump(mode="json")
                    for component in request.candidate
                ],
            )
            entries.append(
                GepaEffectTranscriptEntry(
                    invocation_ordinal=ordinal,
                    effect_kind=kind,
                    semantic_candidate_identity_hash=candidate_hash,
                    upstream_candidate_index=(
                        request.upstream_candidate_index
                    ),
                    request_ref=self._typed_ref(request_ref),
                    result_ref=self._typed_ref(result_ref),
                    data_ids=data_ids,
                    component_names=component_names,
                    evidence_refs=evidence_refs,
                )
            )
        return GepaEffectTranscript(
            context=context,
            entries=tuple(entries),
            score_mismatch_evidence=score_mismatch_evidence,
            skipped_mutations=skipped_mutations,
        )

    @staticmethod
    def _effect_evidence_refs(
        result: GepaEffectResult,
    ) -> tuple[TypedRef, ...]:
        refs: list[TypedRef] = []
        if isinstance(result, GepaEvaluationEffectResult):
            for row in result.rows:
                refs.extend(row.evidence_refs)
                refs.extend(row.provider_attempt_refs)
                if row.failure_ref is not None:
                    refs.append(row.failure_ref)
                if row.trajectory is not None:
                    refs.extend(row.trajectory.source_refs)
                    for traces in row.trajectory.component_records.values():
                        for trace in traces:
                            refs.extend(trace.source_refs)
        else:
            refs.extend(result.provider_attempt_refs)
        unique: list[TypedRef] = []
        seen: set[tuple[str, str]] = set()
        for ref in refs:
            key = (ref.schema_name, ref.content_hash)
            if key not in seen:
                seen.add(key)
                unique.append(ref)
        return tuple(unique)

    def persist_transcript(
        self,
        transcript: GepaEffectTranscript,
    ) -> TypedRef:
        reference, _ = self._store.put(
            GEPA_EFFECT_TRANSCRIPT_SCHEMA,
            transcript.model_dump(mode="json"),
        )
        key = (
            "whetstone.gepa.effect_transcript:"
            f"{transcript.context.identity_hash()}"
        )
        try:
            self._store.bind(key, reference)
        except BindingConflictError:
            pass
        bound = self._store.resolve(key)
        if bound != reference:
            raise GepaEffectConflictError(
                "GEPA run already has another terminal effect transcript"
            )
        return self._typed_ref(reference)


__all__ = [
    "GEPA_EFFECT_CONTEXT_SCHEMA",
    "GEPA_EFFECT_CONTEXT_SCHEMA_VERSION",
    "GEPA_EFFECT_SLOT_SCHEMA",
    "GEPA_EFFECT_SLOT_SCHEMA_VERSION",
    "GEPA_EFFECT_TRANSCRIPT_SCHEMA",
    "GEPA_EFFECT_TRANSCRIPT_SCHEMA_VERSION",
    "GEPA_EVALUATION_EFFECT_SCHEMA",
    "GEPA_EVALUATION_EFFECT_SCHEMA_VERSION",
    "GEPA_EVALUATION_REQUEST_RECORD_SCHEMA",
    "GEPA_EVALUATION_RESULT_RECORD_SCHEMA",
    "GEPA_PROPOSAL_EFFECT_SCHEMA",
    "GEPA_PROPOSAL_EFFECT_SCHEMA_VERSION",
    "GEPA_PROPOSAL_REQUEST_RECORD_SCHEMA",
    "GEPA_PROPOSAL_RESULT_RECORD_SCHEMA",
    "GepaCandidateComponent",
    "GepaComponentTraceProjection",
    "GepaDataInstance",
    "GepaSkippedMutation",
    "GepaEffectBroker",
    "GepaEffectConflictError",
    "GepaEffectContext",
    "GepaEffectRecorder",
    "GepaEffectRequest",
    "GepaEffectResult",
    "GepaEffectSlot",
    "GepaEffectTranscript",
    "GepaEffectTranscriptEntry",
    "GepaEvalAuthorityBinding",
    "GepaEvaluationEffectAuthority",
    "GepaEvaluationEffectRequest",
    "GepaEvaluationEffectResult",
    "GepaEvaluationRow",
    "GepaProposalAuthorityBinding",
    "GepaProposalEffectAuthority",
    "GepaProposalEffectRequest",
    "GepaProposalEffectResult",
    "GepaScoreMismatchEvidence",
    "GepaTrajectoryProjection",
]
