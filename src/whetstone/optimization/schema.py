"""Serialization contracts for durable optimization."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from dr_code.eval import EvalConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.identity import (
    TypedRef,
    compute_identity_hash,
    reject_non_json,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.optimization.tool_store import ToolCallStoreEntry
from whetstone.optimization.tools import ToolConfig

__all__ = [
    "CANDIDATE_RECORD_SCHEMA",
    "EVAL_CONFIG_RECORD_SCHEMA",
    "OPTIMIZATION_RESULT_SCHEMA",
    "OPTIMIZATION_RUN_SCHEMA",
    "STEP_REQUEST_SCHEMA",
    "STEP_RESULT_SCHEMA",
    "BudgetDelta",
    "BudgetState",
    "Candidate",
    "CandidateRef",
    "EvalConfigRef",
    "EvaluationIntent",
    "IntentOutcome",
    "IntentResolution",
    "OptimizationProposal",
    "OptimizationResult",
    "OptimizationRun",
    "OptimizationStepRequest",
    "OptimizationStepResult",
    "OutputContract",
    "ResolutionClass",
    "ResolutionDetail",
    "StepKind",
    "StepMode",
    "StepStatus",
    "ToolEvidence",
    "candidate_reference",
    "eval_config_reference",
    "optimization_result_reference",
    "step_request_reference",
    "step_result_reference",
]

CANDIDATE_RECORD_SCHEMA = "whetstone.optimization_candidate"
EVAL_CONFIG_RECORD_SCHEMA = "dr_code.eval_config"
OPTIMIZATION_RUN_SCHEMA = "whetstone.optimization_run"
OPTIMIZATION_RUN_SCHEMA_VERSION = 1
STEP_REQUEST_SCHEMA = "whetstone.optimization_step_request"
STEP_RESULT_SCHEMA = "whetstone.optimization_step_result"
OPTIMIZATION_RESULT_SCHEMA = "whetstone.optimization_result"
CANDIDATE_IDENTITY_SCHEMA = "whetstone.optimization_candidate"
CANDIDATE_IDENTITY_SCHEMA_VERSION = 1


class StepMode(StrEnum):
    PURE = "pure"
    PROPOSAL_ONLY = "proposal_only"
    TOOL_USING = "tool_using"


class StepKind(StrEnum):
    IDENTITY = "identity"
    PROPOSAL = "proposal"
    TOOL = "tool"


class StepStatus(StrEnum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    FAILED = "failed"


class IntentOutcome(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class ResolutionClass(StrEnum):
    MEASURED = "measured"
    VALIDATION = "validation"
    UNSCORABLE = "unscorable"
    PROVIDER = "provider"
    INFRASTRUCTURE = "infrastructure"


class Candidate(BaseModel):
    """Identity-bearing candidate record persisted by the harness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: StrictStr
    base_ref: StrictStr
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> Candidate:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.base_ref:
            raise ValueError("base_ref must be non-empty")
        reject_non_json(self.payload, field="payload")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "base_ref": self.base_ref,
            "payload": self.payload,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=CANDIDATE_IDENTITY_SCHEMA,
            schema_version=CANDIDATE_IDENTITY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CandidateRef(BaseModel):
    """Exact typed candidate record and its persisted content reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: Candidate
    record_ref: TypedRef
    identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> CandidateRef:
        if self.record_ref != typed_ref_for_record(
            CANDIDATE_RECORD_SCHEMA, self.record.record_content()
        ):
            raise ValueError(
                "candidate record_ref must address the exact candidate record"
            )
        if self.identity_hash != self.record.identity_hash():
            raise ValueError(
                "candidate identity_hash must match the exact candidate record"
            )
        return self


def candidate_reference(candidate: Candidate) -> CandidateRef:
    return CandidateRef(
        record=candidate,
        record_ref=typed_ref_for_record(
            CANDIDATE_RECORD_SCHEMA, candidate.record_content()
        ),
        identity_hash=candidate.identity_hash(),
    )


class EvalConfigRef(BaseModel):
    """Exact typed Eval Config record and persisted record reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvalConfig
    record_ref: TypedRef
    identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> EvalConfigRef:
        require_full_hash(self.identity_hash, field="identity_hash")
        if self.identity_hash != self.record.config_identity_hash:
            raise ValueError(
                "Eval Config identity_hash must match the exact typed record"
            )
        expected = typed_ref_for_record(
            EVAL_CONFIG_RECORD_SCHEMA, self.record.model_dump(mode="json")
        )
        if self.record_ref != expected:
            raise ValueError(
                "Eval Config record_ref must address the exact typed record"
            )
        return self


def eval_config_reference(eval_config: EvalConfig) -> EvalConfigRef:
    return EvalConfigRef(
        record=eval_config,
        record_ref=typed_ref_for_record(
            EVAL_CONFIG_RECORD_SCHEMA, eval_config.model_dump(mode="json")
        ),
        identity_hash=eval_config.config_identity_hash,
    )


class OutputContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    returned_proposal_count: StrictInt
    require_distinct_bases: StrictBool = False

    @model_validator(mode="after")
    def _validate(self) -> OutputContract:
        if self.returned_proposal_count < 0:
            raise ValueError("returned_proposal_count cannot be negative")
        return self


class BudgetDelta(BaseModel):
    """The adapter-reported consumption for one invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consumed: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> BudgetDelta:
        if any(value < 0 for value in self.consumed.values()):
            raise ValueError("budget deltas cannot be negative")
        if any(not label for label in self.consumed):
            raise ValueError("budget labels must be non-empty")
        return self


class BudgetState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    consumed: dict[str, int] = Field(default_factory=dict)
    remaining: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> BudgetState:
        for label, value in {**self.consumed, **self.remaining}.items():
            if not label:
                raise ValueError("budget labels must be non-empty")
            if value < 0:
                raise ValueError(
                    f"budget value for {label!r} cannot be negative"
                )
        return self

    def debit(self, delta: BudgetDelta) -> BudgetState:
        consumed = dict(self.consumed)
        remaining = dict(self.remaining)
        for label, amount in delta.consumed.items():
            if label not in remaining:
                raise ValueError(
                    f"adapter consumed undeclared budget {label!r}"
                )
            if amount > remaining[label]:
                raise ValueError(
                    f"adapter consumed {amount} {label!r}, but only "
                    f"{remaining[label]} remains"
                )
            consumed[label] = consumed.get(label, 0) + amount
            remaining[label] -= amount
        return BudgetState(consumed=consumed, remaining=remaining)


class EvaluationIntent(BaseModel):
    """Measurement request with exact typed candidate and Eval Config refs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_id: StrictStr
    candidate: CandidateRef
    target_eval_config: EvalConfigRef
    context_role: EvaluationRole
    context_policy_ref: StrictStr | None = None
    purpose: StrictStr
    run_id: StrictStr
    step_index: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> EvaluationIntent:
        if not self.intent_id:
            raise ValueError("intent_id must be non-empty")
        if not self.purpose:
            raise ValueError("purpose must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        return self


class OptimizationRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    optimizer_config_hash: StrictStr
    adapter_key: StrictStr
    mode: StepMode
    output_contract: OutputContract
    tool_configs: tuple[ToolConfig, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> OptimizationRun:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.adapter_key:
            raise ValueError("adapter_key must be non-empty")
        require_full_hash(
            self.optimizer_config_hash, field="optimizer_config_hash"
        )
        if self.mode is not StepMode.TOOL_USING and self.tool_configs:
            raise ValueError("only tool-using runs may carry Tool Configs")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "optimizer_config_hash": self.optimizer_config_hash,
            "adapter_key": self.adapter_key,
            "mode": self.mode.value,
            "output_contract": self.output_contract.model_dump(mode="json"),
            "tool_configs": [cfg.identity_hash() for cfg in self.tool_configs],
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=OPTIMIZATION_RUN_SCHEMA,
            schema_version=OPTIMIZATION_RUN_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class OptimizationStepRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    step_id: StrictStr
    optimizer_config_hash: StrictStr
    adapter_key: StrictStr
    mode: StepMode
    kind: StepKind
    kind_label: StrictStr | None = None
    step_index: StrictInt
    prior_step_result_ref: TypedRef | None = None
    prior_state_ref: TypedRef | None = None
    prior_history_ref: TypedRef | None = None
    candidates: tuple[Candidate, ...] = ()
    pools: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetState = Field(default_factory=BudgetState)
    output_contract: OutputContract
    tool_configs: tuple[ToolConfig, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepRequest:
        if not self.run_id or not self.step_id or not self.adapter_key:
            raise ValueError(
                "run_id, step_id, and adapter_key must be non-empty"
            )
        require_full_hash(
            self.optimizer_config_hash, field="optimizer_config_hash"
        )
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        reject_non_json(self.pools, field="pools")
        reject_non_json(self.hyperparameters, field="hyperparameters")
        if self.step_index == 0 and self.prior_step_result_ref is not None:
            raise ValueError(
                "the initial Step Request carries no prior Step Result ref"
            )
        if self.step_index > 0 and self.prior_step_result_ref is None:
            raise ValueError(
                "a noninitial Step Request must reference the prior result"
            )
        if self.mode is StepMode.PURE and self.kind is not StepKind.IDENTITY:
            raise ValueError("a pure step must be the identity kind")
        if self.mode is StepMode.TOOL_USING and not self.tool_configs:
            raise ValueError("a tool-using request requires a Tool Config")
        if self.mode is not StepMode.TOOL_USING and self.tool_configs:
            raise ValueError(
                "only a tool-using request may carry Tool Configs"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ResolutionDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    classification: ResolutionClass
    message: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> ResolutionDetail:
        if not self.message:
            raise ValueError("resolution detail message must be non-empty")
        return self


class IntentResolution(BaseModel):
    """Typed terminal outcome for one exact Evaluation Intent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: EvaluationIntent
    outcome: IntentOutcome
    detail: ResolutionDetail
    evaluation_evidence_refs: tuple[TypedRef, ...] = ()
    resolved_eval_config: EvalConfigRef
    reward_ref: TypedRef | None = None

    @model_validator(mode="after")
    def _validate(self) -> IntentResolution:
        if self.resolved_eval_config != self.intent.target_eval_config:
            raise ValueError(
                "an Intent may resolve only under its exact target Eval Config"
            )
        if (
            self.outcome is not IntentOutcome.REJECTED
            and not self.evaluation_evidence_refs
        ):
            raise ValueError(
                "completed/failed resolution requires execution evidence"
            )
        if (
            self.outcome is IntentOutcome.REJECTED
            and self.evaluation_evidence_refs
        ):
            raise ValueError(
                "pre-execution rejection must not carry execution evidence"
            )
        if (
            self.outcome is IntentOutcome.COMPLETED
            and self.detail.classification is not ResolutionClass.MEASURED
        ):
            raise ValueError(
                "completed resolution must be classified measured"
            )
        if (
            self.outcome is IntentOutcome.REJECTED
            and self.detail.classification
            not in {ResolutionClass.VALIDATION, ResolutionClass.UNSCORABLE}
        ):
            raise ValueError(
                "rejection must be classified validation or unscorable"
            )
        return self


class ToolEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_result_ref: TypedRef
    store_entry: ToolCallStoreEntry

    @model_validator(mode="after")
    def _validate(self) -> ToolEvidence:
        entry_ref = self.store_entry.tool_result_ref
        if entry_ref is not None and entry_ref != self.tool_result_ref:
            raise ValueError(
                "Tool Result ref must match the completed store entry"
            )
        if entry_ref is None and self.store_entry.refusal is None:
            raise ValueError(
                "Tool evidence requires a completed or refused entry"
            )
        return self


class OptimizationStepResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    step_id: StrictStr
    step_index: StrictInt
    request_ref: TypedRef
    proposed_candidates: tuple[CandidateRef, ...] = ()
    accepted_candidates: tuple[CandidateRef, ...] = ()
    resolved_intents: tuple[IntentResolution, ...] = ()
    tool_evidence: tuple[ToolEvidence, ...] = ()
    state_ref: TypedRef | None = None
    history_ref: TypedRef | None = None
    budget_delta: BudgetDelta = Field(default_factory=BudgetDelta)
    budget: BudgetState = Field(default_factory=BudgetState)
    status: StepStatus
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepResult:
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        if self.request_ref.schema_name != STEP_REQUEST_SCHEMA:
            raise ValueError("request_ref must be a typed Step Request ref")
        if self.resolved_intents and self.tool_evidence:
            raise ValueError(
                "a Step Result carries intent or tool evidence, never both"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OptimizationProposal(BaseModel):
    """An ordered terminal proposal composed from a persisted candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef


class OptimizationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    proposals: tuple[OptimizationProposal, ...]
    step_result_refs: tuple[TypedRef, ...]
    status: StepStatus
    cost: dict[str, Any] = Field(default_factory=dict)
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> OptimizationResult:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.step_result_refs:
            raise ValueError(
                "an Optimization Result references every ordered Step Result"
            )
        if any(
            ref.schema_name != STEP_RESULT_SCHEMA
            for ref in self.step_result_refs
        ):
            raise ValueError(
                "step_result_refs must be typed Step Result references"
            )
        if self.status is StepStatus.FAILED and self.proposals:
            raise ValueError(
                "a failed Optimization Result claims no proposals"
            )
        reject_non_json(self.cost, field="cost")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def step_request_reference(request: OptimizationStepRequest) -> TypedRef:
    return typed_ref_for_record(STEP_REQUEST_SCHEMA, request.record_content())


def step_result_reference(result: OptimizationStepResult) -> TypedRef:
    return typed_ref_for_record(STEP_RESULT_SCHEMA, result.record_content())


def optimization_result_reference(result: OptimizationResult) -> TypedRef:
    return typed_ref_for_record(
        OPTIMIZATION_RESULT_SCHEMA, result.record_content()
    )
