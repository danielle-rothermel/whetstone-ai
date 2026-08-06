from __future__ import annotations

from collections import Counter
from enum import UNIQUE, StrEnum, verify
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from whetstone.core.effects.authority import TerminalOutcome
from whetstone.core.identity import (
    IdentityHash,
    IdentityRef,
    ImmutableJsonObject,
    NonEmptyId,
    NonNegativeInt,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA as _EVALUATION_EVIDENCE_SCHEMA,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_FAILURE_SCHEMA as _EVALUATION_FAILURE_SCHEMA,
)
from whetstone.experiment import binding as _binding
from whetstone.experiment import candidate as _candidate
from whetstone.experiment import reward as _reward
from whetstone.optimization.proposal.mutation import (
    DiffCheckError,
    diff_check,
    validate_candidate_template,
)
from whetstone.optimization.tools.admission import (
    ToolCallState,
    ToolCallStoreEntry,
)
from whetstone.optimization.tools.contracts import (
    ToolCapacityScope,
    ToolConfigRef,
    ToolResultRef,
)

__all__ = [
    "INTENT_RESOLUTION_SCHEMA",
    "INTENT_RESOLUTION_SCHEMA_VERSION",
    "OPTIMIZATION_RESULT_SCHEMA",
    "OPTIMIZATION_RUN_SCHEMA",
    "OPTIMIZATION_RUN_SCHEMA_VERSION",
    "STEP_REQUEST_SCHEMA",
    "STEP_RESULT_SCHEMA",
    "BudgetDelta",
    "BudgetState",
    "EvaluationIntent",
    "IntentOutcome",
    "IntentResolution",
    "OptimizationProposal",
    "OptimizationResult",
    "OptimizationRun",
    "OptimizationRunRef",
    "OptimizationStepRequest",
    "OptimizationStepRequestRef",
    "OptimizationStepResult",
    "OptimizationStepResultRef",
    "OutputContract",
    "ResolutionClass",
    "ResolutionDetail",
    "StepKind",
    "StepMode",
    "StepStatus",
    "ToolEvidence",
    "optimization_result_reference",
    "optimization_run_reference",
    "step_request_reference",
    "step_result_reference",
]

INTENT_RESOLUTION_SCHEMA = "whetstone.optimization_intent_resolution"
INTENT_RESOLUTION_SCHEMA_VERSION = 2
OPTIMIZATION_RUN_SCHEMA = "whetstone.optimization_run"
OPTIMIZATION_RUN_SCHEMA_VERSION = 1
STEP_REQUEST_SCHEMA = "whetstone.optimization_step_request"
STEP_RESULT_SCHEMA = "whetstone.optimization_step_result"
OPTIMIZATION_RESULT_SCHEMA = "whetstone.optimization_result"


def _require_ordered_sequence(value: Any, info: ValidationInfo) -> Any:
    """Accept only the two deliberate Python representations of JSON arrays."""
    if type(value) not in (list, tuple):
        raise ValueError(
            f"{info.field_name} must be an ordered tuple or JSON array"
        )
    return value


@verify(UNIQUE)
class StepMode(StrEnum):
    PURE = "pure"
    PROPOSAL_ONLY = "proposal_only"
    TOOL_USING = "tool_using"


@verify(UNIQUE)
class StepKind(StrEnum):
    IDENTITY = "identity"
    PROPOSAL = "proposal"
    TOOL = "tool"


@verify(UNIQUE)
class StepStatus(StrEnum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    FAILED = "failed"


@verify(UNIQUE)
class IntentOutcome(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


@verify(UNIQUE)
class ResolutionClass(StrEnum):
    MEASURED = "measured"
    VALIDATION = "validation"
    UNSCORABLE = "unscorable"
    PROVIDER = "provider"
    INFRASTRUCTURE = "infrastructure"


class OutputContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    returned_proposal_count: NonNegativeInt
    require_distinct_bases: StrictBool = False


def _validate_budget_values(
    values: ImmutableJsonObject, *, field: str
) -> None:
    for label, value in values.items():
        if not label:
            raise ValueError("budget labels must be non-empty")
        if type(value) is not int:
            raise ValueError(
                f"{field} budget value for {label!r} must be a strict integer"
            )
        if value < 0:
            raise ValueError(
                f"{field} budget value for {label!r} cannot be negative"
            )


def _budget_dict(values: ImmutableJsonObject) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, value in values.items():
        if type(value) is not int:
            raise TypeError("validated budget value is not an integer")
        result[label] = value
    return result


class BudgetDelta(BaseModel):
    """The adapter-reported nonnegative consumption for one invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consumed: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    @model_validator(mode="after")
    def _validate(self) -> BudgetDelta:
        _validate_budget_values(self.consumed, field="consumed")
        return self


class BudgetState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    consumed: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    remaining: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    @model_validator(mode="after")
    def _validate(self) -> BudgetState:
        # Validate the two maps independently: overlapping labels must never
        # let a valid remaining value mask an invalid consumed value.
        _validate_budget_values(self.consumed, field="consumed")
        _validate_budget_values(self.remaining, field="remaining")
        return self

    def debit(self, delta: BudgetDelta) -> BudgetState:
        consumed = _budget_dict(self.consumed)
        remaining = _budget_dict(self.remaining)
        for label, raw_amount in delta.consumed.items():
            if type(raw_amount) is not int:
                raise TypeError("validated budget delta is not an integer")
            amount = raw_amount
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
    """Measurement request with exact candidate, config, and policy binding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_id: NonEmptyId
    candidate: _candidate.CandidateRef
    target_eval_config: _binding.EvalConfigRef
    evaluation_binding: _binding.EvaluationBinding
    purpose: NonEmptyId
    run_id: NonEmptyId
    step_index: NonNegativeInt
    expected_reward_policy_hash: IdentityHash | None = None

    @model_validator(mode="after")
    def _validate(self) -> EvaluationIntent:
        if self.target_eval_config != self.evaluation_binding.eval_config:
            raise ValueError(
                "Evaluation Intent target Eval Config must match its exact "
                "Evaluation Binding"
            )
        if self.evaluation_binding.role is EvaluationRole.INTERNAL:
            if self.expected_reward_policy_hash is None:
                raise ValueError(
                    "an internal proposal Evaluation Intent requires its "
                    "expected Reward Policy hash"
                )
        elif self.expected_reward_policy_hash is not None:
            raise ValueError(
                "an official Evaluation Intent must not expect a Reward Policy"
            )
        return self


class OptimizationRun(BaseModel):
    """An immutable run envelope independent of harness binding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: NonEmptyId
    optimizer_config: IdentityRef
    adapter_key: OpaqueKey
    mode: StepMode
    terminal_output_contract: OutputContract
    template_render_contract: _candidate.TemplateRenderContract
    reward_policy: _reward.RewardPolicy | None = None
    tool_configs: tuple[ToolConfigRef, ...] = ()

    @field_validator("tool_configs", mode="before")
    @classmethod
    def _validate_tool_configs(cls, value: Any, info: ValidationInfo) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> OptimizationRun:
        if self.mode is StepMode.PROPOSAL_ONLY:
            if self.reward_policy is None:
                raise ValueError(
                    "a proposal-only run requires an exact Reward Policy"
                )
        elif self.reward_policy is not None:
            raise ValueError(
                "only a proposal-only run may carry a Reward Policy"
            )
        if self.mode is not StepMode.TOOL_USING and self.tool_configs:
            raise ValueError("only tool-using runs may carry Tool Configs")
        if self.mode is StepMode.TOOL_USING and not self.tool_configs:
            raise ValueError("a tool-using run requires a Tool Config")
        hashes = [config.identity_hash for config in self.tool_configs]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Optimization Run Tool Configs must be unique")
        return self

    def identity_payload(self) -> dict[str, Any]:
        # These persisted identity keys are an explicit wire contract. Never
        # derive them by iterating over model fields.
        return {
            "run_id": self.run_id,
            "optimizer_config": self.optimizer_config.model_dump(mode="json"),
            "adapter_key": self.adapter_key,
            "mode": self.mode.value,
            "terminal_output_contract": (
                self.terminal_output_contract.model_dump(mode="json")
            ),
            "template_render_contract": (
                self.template_render_contract.model_dump(mode="json")
            ),
            "reward_policy": (
                None
                if self.reward_policy is None
                else self.reward_policy.model_dump(mode="json")
            ),
            "tool_configs": [
                config.model_dump(mode="json") for config in self.tool_configs
            ],
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=OPTIMIZATION_RUN_SCHEMA,
            schema_version=OPTIMIZATION_RUN_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OptimizationRunRef(BaseModel):
    """Exact Optimization Run record and identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: OptimizationRun
    record_ref: TypedRef
    identity_hash: IdentityHash

    @model_validator(mode="after")
    def _validate(self) -> OptimizationRunRef:
        expected_ref = typed_ref_for_record(
            OPTIMIZATION_RUN_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected_ref:
            raise ValueError(
                "Optimization Run record_ref must address the exact run"
            )
        if self.identity_hash != self.record.identity_hash():
            raise ValueError(
                "Optimization Run identity_hash must match the exact run"
            )
        return self


def optimization_run_reference(run: OptimizationRun) -> OptimizationRunRef:
    return OptimizationRunRef(
        record=run,
        record_ref=typed_ref_for_record(
            OPTIMIZATION_RUN_SCHEMA, run.record_content()
        ),
        identity_hash=run.identity_hash(),
    )


class OptimizationStepRequest(BaseModel):
    """One step bound to the exact immutable Optimization Run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: OptimizationRunRef
    step_id: NonEmptyId
    kind: StepKind
    kind_label: NonEmptyId | None = None
    step_index: NonNegativeInt
    prior_step_result_ref: TypedRef | None = None
    prior_state_ref: TypedRef | None = None
    prior_history_ref: TypedRef | None = None
    candidates: tuple[_candidate.Candidate, ...] = ()
    pools: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    hyperparameters: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    budget: BudgetState = Field(default_factory=BudgetState)
    step_output_contract: OutputContract

    @field_validator("candidates", mode="before")
    @classmethod
    def _validate_candidates(cls, value: Any, info: ValidationInfo) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepRequest:
        if self.step_index == 0 and any(
            ref is not None
            for ref in (
                self.prior_step_result_ref,
                self.prior_state_ref,
                self.prior_history_ref,
            )
        ):
            raise ValueError("the initial Step Request carries no prior refs")
        if self.step_index > 0 and self.prior_step_result_ref is None:
            raise ValueError(
                "a noninitial Step Request must reference the prior result"
            )
        if (
            self.prior_step_result_ref is not None
            and self.prior_step_result_ref.schema_name != STEP_RESULT_SCHEMA
        ):
            raise ValueError(
                "prior_step_result_ref must be a typed Step Result ref"
            )
        if (
            self.run.record.mode is StepMode.PURE
            and self.kind is not StepKind.IDENTITY
        ):
            raise ValueError("a pure step must be the identity kind")
        for candidate in self.candidates:
            try:
                validate_candidate_template(candidate=candidate, run=self.run)
            except ValueError as error:
                raise ValueError(
                    "every Step Request candidate must satisfy the exact run "
                    f"template contract: {error}"
                ) from error
        return self

    @property
    def run_id(self) -> NonEmptyId:
        return self.run.record.run_id

    @property
    def adapter_key(self) -> OpaqueKey:
        return self.run.record.adapter_key

    @property
    def mode(self) -> StepMode:
        return self.run.record.mode

    @property
    def tool_configs(self) -> tuple[ToolConfigRef, ...]:
        return self.run.record.tool_configs

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OptimizationStepRequestRef(BaseModel):
    """Exact Optimization Step Request record and content reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: OptimizationStepRequest
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepRequestRef:
        expected_ref = typed_ref_for_record(
            STEP_REQUEST_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected_ref:
            raise ValueError(
                "Step Request record_ref must address the exact request"
            )
        return self


class ResolutionDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    classification: ResolutionClass
    message: NonEmptyId


class IntentResolution(BaseModel):
    """Typed terminal outcome for one exact Evaluation Intent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    intent: EvaluationIntent
    outcome: IntentOutcome
    detail: ResolutionDetail
    evaluation_result_ref: TypedRef | None = None
    reward_evidence_refs: tuple[TypedRef, ...] = ()
    resolved_eval_config: _binding.EvalConfigRef
    reward_ref: _reward.RewardRef | None = None
    terminal_failure: TerminalFailure | None = None

    @field_validator("reward_evidence_refs", mode="before")
    @classmethod
    def _validate_reward_evidence_refs(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> IntentResolution:
        if self.resolved_eval_config != self.intent.target_eval_config:
            raise ValueError(
                "an Intent may resolve only under its exact target Eval Config"
            )
        if (
            self.outcome is not IntentOutcome.REJECTED
            and self.evaluation_result_ref is None
        ):
            raise ValueError(
                "completed/failed resolution requires an Evaluation Result"
            )
        if (
            self.outcome is IntentOutcome.REJECTED
            and self.evaluation_result_ref is not None
        ):
            raise ValueError(
                "pre-execution rejection must not carry an Evaluation Result"
            )
        expected_result_schema = {
            IntentOutcome.COMPLETED: _EVALUATION_EVIDENCE_SCHEMA,
            IntentOutcome.FAILED: _EVALUATION_FAILURE_SCHEMA,
        }.get(self.outcome)
        if (
            self.evaluation_result_ref is not None
            and self.evaluation_result_ref.schema_name
            != expected_result_schema
        ):
            raise ValueError(
                f"{self.outcome.value} evaluation_result_ref must use "
                f"schema {expected_result_schema!r}"
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
        if (self.outcome is IntentOutcome.FAILED) != (
            self.terminal_failure is not None
        ):
            raise ValueError(
                "a failed Intent Resolution requires exactly one shared "
                "terminal failure"
            )
        if (
            self.outcome is not IntentOutcome.COMPLETED
            and self.reward_ref is not None
        ):
            raise ValueError(
                "only a completed Intent Resolution may carry a Reward"
            )
        if (
            self.intent.evaluation_binding.role is EvaluationRole.OFFICIAL
            and self.reward_ref is not None
        ):
            raise ValueError(
                "an official Intent Resolution must not carry a Reward"
            )
        if (
            self.outcome is IntentOutcome.COMPLETED
            and self.intent.evaluation_binding.role is EvaluationRole.INTERNAL
            and self.reward_ref is None
        ):
            raise ValueError(
                "a completed internal Intent Resolution requires a Reward"
            )
        if self.reward_ref is None and self.reward_evidence_refs:
            raise ValueError(
                "a rewardless Intent Resolution must not carry Reward "
                "evidence refs"
            )
        if self.reward_ref is not None:
            reward = self.reward_ref.record
            if reward.evidence_refs != self.reward_evidence_refs:
                raise ValueError(
                    "Reward evidence_refs must exactly equal the ordered "
                    "Intent Resolution reward_evidence_refs"
                )
            if (
                reward.reward_policy_hash
                != self.intent.expected_reward_policy_hash
            ):
                raise ValueError(
                    "Reward Policy must match the Evaluation Intent expected "
                    "Reward Policy hash"
                )
        return self


class ToolEvidence(BaseModel):
    """Exact Tool Result composed with one terminal store projection.

    This model proves internal composition only. ToolCallStore owns I/O
    verification that the EffectTerminal came from its effect authority.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: ToolResultRef
    store_entry: ToolCallStoreEntry

    @model_validator(mode="after")
    def _validate(self) -> ToolEvidence:
        entry = self.store_entry
        result = self.result.record
        entry_call_ref = getattr(entry, "tool_call_ref", None)
        if entry_call_ref != result.call.record_ref:
            raise ValueError(
                "Tool Evidence store entry must cite the exact Tool Call"
            )
        entry_result_ref = getattr(entry, "tool_result_ref", None)
        if entry_result_ref != self.result.record_ref:
            raise ValueError(
                "Tool Evidence store entry must cite the exact Tool Result"
            )
        if entry.state is ToolCallState.REFUSED:
            if (
                result.output is not None
                or result.terminal_failure is not None
            ):
                raise ValueError(
                    "refused Tool Evidence result has no output or terminal "
                    "failure"
                )
            if entry.refusal != result.refusal:
                raise ValueError(
                    "refused Tool Evidence requires the exact Tool Result "
                    "refusal"
                )
            return self
        if entry.state is not ToolCallState.COMPLETED:
            raise ValueError("Tool Evidence requires a terminal store entry")
        terminal = entry.effect_terminal
        if terminal is None:
            raise ValueError(
                "completed Tool Call Store entry lost its EffectTerminal"
            )
        if terminal.outcome is TerminalOutcome.SUCCEEDED:
            if (
                result.output is None
                or result.refusal is not None
                or result.terminal_failure is not None
            ):
                raise ValueError(
                    "succeeded EffectTerminal requires the exact successful "
                    "Tool Result"
                )
        elif terminal.outcome is TerminalOutcome.FAILED:
            if (
                result.output is not None
                or result.refusal is not None
                or result.terminal_failure != terminal.failure
            ):
                raise ValueError(
                    "failed EffectTerminal failure must exactly equal the "
                    "Tool Result terminal failure"
                )
        else:
            raise ValueError(
                "recovery-required effects have no completed Tool Evidence"
            )
        if entry.capacity_debit_ordinal != result.provenance_ordinal:
            raise ValueError(
                "completed Tool Evidence capacity debit ordinal must exactly "
                "equal the Tool Result provenance ordinal"
            )
        return self


class OptimizationStepResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: OptimizationStepRequestRef
    proposed_candidates: tuple[_candidate.CandidateRef, ...] = ()
    accepted_candidates: tuple[_candidate.CandidateRef, ...] = ()
    resolved_intents: tuple[IntentResolution, ...] = ()
    tool_evidence: tuple[ToolEvidence, ...] = ()
    state_ref: TypedRef | None = None
    history_ref: TypedRef | None = None
    budget_delta: BudgetDelta = Field(default_factory=BudgetDelta)
    budget: BudgetState = Field(default_factory=BudgetState)
    status: StepStatus
    terminal_failure: TerminalFailure | None = None
    provenance_note: NonEmptyId | None = None
    provenance_ordinal: NonNegativeInt | None = None

    @field_validator(
        "proposed_candidates",
        "accepted_candidates",
        "resolved_intents",
        "tool_evidence",
        mode="before",
    )
    @classmethod
    def _validate_ordered_outputs(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepResult:
        request = self.request.record
        request_candidate_refs = tuple(
            _candidate.candidate_reference(candidate)
            for candidate in request.candidates
        )
        for field_name, candidates in (
            ("proposed", self.proposed_candidates),
            ("accepted", self.accepted_candidates),
        ):
            for candidate in candidates:
                try:
                    validate_candidate_template(
                        candidate=candidate.record,
                        run=request.run,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"every {field_name} candidate must satisfy the exact "
                        f"run template contract: {error}"
                    ) from error
        if self.resolved_intents and self.tool_evidence:
            raise ValueError(
                "a Step Result carries intent or tool evidence, never both"
            )
        if request.mode is StepMode.PURE and (
            self.resolved_intents or self.tool_evidence
        ):
            raise ValueError(
                "a pure Step Result carries no execution evidence"
            )
        if request.mode is StepMode.PROPOSAL_ONLY and self.tool_evidence:
            raise ValueError(
                "a proposal-only Step Result carries only Intent Resolutions"
            )
        if request.mode is StepMode.TOOL_USING and self.resolved_intents:
            raise ValueError(
                "a tool-using Step Result carries only Tool Evidence"
            )
        request_candidate_multiset = Counter(
            candidate.record_ref for candidate in request_candidate_refs
        )
        proposed_candidate_multiset = Counter(
            candidate.record_ref for candidate in self.proposed_candidates
        )
        accepted_candidate_multiset = Counter(
            candidate.record_ref for candidate in self.accepted_candidates
        )
        if request.mode is StepMode.PURE:
            if proposed_candidate_multiset != request_candidate_multiset:
                raise ValueError(
                    "a pure Step Result proposed candidate multiset must "
                    "exactly equal the request candidate multiset"
                )
            if (
                self.status is not StepStatus.FAILED
                and accepted_candidate_multiset != request_candidate_multiset
            ):
                raise ValueError(
                    "a pure Step Result accepted candidate multiset must "
                    "exactly equal the request candidate multiset"
                )
        if request.mode is not StepMode.PURE:
            request_bases = {
                candidate.record_ref: candidate.record
                for candidate in request_candidate_refs
            }
            for proposed in self.proposed_candidates:
                base = request_bases.get(proposed.record.base_ref)
                if base is None:
                    raise ValueError(
                        "every effectful proposed candidate must bind an "
                        "exact request base"
                    )
                try:
                    diff_check(base=base, proposed=proposed.record)
                except DiffCheckError as error:
                    raise ValueError(
                        "every effectful proposed candidate must satisfy the "
                        f"canonical Mutation Surface: {error}"
                    ) from error
        allowed_candidates = request_candidate_refs + self.proposed_candidates
        intent_ids = tuple(
            resolution.intent.intent_id for resolution in self.resolved_intents
        )
        if len(set(intent_ids)) != len(intent_ids):
            raise ValueError(
                "a Step Result must not carry duplicate Evaluation Intent IDs"
            )
        exact_intents = tuple(
            resolution.intent.model_dump_json()
            for resolution in self.resolved_intents
        )
        if len(set(exact_intents)) != len(exact_intents):
            raise ValueError(
                "a Step Result must not carry duplicate Intent Resolutions"
            )
        for resolution in self.resolved_intents:
            intent = resolution.intent
            if intent.run_id != request.run_id:
                raise ValueError(
                    "every resolved Intent must belong to the exact request "
                    "run"
                )
            if intent.step_index != request.step_index:
                raise ValueError(
                    "every resolved Intent must belong to the exact request "
                    "step"
                )
            if not any(
                intent.candidate == candidate
                for candidate in allowed_candidates
            ):
                raise ValueError(
                    "every resolved Intent must cite an exact request, "
                    "or proposed candidate"
                )
            reward_policy = request.run.record.reward_policy
            if reward_policy is None:
                raise ValueError(
                    "a resolved proposal Intent requires the exact run Reward "
                    "Policy"
                )
            if (
                intent.evaluation_binding.role is EvaluationRole.INTERNAL
                and intent.expected_reward_policy_hash
                != reward_policy.identity_hash()
            ):
                raise ValueError(
                    "every resolved Intent must expect the exact run Reward "
                    "Policy"
                )
            if (
                resolution.reward_ref is not None
                and resolution.reward_ref.record.reward_policy != reward_policy
            ):
                raise ValueError(
                    "every resolved Reward must carry the exact run Reward "
                    "Policy"
                )
        for evidence in self.tool_evidence:
            call = evidence.result.record.call.record
            if call.tool_config not in request.tool_configs:
                raise ValueError(
                    "every Tool Evidence call must use an exact request Tool "
                    "Config"
                )
            binding = call.capacity_binding
            expected_subject = (
                request.run.record_ref
                if binding.scope is ToolCapacityScope.RUN
                else (
                    self.request.record_ref
                    if binding.scope is ToolCapacityScope.STEP
                    else None
                )
            )
            if binding.subject_ref != expected_subject:
                raise ValueError(
                    "every Tool Evidence call must bind exact request "
                    "capacity authority"
                )
        if (self.status is StepStatus.FAILED) != (
            self.terminal_failure is not None
        ):
            raise ValueError(
                "a failed Step Result requires exactly one shared terminal "
                "failure"
            )
        if self.status is StepStatus.FAILED and self.accepted_candidates:
            raise ValueError(
                "a failed Step Result claims no accepted candidates"
            )
        if self.status is StepStatus.FAILED:
            nested_failures = tuple(
                resolution.terminal_failure
                for resolution in self.resolved_intents
                if resolution.terminal_failure is not None
            ) + tuple(
                evidence.result.record.terminal_failure
                for evidence in self.tool_evidence
                if evidence.result.record.terminal_failure is not None
            )
            if any(
                failure != self.terminal_failure for failure in nested_failures
            ):
                raise ValueError(
                    "every nested terminal failure must equal the exact "
                    "outer Step failure"
                )
        if self.status is not StepStatus.FAILED:
            contract = request.step_output_contract
            if (
                len(self.accepted_candidates)
                != contract.returned_proposal_count
            ):
                raise ValueError(
                    "Step Result violates returned proposal cardinality"
                )
            if contract.require_distinct_bases:
                bases = [
                    (
                        candidate.record.base_ref.schema_name,
                        candidate.record.base_ref.content_hash,
                    )
                    for candidate in self.accepted_candidates
                ]
                if len(bases) != len(set(bases)):
                    raise ValueError(
                        "Step Result violates the distinct-base output "
                        "contract"
                    )
        if accepted_candidate_multiset - proposed_candidate_multiset:
            raise ValueError(
                "accepted candidate multiset must be contained in proposed "
                "candidate multiset"
            )
        if (
            self.status is StepStatus.COMPLETE
            and self.request.record.step_output_contract
            != self.request.record.run.record.terminal_output_contract
        ):
            raise ValueError(
                "a COMPLETE Step must use the run terminal output contract"
            )
        expected_budget = request.budget.debit(self.budget_delta)
        if self.budget != expected_budget:
            raise ValueError(
                "Step Result budget must exactly debit its Step Request budget"
            )
        return self

    @property
    def run_id(self) -> NonEmptyId:
        return self.request.record.run_id

    @property
    def step_id(self) -> NonEmptyId:
        return self.request.record.step_id

    @property
    def step_index(self) -> NonNegativeInt:
        return self.request.record.step_index

    @property
    def request_ref(self) -> TypedRef:
        return self.request.record_ref

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OptimizationStepResultRef(BaseModel):
    """Exact Optimization Step Result record and content reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: OptimizationStepResult
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepResultRef:
        expected_ref = typed_ref_for_record(
            STEP_RESULT_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected_ref:
            raise ValueError(
                "Step Result record_ref must address the exact result"
            )
        return self


class OptimizationProposal(BaseModel):
    """An ordered terminal proposal composed from a persisted candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: _candidate.CandidateRef


class OptimizationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run: OptimizationRunRef
    proposals: tuple[OptimizationProposal, ...]
    step_results: tuple[OptimizationStepResultRef, ...]
    cost: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    terminal_failure: TerminalFailure | None = None
    provenance_note: NonEmptyId | None = None
    provenance_ordinal: NonNegativeInt | None = None

    @field_validator("proposals", "step_results", mode="before")
    @classmethod
    def _validate_ordered_terminal_fields(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> OptimizationResult:
        if not self.step_results:
            raise ValueError(
                "an Optimization Result composes every ordered Step Result"
            )
        for index, exact_result in enumerate(self.step_results):
            result = exact_result.record
            request = result.request.record
            if result.step_index != index:
                raise ValueError(
                    "Step Result indices must be contiguous from zero"
                )
            if request.run != self.run:
                raise ValueError(
                    "every Step Result request must belong to the exact run"
                )
            is_final = index == len(self.step_results) - 1
            if not is_final and result.status is not StepStatus.CONTINUE:
                raise ValueError(
                    "only the final Step Result may have terminal status"
                )
            if index == 0:
                continue
            prior = self.step_results[index - 1]
            prior_result = prior.record
            if request.prior_step_result_ref != prior.record_ref:
                raise ValueError(
                    "each later Step Request must cite the prior exact result"
                )
            if request.prior_state_ref != prior_result.state_ref:
                raise ValueError(
                    "each later Step Request must carry the prior exact state"
                )
            if request.prior_history_ref != prior_result.history_ref:
                raise ValueError(
                    "each later Step Request must carry the prior exact "
                    "history"
                )
            if request.budget != prior_result.budget:
                raise ValueError(
                    "each later Step Request must carry the prior exact budget"
                )

        last = self.step_results[-1].record
        if last.status is StepStatus.CONTINUE:
            raise ValueError("the final Step Result must have terminal status")
        if self.terminal_failure != last.terminal_failure:
            raise ValueError(
                "Optimization Result failure must match the final Step Result"
            )
        if last.status is StepStatus.FAILED and self.proposals:
            raise ValueError(
                "a failed Optimization Result claims no proposals"
            )
        if last.status is StepStatus.COMPLETE:
            proposal_candidates = tuple(
                proposal.candidate for proposal in self.proposals
            )
            if proposal_candidates != last.accepted_candidates:
                raise ValueError(
                    "successful proposals must exactly derive from final "
                    "accepted candidates"
                )
            contract = self.run.record.terminal_output_contract
            if len(self.proposals) != contract.returned_proposal_count:
                raise ValueError(
                    "Optimization Result violates terminal proposal "
                    "cardinality"
                )
            if contract.require_distinct_bases:
                bases = [
                    (
                        proposal.candidate.record.base_ref.schema_name,
                        proposal.candidate.record.base_ref.content_hash,
                    )
                    for proposal in self.proposals
                ]
                if len(bases) != len(set(bases)):
                    raise ValueError(
                        "Optimization Result violates the terminal "
                        "distinct-base contract"
                    )
        return self

    @property
    def run_id(self) -> NonEmptyId:
        return self.run.record.run_id

    @property
    def step_result_refs(self) -> tuple[TypedRef, ...]:
        return tuple(result.record_ref for result in self.step_results)

    @property
    def status(self) -> StepStatus:
        return self.step_results[-1].record.status

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def step_request_reference(
    request: OptimizationStepRequest,
) -> OptimizationStepRequestRef:
    return OptimizationStepRequestRef(
        record=request,
        record_ref=typed_ref_for_record(
            STEP_REQUEST_SCHEMA, request.record_content()
        ),
    )


def step_result_reference(
    result: OptimizationStepResult,
) -> OptimizationStepResultRef:
    return OptimizationStepResultRef(
        record=result,
        record_ref=typed_ref_for_record(
            STEP_RESULT_SCHEMA, result.record_content()
        ),
    )


def optimization_result_reference(result: OptimizationResult) -> TypedRef:
    return typed_ref_for_record(
        OPTIMIZATION_RESULT_SCHEMA, result.record_content()
    )
