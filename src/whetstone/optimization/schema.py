"""The durable Optimization Step protocol schemas and identities.

Five immutable schemas, per Workstream 7 and the vocabulary:

* :class:`OptimizationRun` — the umbrella Run identity (run/config identities,
  step-mode envelope, output contract, optional Tool Configs). It freezes the
  run-wide inputs the initial Step Request carries.

* :class:`OptimizationStepRequest` — the immutable input to one Step:
  run/step/config identities, kind + ordered index, prior Step Result / state
  / history :class:`TypedRef`s, candidate or pool inputs, hyperparameters,
  budget state, output contract, and optional serialized Tool Configs (each
  with its typed Tool Definition ref + Identity Hash). It **rejects** runtime
  handles, clients, and closures — validation only accepts strict-JSON pydantic
  values, so a Runtime Tool Handle (a plain non-JSON object) cannot be carried.

* :class:`EvaluationIntent` — an immutable measurement request: candidate
  identity, exact target Eval Config ref + Identity Hash, Evaluation Context
  role/policy, purpose, run/step correlation. It has **no** result/score field.

* :class:`OptimizationStepResult` — the immutable terminal result for one Step:
  the request ref + Content Hash; proposed/accepted candidates; resolved
  intents + evaluation-evidence refs OR Tool Results + Tool Call Store Entries;
  a state/history delta/snapshot ref; budgets consumed/remaining; and exactly
  one status (``continue`` | ``complete`` | ``failed``).

* :class:`OptimizationResult` — the terminal run output: ordered proposals,
  every ordered Step Result ref, record-local provenance, and cost; it makes no
  official-evaluation claim.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

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
    "OPTIMIZATION_RESULT_SCHEMA",
    "OPTIMIZATION_RUN_SCHEMA",
    "OPTIMIZATION_RUN_SCHEMA_VERSION",
    "STEP_REQUEST_SCHEMA",
    "STEP_RESULT_SCHEMA",
    "BudgetState",
    "Candidate",
    "EvaluationIntent",
    "IntentResolution",
    "OptimizationProposal",
    "OptimizationResult",
    "OptimizationRun",
    "OptimizationStepRequest",
    "OptimizationStepResult",
    "OutputContract",
    "StepKind",
    "StepMode",
    "StepStatus",
    "ToolEvidence",
    "step_request_reference",
    "step_result_reference",
]

OPTIMIZATION_RUN_SCHEMA = "whetstone.optimization_run"
OPTIMIZATION_RUN_SCHEMA_VERSION = 1
# Stored records (Content Hash), not identities.
STEP_REQUEST_SCHEMA = "whetstone.optimization_step_request"
STEP_RESULT_SCHEMA = "whetstone.optimization_step_result"
OPTIMIZATION_RESULT_SCHEMA = "whetstone.optimization_result"


class StepMode(StrEnum):
    """The declared execution mode the harness dispatches on."""

    PURE = "pure"
    PROPOSAL_ONLY = "proposal_only"
    TOOL_USING = "tool_using"


class StepKind(StrEnum):
    """Algorithm-neutral step kinds recognized by the harness.

    Algorithm-specific kinds (bootstrap, minibatch, promotion_full, ...) are
    carried as free ``kind_label`` strings on the request; these are the
    harness-level structural kinds every adapter shares.
    """

    IDENTITY = "identity"
    PROPOSAL = "proposal"
    TOOL = "tool"


class StepStatus(StrEnum):
    """Exactly one terminal status per Step Result."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    FAILED = "failed"


class Candidate(BaseModel):
    """A candidate carried into/out of a Step (identity-bearing).

    ``candidate_id`` is the stable optimizer-facing ID; ``payload`` is the
    mutation-surface assignment (encoder ``{user_prompt_template}`` in these
    runs) plus its explicit base. The identity hash is over the full record so
    an unchanged candidate round-trips to the same identity.
    """

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


class OutputContract(BaseModel):
    """The declared output contract a Step Result must satisfy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    returned_proposal_count: StrictInt
    require_distinct_bases: StrictBool = False

    @model_validator(mode="after")
    def _validate(self) -> OutputContract:
        if self.returned_proposal_count < 0:
            raise ValueError("returned_proposal_count cannot be negative")
        return self


class BudgetState(BaseModel):
    """Consumed/remaining budget state carried through immutable Results.

    Budget advances only through immutable Step Results: the next request's
    ``remaining`` is the prior Result's remaining, never recomputed from
    process memory.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    consumed: dict[str, int] = Field(default_factory=dict)
    remaining: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> BudgetState:
        for label, value in {**self.consumed, **self.remaining}.items():
            if value < 0:
                raise ValueError(
                    f"budget value for {label!r} cannot be negative"
                )
        return self


class EvaluationIntent(BaseModel):
    """Immutable measurement request. Carries no result or score.

    Fixes the candidate identity, the exact target Eval Config ref + Identity
    Hash, the Evaluation Context role and policy, the purpose, and the run/step
    correlation. Evaluation under any other Eval Config does not resolve it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_id: StrictStr
    candidate_id: StrictStr
    # Exact target Eval Config reference + Identity Hash.
    target_eval_config_ref: StrictStr
    target_eval_config_hash: StrictStr
    # Evaluation Context role/policy the resolution must use.
    context_role: EvaluationRole
    context_policy_ref: StrictStr | None = None
    purpose: StrictStr
    # Run/step correlation.
    run_id: StrictStr
    step_index: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> EvaluationIntent:
        if not self.intent_id:
            raise ValueError("intent_id must be non-empty")
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        require_full_hash(
            self.target_eval_config_hash, field="target_eval_config_hash"
        )
        if not self.purpose:
            raise ValueError("purpose must be non-empty")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        # No result/score field exists on this schema by construction; the
        # forbidden-extra config makes any attempt to carry one fail.
        return self


class OptimizationRun(BaseModel):
    """The umbrella Optimization Run identity and frozen run-wide inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    optimizer_config_hash: StrictStr
    mode: StepMode
    output_contract: OutputContract
    # Serialized Tool Configs for tool-using runs (∅ for proposal-only/pure).
    tool_configs: tuple[ToolConfig, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> OptimizationRun:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        require_full_hash(
            self.optimizer_config_hash, field="optimizer_config_hash"
        )
        if self.mode is not StepMode.TOOL_USING and self.tool_configs:
            raise ValueError(
                "only tool-using runs may carry Tool Configs"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "optimizer_config_hash": self.optimizer_config_hash,
            "mode": self.mode.value,
            "output_contract": self.output_contract.model_dump(mode="json"),
            "tool_configs": [
                cfg.identity_hash() for cfg in self.tool_configs
            ],
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=OPTIMIZATION_RUN_SCHEMA,
            schema_version=OPTIMIZATION_RUN_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class OptimizationStepRequest(BaseModel):
    """Immutable input to one Optimization Step.

    Rejects any runtime handle/client/closure: every field is a strict-JSON
    pydantic value, so a Runtime Tool Handle (a plain non-JSON object) cannot
    be assigned. Tools are carried only as serialized :class:`ToolConfig`s,
    each with its typed Tool Definition ref + Identity Hash.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Run / step / config identities.
    run_id: StrictStr
    step_id: StrictStr
    optimizer_config_hash: StrictStr

    # Step kind + ordered index.
    mode: StepMode
    kind: StepKind
    kind_label: StrictStr | None = None
    step_index: StrictInt

    # Immutable prior Step Result / state / history references (Content Hash).
    prior_step_result_ref: TypedRef | None = None
    prior_state_ref: TypedRef | None = None
    prior_history_ref: TypedRef | None = None

    # Candidate or pool inputs.
    candidates: tuple[Candidate, ...] = ()
    pools: dict[str, Any] = Field(default_factory=dict)

    # Algorithm hyperparameters and budget state.
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetState = Field(default_factory=BudgetState)

    # Output contract.
    output_contract: OutputContract

    # Optional serialized Tool Configs (tool-using only).
    tool_configs: tuple[ToolConfig, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepRequest:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.step_id:
            raise ValueError("step_id must be non-empty")
        require_full_hash(
            self.optimizer_config_hash, field="optimizer_config_hash"
        )
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")

        # Reject any mutable process object smuggled into a free-form field:
        # pools/hyperparameters must be strict finite JSON.
        reject_non_json(self.pools, field="pools")
        reject_non_json(self.hyperparameters, field="hyperparameters")

        # Ordering rule: the initial step (index 0) carries no prior refs; a
        # noninitial step MUST reference the exact prior Step Result.
        if self.step_index == 0:
            if self.prior_step_result_ref is not None:
                raise ValueError(
                    "the initial Step Request (index 0) carries no prior "
                    "Step Result reference"
                )
        elif self.prior_step_result_ref is None:
            raise ValueError(
                "a noninitial Step Request must reference the exact prior "
                "Step Result"
            )

        # Mode/kind coherence.
        if self.mode is StepMode.PURE and self.kind is not StepKind.IDENTITY:
            raise ValueError("a pure step must be the identity kind")
        if self.mode is StepMode.TOOL_USING and not self.tool_configs:
            raise ValueError(
                "a tool-using Step Request must carry at least one Tool Config"
            )
        if self.mode is not StepMode.TOOL_USING and self.tool_configs:
            raise ValueError(
                "only a tool-using Step Request may carry Tool Configs"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class IntentResolution(BaseModel):
    """A resolved Evaluation Intent + its evaluation-evidence references.

    Whetstone evaluates the Intent OUTSIDE the optimizer invocation, under the
    exact target Eval Config; this pairs the Intent with the resulting
    measurement/aggregate evidence refs. It carries no score on the Intent
    itself — the evidence lives behind the refs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: EvaluationIntent
    evaluation_evidence_refs: tuple[TypedRef, ...]
    # The Eval Config the resolution actually used — MUST equal the Intent's
    # target, proving the Intent resolved only under its exact target.
    resolved_eval_config_hash: StrictStr
    # Optional produced internal Reward reference/citation content hash.
    reward_content_hash: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> IntentResolution:
        if not self.evaluation_evidence_refs:
            raise ValueError(
                "a resolved Intent must reference its evaluation evidence"
            )
        target_hash = self.intent.target_eval_config_hash
        if self.resolved_eval_config_hash != target_hash:
            raise ValueError(
                "an Evaluation Intent resolves only under its exact target "
                "Eval Config identity; resolved "
                f"{self.resolved_eval_config_hash!r} != target "
                f"{self.intent.target_eval_config_hash!r}"
            )
        return self


class ToolEvidence(BaseModel):
    """A Tool Result reference + its Tool Call Store Entry.

    Every Tool Result used by a tool-using Step is referenced as evidence by
    that Step Result, together with the authoritative Tool Call Store Entry
    that records its acceptance / completion / refusal. A *completed* Entry's
    terminal Tool Result ref MUST equal the referenced Tool Result; a *refused*
    Entry carries no completed Tool Result ref but still references the
    persisted refusal Tool Result as evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_result_ref: TypedRef
    store_entry: ToolCallStoreEntry

    @model_validator(mode="after")
    def _validate(self) -> ToolEvidence:
        entry_result_ref = self.store_entry.tool_result_ref
        if entry_result_ref is not None:
            # Completed entry: its terminal ref must match the evidence ref.
            if entry_result_ref != self.tool_result_ref:
                raise ValueError(
                    "Tool evidence Tool Result ref must match the Store "
                    "Entry's terminal Tool Result ref"
                )
        elif self.store_entry.refusal is None:
            # An accepted-but-not-completed entry is not terminal evidence.
            raise ValueError(
                "Tool evidence references a completed or refused Store Entry, "
                "not an accepted-only one"
            )
        return self


class OptimizationStepResult(BaseModel):
    """Immutable terminal result for one Optimization Step.

    Carries the request ref + Content Hash; proposed/accepted candidates;
    resolved intents + evaluation evidence OR Tool Results + Store Entries; a
    state/history delta or snapshot ref; budgets consumed/remaining; and
    exactly one status. Never updated in place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    step_id: StrictStr
    step_index: StrictInt

    # Request Object Reference + Content Hash.
    request_ref: TypedRef

    # Proposed and accepted candidates.
    proposed_candidates: tuple[Candidate, ...] = ()
    accepted_candidates: tuple[Candidate, ...] = ()

    # Proposal-only evidence: resolved Evaluation Intents + evidence refs.
    resolved_intents: tuple[IntentResolution, ...] = ()

    # Tool-using evidence: every Tool Result ref + Tool Call Store Entry used.
    tool_evidence: tuple[ToolEvidence, ...] = ()

    # State/history delta or snapshot reference.
    state_ref: TypedRef | None = None
    history_ref: TypedRef | None = None

    # Budget accounting.
    budget: BudgetState = Field(default_factory=BudgetState)

    # Exactly one status.
    status: StepStatus

    # Record-local provenance.
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> OptimizationStepResult:
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        if self.request_ref.schema_name != STEP_REQUEST_SCHEMA:
            raise ValueError(
                "request_ref must be a typed Step Request reference"
            )
        # Tool-using and proposal-only evidence are mutually exclusive by
        # dispatch; a pure step carries neither.
        if self.resolved_intents and self.tool_evidence:
            raise ValueError(
                "a Step Result carries resolved Intents OR Tool evidence, "
                "never both"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OptimizationProposal(BaseModel):
    """An ordered terminal proposal: base + mutation-surface payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: StrictStr
    base_ref: StrictStr
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> OptimizationProposal:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.base_ref:
            raise ValueError("base_ref must be non-empty")
        return self


class OptimizationResult(BaseModel):
    """Immutable terminal run output. Makes no official-evaluation claim.

    Carries the ordered proposals, every ordered Step Result reference,
    record-local provenance, and cost. Official evaluation follows downstream;
    this record never substitutes optimizer-internal evaluation for it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    proposals: tuple[OptimizationProposal, ...]
    step_result_refs: tuple[TypedRef, ...]
    status: StepStatus
    # Advisory run cost (control tokens / wall clock / usage), never a score.
    cost: dict[str, Any] = Field(default_factory=dict)
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> OptimizationResult:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.status is StepStatus.FAILED:
            # A failed run makes no proposal claim (cardinality/contract
            # failure); it blocks official materialization downstream.
            return self
        if not self.step_result_refs:
            raise ValueError(
                "a completed Optimization Result references every ordered "
                "Step Result"
            )
        for ref in self.step_result_refs:
            if ref.schema_name != STEP_RESULT_SCHEMA:
                raise ValueError(
                    "step_result_refs must be typed Step Result references"
                )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def step_request_reference(request: OptimizationStepRequest) -> TypedRef:
    """The typed Object Reference (by Content Hash) for a Step Request."""
    return typed_ref_for_record(
        STEP_REQUEST_SCHEMA, request.record_content()
    )


def step_result_reference(result: OptimizationStepResult) -> TypedRef:
    """The typed Object Reference (by Content Hash) for a Step Result."""
    return typed_ref_for_record(STEP_RESULT_SCHEMA, result.record_content())
