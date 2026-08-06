from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    ImmutableJsonObject,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.optimization.contracts import OptimizationRunRef
from whetstone.optimization.miprov2.bootstrap import (
    BootstrapAttemptPlan,
    BootstrapCompilerState,
    BootstrapFoldEvidence,
    BootstrapRolloutResult,
    FewshotCandidatePlan,
    FewshotCandidatePlanningResult,
    FewshotSeedKind,
    create_fewshot_candidate_plans,
    fold_bootstrap_result,
    initial_compiler_state,
    materialize_bootstrap_demo_set,
    materialize_labels_only_demo_set,
    materialize_reset_demo_set,
    next_bootstrap_attempt,
)
from whetstone.optimization.miprov2.control import (
    MIPROV2_CANDIDATE_RENDERER_VERSION,
    MIPROV2_RESULT_SCHEMA,
    MIPROV2_RESULT_SCHEMA_VERSION,
    MIPROV2_STATE_SCHEMA,
    MIPROV2_STATE_SCHEMA_VERSION,
    Miprov2Control,
    _deep_revalidate_model,
)
from whetstone.optimization.miprov2.demo import (
    ComponentDemoSequence,
    ComponentDemoSet,
    LabeledTaskDemo,
    proposal_demo_context,
    study_demo_context,
)
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvaluationExecutionPolicy,
)
from whetstone.optimization.miprov2.evidence import (
    Miprov2ResolvedEvaluation,
)
from whetstone.optimization.miprov2.proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
    Miprov2ProposalRequest,
    Miprov2ProposalResponse,
    Miprov2ProposalState,
    fold_proposal_response,
    plan_next_proposal_request,
    proposal_candidates_from_demo_sets,
    start_miprov2_proposal,
)
from whetstone.optimization.miprov2.render import candidate_from_components
from whetstone.optimization.miprov2.rng import (
    Miprov2DurableBindings,
    Miprov2RngCheckpoint,
)
from whetstone.optimization.miprov2.study import (
    MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
    MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
    Miprov2CandidateAssemblyBinding,
    Miprov2CandidateRendering,
    Miprov2ComponentSelection,
    Miprov2EvaluationObservation,
    Miprov2ParameterSpace,
    Miprov2Study,
    Miprov2StudySchedule,
    PromotionCandidate,
    StudySuggestion,
    StudyTranscript,
    TrialParams,
)

MIPROV2_CANDIDATE_RENDERER_SCHEMA = "whetstone.miprov2_candidate_rendering"
MIPROV2_CANDIDATE_RENDERER_SCHEMA_VERSION = 1
MIPROV2_RUNTIME_SCHEMA = MIPROV2_STATE_SCHEMA
MIPROV2_RUNTIME_SCHEMA_VERSION = MIPROV2_STATE_SCHEMA_VERSION

Miprov2Phase = Literal[
    "bootstrap",
    "proposal",
    "baseline",
    "sample",
    "promotion",
    "complete",
    "failed",
]


def _validate_component_field_order(
    value: Mapping[StrictStr, Sequence[StrictStr]],
) -> Mapping[StrictStr, Sequence[StrictStr]]:
    for component_id, fields in value.items():
        if not component_id:
            raise ValueError("component field-order ID must not be empty")
        ordered_fields = tuple(fields)
        if not ordered_fields or any(not field for field in ordered_fields):
            raise ValueError("component field-order names must not be empty")
        if len(ordered_fields) != len(set(ordered_fields)):
            raise ValueError("component field-order names must be unique")
    return value


type Miprov2ComponentFieldOrder = Annotated[
    Mapping[StrictStr, Sequence[StrictStr]],
    AfterValidator(_validate_component_field_order),
]

_COMPONENT_FIELD_ORDER_ADAPTER = TypeAdapter(Miprov2ComponentFieldOrder)


@dataclass(frozen=True)
class Miprov2ReplayProjection:
    """All redundant runtime fields derived from immutable inputs/evidence."""

    rng_checkpoint: Miprov2RngCheckpoint
    bootstrap: tuple[
        int,
        BootstrapCompilerState | None,
        tuple[ComponentDemoSet, ...],
        BootstrapAttemptPlan | None,
    ]
    completed_effects: tuple[Miprov2CompletedEffect, ...]
    fully_evaluated_candidates: tuple[CandidateRef, ...]
    phase: Miprov2Phase
    pending_evaluation_spec: Miprov2EvaluationSpec | None
    pending_eval_binding_request: Miprov2EvalConfigBindingRequest | None
    pending_evaluation: Miprov2EvaluationEffect | None
    instruction_pools: tuple[tuple[str, ...], ...]
    study_demo_candidates: tuple[ComponentDemoSet, ...] | None
    winner: CandidateRef | None
    winner_score: float | None


Miprov2EffectKind = Literal[
    "eval_config_binding",
    "bootstrap_rollout",
    "proposal_model",
    "baseline_evaluation",
    "sample_evaluation",
    "promotion_evaluation",
]


def _instruction_identity(instruction: str) -> str:
    return compute_identity_hash(
        schema="whetstone.miprov2_instruction",
        schema_version=1,
        payload={"instruction": instruction},
    )


def _component_demo_projection(
    demo_set: ComponentDemoSet,
    component_id: str,
) -> ComponentDemoSet:
    """Return the exact predictor-specific categorical demo value."""

    return ComponentDemoSet(
        candidate_seed=demo_set.candidate_seed,
        components=(
            ComponentDemoSequence(
                component_id=component_id,
                demos=demo_set.demos_for(component_id),
            ),
        ),
    )


class Miprov2EvaluationEffect(BaseModel):
    """One identity-bound evaluation to be executed by the durable harness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    ordinal: StrictInt
    purpose: Literal[
        "miprov2_baseline",
        "miprov2_sample",
        "miprov2_promotion",
    ]
    candidate: Candidate
    categorical_combination_identity_hash: StrictStr
    task_batch_identities: tuple[StrictStr, ...]
    eval_config: EvalConfigRef
    execution_policy: Miprov2EvaluationExecutionPolicy
    reward_policy_hash: StrictStr
    suggestion: StudySuggestion | None = None
    promotion_candidate: PromotionCandidate | None = None
    candidate_assembly: Miprov2CandidateAssemblyBinding | None = None

    @model_validator(mode="after")
    def _validate_effect(self) -> Miprov2EvaluationEffect:
        if not self.run_id or self.ordinal < 0:
            raise ValueError("evaluation run and ordinal are required")
        require_full_hash(
            self.categorical_combination_identity_hash,
            field="categorical_combination_identity_hash",
        )
        require_full_hash(
            self.reward_policy_hash,
            field="reward_policy_hash",
        )
        if not self.task_batch_identities:
            raise ValueError("evaluation task batch cannot be empty")
        for task_identity in self.task_batch_identities:
            require_full_hash(task_identity, field="task_batch_identity")
        if self.purpose == "miprov2_sample" and self.suggestion is None:
            raise ValueError(
                "sample evaluation requires its Optuna suggestion"
            )
        if self.purpose != "miprov2_sample" and self.suggestion is not None:
            raise ValueError("only sample evaluation carries a suggestion")
        if (self.purpose == "miprov2_promotion") != (
            self.promotion_candidate is not None
        ):
            raise ValueError(
                "promotion evaluation requires its selected promotion"
            )
        if (self.purpose == "miprov2_baseline") != (
            self.candidate_assembly is None
        ):
            raise ValueError(
                "only non-baseline evaluation requires candidate assembly"
            )
        return self

    def identity_hash(self) -> str:
        # Eval Config derivation is itself keyed by this identity, so the
        # identity is exactly the pre-derivation evaluation specification.
        return Miprov2EvaluationSpec(
            run_id=self.run_id,
            ordinal=self.ordinal,
            purpose=self.purpose,
            candidate=self.candidate,
            categorical_combination_identity_hash=(
                self.categorical_combination_identity_hash
            ),
            task_batch_identities=self.task_batch_identities,
            suggestion=self.suggestion,
            promotion_candidate=self.promotion_candidate,
            candidate_assembly=self.candidate_assembly,
            execution_policy=self.execution_policy,
        ).identity_hash()


class Miprov2EvaluationSpec(BaseModel):
    """Evaluation identity before its exact subset Eval Config is derived."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    ordinal: StrictInt
    purpose: Literal[
        "miprov2_baseline",
        "miprov2_sample",
        "miprov2_promotion",
    ]
    candidate: Candidate
    categorical_combination_identity_hash: StrictStr
    task_batch_identities: tuple[StrictStr, ...]
    execution_policy: Miprov2EvaluationExecutionPolicy
    suggestion: StudySuggestion | None = None
    promotion_candidate: PromotionCandidate | None = None
    candidate_assembly: Miprov2CandidateAssemblyBinding | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> Miprov2EvaluationSpec:
        if not self.run_id or self.ordinal < 0:
            raise ValueError("evaluation spec run and ordinal are required")
        require_full_hash(
            self.categorical_combination_identity_hash,
            field="categorical_combination_identity_hash",
        )
        if not self.task_batch_identities:
            raise ValueError("evaluation spec task batch cannot be empty")
        if self.purpose == "miprov2_sample" and self.suggestion is None:
            raise ValueError("sample evaluation requires its suggestion")
        if self.purpose != "miprov2_sample" and self.suggestion is not None:
            raise ValueError("only sample evaluation carries a suggestion")
        if (self.purpose == "miprov2_promotion") != (
            self.promotion_candidate is not None
        ):
            raise ValueError("promotion evaluation requires selection")
        if (self.purpose == "miprov2_baseline") != (
            self.candidate_assembly is None
        ):
            raise ValueError(
                "only non-baseline evaluation requires candidate assembly"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are an explicit wire contract.
        return {
            "run_id": self.run_id,
            "ordinal": self.ordinal,
            "purpose": self.purpose,
            "candidate": self.candidate.model_dump(mode="json"),
            "categorical_combination_identity_hash": (
                self.categorical_combination_identity_hash
            ),
            "task_batch_identities": list(self.task_batch_identities),
            "execution_policy": self.execution_policy.model_dump(mode="json"),
            "suggestion": (
                None
                if self.suggestion is None
                else self.suggestion.model_dump(mode="json")
            ),
            "promotion_candidate": (
                None
                if self.promotion_candidate is None
                else self.promotion_candidate.model_dump(mode="json")
            ),
            "candidate_assembly": (
                None
                if self.candidate_assembly is None
                else self.candidate_assembly.model_dump(mode="json")
            ),
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="whetstone.miprov2_evaluation_spec",
            schema_version=1,
            payload=self.identity_payload(),
        )


class Miprov2EffectBudget(BaseModel):
    """Hard effect ceilings checked immediately before every external call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_rollouts: StrictInt
    proposal_calls: StrictInt
    evaluations: StrictInt
    task_rows: StrictInt

    @model_validator(mode="after")
    def _validate_budget(self) -> Miprov2EffectBudget:
        if (
            min(
                self.bootstrap_rollouts,
                self.proposal_calls,
                self.evaluations,
                self.task_rows,
            )
            < 0
        ):
            raise ValueError("MIPROv2 effect budgets cannot be negative")
        return self


class Miprov2CompletedEffect(BaseModel):
    """Append-only proof that one planned external effect was folded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["bootstrap_rollouts", "proposal_calls", "evaluations"]
    identity_hash: StrictStr
    task_rows: StrictInt = 0

    @model_validator(mode="after")
    def _validate_effect(self) -> Miprov2CompletedEffect:
        require_full_hash(self.identity_hash, field="effect_identity_hash")
        if self.task_rows < 0:
            raise ValueError("completed effect task_rows cannot be negative")
        if self.kind == "proposal_calls" and self.task_rows:
            raise ValueError("proposal calls do not account task rows")
        return self


class Miprov2ScoreObservation(BaseModel):
    """One DSPy ``score_data`` row in execution order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["baseline", "sample", "promotion"]
    trial_number: StrictInt
    candidate: CandidateRef
    score: float
    full_eval: bool
    cumulative_evaluation_calls: StrictInt

    @model_validator(mode="after")
    def _validate_observation(self) -> Miprov2ScoreObservation:
        if self.trial_number < 0 or self.cumulative_evaluation_calls <= 0:
            raise ValueError("score observation ordinals must be valid")
        if not math.isfinite(self.score):
            raise ValueError("score observation must be finite")
        return self


class Miprov2TrialLog(BaseModel):
    """Reference-equivalent ordered trial log without filesystem paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    log_key: StrictInt
    source: Literal["baseline", "sample", "promotion"]
    optuna_trial_number: StrictInt
    params: TrialParams | None = None
    candidate: CandidateRef
    minibatch_score: float | None = None
    full_score: float | None = None
    cumulative_evaluation_calls: StrictInt

    @model_validator(mode="after")
    def _validate_log(self) -> Miprov2TrialLog:
        if (
            self.log_key <= 0
            or self.optuna_trial_number < 0
            or self.cumulative_evaluation_calls <= 0
        ):
            raise ValueError("trial log ordinals must be valid")
        if (self.minibatch_score is None) == (self.full_score is None):
            raise ValueError("trial log requires exactly one score kind")
        if (self.params is not None) != (self.source == "sample"):
            raise ValueError(
                "only sampled trial logs carry categorical parameters"
            )
        score = (
            self.minibatch_score
            if self.minibatch_score is not None
            else self.full_score
        )
        assert score is not None
        if not math.isfinite(score):
            raise ValueError("trial log score must be finite")
        return self


class Miprov2ScoredCandidate(BaseModel):
    """One stable score-sorted DSPy candidate-program projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateRef
    score: float
    source: Literal["baseline", "sample", "promotion"]
    trial_number: StrictInt

    @model_validator(mode="after")
    def _validate_candidate(self) -> Miprov2ScoredCandidate:
        if self.trial_number < 0 or not math.isfinite(self.score):
            raise ValueError("scored candidate must have valid score metadata")
        return self


class Miprov2TerminalStats(BaseModel):
    """Detailed DSPy-style run artifacts retained only with track_stats."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_transcript: StudyTranscript
    fully_evaluated_candidates: tuple[CandidateRef, ...]
    completed_effects: tuple[Miprov2CompletedEffect, ...]
    instruction_pools: tuple[tuple[StrictStr, ...], ...]
    demo_candidates: tuple[ComponentDemoSet, ...] | None
    effect_counts: ImmutableJsonObject
    trial_logs: tuple[Miprov2TrialLog, ...]
    cumulative_evaluation_calls: StrictInt
    score_data: tuple[Miprov2ScoreObservation, ...]
    mb_candidate_programs: tuple[Miprov2ScoredCandidate, ...]
    candidate_programs: tuple[Miprov2ScoredCandidate, ...]
    prompt_model_total_calls: Literal[0] = 0
    total_calls: Literal[0] = 0

    def model_post_init(self, _context: Any) -> None:
        if not isinstance(self.effect_counts, ImmutableJsonObject):
            object.__setattr__(
                self,
                "effect_counts",
                ImmutableJsonObject(self.effect_counts),
            )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if deep:
            payload = self.model_dump(mode="json")
            payload.update(update or {})
            return type(self).model_validate(payload)
        copied = super().model_copy(update=update, deep=deep)
        copied.model_post_init(None)
        return copied

    @model_validator(mode="after")
    def _validate_stats(self) -> Miprov2TerminalStats:
        if not self.score_data or not self.trial_logs:
            raise ValueError("tracked stats require baseline observations")
        if self.score_data[0].source != "baseline":
            raise ValueError("score_data must begin with baseline")
        if self.cumulative_evaluation_calls != (
            self.score_data[-1].cumulative_evaluation_calls
        ):
            raise ValueError("evaluation call total conflicts with score_data")
        expected_full = tuple(
            sorted(
                (
                    Miprov2ScoredCandidate(
                        candidate=item.candidate,
                        score=item.score,
                        source=item.source,
                        trial_number=item.trial_number,
                    )
                    for item in self.score_data
                    if item.full_eval
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        expected_mb = tuple(
            sorted(
                (
                    Miprov2ScoredCandidate(
                        candidate=item.candidate,
                        score=item.score,
                        source=item.source,
                        trial_number=item.trial_number,
                    )
                    for item in self.score_data
                    if not item.full_eval
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        if (
            self.candidate_programs != expected_full
            or self.mb_candidate_programs != expected_mb
        ):
            raise ValueError("candidate program collections are not canonical")
        return self


class Miprov2TerminalResult(BaseModel):
    """Typed terminal projection matching the public track_stats control."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: Literal["whetstone.miprov2_result"] = MIPROV2_RESULT_SCHEMA
    schema_version: Literal[1] = MIPROV2_RESULT_SCHEMA_VERSION
    winner: CandidateRef
    winner_score: float
    track_stats: bool
    stats: Miprov2TerminalStats | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Miprov2TerminalResult:
        if not math.isfinite(self.winner_score):
            raise ValueError("terminal winner score must be finite")
        if self.track_stats != (self.stats is not None):
            raise ValueError(
                "terminal detailed stats must follow track_stats exactly"
            )
        return self


class Miprov2PendingSample(BaseModel):
    """Sample evidence retained while a promotion runs before ``tell``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suggestion: StudySuggestion
    score: float
    evaluation: Miprov2EvaluationObservation
    candidate_identity_hash: StrictStr
    candidate_assembly: Miprov2CandidateAssemblyBinding

    @model_validator(mode="after")
    def _validate_pending(self) -> Miprov2PendingSample:
        require_full_hash(
            self.candidate_identity_hash,
            field="candidate_identity_hash",
        )
        if not math.isfinite(self.score):
            raise ValueError("pending sample score must be finite")
        return self


class Miprov2State(BaseModel):
    """Complete immutable state of one MIPROv2 run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: Literal["whetstone.miprov2_runtime"] = MIPROV2_RUNTIME_SCHEMA
    schema_version: Literal[4] = MIPROV2_RUNTIME_SCHEMA_VERSION
    run_id: StrictStr
    run: OptimizationRunRef
    control: Miprov2Control
    bindings: Miprov2DurableBindings
    rng_checkpoint: Miprov2RngCheckpoint
    labeled_trainset: tuple[LabeledTaskDemo, ...]
    proposal_components: tuple[Miprov2PromptComponent, ...]
    proposal_trainset: tuple[Miprov2DatasetExample, ...]
    component_field_order: ImmutableJsonObject
    input_data_identity_hash: StrictStr
    budget: Miprov2EffectBudget

    phase: Miprov2Phase = "bootstrap"
    bootstrap_plans: tuple[FewshotCandidatePlan, ...]
    bootstrap_plan_index: StrictInt = 0
    bootstrap_state: BootstrapCompilerState | None = None
    demo_candidates: tuple[ComponentDemoSet, ...] = ()
    bootstrap_evidence: tuple[BootstrapFoldEvidence, ...] = ()
    proposal_state: Miprov2ProposalState | None = None
    instruction_pools: tuple[tuple[StrictStr, ...], ...] = ()
    study_demo_candidates: tuple[ComponentDemoSet, ...] | None = None
    study_transcript: StudyTranscript | None = None
    pending_bootstrap: BootstrapAttemptPlan | None = None
    pending_bootstrap_candidate: CandidateRef | None = None
    pending_proposal: Miprov2ProposalRequest | None = None
    pending_evaluation_spec: Miprov2EvaluationSpec | None = None
    pending_eval_binding_request: Miprov2EvalConfigBindingRequest | None = None
    resolved_eval_binding: Miprov2EvalConfigBinding | None = None
    pending_evaluation: Miprov2EvaluationEffect | None = None
    pending_sample: Miprov2PendingSample | None = None
    fully_evaluated_candidates: tuple[CandidateRef, ...] = ()
    accepted_candidate: Candidate | None = None
    accepted_candidate_ref: CandidateRef | None = None
    terminal_result: Miprov2TerminalResult | None = None
    completed_effects: tuple[Miprov2CompletedEffect, ...] = ()
    failure: StrictStr | None = None

    @field_validator("component_field_order", mode="before")
    @classmethod
    def _normalize_component_field_order(
        cls,
        value: Any,
    ) -> ImmutableJsonObject:
        if isinstance(value, ImmutableJsonObject):
            value = value.to_json()
        elif isinstance(value, Mapping):
            value = dict(value.items())
        validated = _COMPONENT_FIELD_ORDER_ADAPTER.validate_python(
            value,
            strict=True,
        )
        return ImmutableJsonObject(
            {
                component_id: list(fields)
                for component_id, fields in validated.items()
            }
        )

    def model_post_init(self, _context: Any) -> None:
        if not isinstance(self.component_field_order, ImmutableJsonObject):
            object.__setattr__(
                self,
                "component_field_order",
                type(self)._normalize_component_field_order(
                    self.component_field_order
                ),
            )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        replacements = update or {}
        for field_name in replacements:
            field = type(self).model_fields.get(field_name)
            if field is None:
                raise ValueError(f"unknown MIPROv2 state field {field_name!r}")
        values: dict[str, Any] = {}
        for field_name in type(self).model_fields:
            value = replacements.get(field_name, getattr(self, field_name))
            if field_name == "component_field_order":
                value = type(self)._normalize_component_field_order(value)
            values[field_name] = value
        copied = _deep_revalidate_model(type(self), values)
        object.__setattr__(
            copied,
            "__pydantic_fields_set__",
            self.__pydantic_fields_set__ | replacements.keys(),
        )
        return copied

    @classmethod
    def model_construct(
        cls,
        _fields_set: set[str] | None = None,
        **values: Any,
    ) -> Self:
        """Validate construction and defensively freeze all containers."""

        del _fields_set
        if "component_field_order" in values:
            values["component_field_order"] = (
                cls._normalize_component_field_order(
                    values["component_field_order"]
                )
            )
        return _deep_revalidate_model(cls, values)

    @model_validator(mode="after")
    def _validate_state(self) -> Miprov2State:
        if not self.run_id:
            raise ValueError("MIPROv2 run_id cannot be empty")
        if self.run.record.run_id != self.run_id:
            raise ValueError("MIPROv2 run_id conflicts with the exact run")
        if self.run.record.optimizer_config != self.control.reference():
            raise ValueError("MIPROv2 run conflicts with resolved control")
        if (
            self.run.record.template_render_contract
            != self.control.template_render_contract
        ):
            raise ValueError(
                "MIPROv2 run render contract conflicts with control"
            )
        if self.input_data_identity_hash != _input_data_identity(
            control=self.control,
            labeled_trainset=self.labeled_trainset,
            proposal_components=self.proposal_components,
            proposal_trainset=self.proposal_trainset,
            component_field_order=self.component_field_order,
        ):
            raise ValueError(
                "runtime inputs conflict with their exact content binding"
            )
        if self.bindings.control_identity_hash != self.control.identity_hash():
            raise ValueError("runtime bindings do not match resolved control")
        expected_bindings = (
            self.control.prompt_model.identity_hash(),
            self.control.task_model_identity_hash,
            self.control.provider_execution_policy_hash,
            self.control.prompt_adapter_identity_hash,
            self.control.base_candidate.identity_hash,
            self.control.teacher_candidate.identity_hash,
        )
        actual_bindings = (
            self.bindings.prompt_route_identity_hash,
            self.bindings.task_route_identity_hash,
            self.bindings.execution_policy_identity_hash,
            self.bindings.prompt_adapter_identity_hash,
            self.bindings.base_candidate_identity_hash,
            self.bindings.teacher_candidate_identity_hash,
        )
        if actual_bindings != expected_bindings:
            raise ValueError("runtime routes conflict with resolved control")
        component_ids = self.control.component_ids
        if (
            tuple(
                component.component_id
                for component in self.proposal_components
            )
            != component_ids
        ):
            raise ValueError(
                "proposal components conflict with program layout"
            )
        if any(
            component.template_render_contract
            != self.control.template_render_contract
            for component in self.proposal_components
        ):
            raise ValueError(
                "proposal component render contracts conflict with the run"
            )
        if set(self.component_field_order) != set(component_ids):
            raise ValueError("component field order conflicts with layout")
        expected_tasks = self.control.trainset_task_identities
        if (
            tuple(item.source_task_identity for item in self.labeled_trainset)
            != expected_tasks
        ):
            raise ValueError(
                "labeled trainset conflicts with resolved control"
            )
        if (
            tuple(item.task_identity for item in self.proposal_trainset)
            != expected_tasks
        ):
            raise ValueError(
                "proposal trainset conflicts with resolved control"
            )
        if not 0 <= self.bootstrap_plan_index <= len(self.bootstrap_plans):
            raise ValueError("bootstrap plan cursor is outside its plan set")
        pending = (
            self.pending_bootstrap,
            self.pending_proposal,
            self.pending_evaluation,
        )
        if sum(item is not None for item in pending) > 1:
            raise ValueError("runtime can expose at most one external effect")
        if (self.pending_bootstrap is None) != (
            self.pending_bootstrap_candidate is None
        ):
            raise ValueError(
                "bootstrap attempt and materialized teacher must be paired"
            )
        if (
            self.pending_bootstrap is not None
            and self.pending_bootstrap_candidate is not None
        ):
            plan = self.bootstrap_plans[self.bootstrap_plan_index]
            expected_teacher = _materialize_bootstrap_teacher(
                state=self,
                plan=plan,
                attempt=self.pending_bootstrap,
            )
            if self.pending_bootstrap_candidate != expected_teacher:
                raise ValueError("pending bootstrap teacher is not canonical")
        expected_planning = create_fewshot_candidate_plans(
            bindings=self.bindings,
            component_ids=self.control.component_ids,
            trainset_task_identities=(self.control.trainset_task_identities),
            num_candidate_sets=self.control.num_fewshot_candidates,
            max_bootstrapped_demos=self.control.max_bootstrapped_demos,
            max_labeled_demos=self.control.max_labeled_demos,
            max_errors=self.control.max_errors,
            metric_threshold=self.control.metric_threshold,
            explicit_teacher=(
                self.control.teacher_candidate.identity_hash
                != self.control.base_candidate.identity_hash
            ),
            teacher_compiled=_teacher_compiled(self.control),
            rng_checkpoint=_initial_bootstrap_rng(self.control),
            zeroshot_opt=self.control.zeroshot_opt,
        )
        if self.bootstrap_plans != expected_planning.plans:
            raise ValueError(
                "bootstrap plans are not the canonical control replay"
            )
        replay = replay_miprov2_state(self, expected_planning)
        if self.rng_checkpoint != replay.rng_checkpoint:
            raise ValueError(
                "runtime RNG checkpoint is not the canonical evidence replay"
            )
        if (
            self.pending_evaluation_spec is not None
            and self.pending_evaluation_spec != replay.pending_evaluation_spec
        ):
            raise ValueError(
                "pending evaluation spec is not the canonical study replay"
            )
        if (
            self.bootstrap_plan_index,
            self.bootstrap_state,
            self.demo_candidates,
            self.pending_bootstrap,
        ) != replay.bootstrap:
            raise ValueError(
                "bootstrap phase state is not the canonical evidence replay"
            )
        completed_hashes = tuple(
            effect.identity_hash for effect in self.completed_effects
        )
        if len(set(completed_hashes)) != len(completed_hashes):
            raise ValueError("completed effect identities must be unique")
        for effect in self.completed_effects:
            if effect.kind == "bootstrap_rollouts" and effect.task_rows != 1:
                raise ValueError("bootstrap ledger entries require one task")
            if effect.kind == "evaluations" and effect.task_rows <= 0:
                raise ValueError("evaluation ledger entries require tasks")
        if self.completed_effects != replay.completed_effects:
            raise ValueError(
                "completed-effect ledger is not the canonical evidence replay"
            )
        if (
            self.fully_evaluated_candidates
            != replay.fully_evaluated_candidates
        ):
            raise ValueError(
                "fully evaluated candidates are not the canonical study replay"
            )
        if self.phase != replay.phase:
            raise ValueError(
                "runtime phase is not the canonical evidence replay"
            )
        if self.pending_proposal is not None:
            if self.proposal_state is None:
                raise ValueError(
                    "pending proposal requires canonical proposal state"
                )
            expected_proposal = plan_next_proposal_request(
                self.proposal_state
            ).request
            if self.pending_proposal != expected_proposal:
                raise ValueError(
                    "pending proposal is not the canonical proposal replay"
                )
        if (
            self.proposal_state is not None
            and self.proposal_state.optimization_run_identity_hash
            != self.run.identity_hash
        ):
            raise ValueError(
                "proposal state belongs to another optimization run"
            )
        if self.instruction_pools != replay.instruction_pools:
            raise ValueError(
                "instruction pools are not the canonical proposal replay"
            )
        if self.study_demo_candidates != replay.study_demo_candidates:
            raise ValueError(
                "study demos are not the canonical bootstrap projection"
            )
        for label in (
            "bootstrap_rollouts",
            "proposal_calls",
            "evaluations",
            "task_rows",
        ):
            if self.effect_counts[label] > getattr(self.budget, label):
                raise ValueError("completed effects exceed the durable budget")
        binding_subjects = (
            self.pending_bootstrap,
            self.pending_evaluation_spec,
        )
        if (
            self.pending_eval_binding_request is not None
            and sum(item is not None for item in binding_subjects) != 1
        ):
            raise ValueError(
                "pending Eval Config request requires exactly one subject"
            )
        if self.pending_eval_binding_request is not None:
            if (
                self.pending_eval_binding_request
                != replay.pending_eval_binding_request
            ):
                raise ValueError(
                    "pending Eval Config request is not the canonical subject "
                    "derivation"
                )
        if self.resolved_eval_binding is not None:
            if sum(item is not None for item in binding_subjects) != 1:
                raise ValueError(
                    "resolved Eval Config requires exactly one subject"
                )
            if (
                self.resolved_eval_binding.request
                != replay.pending_eval_binding_request
            ):
                raise ValueError(
                    "resolved Eval Config request is not canonical for its "
                    "subject"
                )
        if (
            self.pending_evaluation is not None
            and self.pending_evaluation != replay.pending_evaluation
        ):
            raise ValueError(
                "pending evaluation is incoherent with its exact spec"
            )
        if self.pending_bootstrap is not None and self.phase != "bootstrap":
            raise ValueError("bootstrap effects require bootstrap phase")
        if self.pending_proposal is not None and self.phase != "proposal":
            raise ValueError("proposal effects require proposal phase")
        if self.pending_evaluation_spec is not None:
            expected_phase = {
                "miprov2_baseline": "baseline",
                "miprov2_sample": "sample",
                "miprov2_promotion": "promotion",
            }[self.pending_evaluation_spec.purpose]
            if self.phase != expected_phase:
                raise ValueError(
                    "evaluation purpose conflicts with runtime phase"
                )
        if (self.pending_sample is not None) != (self.phase == "promotion"):
            raise ValueError(
                "only promotion phase carries its pending minibatch sample"
            )
        if self.phase in {"complete", "failed"} and any(
            item is not None
            for item in (
                self.pending_bootstrap,
                self.pending_proposal,
                self.pending_evaluation_spec,
                self.pending_eval_binding_request,
                self.resolved_eval_binding,
                self.pending_evaluation,
                self.pending_sample,
            )
        ):
            raise ValueError("terminal states cannot carry pending effects")
        if self.phase == "complete" and (
            self.accepted_candidate is None
            or self.accepted_candidate_ref is None
            or self.terminal_result is None
        ):
            raise ValueError("complete state requires exactly one winner")
        if self.phase != "complete" and (
            self.accepted_candidate is not None
            or self.accepted_candidate_ref is not None
            or self.terminal_result is not None
        ):
            raise ValueError("only complete state carries a winner")
        if (
            self.accepted_candidate is not None
            and self.accepted_candidate_ref is not None
            and self.accepted_candidate_ref
            != candidate_reference(self.accepted_candidate)
        ):
            raise ValueError("accepted candidate ref is not its exact record")
        if (
            self.terminal_result is not None
            and self.accepted_candidate_ref is not None
            and self.terminal_result.winner != self.accepted_candidate_ref
        ):
            raise ValueError("terminal result cites another winner")
        if self.phase == "complete":
            assert self.accepted_candidate_ref is not None
            assert self.terminal_result is not None
            if (
                self.accepted_candidate_ref != replay.winner
                or self.terminal_result.winner_score != replay.winner_score
            ):
                raise ValueError(
                    "terminal winner is not the canonical best full evaluation"
                )
        if (
            self.terminal_result is not None
            and self.terminal_result.track_stats
            and self.terminal_result.stats != _terminal_statistics(self)
        ):
            raise ValueError("terminal statistics are not canonical")
        if self.phase == "failed" and not self.failure:
            raise ValueError("failed state requires detail")
        if self.phase != "failed" and self.failure is not None:
            raise ValueError("only failed state carries failure detail")
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_RUNTIME_SCHEMA,
            schema_version=MIPROV2_RUNTIME_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are an explicit wire contract. Never derive
        # them by iterating over model fields.
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run": self.run.model_dump(mode="json"),
            "control": self.control.model_dump(mode="json"),
            "bindings": self.bindings.model_dump(mode="json"),
            "rng_checkpoint": self.rng_checkpoint.model_dump(mode="json"),
            "labeled_trainset": [
                item.model_dump(mode="json") for item in self.labeled_trainset
            ],
            "proposal_components": [
                item.model_dump(mode="json")
                for item in self.proposal_components
            ],
            "proposal_trainset": [
                item.model_dump(mode="json") for item in self.proposal_trainset
            ],
            "component_field_order": self.component_field_order.to_json(),
            "input_data_identity_hash": self.input_data_identity_hash,
            "budget": self.budget.model_dump(mode="json"),
            "phase": self.phase,
            "bootstrap_plans": [
                item.model_dump(mode="json") for item in self.bootstrap_plans
            ],
            "bootstrap_plan_index": self.bootstrap_plan_index,
            "bootstrap_state": (
                None
                if self.bootstrap_state is None
                else self.bootstrap_state.model_dump(mode="json")
            ),
            "demo_candidates": [
                item.model_dump(mode="json") for item in self.demo_candidates
            ],
            "bootstrap_evidence": [
                item.model_dump(mode="json")
                for item in self.bootstrap_evidence
            ],
            "proposal_state": (
                None
                if self.proposal_state is None
                else self.proposal_state.model_dump(mode="json")
            ),
            "instruction_pools": [
                list(pool) for pool in self.instruction_pools
            ],
            "study_demo_candidates": (
                None
                if self.study_demo_candidates is None
                else [
                    item.model_dump(mode="json")
                    for item in self.study_demo_candidates
                ]
            ),
            "study_transcript": (
                None
                if self.study_transcript is None
                else self.study_transcript.model_dump(mode="json")
            ),
            "pending_bootstrap": (
                None
                if self.pending_bootstrap is None
                else self.pending_bootstrap.model_dump(mode="json")
            ),
            "pending_bootstrap_candidate": (
                None
                if self.pending_bootstrap_candidate is None
                else self.pending_bootstrap_candidate.model_dump(mode="json")
            ),
            "pending_proposal": (
                None
                if self.pending_proposal is None
                else self.pending_proposal.model_dump(mode="json")
            ),
            "pending_evaluation_spec": (
                None
                if self.pending_evaluation_spec is None
                else self.pending_evaluation_spec.model_dump(mode="json")
            ),
            "pending_eval_binding_request": (
                None
                if self.pending_eval_binding_request is None
                else self.pending_eval_binding_request.model_dump(mode="json")
            ),
            "resolved_eval_binding": (
                None
                if self.resolved_eval_binding is None
                else self.resolved_eval_binding.model_dump(mode="json")
            ),
            "pending_evaluation": (
                None
                if self.pending_evaluation is None
                else self.pending_evaluation.model_dump(mode="json")
            ),
            "pending_sample": (
                None
                if self.pending_sample is None
                else self.pending_sample.model_dump(mode="json")
            ),
            "fully_evaluated_candidates": [
                item.model_dump(mode="json")
                for item in self.fully_evaluated_candidates
            ],
            "accepted_candidate": (
                None
                if self.accepted_candidate is None
                else self.accepted_candidate.model_dump(mode="json")
            ),
            "accepted_candidate_ref": (
                None
                if self.accepted_candidate_ref is None
                else self.accepted_candidate_ref.model_dump(mode="json")
            ),
            "terminal_result": (
                None
                if self.terminal_result is None
                else self.terminal_result.model_dump(mode="json")
            ),
            "completed_effects": [
                item.model_dump(mode="json") for item in self.completed_effects
            ],
            "failure": self.failure,
        }

    @property
    def effect_counts(self) -> dict[str, int]:
        counts = {
            kind: sum(effect.kind == kind for effect in self.completed_effects)
            for kind in (
                "bootstrap_rollouts",
                "proposal_calls",
                "evaluations",
            )
        }
        counts["task_rows"] = sum(
            effect.task_rows for effect in self.completed_effects
        )
        return counts


class Miprov2DriverPlan(BaseModel):
    """One pure planning checkpoint with zero or one external effect."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: Miprov2State
    kind: Miprov2EffectKind | Literal["complete"]
    eval_config_binding: Miprov2EvalConfigBindingRequest | None = None
    bootstrap_rollout: BootstrapAttemptPlan | None = None
    proposal_request: Miprov2ProposalRequest | None = None
    evaluation: Miprov2EvaluationEffect | None = None
    accepted_candidate: Candidate | None = None

    @model_validator(mode="after")
    def _validate_plan(self) -> Miprov2DriverPlan:
        effects = (
            self.eval_config_binding,
            self.bootstrap_rollout,
            self.proposal_request,
            self.evaluation,
        )
        if self.kind == "complete":
            if any(item is not None for item in effects):
                raise ValueError("complete plan cannot contain an effect")
            if self.accepted_candidate is None:
                raise ValueError("complete plan requires its one winner")
        elif sum(item is not None for item in effects) != 1:
            raise ValueError("effect plan must contain exactly one effect")
        return self


def _input_data_identity(
    *,
    control: Miprov2Control,
    labeled_trainset: tuple[LabeledTaskDemo, ...],
    proposal_components: tuple[Miprov2PromptComponent, ...],
    proposal_trainset: tuple[Miprov2DatasetExample, ...],
    component_field_order: Mapping[str, object],
) -> str:
    return compute_identity_hash(
        schema="whetstone.miprov2_runtime_inputs",
        schema_version=1,
        payload={
            "control_identity_hash": control.identity_hash(),
            "labeled_trainset": [
                item.model_dump(mode="json") for item in labeled_trainset
            ],
            "proposal_components": [
                item.model_dump(mode="json") for item in proposal_components
            ],
            "proposal_trainset": [
                item.model_dump(mode="json") for item in proposal_trainset
            ],
            "component_field_order": {
                component_id: list(cast("tuple[str, ...]", fields))
                for component_id, fields in component_field_order.items()
            },
        },
    )


def _teacher_compiled(control: Miprov2Control) -> bool:
    return control.teacher_compiled


def _provider_parameters(
    teacher_settings: dict[str, object],
    *,
    temperature: float | None = None,
) -> dict[str, object]:
    """Translate DSPy LM kwargs to Whetstone's typed provider parameters."""

    parameters: dict[str, object] = {}
    extra_body: dict[str, object] = {}
    supplied_extra_body = teacher_settings.get("extra_body")
    if supplied_extra_body is not None:
        if not isinstance(supplied_extra_body, dict):
            raise ValueError("teacher_settings.extra_body must be an object")
        for key, value in supplied_extra_body.items():
            if not isinstance(key, str):
                raise ValueError(
                    "teacher_settings.extra_body keys must be strings"
                )
            extra_body[key] = value
    for key, value in teacher_settings.items():
        if key == "extra_body":
            continue
        if key in {"temperature", "token_limit", "reasoning"}:
            parameters[key] = value
        else:
            if key in extra_body:
                raise ValueError(
                    f"teacher setting conflicts with extra_body key {key!r}"
                )
            extra_body[key] = value
    if temperature is not None:
        parameters["temperature"] = temperature
    if extra_body:
        parameters["extra_body"] = extra_body
    return parameters


def _execution_policy(
    control: Miprov2Control,
    *,
    bootstrap_attempt: BootstrapAttemptPlan | None = None,
) -> Miprov2EvaluationExecutionPolicy:
    teacher_settings = (
        control.model_dump(mode="json")["teacher_settings"]
        if bootstrap_attempt is not None
        else {}
    )
    return Miprov2EvaluationExecutionPolicy(
        num_threads=control.num_threads,
        max_errors=control.max_errors,
        provide_traceback=control.provide_traceback,
        task_model_identity_hash=control.task_model_identity_hash,
        provider_execution_policy_hash=(
            control.provider_execution_policy_hash
        ),
        provider_parameters=_provider_parameters(
            teacher_settings,
            temperature=(
                None
                if bootstrap_attempt is None
                else bootstrap_attempt.temperature
            ),
        ),
        rollout_id=(
            None if bootstrap_attempt is None else bootstrap_attempt.rollout_id
        ),
        copy_task_model=(
            False
            if bootstrap_attempt is None
            else bootstrap_attempt.copy_task_model
        ),
    )


def _terminal_statistics(state: Miprov2State) -> Miprov2TerminalStats:
    """Project the frozen DSPy statistics surface from durable evidence."""

    transcript = state.study_transcript
    if transcript is None:
        raise ValueError("terminal statistics require a study transcript")
    evaluation_calls = len(
        transcript.baseline.evaluation.task_batch_identities
    )
    score_data: list[Miprov2ScoreObservation] = [
        Miprov2ScoreObservation(
            source="baseline",
            trial_number=0,
            candidate=transcript.baseline.evaluated_base_candidate,
            score=transcript.baseline.score,
            full_eval=True,
            cumulative_evaluation_calls=evaluation_calls,
        )
    ]
    trial_logs: list[Miprov2TrialLog] = [
        Miprov2TrialLog(
            log_key=1,
            source="baseline",
            optuna_trial_number=0,
            candidate=transcript.baseline.evaluated_base_candidate,
            full_score=transcript.baseline.score,
            cumulative_evaluation_calls=evaluation_calls,
        )
    ]
    for sample in transcript.samples:
        evaluation_calls += len(sample.evaluation.task_batch_identities)
        sample_candidate = sample.candidate_assembly.candidate
        score_data.append(
            Miprov2ScoreObservation(
                source="sample",
                trial_number=sample.trial_number,
                candidate=sample_candidate,
                score=sample.score,
                full_eval=sample.batch_full_evaluation,
                cumulative_evaluation_calls=evaluation_calls,
            )
        )
        trial_logs.append(
            Miprov2TrialLog(
                log_key=sample.trial_number + 1,
                source="sample",
                optuna_trial_number=sample.trial_number,
                params=sample.params,
                candidate=sample_candidate,
                minibatch_score=(
                    sample.score if transcript.schedule.minibatch else None
                ),
                full_score=(
                    None if transcript.schedule.minibatch else sample.score
                ),
                cumulative_evaluation_calls=evaluation_calls,
            )
        )
        promotion = sample.promotion
        if promotion is None:
            continue
        evaluation_calls += len(promotion.evaluation.task_batch_identities)
        promotion_candidate = promotion.candidate_assembly.candidate
        score_data.append(
            Miprov2ScoreObservation(
                source="promotion",
                trial_number=promotion.trial_number,
                candidate=promotion_candidate,
                score=promotion.full_score,
                full_eval=True,
                cumulative_evaluation_calls=evaluation_calls,
            )
        )
        trial_logs.append(
            Miprov2TrialLog(
                log_key=sample.trial_number + 2,
                source="promotion",
                optuna_trial_number=promotion.trial_number,
                candidate=promotion_candidate,
                full_score=promotion.full_score,
                cumulative_evaluation_calls=evaluation_calls,
            )
        )
    full = tuple(
        sorted(
            (
                Miprov2ScoredCandidate(
                    candidate=item.candidate,
                    score=item.score,
                    source=item.source,
                    trial_number=item.trial_number,
                )
                for item in score_data
                if item.full_eval
            ),
            key=lambda item: item.score,
            reverse=True,
        )
    )
    minibatch = tuple(
        sorted(
            (
                Miprov2ScoredCandidate(
                    candidate=item.candidate,
                    score=item.score,
                    source=item.source,
                    trial_number=item.trial_number,
                )
                for item in score_data
                if not item.full_eval
            ),
            key=lambda item: item.score,
            reverse=True,
        )
    )
    return Miprov2TerminalStats(
        study_transcript=transcript,
        fully_evaluated_candidates=state.fully_evaluated_candidates,
        completed_effects=state.completed_effects,
        instruction_pools=state.instruction_pools,
        demo_candidates=state.study_demo_candidates,
        effect_counts=state.effect_counts,
        trial_logs=tuple(trial_logs),
        cumulative_evaluation_calls=evaluation_calls,
        score_data=tuple(score_data),
        mb_candidate_programs=minibatch,
        candidate_programs=full,
    )


def _initial_bootstrap_rng(control: Miprov2Control) -> Miprov2RngCheckpoint:
    pre_auto_valset_size = (
        len(control.source_valset_task_identities)
        if control.source_valset_task_identities is not None
        else (
            len(control.source_trainset_task_identities)
            - len(control.trainset_task_identities)
        )
    )
    return Miprov2RngCheckpoint.after_validation_sampling(
        seed=control.seed,
        population_size=pre_auto_valset_size,
        sample_indices=control.auto_validation_sample_indices,
    )


def _ordered_labeled_for_plan(
    state: Miprov2State,
    plan: FewshotCandidatePlan,
) -> tuple[LabeledTaskDemo, ...]:
    by_identity = {
        item.source_task_identity: item for item in state.labeled_trainset
    }
    try:
        return tuple(
            by_identity[identity] for identity in plan.trainset_task_identities
        )
    except KeyError as exc:
        raise ValueError(
            "bootstrap plan cites a foreign labeled task"
        ) from exc


def _canonical_bootstrap_projection(
    state: Miprov2State,
    planning: FewshotCandidatePlanningResult,
) -> tuple[
    int,
    BootstrapCompilerState | None,
    tuple[ComponentDemoSet, ...],
    BootstrapAttemptPlan | None,
]:
    """Replay all bootstrap evidence and validate cursor/demo checkpoints."""

    plans = planning.plans
    events = state.bootstrap_evidence
    event_index = 0
    expected_demos: list[ComponentDemoSet] = []
    current_compiler: BootstrapCompilerState | None = None
    current_attempt: BootstrapAttemptPlan | None = None
    for plan_index, plan in enumerate(plans):
        plan_events: list[BootstrapFoldEvidence] = []
        while (
            event_index < len(events)
            and events[event_index].attempt.plan_identity_hash
            == plan.identity_hash()
        ):
            plan_events.append(events[event_index])
            event_index += 1
        if plan_index < state.bootstrap_plan_index:
            if plan.kind is FewshotSeedKind.RESET:
                demo_set = materialize_reset_demo_set(
                    plan=plan,
                    component_ids=state.control.component_ids,
                )
            elif plan.kind is FewshotSeedKind.LABELS_ONLY:
                demo_set = materialize_labels_only_demo_set(
                    plan=plan,
                    labeled_trainset=_ordered_labeled_for_plan(state, plan),
                )
            else:
                compiler = initial_compiler_state(plan)
                for event in plan_events:
                    compiler = fold_bootstrap_result(
                        plan=plan,
                        state=compiler,
                        attempt=event.attempt,
                        result=event.result,
                        metric_threshold=state.control.metric_threshold,
                        component_ids=state.control.component_ids,
                    )
                try:
                    unfinished = next_bootstrap_attempt(plan, compiler)
                except Exception as exc:
                    raise ValueError(
                        "finished bootstrap plan has terminal failure"
                    ) from exc
                if unfinished is not None:
                    raise ValueError(
                        "bootstrap cursor skips an unfinished compiler"
                    )
                demo_set = materialize_bootstrap_demo_set(
                    plan=plan,
                    state=compiler,
                    labeled_trainset=_ordered_labeled_for_plan(state, plan),
                    component_ids=state.control.component_ids,
                )
            expected_demos.append(demo_set)
            continue
        if plan_index == state.bootstrap_plan_index:
            if plan.kind is not FewshotSeedKind.BOOTSTRAP:
                if (
                    state.bootstrap_plan_index != 0
                    or state.demo_candidates
                    or state.bootstrap_evidence
                ):
                    raise ValueError(
                        "runtime cannot pause between effect-free bootstrap "
                        "plans"
                    )
                break
            compiler = initial_compiler_state(plan)
            for event in plan_events:
                compiler = fold_bootstrap_result(
                    plan=plan,
                    state=compiler,
                    attempt=event.attempt,
                    result=event.result,
                    metric_threshold=state.control.metric_threshold,
                    component_ids=state.control.component_ids,
                )
            current_compiler = compiler if plan_events else None
            current_attempt = state.pending_bootstrap
            if current_attempt is not None:
                expected_attempt = next_bootstrap_attempt(plan, compiler)
                if current_attempt != expected_attempt:
                    raise ValueError(
                        "pending bootstrap attempt is not canonical"
                    )
                current_compiler = compiler
            break
        if plan_events:
            raise ValueError("bootstrap evidence belongs to a future plan")
    if event_index != len(events):
        raise ValueError("bootstrap evidence order does not match plans")
    if tuple(expected_demos) != state.demo_candidates:
        raise ValueError("demo candidates are not canonical bootstrap output")
    if state.bootstrap_state != current_compiler:
        raise ValueError("bootstrap compiler state conflicts with evidence")
    if state.phase != "bootstrap" and (
        state.bootstrap_plan_index != len(plans)
        or len(state.demo_candidates) != len(plans)
        or state.bootstrap_state is not None
        or state.pending_bootstrap is not None
    ):
        raise ValueError(
            "post-bootstrap phase has incomplete bootstrap replay"
        )
    return (
        state.bootstrap_plan_index,
        state.bootstrap_state,
        state.demo_candidates,
        current_attempt,
    )


def _canonical_completed_effects(
    state: Miprov2State,
) -> tuple[Miprov2CompletedEffect, ...]:
    effects: list[Miprov2CompletedEffect] = [
        Miprov2CompletedEffect(
            kind="bootstrap_rollouts",
            identity_hash=event.attempt.identity_hash(),
            task_rows=1,
        )
        for event in state.bootstrap_evidence
    ]
    if state.proposal_state is not None:
        effects.extend(
            Miprov2CompletedEffect(
                kind="proposal_calls",
                identity_hash=item.request.identity_hash,
            )
            for item in state.proposal_state.evidence
        )
    transcript = state.study_transcript
    if transcript is not None:
        bindings = [transcript.baseline.evaluation]
        for sample in transcript.samples:
            bindings.append(sample.evaluation)
            if sample.promotion is not None:
                bindings.append(sample.promotion.evaluation)
        effects.extend(
            Miprov2CompletedEffect(
                kind="evaluations",
                identity_hash=binding.effect_identity_hash,
                task_rows=len(binding.task_batch_identities),
            )
            for binding in bindings
        )
    if state.pending_sample is not None:
        binding = state.pending_sample.evaluation
        effects.append(
            Miprov2CompletedEffect(
                kind="evaluations",
                identity_hash=binding.effect_identity_hash,
                task_rows=len(binding.task_batch_identities),
            )
        )
    return tuple(effects)


def _canonical_fully_evaluated_candidates(
    state: Miprov2State,
) -> tuple[CandidateRef, ...]:
    transcript = state.study_transcript
    if transcript is None:
        return ()
    candidates: list[CandidateRef] = [
        transcript.baseline.evaluated_base_candidate
    ]
    for sample in transcript.samples:
        if not state.control.minibatch:
            candidates.append(sample.candidate_assembly.candidate)
        if sample.promotion is not None:
            candidates.append(sample.promotion.candidate_assembly.candidate)
    return tuple(candidates)


def _canonical_phase(state: Miprov2State) -> Miprov2Phase:
    if state.failure is not None:
        return "failed"
    if state.terminal_result is not None:
        return "complete"
    if state.bootstrap_plan_index < len(state.bootstrap_plans):
        return "bootstrap"
    if not state.instruction_pools:
        return "proposal"
    if state.study_transcript is None:
        return "baseline"
    if state.pending_sample is not None:
        return "promotion"
    return "sample"


def _canonical_runtime_rng(
    state: Miprov2State,
    planning: FewshotCandidatePlanningResult,
    *,
    include_pending_spec: bool = True,
) -> Miprov2RngCheckpoint:
    checkpoint = planning.rng_checkpoint
    if state.proposal_state is not None:
        if state.proposal_state.initial_rng_checkpoint != checkpoint:
            raise ValueError(
                "proposal RNG does not start at canonical bootstrap cursor"
            )
        proposal_replay = start_miprov2_proposal(
            bindings=state.proposal_state.bindings,
            optimization_run_identity_hash=state.run.identity_hash,
            components=state.proposal_state.components,
            trainset=state.proposal_state.trainset,
            demo_candidates=state.proposal_state.demo_candidates,
            num_candidates=state.proposal_state.num_candidates,
            view_data_batch_size=state.proposal_state.view_data_batch_size,
            init_temperature=state.proposal_state.init_temperature,
            data_aware=state.proposal_state.initial_data_aware,
            program_aware=state.proposal_state.program_aware,
            tip_aware=state.proposal_state.tip_aware,
            fewshot_aware=state.proposal_state.fewshot_aware,
            rng_checkpoint=checkpoint,
        )
        for evidence in state.proposal_state.evidence:
            planned = plan_next_proposal_request(proposal_replay)
            if planned.request != evidence.request:
                raise ValueError(
                    "proposal RNG replay encountered foreign evidence"
                )
            proposal_replay = fold_proposal_response(
                planned.state,
                evidence.response,
            )
        checkpoint = proposal_replay.rng_checkpoint
    if not state.control.minibatch or state.control.minibatch_size >= len(
        state.control.valset_task_identities
    ):
        return checkpoint
    batches: list[tuple[str, ...]] = []
    if state.study_transcript is not None:
        batches.extend(
            sample.evaluation.task_batch_identities
            for sample in state.study_transcript.samples
        )
    if state.pending_sample is not None:
        batches.append(state.pending_sample.evaluation.task_batch_identities)
    elif (
        include_pending_spec
        and state.pending_evaluation_spec is not None
        and state.pending_evaluation_spec.purpose == "miprov2_sample"
    ):
        batches.append(state.pending_evaluation_spec.task_batch_identities)
    population = state.control.valset_task_identities
    for batch in batches:
        rng = checkpoint.state.restore()
        actual = tuple(rng.sample(population, state.control.minibatch_size))
        if actual != batch:
            raise ValueError(
                "sample task batch does not match canonical shared RNG"
            )
        checkpoint = checkpoint.append(
            rng=rng,
            phase="evaluation",
            operation="sample",
            arguments=(population, state.control.minibatch_size),
            result=actual,
        )
    return checkpoint


def _canonical_pending_evaluation_spec(
    state: Miprov2State,
    planning: FewshotCandidatePlanningResult,
) -> Miprov2EvaluationSpec | None:
    """Derive the only evaluation spec the study may expose next."""

    if state.pending_evaluation_spec is None:
        return None
    driver = Miprov2Driver()
    if state.phase == "baseline":
        space = driver._space(state)
        params = space.baseline_params
        return Miprov2EvaluationSpec(
            run_id=state.run_id,
            ordinal=0,
            purpose="miprov2_baseline",
            candidate=state.control.base_candidate.record,
            categorical_combination_identity_hash=(
                space.combination_identity_hash(params)
            ),
            task_batch_identities=state.control.valset_task_identities,
            execution_policy=_execution_policy(state.control),
        )
    if state.phase == "sample":
        if state.study_transcript is None:
            raise ValueError(
                "sample phase requires canonical study transcript"
            )
        study = driver._study(state)
        suggestion = study.suggest_next(state.study_transcript)
        assembly = driver._assemble(
            state,
            suggestion.params,
            suggestion.candidate_combination_identity_hash,
        )
        checkpoint = _canonical_runtime_rng(
            state,
            planning,
            include_pending_spec=False,
        )
        if state.control.minibatch and state.control.minibatch_size < len(
            state.control.valset_task_identities
        ):
            rng = checkpoint.state.restore()
            batch = tuple(
                rng.sample(
                    state.control.valset_task_identities,
                    state.control.minibatch_size,
                )
            )
        else:
            batch = state.control.valset_task_identities
        return Miprov2EvaluationSpec(
            run_id=state.run_id,
            ordinal=len(state.completed_effects),
            purpose="miprov2_sample",
            candidate=assembly.candidate.record,
            categorical_combination_identity_hash=(
                suggestion.candidate_combination_identity_hash
            ),
            task_batch_identities=batch,
            execution_policy=_execution_policy(state.control),
            suggestion=suggestion,
            candidate_assembly=assembly,
        )
    if state.phase == "promotion":
        if state.study_transcript is None or state.pending_sample is None:
            raise ValueError(
                "promotion phase requires canonical study and sample"
            )
        study = driver._study(state)
        promotion = study.promotion_candidate(
            state.study_transcript,
            state.pending_sample.suggestion,
            score=state.pending_sample.score,
            evaluation=state.pending_sample.evaluation,
            candidate_assembly=state.pending_sample.candidate_assembly,
        )
        if promotion is None:
            raise ValueError("promotion phase has no canonical candidate")
        assembly = promotion.candidate_assembly
        return Miprov2EvaluationSpec(
            run_id=state.run_id,
            ordinal=len(state.completed_effects),
            purpose="miprov2_promotion",
            candidate=assembly.candidate.record,
            categorical_combination_identity_hash=(
                promotion.candidate_combination_identity_hash
            ),
            task_batch_identities=state.control.valset_task_identities,
            execution_policy=_execution_policy(state.control),
            promotion_candidate=promotion,
            candidate_assembly=assembly,
        )
    raise ValueError("only study evaluation phases carry evaluation specs")


def _canonical_eval_binding_request(
    state: Miprov2State,
) -> Miprov2EvalConfigBindingRequest:
    if state.pending_bootstrap is not None:
        attempt = state.pending_bootstrap
        return Miprov2EvalConfigBindingRequest(
            control_identity_hash=state.control.identity_hash(),
            source_eval_config=state.control.bootstrap_eval_source,
            purpose="bootstrap",
            effect_identity_hash=attempt.identity_hash(),
            execution_policy=_execution_policy(
                state.control,
                bootstrap_attempt=attempt,
            ),
            task_batch_identities=(attempt.task_identity,),
        )
    spec = state.pending_evaluation_spec
    if spec is None:
        raise ValueError("Eval Config request has no canonical subject")
    purpose: Literal["baseline", "sample", "promotion"]
    if spec.purpose == "miprov2_baseline":
        purpose = "baseline"
    elif spec.purpose == "miprov2_sample":
        purpose = "sample"
    else:
        purpose = "promotion"
    return Miprov2EvalConfigBindingRequest(
        control_identity_hash=state.control.identity_hash(),
        source_eval_config=state.control.validation_eval_source,
        purpose=purpose,
        effect_identity_hash=spec.identity_hash(),
        execution_policy=spec.execution_policy,
        task_batch_identities=spec.task_batch_identities,
    )


def replay_miprov2_state(
    state: Miprov2State,
    planning: FewshotCandidatePlanningResult,
) -> Miprov2ReplayProjection:
    """Replay the canonical projection from immutable inputs and evidence."""

    instruction_pools = (
        ()
        if state.proposal_state is None
        else state.proposal_state.instruction_pools
    )
    study_demos = (
        study_demo_context(
            state.demo_candidates,
            zeroshot_opt=state.control.zeroshot_opt,
        )
        if instruction_pools
        else None
    )
    pending_spec = _canonical_pending_evaluation_spec(state, planning)
    pending_request = (
        _canonical_eval_binding_request(state)
        if (
            state.pending_bootstrap is not None
            or state.pending_evaluation_spec is not None
        )
        else None
    )
    pending_evaluation: Miprov2EvaluationEffect | None = None
    if pending_spec is not None and state.resolved_eval_binding is not None:
        pending_evaluation = Miprov2EvaluationEffect(
            run_id=pending_spec.run_id,
            ordinal=pending_spec.ordinal,
            purpose=pending_spec.purpose,
            candidate=pending_spec.candidate,
            categorical_combination_identity_hash=(
                pending_spec.categorical_combination_identity_hash
            ),
            task_batch_identities=pending_spec.task_batch_identities,
            eval_config=state.resolved_eval_binding.eval_config,
            execution_policy=pending_spec.execution_policy,
            reward_policy_hash=state.control.reward_policy_hash,
            suggestion=pending_spec.suggestion,
            promotion_candidate=pending_spec.promotion_candidate,
            candidate_assembly=pending_spec.candidate_assembly,
        )
    winner_ref: CandidateRef | None = None
    winner_score: float | None = None
    if state.terminal_result is not None:
        if state.study_transcript is None:
            raise ValueError("terminal state has no canonical study")
        winner = (
            Miprov2Driver()
            ._study(state)
            .best_full_evaluation(state.study_transcript)
        )
        winner_ref = next(
            (
                candidate
                for candidate in _canonical_fully_evaluated_candidates(state)
                if candidate.identity_hash
                == winner.evaluated_candidate_identity_hash
            ),
            None,
        )
        if winner_ref is None:
            raise ValueError("canonical winner has no evaluated CandidateRef")
        winner_score = winner.score
    return Miprov2ReplayProjection(
        rng_checkpoint=_canonical_runtime_rng(state, planning),
        bootstrap=_canonical_bootstrap_projection(state, planning),
        completed_effects=_canonical_completed_effects(state),
        fully_evaluated_candidates=(
            _canonical_fully_evaluated_candidates(state)
        ),
        phase=_canonical_phase(state),
        pending_evaluation_spec=pending_spec,
        pending_eval_binding_request=pending_request,
        pending_evaluation=pending_evaluation,
        instruction_pools=instruction_pools,
        study_demo_candidates=study_demos,
        winner=winner_ref,
        winner_score=winner_score,
    )


def _materialize_bootstrap_teacher(
    *,
    state: Miprov2State,
    plan: FewshotCandidatePlan,
    attempt: BootstrapAttemptPlan,
) -> CandidateRef:
    """Apply the teacher plan through the sole candidate mutation surface."""

    teacher_plan = plan.teacher
    if teacher_plan is None:
        raise ValueError("bootstrap plan has no teacher preparation")
    source = state.control.teacher_candidate
    selection = teacher_plan.labeled_selection
    components: list[dict[str, object]] = []
    for component_index, spec in enumerate(state.control.component_specs):
        examples: list[dict[str, object]] = []
        if selection is not None:
            for index in selection.per_component_task_indices[component_index]:
                item = state.labeled_trainset[index]
                if item.source_task_identity == attempt.task_identity:
                    continue
                examples.append(
                    {
                        "inputs": item.inputs_for(spec.component_id),
                        "outputs": item.outputs_for(spec.component_id),
                    }
                )
        instruction = source.record.payload["user_prompt_template"]
        assert isinstance(instruction, str)
        components.append(
            {
                "component_id": spec.component_id,
                "instruction_index": component_index,
                "instruction": instruction,
                "instruction_identity_hash": _instruction_identity(
                    instruction
                ),
                "demo_index": component_index if examples else None,
                "demo_set": examples or None,
                "demo_identity_hash": (
                    compute_identity_hash(
                        schema="whetstone.miprov2_teacher_examples",
                        schema_version=1,
                        payload=examples,
                    )
                    if examples
                    else None
                ),
            }
        )
    identity = compute_identity_hash(
        schema="whetstone.miprov2_bootstrap_teacher_execution",
        schema_version=2,
        payload={
            "source_candidate_identity_hash": source.identity_hash,
            "plan_identity_hash": plan.identity_hash(),
            "attempt_identity_hash": attempt.identity_hash(),
            "components": components,
        },
    )
    return candidate_reference(
        candidate_from_components(
            base=source,
            candidate_id=f"miprov2-teacher-{identity[:24]}",
            components=components,
            run=state.run,
        )
    )


def render_miprov2_candidate(
    *,
    run: OptimizationRunRef,
    control: Miprov2Control,
    instruction_pools: tuple[tuple[str, ...], ...],
    demo_candidates: tuple[ComponentDemoSet, ...] | None,
    params: TrialParams,
    categorical_combination_identity_hash: str,
) -> Candidate:
    """Render one categorical program into ``user_prompt_template`` only."""

    rendering = _miprov2_candidate_rendering(
        control=control,
        instruction_pools=instruction_pools,
        demo_candidates=demo_candidates,
        params=params,
        categorical_combination_identity_hash=(
            categorical_combination_identity_hash
        ),
    )
    return candidate_from_components(
        base=control.base_candidate,
        candidate_id=f"miprov2-{rendering.identity_hash()[:24]}",
        components=rendering.model_dump(mode="json")["components"],
        run=run,
    )


def _miprov2_candidate_rendering(
    *,
    control: Miprov2Control,
    instruction_pools: tuple[tuple[str, ...], ...],
    demo_candidates: tuple[ComponentDemoSet, ...] | None,
    params: TrialParams,
    categorical_combination_identity_hash: str,
) -> Miprov2CandidateRendering:
    """Bind exact categorical selections before candidate composition."""

    values = dict(params)
    specs = control.component_specs
    if len(instruction_pools) != len(specs):
        raise ValueError("instruction pools do not match component count")
    if demo_candidates is not None and not demo_candidates:
        raise ValueError("demo candidate pool cannot be empty")
    instruction_hashes = tuple(
        tuple(_instruction_identity(item) for item in pool)
        for pool in instruction_pools
    )
    demo_hashes = (
        None
        if demo_candidates is None
        else tuple(
            tuple(
                _component_demo_projection(
                    item,
                    spec.component_id,
                ).identity_hash()
                for item in demo_candidates
            )
            for spec in specs
        )
    )
    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=instruction_hashes,
        demo_pool_identity_hashes=demo_hashes,
    )
    if (
        space.combination_identity_hash(params)
        != categorical_combination_identity_hash
    ):
        raise ValueError(
            "categorical combination identity conflicts with selections"
        )
    selections: list[Miprov2ComponentSelection] = []
    for index, (spec, pool) in enumerate(
        zip(specs, instruction_pools, strict=True)
    ):
        instruction_index = values[f"{index}_predictor_instruction"]
        try:
            instruction = pool[instruction_index]
        except IndexError as exc:
            raise ValueError(
                "instruction category is outside its pool"
            ) from exc
        demo_index: int | None = None
        demo_set: ComponentDemoSet | None = None
        demo_hash: str | None = None
        if demo_candidates is not None:
            demo_index = values[f"{index}_predictor_demos"]
            try:
                demo_set = _component_demo_projection(
                    demo_candidates[demo_index],
                    spec.component_id,
                )
            except IndexError as exc:
                raise ValueError("demo category is outside its pool") from exc
            demo_hash = demo_set.identity_hash()
        selections.append(
            Miprov2ComponentSelection(
                component_id=spec.component_id,
                instruction_index=instruction_index,
                instruction=instruction,
                instruction_identity_hash=_instruction_identity(instruction),
                demo_index=demo_index,
                demo_set=demo_set,
                demo_identity_hash=demo_hash,
            )
        )
    return Miprov2CandidateRendering(
        control_identity_hash=control.identity_hash(),
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        categorical_combination_identity_hash=(
            categorical_combination_identity_hash
        ),
        components=tuple(selections),
    )


class Miprov2Driver:
    """Pure, crash-safe orchestration over the exact algorithm primitives."""

    def start(
        self,
        *,
        run: OptimizationRunRef,
        control: Miprov2Control,
        bindings: Miprov2DurableBindings,
        labeled_trainset: tuple[LabeledTaskDemo, ...],
        proposal_components: tuple[Miprov2PromptComponent, ...],
        proposal_trainset: tuple[Miprov2DatasetExample, ...],
        component_field_order: dict[str, tuple[str, ...]],
        budget: Miprov2EffectBudget,
    ) -> Miprov2State:
        """Bind resolved control and consume no external effects."""

        if len(control.component_ids) != 1:
            raise ValueError(
                "integrated MIPROv2 currently requires exactly one executable "
                "prompt component because the Whetstone rollout primitive "
                "exposes only one provider trace"
            )
        rng = _initial_bootstrap_rng(control)
        planned = create_fewshot_candidate_plans(
            bindings=bindings,
            component_ids=control.component_ids,
            trainset_task_identities=control.trainset_task_identities,
            num_candidate_sets=control.num_fewshot_candidates,
            max_bootstrapped_demos=control.max_bootstrapped_demos,
            max_labeled_demos=control.max_labeled_demos,
            max_errors=control.max_errors,
            metric_threshold=control.metric_threshold,
            explicit_teacher=(
                control.teacher_candidate.identity_hash
                != control.base_candidate.identity_hash
            ),
            teacher_compiled=_teacher_compiled(control),
            rng_checkpoint=rng,
            zeroshot_opt=control.zeroshot_opt,
        )
        return Miprov2State(
            run_id=run.record.run_id,
            run=run,
            control=control,
            bindings=bindings,
            rng_checkpoint=planned.rng_checkpoint,
            labeled_trainset=labeled_trainset,
            proposal_components=proposal_components,
            proposal_trainset=proposal_trainset,
            component_field_order=component_field_order,
            input_data_identity_hash=_input_data_identity(
                control=control,
                labeled_trainset=labeled_trainset,
                proposal_components=proposal_components,
                proposal_trainset=proposal_trainset,
                component_field_order=component_field_order,
            ),
            budget=budget,
            bootstrap_plans=planned.plans,
        )

    def plan(self, state: Miprov2State) -> Miprov2DriverPlan:
        """Advance pure phases until one effect or the strict winner."""

        state = self._validated(state)
        if state.phase == "failed":
            raise ValueError(state.failure)
        if state.phase == "complete":
            assert state.accepted_candidate is not None
            return Miprov2DriverPlan(
                state=state,
                kind="complete",
                accepted_candidate=state.accepted_candidate,
            )
        if state.pending_eval_binding_request is not None:
            if state.pending_eval_binding_request.purpose != "bootstrap":
                self._require_budget(state, "evaluations")
            return Miprov2DriverPlan(
                state=state,
                kind="eval_config_binding",
                eval_config_binding=state.pending_eval_binding_request,
            )
        if state.pending_bootstrap is not None:
            if state.resolved_eval_binding is None:
                raise ValueError(
                    "bootstrap attempt has no exact subset Eval Config"
                )
            self._require_budget(state, "bootstrap_rollouts")
            return Miprov2DriverPlan(
                state=state,
                kind="bootstrap_rollout",
                bootstrap_rollout=state.pending_bootstrap,
            )
        if state.pending_proposal is not None:
            self._require_budget(state, "proposal_calls")
            return Miprov2DriverPlan(
                state=state,
                kind="proposal_model",
                proposal_request=state.pending_proposal,
            )
        if state.pending_evaluation is not None:
            self._require_budget(state, "evaluations")
            purpose = state.pending_evaluation.purpose
            if purpose == "miprov2_baseline":
                kind: Miprov2EffectKind = "baseline_evaluation"
            elif purpose == "miprov2_sample":
                kind = "sample_evaluation"
            else:
                kind = "promotion_evaluation"
            return Miprov2DriverPlan(
                state=state,
                kind=kind,
                evaluation=state.pending_evaluation,
            )
        if state.pending_evaluation_spec is not None:
            return self._plan_evaluation_spec(
                state, state.pending_evaluation_spec
            )
        if state.phase == "bootstrap":
            return self._plan_bootstrap(state)
        if state.phase == "proposal":
            return self._plan_proposal(state)
        if state.phase == "baseline":
            return self._plan_baseline(state)
        if state.phase == "sample":
            return self._plan_sample(state)
        if state.phase == "promotion":
            return self._plan_promotion(state)
        raise AssertionError(f"unhandled MIPROv2 phase {state.phase!r}")

    def fold_eval_config_binding(
        self,
        state: Miprov2State,
        binding: Miprov2EvalConfigBinding,
    ) -> Miprov2State:
        """Fold the exact ordered-subset config before issuing its Intent."""

        state = self._validated(state)
        request = state.pending_eval_binding_request
        if request is None:
            raise ValueError("no Eval Config binding is pending")
        if binding.request != request:
            raise ValueError("Eval Config binding belongs to another request")
        return state.model_copy(
            update={
                "pending_eval_binding_request": None,
                "resolved_eval_binding": binding,
            }
        )

    def fold_bootstrap(
        self,
        state: Miprov2State,
        result: BootstrapRolloutResult,
    ) -> Miprov2State:
        state = self._validated(state)
        attempt = state.pending_bootstrap
        compiler = state.bootstrap_state
        if attempt is None or compiler is None:
            raise ValueError("no bootstrap rollout is pending")
        self._require_new_effect(state, attempt.identity_hash())
        plan = state.bootstrap_plans[state.bootstrap_plan_index]
        advanced = fold_bootstrap_result(
            plan=plan,
            state=compiler,
            attempt=attempt,
            result=result,
            metric_threshold=state.control.metric_threshold,
            component_ids=state.control.component_ids,
        )
        return state.model_copy(
            update={
                "bootstrap_state": advanced,
                "bootstrap_evidence": (
                    *state.bootstrap_evidence,
                    advanced.evidence[-1],
                ),
                "pending_bootstrap": None,
                "pending_bootstrap_candidate": None,
                "resolved_eval_binding": None,
                "completed_effects": (
                    *state.completed_effects,
                    Miprov2CompletedEffect(
                        kind="bootstrap_rollouts",
                        identity_hash=attempt.identity_hash(),
                        task_rows=1,
                    ),
                ),
            }
        )

    def fold_proposal(
        self,
        state: Miprov2State,
        response: Miprov2ProposalResponse,
    ) -> Miprov2State:
        state = self._validated(state)
        request = state.pending_proposal
        proposal_state = state.proposal_state
        if request is None or proposal_state is None:
            raise ValueError("no proposal-model call is pending")
        self._require_new_effect(state, request.identity_hash)
        advanced = fold_proposal_response(proposal_state, response)
        return state.model_copy(
            update={
                "proposal_state": advanced,
                "rng_checkpoint": advanced.rng_checkpoint,
                "pending_proposal": None,
                "completed_effects": (
                    *state.completed_effects,
                    Miprov2CompletedEffect(
                        kind="proposal_calls",
                        identity_hash=request.identity_hash,
                    ),
                ),
            }
        )

    def fold_evaluation(
        self,
        state: Miprov2State,
        resolved: Miprov2ResolvedEvaluation,
    ) -> Miprov2State:
        """Fold canonical evidence using DSPy's ``round(score*100, 2)``."""

        state = self._validated(state)
        effect = state.pending_evaluation
        if effect is None:
            raise ValueError("no evaluation is pending")
        effect_identity = effect.identity_hash()
        self._require_new_effect(state, effect_identity)
        evaluation = resolved.evaluation
        if (
            resolved.context.effect_identity_hash,
            evaluation.run_id,
            evaluation.purpose,
            evaluation.candidate.identity_hash,
            evaluation.task_batch_identities,
            evaluation.eval_config,
            evaluation.reward_policy_hash,
        ) != (
            effect_identity,
            effect.run_id,
            effect.purpose,
            effect.candidate.identity_hash(),
            effect.task_batch_identities,
            effect.eval_config,
            effect.reward_policy_hash,
        ):
            raise ValueError("evaluation result does not match pending effect")
        normalized = resolved.normalized_score
        if not math.isfinite(normalized):
            raise ValueError("MIPROv2 evaluation score must be finite")
        evaluated_ref = candidate_reference(effect.candidate)
        fully_evaluated = state.fully_evaluated_candidates
        if effect.purpose in {"miprov2_baseline", "miprov2_promotion"} or (
            effect.purpose == "miprov2_sample" and not state.control.minibatch
        ):
            fully_evaluated = (*fully_evaluated, evaluated_ref)
        completed = (
            *state.completed_effects,
            Miprov2CompletedEffect(
                kind="evaluations",
                identity_hash=effect_identity,
                task_rows=resolved.row_accounting.planned,
            ),
        )
        study = self._study(state)
        if effect.purpose == "miprov2_baseline":
            transcript = study.initial_transcript(
                baseline_score=normalized,
                baseline_evaluation=evaluation,
            )
            return state.model_copy(
                update={
                    "phase": "sample",
                    "study_transcript": transcript,
                    "pending_evaluation": None,
                    "pending_evaluation_spec": None,
                    "resolved_eval_binding": None,
                    "fully_evaluated_candidates": fully_evaluated,
                    "completed_effects": completed,
                }
            )
        if effect.purpose == "miprov2_sample":
            assert effect.suggestion is not None
            assert effect.candidate_assembly is not None
            assert state.study_transcript is not None
            promotion = study.promotion_candidate(
                state.study_transcript,
                effect.suggestion,
                score=normalized,
                evaluation=evaluation,
                candidate_assembly=effect.candidate_assembly,
            )
            pending = Miprov2PendingSample(
                suggestion=effect.suggestion,
                score=normalized,
                evaluation=evaluation,
                candidate_identity_hash=effect.candidate.identity_hash(),
                candidate_assembly=effect.candidate_assembly,
            )
            if promotion is not None:
                return state.model_copy(
                    update={
                        "phase": "promotion",
                        "pending_evaluation": None,
                        "pending_evaluation_spec": None,
                        "resolved_eval_binding": None,
                        "pending_sample": pending,
                        "fully_evaluated_candidates": fully_evaluated,
                        "completed_effects": completed,
                    }
                )
            transcript = study.record_sample(
                state.study_transcript,
                effect.suggestion,
                score=normalized,
                evaluation=evaluation,
                candidate_assembly=effect.candidate_assembly,
            )
            return state.model_copy(
                update={
                    "phase": "sample",
                    "study_transcript": transcript,
                    "pending_evaluation": None,
                    "pending_evaluation_spec": None,
                    "resolved_eval_binding": None,
                    "fully_evaluated_candidates": fully_evaluated,
                    "completed_effects": completed,
                }
            )
        assert state.pending_sample is not None
        assert state.study_transcript is not None
        pending = state.pending_sample
        transcript = study.record_sample(
            state.study_transcript,
            pending.suggestion,
            score=pending.score,
            evaluation=pending.evaluation,
            candidate_assembly=pending.candidate_assembly,
            promotion_full_score=normalized,
            promotion_evaluation=evaluation,
        )
        return state.model_copy(
            update={
                "phase": "sample",
                "study_transcript": transcript,
                "pending_evaluation": None,
                "pending_evaluation_spec": None,
                "resolved_eval_binding": None,
                "pending_sample": None,
                "fully_evaluated_candidates": fully_evaluated,
                "completed_effects": completed,
            }
        )

    def _plan_bootstrap(
        self,
        state: Miprov2State,
    ) -> Miprov2DriverPlan:
        plan_index = state.bootstrap_plan_index
        compiler_state = state.bootstrap_state
        demo_candidates = state.demo_candidates
        while plan_index < len(state.bootstrap_plans):
            plan = state.bootstrap_plans[plan_index]
            ordered = self._ordered_labeled_trainset(state, plan)
            if plan.kind is FewshotSeedKind.RESET:
                demos = materialize_reset_demo_set(
                    plan=plan,
                    component_ids=state.control.component_ids,
                )
                plan_index += 1
                compiler_state = None
                demo_candidates = (*demo_candidates, demos)
                continue
            if plan.kind is FewshotSeedKind.LABELS_ONLY:
                demos = materialize_labels_only_demo_set(
                    plan=plan,
                    labeled_trainset=ordered,
                )
                plan_index += 1
                compiler_state = None
                demo_candidates = (*demo_candidates, demos)
                continue
            compiler = compiler_state or initial_compiler_state(plan)
            attempt = next_bootstrap_attempt(plan, compiler)
            if attempt is None:
                demos = materialize_bootstrap_demo_set(
                    plan=plan,
                    state=compiler,
                    labeled_trainset=ordered,
                    component_ids=state.control.component_ids,
                )
                plan_index += 1
                compiler_state = None
                demo_candidates = (*demo_candidates, demos)
                continue
            self._require_budget(state, "bootstrap_rollouts")
            binding_request = Miprov2EvalConfigBindingRequest(
                control_identity_hash=state.control.identity_hash(),
                source_eval_config=state.control.bootstrap_eval_source,
                purpose="bootstrap",
                effect_identity_hash=attempt.identity_hash(),
                execution_policy=_execution_policy(
                    state.control,
                    bootstrap_attempt=attempt,
                ),
                task_batch_identities=(attempt.task_identity,),
            )
            planned = state.model_copy(
                update={
                    "bootstrap_plan_index": plan_index,
                    "bootstrap_state": compiler,
                    "demo_candidates": demo_candidates,
                    "pending_bootstrap": attempt,
                    "pending_bootstrap_candidate": (
                        _materialize_bootstrap_teacher(
                            state=state,
                            plan=plan,
                            attempt=attempt,
                        )
                    ),
                    "pending_eval_binding_request": binding_request,
                }
            )
            return Miprov2DriverPlan(
                state=planned,
                kind="eval_config_binding",
                eval_config_binding=binding_request,
            )
        return self._plan_proposal(
            state.model_copy(
                update={
                    "phase": "proposal",
                    "bootstrap_plan_index": plan_index,
                    "bootstrap_state": None,
                    "pending_bootstrap": None,
                    "pending_bootstrap_candidate": None,
                    "demo_candidates": demo_candidates,
                }
            )
        )

    def _plan_proposal(
        self,
        state: Miprov2State,
    ) -> Miprov2DriverPlan:
        proposal_state = state.proposal_state
        if proposal_state is None:
            context = proposal_demo_context(
                state.demo_candidates,
                zeroshot_opt=state.control.zeroshot_opt,
            )
            bridged = proposal_candidates_from_demo_sets(
                context,
                components=state.proposal_components,
                component_field_order={
                    component_id: cast("tuple[str, ...]", fields)
                    for component_id, fields in (
                        state.component_field_order.items()
                    )
                },
            )
            proposal_state = start_miprov2_proposal(
                bindings=state.bindings,
                optimization_run_identity_hash=state.run.identity_hash,
                components=state.proposal_components,
                trainset=state.proposal_trainset,
                demo_candidates=bridged,
                num_candidates=state.control.num_instruct_candidates,
                view_data_batch_size=state.control.view_data_batch_size,
                init_temperature=state.control.init_temperature,
                data_aware=state.control.data_aware_proposer,
                program_aware=state.control.program_aware_proposer,
                tip_aware=state.control.tip_aware_proposer,
                fewshot_aware=state.control.fewshot_aware_proposer,
                rng_checkpoint=state.rng_checkpoint,
            )
        plan = plan_next_proposal_request(proposal_state)
        if plan.request is not None:
            self._require_budget(state, "proposal_calls")
            planned = state.model_copy(
                update={
                    "proposal_state": plan.state,
                    "pending_proposal": plan.request,
                }
            )
            return Miprov2DriverPlan(
                state=planned,
                kind="proposal_model",
                proposal_request=plan.request,
            )
        demos = study_demo_context(
            state.demo_candidates,
            zeroshot_opt=state.control.zeroshot_opt,
        )
        advanced = state.model_copy(
            update={
                "phase": "baseline",
                "proposal_state": plan.state,
                "rng_checkpoint": plan.state.rng_checkpoint,
                "instruction_pools": plan.state.instruction_pools,
                "study_demo_candidates": demos,
            }
        )
        return self._plan_baseline(advanced)

    def _plan_baseline(
        self,
        state: Miprov2State,
    ) -> Miprov2DriverPlan:
        space = self._space(state)
        params = space.baseline_params
        combination = space.combination_identity_hash(params)
        # DSPy evaluates an untouched deepcopy of the original program before
        # any trial.  Trial-zero params remain all-zero categorical values,
        # but the exact evaluated record is the bound base Candidate.
        candidate = state.control.base_candidate.record
        spec = Miprov2EvaluationSpec(
            run_id=state.run_id,
            ordinal=0,
            purpose="miprov2_baseline",
            candidate=candidate,
            categorical_combination_identity_hash=combination,
            task_batch_identities=state.control.valset_task_identities,
            execution_policy=_execution_policy(state.control),
        )
        return self._plan_evaluation_spec(state, spec)

    def _plan_sample(
        self,
        state: Miprov2State,
    ) -> Miprov2DriverPlan:
        assert state.study_transcript is not None
        study = self._study(state)
        if len(state.study_transcript.samples) >= state.control.num_trials:
            winner = study.best_full_evaluation(state.study_transcript)
            try:
                candidate_ref = next(
                    item
                    for item in state.fully_evaluated_candidates
                    if item.identity_hash
                    == winner.evaluated_candidate_identity_hash
                )
            except StopIteration as exc:
                raise ValueError(
                    "winning full evaluation has no exact CandidateRef"
                ) from exc
            candidate = candidate_ref.record
            stats = (
                _terminal_statistics(state)
                if state.control.track_stats
                else None
            )
            terminal_result = Miprov2TerminalResult(
                winner=candidate_ref,
                winner_score=winner.score,
                track_stats=state.control.track_stats,
                stats=stats,
            )
            completed = state.model_copy(
                update={
                    "phase": "complete",
                    "accepted_candidate": candidate,
                    "accepted_candidate_ref": candidate_ref,
                    "terminal_result": terminal_result,
                }
            )
            return Miprov2DriverPlan(
                state=completed,
                kind="complete",
                accepted_candidate=candidate,
            )
        suggestion = study.suggest_next(state.study_transcript)
        assembly = self._assemble(
            state,
            suggestion.params,
            suggestion.candidate_combination_identity_hash,
        )
        candidate = assembly.candidate.record
        checkpoint = state.rng_checkpoint
        if state.control.minibatch and state.control.minibatch_size < len(
            state.control.valset_task_identities
        ):
            rng = checkpoint.state.restore()
            population = state.control.valset_task_identities
            batch = tuple(rng.sample(population, state.control.minibatch_size))
            checkpoint = checkpoint.append(
                rng=rng,
                phase="evaluation",
                operation="sample",
                arguments=(population, state.control.minibatch_size),
                result=batch,
            )
        else:
            batch = state.control.valset_task_identities
        spec = Miprov2EvaluationSpec(
            run_id=state.run_id,
            ordinal=len(state.completed_effects),
            purpose="miprov2_sample",
            candidate=candidate,
            categorical_combination_identity_hash=(
                suggestion.candidate_combination_identity_hash
            ),
            task_batch_identities=batch,
            execution_policy=_execution_policy(state.control),
            suggestion=suggestion,
            candidate_assembly=assembly,
        )
        return self._plan_evaluation_spec(
            state,
            spec,
            rng_checkpoint=checkpoint,
        )

    def _plan_promotion(
        self,
        state: Miprov2State,
    ) -> Miprov2DriverPlan:
        assert state.study_transcript is not None
        assert state.pending_sample is not None
        study = self._study(state)
        pending = state.pending_sample
        promotion = study.promotion_candidate(
            state.study_transcript,
            pending.suggestion,
            score=pending.score,
            evaluation=pending.evaluation,
            candidate_assembly=pending.candidate_assembly,
        )
        if promotion is None:
            raise ValueError("promotion phase has no due candidate")
        assembly = promotion.candidate_assembly
        candidate = assembly.candidate.record
        spec = Miprov2EvaluationSpec(
            run_id=state.run_id,
            ordinal=len(state.completed_effects),
            purpose="miprov2_promotion",
            candidate=candidate,
            categorical_combination_identity_hash=(
                promotion.candidate_combination_identity_hash
            ),
            task_batch_identities=state.control.valset_task_identities,
            execution_policy=_execution_policy(state.control),
            promotion_candidate=promotion,
            candidate_assembly=assembly,
        )
        return self._plan_evaluation_spec(state, spec)

    def _plan_evaluation_spec(
        self,
        state: Miprov2State,
        spec: Miprov2EvaluationSpec,
        *,
        rng_checkpoint: Miprov2RngCheckpoint | None = None,
    ) -> Miprov2DriverPlan:
        self._require_budget(state, "evaluations")
        binding = state.resolved_eval_binding
        if binding is None:
            if spec.purpose == "miprov2_baseline":
                purpose = "baseline"
            elif spec.purpose == "miprov2_sample":
                purpose = "sample"
            else:
                purpose = "promotion"
            request = Miprov2EvalConfigBindingRequest(
                control_identity_hash=state.control.identity_hash(),
                source_eval_config=state.control.validation_eval_source,
                purpose=purpose,
                effect_identity_hash=spec.identity_hash(),
                execution_policy=spec.execution_policy,
                task_batch_identities=spec.task_batch_identities,
            )
            updates: dict[str, Any] = {
                "pending_evaluation_spec": spec,
                "pending_eval_binding_request": request,
            }
            if rng_checkpoint is not None:
                updates["rng_checkpoint"] = rng_checkpoint
            planned = state.model_copy(update=updates)
            return Miprov2DriverPlan(
                state=planned,
                kind="eval_config_binding",
                eval_config_binding=request,
            )
        if (
            binding.request.effect_identity_hash != spec.identity_hash()
            or binding.request.task_batch_identities
            != spec.task_batch_identities
            or binding.request.execution_policy != spec.execution_policy
        ):
            raise ValueError(
                "resolved Eval Config binding conflicts with evaluation spec"
            )
        effect = Miprov2EvaluationEffect(
            run_id=spec.run_id,
            ordinal=spec.ordinal,
            purpose=spec.purpose,
            candidate=spec.candidate,
            categorical_combination_identity_hash=(
                spec.categorical_combination_identity_hash
            ),
            task_batch_identities=spec.task_batch_identities,
            eval_config=binding.eval_config,
            execution_policy=spec.execution_policy,
            reward_policy_hash=state.control.reward_policy_hash,
            suggestion=spec.suggestion,
            promotion_candidate=spec.promotion_candidate,
            candidate_assembly=spec.candidate_assembly,
        )
        self._require_budget(state, "evaluations")
        updates = {
            "pending_evaluation_spec": spec,
            "pending_evaluation": effect,
        }
        if rng_checkpoint is not None:
            updates["rng_checkpoint"] = rng_checkpoint
        planned = state.model_copy(update=updates)
        kind: Miprov2EffectKind
        if spec.purpose == "miprov2_baseline":
            kind = "baseline_evaluation"
        elif spec.purpose == "miprov2_sample":
            kind = "sample_evaluation"
        else:
            kind = "promotion_evaluation"
        return Miprov2DriverPlan(
            state=planned,
            kind=kind,
            evaluation=effect,
        )

    def _study(self, state: Miprov2State) -> Miprov2Study:
        space = self._space(state)
        schedule = Miprov2StudySchedule(
            num_trials=state.control.num_trials,
            minibatch=state.control.minibatch,
            minibatch_size=state.control.minibatch_size,
            valset_size=len(state.control.valset_task_identities),
            minibatch_full_eval_steps=(
                state.control.minibatch_full_eval_steps
            ),
        )
        return Miprov2Study(
            seed=state.control.seed,
            space=space,
            schedule=schedule,
            run_id=state.run_id,
            validation_task_identities=state.control.valset_task_identities,
            validation_eval_source=state.control.validation_eval_source,
            reward_policy_hash=state.control.reward_policy_hash,
            optimizer_config=state.control.reference(),
            prompt_adapter_identity_hash=(
                state.control.prompt_adapter_identity_hash
            ),
            expected_base_candidate=state.control.base_candidate,
            program_layout=state.control.program_layout,
            run=state.run,
        )

    def _space(self, state: Miprov2State) -> Miprov2ParameterSpace:
        instructions = tuple(
            tuple(_instruction_identity(item) for item in pool)
            for pool in state.instruction_pools
        )
        demos = state.study_demo_candidates
        demo_hashes = (
            None
            if demos is None
            else tuple(
                tuple(
                    _component_demo_projection(
                        item,
                        component_id,
                    ).identity_hash()
                    for item in demos
                )
                for component_id in state.control.component_ids
            )
        )
        return Miprov2ParameterSpace(
            instruction_pool_identity_hashes=instructions,
            demo_pool_identity_hashes=demo_hashes,
        )

    def _assemble(
        self,
        state: Miprov2State,
        params: TrialParams,
        combination_identity: str,
    ) -> Miprov2CandidateAssemblyBinding:
        rendering = _miprov2_candidate_rendering(
            control=state.control,
            instruction_pools=state.instruction_pools,
            demo_candidates=state.study_demo_candidates,
            params=params,
            categorical_combination_identity_hash=combination_identity,
        )
        candidate = candidate_from_components(
            base=state.control.base_candidate,
            candidate_id=f"miprov2-{rendering.identity_hash()[:24]}",
            components=rendering.model_dump(mode="json")["components"],
            run=state.run,
        )
        program_identity_hash = compute_identity_hash(
            schema=MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
            schema_version=MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
            payload={
                "candidate": candidate_reference(candidate).model_dump(
                    mode="json"
                )
            },
        )
        return Miprov2CandidateAssemblyBinding(
            params=params,
            categorical_combination_identity_hash=combination_identity,
            candidate=candidate_reference(candidate),
            program_identity_hash=program_identity_hash,
            rendering=rendering,
            optimizer_config=state.control.reference(),
            base_candidate=state.control.base_candidate,
            program_layout=state.control.program_layout,
            prompt_adapter_identity_hash=(
                state.control.prompt_adapter_identity_hash
            ),
            run=state.run,
        )

    @staticmethod
    def _ordered_labeled_trainset(
        state: Miprov2State,
        plan: FewshotCandidatePlan,
    ) -> tuple[LabeledTaskDemo, ...]:
        by_identity = {
            task.source_task_identity: task for task in state.labeled_trainset
        }
        try:
            return tuple(
                by_identity[identity]
                for identity in plan.trainset_task_identities
            )
        except KeyError as exc:
            raise ValueError(
                "bootstrap plan references a missing labeled task"
            ) from exc

    @staticmethod
    def _validated(state: Miprov2State) -> Miprov2State:
        return Miprov2State.model_validate(state.model_dump(mode="json"))

    @staticmethod
    def _require_new_effect(
        state: Miprov2State,
        effect_identity_hash: str,
    ) -> None:
        if effect_identity_hash in {
            effect.identity_hash for effect in state.completed_effects
        }:
            raise ValueError("MIPROv2 effect has already been folded")

    @staticmethod
    def _require_budget(
        state: Miprov2State,
        label: Literal[
            "bootstrap_rollouts",
            "proposal_calls",
            "evaluations",
        ],
    ) -> None:
        consumed = state.effect_counts[label]
        if consumed >= getattr(state.budget, label):
            raise ValueError(f"MIPROv2 {label} budget exhausted")


__all__ = [
    "MIPROV2_CANDIDATE_RENDERER_SCHEMA",
    "MIPROV2_CANDIDATE_RENDERER_SCHEMA_VERSION",
    "MIPROV2_CANDIDATE_RENDERER_VERSION",
    "MIPROV2_RUNTIME_SCHEMA",
    "MIPROV2_RUNTIME_SCHEMA_VERSION",
    "Miprov2CandidateRendering",
    "Miprov2CompletedEffect",
    "Miprov2ComponentSelection",
    "Miprov2Driver",
    "Miprov2DriverPlan",
    "Miprov2EffectBudget",
    "Miprov2EffectKind",
    "Miprov2EvaluationEffect",
    "Miprov2PendingSample",
    "Miprov2Phase",
    "Miprov2State",
    "Miprov2TerminalResult",
    "Miprov2TerminalStats",
    "render_miprov2_candidate",
]
