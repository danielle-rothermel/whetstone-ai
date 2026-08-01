"""DSPy-compatible, single-prompt COPRO over durable Whetstone primitives."""

from __future__ import annotations

import math
import statistics
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.copro_control import CoproControl
from whetstone.optimization.effect_authority import ReplayPolicy
from whetstone.optimization.identity import (
    ImmutableJsonObject,
    TerminalFailure,
    TypedRef,
    require_full_hash,
)
from whetstone.optimization.mutation import (
    MUTATION_FIELD,
    DiffCheckError,
    candidate_from_draft,
)
from whetstone.optimization.proposal_prompts import (
    copro_proposal_prompt,
)
from whetstone.optimization.proposer import (
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)
from whetstone.optimization.reward import RewardRef
from whetstone.optimization.schema import (
    EVALUATION_EVIDENCE_SCHEMA,
    BudgetDelta,
    Candidate,
    CandidateRef,
    EvaluationBinding,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    OptimizationStepRequest,
    ResolutionClass,
    StepMode,
    StepStatus,
    candidate_reference,
)

COPRO_ADAPTER_KEY = "copro"
SEED_PROPOSAL = "seed_proposal"
HISTORY_PROPOSAL = "history_proposal"


class CoproConfig(BaseModel):
    """COPRO hyperparameters, with the DSPy defaults.

    Whetstone binds DSPy's ``prompt_model`` and ``metric`` constructor
    arguments through, respectively, the adapter's exact
    :class:`ProposerConfig` and the control's exact Evaluation Binding.
    They are deliberately not duplicated as loose string hyperparameters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    breadth: StrictInt = 10
    depth: StrictInt = 3
    init_temperature: float = 1.4
    track_stats: StrictBool = False

    @model_validator(mode="after")
    def _validate(self) -> CoproConfig:
        if self.breadth <= 1:
            raise ValueError("COPRO breadth must be greater than 1")
        if self.depth < 1:
            raise ValueError("COPRO depth must be positive")
        if not math.isfinite(self.init_temperature):
            raise ValueError("COPRO init_temperature must be finite")
        return self


class CoproRoundPlan(BaseModel):
    """A pure, serializable description of one COPRO evaluation round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    iteration: StrictInt
    proposal_mode: str
    proposal_count: StrictInt
    include_initial_candidate: StrictBool
    # DSPy presents the selected best attempts from low score to high score.
    prompt_history: tuple[ImmutableJsonObject, ...] = Field(
        default_factory=tuple
    )


class CoproAttempt(BaseModel):
    """One measured candidate occurrence in the append-only COPRO history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    occurrence_ordinal: StrictInt
    round_index: StrictInt
    run_id: StrictStr
    step_index: StrictInt
    intent_id: StrictStr
    candidate: CandidateRef
    evaluation_binding: EvaluationBinding
    reward: float
    expected_reward_policy_hash: StrictStr
    evaluation_result_ref: TypedRef
    reward_evidence_refs: tuple[TypedRef, ...]
    reward_ref: RewardRef

    @field_validator("reward_evidence_refs", mode="before")
    @classmethod
    def _validate_reward_evidence_refs(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        if type(value) not in (list, tuple):
            raise ValueError(
                f"{info.field_name} must be an ordered tuple or JSON array"
            )
        return value

    @model_validator(mode="after")
    def _validate(self) -> CoproAttempt:
        if self.occurrence_ordinal < 0:
            raise ValueError("COPRO occurrence_ordinal cannot be negative")
        if self.round_index < 0:
            raise ValueError("COPRO round_index cannot be negative")
        if not self.run_id or not self.intent_id:
            raise ValueError("COPRO attempt run_id and intent_id are required")
        if self.step_index != self.round_index:
            raise ValueError(
                "COPRO attempt step_index must equal its round_index"
            )
        if not math.isfinite(self.reward):
            raise ValueError("COPRO attempt reward must be finite")
        require_full_hash(
            self.expected_reward_policy_hash,
            field="expected_reward_policy_hash",
        )
        template = self.candidate.record.payload.get(MUTATION_FIELD)
        if not isinstance(template, str) or not template:
            raise ValueError(
                "COPRO attempt candidate requires a non-empty "
                "user_prompt_template"
            )
        if (
            self.evaluation_result_ref.schema_name
            != EVALUATION_EVIDENCE_SCHEMA
        ):
            raise ValueError(
                "COPRO attempt evaluation_result_ref must use schema "
                f"{EVALUATION_EVIDENCE_SCHEMA!r}"
            )
        reward = self.reward_ref.record
        if reward.value != self.reward:
            raise ValueError(
                "COPRO attempt reward must match its exact Reward"
            )
        if reward.reward_policy_hash != self.expected_reward_policy_hash:
            raise ValueError(
                "COPRO attempt Reward Policy must match its expected hash"
            )
        if reward.evidence_refs != self.reward_evidence_refs:
            raise ValueError(
                "COPRO attempt Reward citations must match its exact Reward"
            )
        if self.evaluation_binding.role is not EvaluationRole.INTERNAL:
            raise ValueError("COPRO attempt requires an internal binding")
        return self

    @property
    def candidate_id(self) -> str:
        return self.candidate.record.candidate_id

    @property
    def template(self) -> str:
        value = self.candidate.record.payload[MUTATION_FIELD]
        assert isinstance(value, str)
        return value

    @classmethod
    def from_resolution(
        cls,
        *,
        occurrence_ordinal: int,
        round_index: int,
        resolution: IntentResolution,
        expected_run_id: str,
        expected_evaluation_binding: EvaluationBinding,
        expected_reward_policy_hash: str,
    ) -> CoproAttempt:
        """Bind an externally loaded Reward to one measured resolution."""

        resolution = IntentResolution.model_validate(
            resolution.model_dump(mode="json")
        )
        if resolution.outcome is not IntentOutcome.COMPLETED:
            raise ValueError("COPRO folds only completed measured resolutions")
        if resolution.detail.classification is not ResolutionClass.MEASURED:
            raise ValueError("COPRO folds only measured resolution details")
        if resolution.reward_ref is None:
            raise ValueError("COPRO measured resolution requires Reward ref")
        if resolution.evaluation_result_ref is None:
            raise ValueError(
                "COPRO measured resolution requires an Evaluation Result ref"
            )
        if (
            resolution.intent.evaluation_binding.role
            is not EvaluationRole.INTERNAL
        ):
            raise ValueError("COPRO folds only internal evaluation intents")
        if resolution.intent.run_id != expected_run_id:
            raise ValueError("COPRO resolution belongs to another run")
        if resolution.intent.step_index != round_index:
            raise ValueError("COPRO resolution belongs to another round")
        if resolution.intent.evaluation_binding != expected_evaluation_binding:
            raise ValueError(
                "COPRO resolution uses an unexpected Evaluation Binding"
            )
        if (
            resolution.intent.expected_reward_policy_hash
            != expected_reward_policy_hash
        ):
            raise ValueError(
                "COPRO resolution expects an unexpected Reward Policy"
            )
        reward_ref = resolution.reward_ref
        reward = reward_ref.record
        if reward.reward_policy_hash != expected_reward_policy_hash:
            raise ValueError("COPRO Reward uses an unexpected Reward Policy")
        return cls(
            occurrence_ordinal=occurrence_ordinal,
            round_index=round_index,
            run_id=expected_run_id,
            step_index=resolution.intent.step_index,
            intent_id=resolution.intent.intent_id,
            candidate=resolution.intent.candidate,
            evaluation_binding=resolution.intent.evaluation_binding,
            reward=reward.value,
            expected_reward_policy_hash=reward.reward_policy_hash,
            evaluation_result_ref=resolution.evaluation_result_ref,
            reward_evidence_refs=resolution.reward_evidence_refs,
            reward_ref=reward_ref,
        )

    def prompt_entry(self) -> ImmutableJsonObject:
        return ImmutableJsonObject(
            {
                "occurrence_ordinal": self.occurrence_ordinal,
                "candidate_id": self.candidate_id,
                "template": self.template,
                "reward": self.reward,
            }
        )


class CoproState(BaseModel):
    """Durable algorithm state reconstructed from measured occurrences."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_candidate: Candidate
    completed_rounds: StrictInt = 0
    attempts: tuple[CoproAttempt, ...] = ()
    total_calls: StrictInt = 0

    @model_validator(mode="after")
    def _validate(self) -> CoproState:
        if self.completed_rounds < 0 or self.total_calls < 0:
            raise ValueError("COPRO state counters cannot be negative")
        if self.total_calls != len(self.attempts):
            raise ValueError("COPRO total_calls must equal folded occurrences")
        return self


class CoproStatisticsSeries(BaseModel):
    """DSPy's statistics keys for one single-prompt predictor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: tuple[int, ...]
    max: tuple[float, ...]
    average: tuple[float, ...]
    min: tuple[float, ...]
    std: tuple[float, ...]


class CoproStatistics(BaseModel):
    """DSPy-equivalent statistics projected from durable round observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_calls: StrictInt
    results_latest: CoproStatisticsSeries
    results_best: CoproStatisticsSeries


class CoproFinalization(BaseModel):
    """Terminal COPRO ranking with unconditional call accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ranked_attempts: tuple[CoproAttempt, ...]
    total_calls: StrictInt
    statistics: CoproStatistics | None = None


def attempt_history_entries(
    request: OptimizationStepRequest,
) -> tuple[CoproAttempt, ...]:
    """Read the append-only measured-attempt stream from a step request."""

    raw = request.pools.get("attempt_history", ())
    if type(raw) is not tuple:
        raise ValueError("attempt_history must be a JSON list")
    attempts: list[CoproAttempt] = []
    for ordinal, item in enumerate(raw):
        if not isinstance(item, ImmutableJsonObject):
            raise ValueError(
                f"attempt_history[{ordinal}] must be a JSON record"
            )
        attempts.append(CoproAttempt.model_validate(item.to_json()))
    return tuple(attempts)


def _unique_measured_attempts(
    entries: tuple[CoproAttempt, ...],
) -> tuple[CoproAttempt, ...]:
    """Keep the best observation per template in first-seen template order.

    DSPy keys its evaluated-candidate mapping by the complete mutable prompt
    assignment. Whetstone's COPRO mutation surface contains only
    ``user_prompt_template``, so template text is the corresponding key.
    Replacing a duplicate with a better observation does not move the key,
    preserving DSPy's stable first-seen ordering for score ties.
    """

    unique: dict[str, CoproAttempt] = {}
    for entry in entries:
        prior = unique.get(entry.template)
        if prior is None or entry.reward > prior.reward:
            unique[entry.template] = entry
    return tuple(unique.values())


def rank_attempt_history(
    entries: tuple[CoproAttempt, ...],
) -> tuple[CoproAttempt, ...]:
    """Return unique measured attempts best-first with stable score ties."""

    unique = _unique_measured_attempts(entries)

    return tuple(sorted(unique, key=lambda entry: -entry.reward))


def _score_summary(scores: list[float]) -> tuple[float, float, float, float]:
    if not scores:
        raise ValueError("COPRO statistics require measured scores")
    return (
        max(scores),
        sum(scores) / len(scores),
        min(scores),
        statistics.pstdev(scores),
    )


class CoproDriver:
    """Pure owner of COPRO round planning, history selection, and ranking."""

    def __init__(self, config: CoproConfig) -> None:
        self.config = config

    def plan_round(
        self,
        *,
        iteration: int,
        initial_candidates: tuple[Candidate, ...],
        attempt_history: tuple[CoproAttempt, ...],
    ) -> CoproRoundPlan:
        if len(initial_candidates) != 1:
            raise ValueError(
                "single-prompt COPRO requires exactly one initial candidate"
            )
        if iteration < 0 or iteration >= self.config.depth:
            raise ValueError("COPRO iteration exceeds configured depth")
        if iteration == 0:
            if rank_attempt_history(attempt_history):
                raise ValueError(
                    "COPRO seed round cannot consume measured history"
                )
            return CoproRoundPlan(
                iteration=iteration,
                proposal_mode=SEED_PROPOSAL,
                proposal_count=self.config.breadth - 1,
                include_initial_candidate=True,
            )

        ranked = rank_attempt_history(attempt_history)
        if not ranked:
            raise ValueError("COPRO history round requires measured history")
        selected_best_first = ranked[: self.config.breadth]
        return CoproRoundPlan(
            iteration=iteration,
            proposal_mode=HISTORY_PROPOSAL,
            proposal_count=self.config.breadth,
            include_initial_candidate=False,
            prompt_history=tuple(
                attempt.prompt_entry()
                for attempt in reversed(selected_best_first)
            ),
        )

    @staticmethod
    def terminal_ranking(
        attempt_history: tuple[CoproAttempt, ...],
    ) -> tuple[CoproAttempt, ...]:
        """Return DSPy's unique, descending-score terminal candidate order."""

        return rank_attempt_history(attempt_history)

    def initial_state(self, initial_candidate: Candidate) -> CoproState:
        return CoproState(initial_candidate=initial_candidate)

    def fold_round(
        self,
        state: CoproState,
        attempts: tuple[CoproAttempt, ...],
    ) -> CoproState:
        """Advance state by one breadth-sized measured occurrence batch."""

        if len(state.attempts) != state.completed_rounds * self.config.breadth:
            raise ValueError(
                "COPRO state occurrence count does not match completed rounds"
            )
        if state.completed_rounds >= self.config.depth:
            raise ValueError("COPRO state already contains configured depth")
        if len(attempts) != self.config.breadth:
            raise ValueError(
                "COPRO round requires exactly breadth measured occurrences"
            )
        start = state.completed_rounds * self.config.breadth
        prior_intent_ids = {attempt.intent_id for attempt in state.attempts}
        expected_run_id = (
            state.attempts[0].run_id if state.attempts else attempts[0].run_id
        )
        expected_evaluation_binding = (
            state.attempts[0].evaluation_binding
            if state.attempts
            else attempts[0].evaluation_binding
        )
        expected_reward_policy_hash = (
            state.attempts[0].expected_reward_policy_hash
            if state.attempts
            else attempts[0].expected_reward_policy_hash
        )
        for offset, attempt in enumerate(attempts):
            expected = start + offset
            if attempt.occurrence_ordinal != expected:
                raise ValueError(
                    "COPRO occurrence ordinals must be contiguous in "
                    "evaluation order"
                )
            if attempt.round_index != state.completed_rounds:
                raise ValueError(
                    "COPRO attempt round_index does not match state"
                )
            if attempt.run_id != expected_run_id:
                raise ValueError("COPRO attempts span multiple runs")
            if attempt.evaluation_binding != expected_evaluation_binding:
                raise ValueError(
                    "COPRO attempts span multiple Evaluation Bindings"
                )
            if (
                attempt.expected_reward_policy_hash
                != expected_reward_policy_hash
            ):
                raise ValueError(
                    "COPRO attempts span multiple Reward Policies"
                )
            if attempt.intent_id in prior_intent_ids:
                raise ValueError("COPRO attempt intent IDs must be unique")
            prior_intent_ids.add(attempt.intent_id)
            initial_fixed = {
                key: value
                for key, value in state.initial_candidate.payload.items()
                if key != MUTATION_FIELD
            }
            attempt_fixed = {
                key: value
                for key, value in attempt.candidate.record.payload.items()
                if key != MUTATION_FIELD
            }
            if attempt_fixed != initial_fixed:
                raise ValueError(
                    "COPRO attempt changes a field outside "
                    "user_prompt_template"
                )
        return CoproState(
            initial_candidate=state.initial_candidate,
            completed_rounds=state.completed_rounds + 1,
            attempts=state.attempts + attempts,
            total_calls=state.total_calls + len(attempts),
        )

    def restore_state(
        self,
        *,
        initial_candidate: Candidate,
        attempts: tuple[CoproAttempt, ...],
    ) -> CoproState:
        """Reconstruct state fail-closed for fresh or restarted controllers."""

        if len(attempts) % self.config.breadth:
            raise ValueError(
                "COPRO history ends with a partial evaluation round"
            )
        state = self.initial_state(initial_candidate)
        for start in range(0, len(attempts), self.config.breadth):
            state = self.fold_round(
                state,
                attempts[start : start + self.config.breadth],
            )
        return state

    def advance(self, state: CoproState) -> CoproRoundPlan:
        """Plan the one next round from exact durable state."""

        if (
            self.restore_state(
                initial_candidate=state.initial_candidate,
                attempts=state.attempts,
            )
            != state
        ):
            raise ValueError(
                "COPRO state does not match its occurrence history"
            )
        if state.completed_rounds >= self.config.depth:
            raise ValueError("COPRO has no round remaining to advance")
        return self.plan_round(
            iteration=state.completed_rounds,
            initial_candidates=(state.initial_candidate,),
            attempt_history=state.attempts,
        )

    def finalize(self, state: CoproState) -> CoproFinalization:
        """Finish only a complete run; call accounting is always returned."""

        if (
            self.restore_state(
                initial_candidate=state.initial_candidate,
                attempts=state.attempts,
            )
            != state
        ):
            raise ValueError(
                "COPRO state does not match its occurrence history"
            )
        if state.completed_rounds != self.config.depth:
            raise ValueError("COPRO cannot finalize before configured depth")
        rounds = tuple(
            state.attempts[start : start + self.config.breadth]
            for start in range(0, len(state.attempts), self.config.breadth)
        )
        return CoproFinalization(
            ranked_attempts=self.terminal_ranking(state.attempts),
            total_calls=state.total_calls,
            statistics=self.statistics(rounds)
            if self.config.track_stats
            else None,
        )

    def statistics(
        self,
        rounds: tuple[tuple[CoproAttempt, ...], ...],
    ) -> CoproStatistics:
        """Project DSPy's optional statistics from occurrence-level rounds.

        ``rounds`` must retain every evaluated occurrence, including
        duplicates. ``results_latest`` therefore summarizes the complete
        breadth-sized round, while ``results_best`` summarizes the top ten
        unique retained observations after that round.
        """

        if len(rounds) != self.config.depth:
            raise ValueError(
                "COPRO statistics require exactly configured depth rounds"
            )
        cumulative: list[CoproAttempt] = []
        latest_summaries: list[tuple[float, float, float, float]] = []
        best_summaries: list[tuple[float, float, float, float]] = []
        total_calls = 0
        for round_entries in rounds:
            if len(round_entries) != self.config.breadth:
                raise ValueError(
                    "COPRO statistics require one breadth-sized batch "
                    "per depth"
                )
            measured_scores = [entry.reward for entry in round_entries]
            total_calls += len(round_entries)
            cumulative.extend(round_entries)
            unique_best = rank_attempt_history(tuple(cumulative))[:10]
            latest_summaries.append(_score_summary(measured_scores))
            best_summaries.append(
                _score_summary([entry.reward for entry in unique_best])
            )
        depths = tuple(range(len(rounds)))

        def series(
            summaries: list[tuple[float, float, float, float]],
        ) -> CoproStatisticsSeries:
            return CoproStatisticsSeries(
                depth=depths,
                max=tuple(item[0] for item in summaries),
                average=tuple(item[1] for item in summaries),
                min=tuple(item[2] for item in summaries),
                std=tuple(item[3] for item in summaries),
            )

        return CoproStatistics(
            total_calls=total_calls,
            results_latest=series(latest_summaries),
            results_best=series(best_summaries),
        )


def _normalize_initial_candidate(
    candidate: Candidate,
    request: OptimizationStepRequest,
) -> Candidate:
    raw = candidate.payload.get(MUTATION_FIELD)
    if not isinstance(raw, str):
        raise ValueError(
            "COPRO initial candidate requires user_prompt_template"
        )
    request.run.record.template_render_contract.validate_template(raw)
    return candidate


def _validate_attempt_placeholders(
    attempts: tuple[CoproAttempt, ...],
    request: OptimizationStepRequest,
) -> None:
    for attempt in attempts:
        try:
            request.run.record.template_render_contract.validate_template(
                attempt.template
            )
        except ValueError as error:
            raise ValueError(
                "COPRO history violates the run Template Render Contract at "
                f"occurrence {attempt.occurrence_ordinal}: {error}"
            ) from error


class CoproAdapter:
    """Plan one COPRO round and emit an exact intent for each candidate."""

    def __init__(
        self,
        *,
        control: CoproControl,
        transport: ProposerTransport,
    ) -> None:
        self._control = control
        self._transport = transport
        self.invocations = 0

    @property
    def key(self) -> str:
        return COPRO_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.NO_REDRIVE

    @property
    def proposer_config(self) -> ProposerConfig:
        return self._control.prompt_model

    @property
    def control(self) -> CoproControl:
        return self._control

    @property
    def provider_execution_policy_hash(self) -> str:
        return self._transport.execution_policy_hash

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return self._transport.prompt_adapter_identity_hash

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[Any, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        if handles:
            raise ValueError("COPRO receives no Runtime Tool Handles")

        if request.run.record.optimizer_config != self._control.reference():
            raise ValueError(
                "COPRO run optimizer_config does not bind the exact control"
            )
        iteration = request.hyperparameters.get("round_index")
        if type(iteration) is not int:
            raise ValueError("COPRO round_index must be an integer")
        expected_hyperparameters = ImmutableJsonObject(
            self._control.step_hyperparameters(iteration=iteration)
        )
        if request.hyperparameters != expected_hyperparameters:
            raise ValueError(
                "COPRO step hyperparameters do not match the exact control"
            )
        config = CoproConfig(
            breadth=self._control.breadth,
            depth=self._control.depth,
            init_temperature=self._control.init_temperature,
            track_stats=self._control.track_stats,
        )
        expected_policy_hash = self._control.provider_execution_policy_hash
        expected_adapter_hash = self._control.prompt_adapter_identity_hash
        if self.provider_execution_policy_hash != expected_policy_hash:
            raise ValueError(
                "COPRO provider execution policy conflicts with request"
            )
        if self.prompt_adapter_identity_hash != expected_adapter_hash:
            raise ValueError(
                "COPRO prompt adapter identity conflicts with request"
            )
        expected_reward_policy_hash = self._control.expected_reward_policy_hash
        run_reward_policy = request.run.record.reward_policy
        if run_reward_policy is None:
            raise ValueError("COPRO requires the exact run Reward Policy")
        if run_reward_policy.identity_hash() != expected_reward_policy_hash:
            raise ValueError(
                "COPRO expected Reward Policy conflicts with the exact run"
            )
        evaluation_binding = self._control.evaluation_binding
        if len(request.candidates) != 1:
            raise ValueError(
                "single-prompt COPRO requires exactly one initial candidate"
            )
        initial = _normalize_initial_candidate(request.candidates[0], request)
        history = attempt_history_entries(request)
        for attempt in history:
            if attempt.run_id != request.run_id:
                raise ValueError("COPRO history belongs to another run")
            if attempt.evaluation_binding != evaluation_binding:
                raise ValueError(
                    "COPRO history uses an unexpected Evaluation Binding"
                )
            if (
                attempt.expected_reward_policy_hash
                != expected_reward_policy_hash
            ):
                raise ValueError(
                    "COPRO history uses an unexpected Reward Policy"
                )
        _validate_attempt_placeholders(history, request)
        driver = CoproDriver(config)
        state = driver.restore_state(
            initial_candidate=initial,
            attempts=history,
        )
        if iteration != state.completed_rounds:
            raise ValueError(
                "COPRO round_index does not match durable measured history"
            )
        plan = driver.advance(state)
        remaining = request.budget.remaining.get("proposal_calls")
        if remaining is not None and type(remaining) is not int:
            raise TypeError(
                "validated proposal_calls budget is not an integer"
            )
        if remaining is not None and remaining < plan.proposal_count:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                terminal_failure=TerminalFailure(
                    code="copro_proposal_budget_exhausted",
                    message="COPRO proposal budget is exhausted",
                    details={
                        "required": plan.proposal_count,
                        "remaining": remaining,
                    },
                ),
                state_delta={
                    "reason": "proposal budget exhausted",
                    "required": plan.proposal_count,
                    "remaining": remaining,
                },
            )

        ranked = driver.terminal_ranking(history)
        base = initial
        context: dict[str, Any] = {
            "prompt_history": [item.to_json() for item in plan.prompt_history],
        }
        proposal_request = ProposalRequest(
            proposal_mode=plan.proposal_mode,
            request_ordinal=iteration,
            base_candidate=candidate_reference(base),
            context=context,
        )
        prompt = copro_proposal_prompt(proposal_request)
        proposal_request = ProposalRequest(
            proposal_mode=proposal_request.proposal_mode,
            request_ordinal=proposal_request.request_ordinal,
            base_candidate=proposal_request.base_candidate,
            context={**context, "proposal_prompt": prompt},
        )
        drafts = self._transport.draft(
            self._control.prompt_model,
            proposal_request,
            plan.proposal_count,
        )

        occurrences: list[tuple[int, Candidate]] = []
        rejected: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        round_start = iteration * config.breadth
        reserved_candidate_ids = {initial.candidate_id}
        for index, draft in enumerate(drafts):
            occurrence_ordinal = round_start + index
            candidate_id = f"copro:{request.run_id}:{occurrence_ordinal}"
            while candidate_id in reserved_candidate_ids:
                candidate_id += ":generated"
            reserved_candidate_ids.add(candidate_id)
            # Match DSPy's candidate normalization. Validation remains a
            # Whetstone post-generation concern, not proposer-prompt content.
            template = draft.template.strip('"').strip()
            disposition = "accepted"
            reason: str | None = None
            try:
                normalized_draft = (
                    draft
                    if draft.failed
                    else draft.model_copy(update={"template": template})
                )
                candidate = candidate_from_draft(
                    base=base,
                    candidate_id=candidate_id,
                    draft=normalized_draft,
                    run=request.run,
                )
            except DiffCheckError as exc:
                disposition = "provider_failed" if draft.failed else "rejected"
                reason = str(exc)
                rejected.append(
                    {
                        "occurrence_ordinal": occurrence_ordinal,
                        "candidate_id": candidate_id,
                        "disposition": disposition,
                        "reason": reason,
                    }
                )
            else:
                occurrences.append((occurrence_ordinal, candidate))
            evidence.append(
                {
                    "occurrence_ordinal": occurrence_ordinal,
                    "candidate_id": candidate_id,
                    "disposition": disposition,
                    "reason": reason,
                    "request": draft.request_evidence.to_json(),
                    "response": draft.response_evidence.to_json(),
                    "usage": draft.usage.to_json(),
                    "cost": draft.cost,
                }
            )

        if plan.include_initial_candidate:
            occurrences.append((round_start + config.breadth - 1, initial))
        for index in range(len(drafts), plan.proposal_count):
            occurrence_ordinal = round_start + index
            evidence.append(
                {
                    "occurrence_ordinal": occurrence_ordinal,
                    "candidate_id": (
                        f"copro:{request.run_id}:{occurrence_ordinal}"
                    ),
                    "disposition": "missing",
                    "reason": "transport returned no draft for paid slot",
                    "request": {},
                    "response": {},
                    "usage": {},
                    "cost": None,
                }
            )
        proposed = [
            candidate
            for _, candidate in occurrences
            if candidate is not initial
        ]
        if (
            len(drafts) != plan.proposal_count
            or len(occurrences) != config.breadth
        ):
            return AdapterOutput(
                proposed_candidates=tuple(proposed),
                proposed_status=StepStatus.FAILED,
                terminal_failure=TerminalFailure(
                    code="copro_proposal_cardinality",
                    message="COPRO proposer failed to fill its round",
                    details={
                        "expected_occurrences": config.breadth,
                        "actual_occurrences": len(occurrences),
                    },
                ),
                budget_delta=BudgetDelta(
                    consumed={"proposal_calls": plan.proposal_count}
                ),
                state_delta={
                    "reason": "proposal cardinality",
                    "rejected": rejected,
                    "round_plan": plan.model_dump(mode="json"),
                    "proposer_evidence": evidence,
                },
            )

        intents: list[EvaluationIntent] = []
        for occurrence_ordinal, candidate in occurrences:
            candidate_ref = candidate_reference(candidate)
            intents.append(
                EvaluationIntent(
                    intent_id=(
                        f"{request.run_id}:{request.step_index}:"
                        f"{occurrence_ordinal}:{candidate_ref.identity_hash}"
                    ),
                    candidate=candidate_ref,
                    target_eval_config=evaluation_binding.eval_config,
                    evaluation_binding=evaluation_binding,
                    purpose=plan.proposal_mode,
                    run_id=request.run_id,
                    step_index=request.step_index,
                    expected_reward_policy_hash=(expected_reward_policy_hash),
                )
            )
        return AdapterOutput(
            proposed_candidates=tuple(proposed),
            accepted_candidates=tuple(proposed),
            evaluation_intents=tuple(intents),
            budget_delta=BudgetDelta(
                consumed={"proposal_calls": plan.proposal_count}
            ),
            # COPRO selection/finalization is always controller-owned after
            # the final round's external resolutions have been folded.
            proposed_status=StepStatus.CONTINUE,
            state_delta={
                "copro_config": config.model_dump(mode="json"),
                "round_plan": plan.model_dump(mode="json"),
                "globally_best_measured": (
                    ranked[0].model_dump(mode="json") if ranked else None
                ),
                "terminal_ranking": [
                    item.model_dump(mode="json") for item in ranked
                ],
                "proposer_evidence": evidence,
            },
            history_delta={
                "prior_entries": [
                    item.model_dump(mode="json") for item in history
                ],
                "proposed_candidate_ids": [
                    candidate.candidate_id for candidate in proposed
                ],
                "occurrence_ordinals": [ordinal for ordinal, _ in occurrences],
            },
        )


__all__ = [
    "COPRO_ADAPTER_KEY",
    "HISTORY_PROPOSAL",
    "SEED_PROPOSAL",
    "CoproAdapter",
    "CoproAttempt",
    "CoproConfig",
    "CoproDriver",
    "CoproFinalization",
    "CoproRoundPlan",
    "CoproState",
    "CoproStatistics",
    "CoproStatisticsSeries",
    "attempt_history_entries",
    "rank_attempt_history",
]
