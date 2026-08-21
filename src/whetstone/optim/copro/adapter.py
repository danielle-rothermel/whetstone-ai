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

from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    ImmutableJsonObject,
    TerminalFailure,
    TypedRef,
    require_full_hash,
)
from whetstone.core.roles import EvalRole
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.core.identity import IdentityRef
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.experiment.reward import RewardRef
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import (
    BudgetDelta,
    OptimEvalRequest,
    IntentOutcome,
    IntentResolution,
    OptimStepRequest,
    ResolutionClass,
    StepMode,
    StepStatus,
)
from whetstone.optim.copro.control import (
    CoproControl,
    CoproProposerConfig,
)
from whetstone.optim.proposal.mutation import (
    DiffCheckError,
    candidate_from_draft,
    resolve_mutation_field,
)
from whetstone.optim.copro.prompts import (
    COPRO_INSTRUCTION_CONTRACT_KEY,
    COPRO_INSTRUCTION_HISTORY_KEY,
    copro_proposal_prompt,
)
from whetstone.optim.proposal.proposer import (
    DurableProposalExecutor,
    ProposalRequest,
    ProposerTransport,
    require_canonical_proposal_executor,
)

COPRO_ADAPTER_KEY = "copro"
SEED_PROPOSAL = "seed_proposal"
HISTORY_PROPOSAL = "history_proposal"


class CoproConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    breadth: StrictInt = 10
    depth: StrictInt = 3
    track_stats: StrictBool = False

    @model_validator(mode="after")
    def _validate(self) -> CoproConfig:
        if self.breadth <= 1:
            raise ValueError("COPRO breadth must be greater than 1")
        if self.depth < 1:
            raise ValueError("COPRO depth must be positive")
        return self


class CoproRoundPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    iteration: StrictInt
    proposal_mode: str
    proposal_count: StrictInt
    include_initial_candidate: StrictBool

    instruction_history: tuple[ImmutableJsonObject, ...] = Field(
        default_factory=tuple
    )


class CoproAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    occurrence_ordinal: StrictInt
    round_index: StrictInt
    run_id: StrictStr
    step_index: StrictInt
    intent_id: StrictStr
    candidate: CandidateRef
    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
    provider_execution_policy_ref: IdentityRef | None = None
    reward: float
    expected_reward_policy_hash: StrictStr
    eval_result_ref: TypedRef
    reward_evidence_refs: tuple[TypedRef, ...]
    reward_ref: RewardRef
    mutation_field: StrictStr = "user_prompt_template"

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
        template = self.candidate.record.payload.get(self.mutation_field)
        if not isinstance(template, str) or not template:
            raise ValueError(
                "COPRO attempt candidate requires a non-empty instruction"
            )
        if (
            self.eval_result_ref.schema_name
            != EVAL_EVIDENCE_SCHEMA
        ):
            raise ValueError(
                "COPRO attempt eval_result_ref must use schema "
                f"{EVAL_EVIDENCE_SCHEMA!r}"
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
        if self.eval_role is not EvalRole.INTERNAL:
            raise ValueError("COPRO attempt requires internal evaluation")
        return self

    @property
    def candidate_id(self) -> str:
        return self.candidate.record.candidate_id

    @property
    def instruction(self) -> str:
        value = self.candidate.record.payload[self.mutation_field]
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
        expected_eval_config_ref: EvalConfigRef,
        expected_eval_role: EvalRole,
        expected_provider_execution_policy_ref: IdentityRef | None,
        expected_reward_policy_hash: str,
        mutation_field: str,
    ) -> CoproAttempt:

        resolution = IntentResolution.model_validate(
            resolution.model_dump(mode="json")
        )
        if resolution.outcome is not IntentOutcome.COMPLETED:
            raise ValueError("COPRO folds only completed measured resolutions")
        if resolution.detail.classification is not ResolutionClass.MEASURED:
            raise ValueError("COPRO folds only measured resolution details")
        if resolution.reward_ref is None:
            raise ValueError("COPRO measured resolution requires Reward ref")
        if resolution.eval_result_ref is None:
            raise ValueError(
                "COPRO measured resolution requires an Evaluation Result ref"
            )
        optim_eval_request = resolution.optim_eval_request
        if optim_eval_request.expected_reward_policy_hash is None:
            raise ValueError("COPRO folds only internal evaluation intents")
        if optim_eval_request.optim_run_id != expected_run_id:
            raise ValueError("COPRO resolution belongs to another run")
        if optim_eval_request.optim_step_index != round_index:
            raise ValueError("COPRO resolution belongs to another round")
        if resolution.resolved_eval_config != expected_eval_config_ref:
            raise ValueError(
                "COPRO resolution uses an unexpected Eval Config"
            )
        if expected_eval_role is not EvalRole.INTERNAL:
            raise ValueError("COPRO folds only internal evaluation intents")
        if (
            optim_eval_request.expected_reward_policy_hash
            != expected_reward_policy_hash
        ):
            raise ValueError(
                "COPRO resolution expects an unexpected Reward Policy"
            )
        reward_ref = resolution.reward_ref
        reward = reward_ref.record
        if reward.reward_policy_hash != expected_reward_policy_hash:
            raise ValueError("COPRO Reward uses an unexpected Reward Policy")
        candidate_ref = candidate_reference(
            optim_eval_request.eval_request.candidate
        )
        return cls(
            occurrence_ordinal=occurrence_ordinal,
            round_index=round_index,
            run_id=expected_run_id,
            step_index=optim_eval_request.optim_step_index,
            intent_id=optim_eval_request.eval_request.request_id,
            candidate=candidate_ref,
            eval_config_ref=resolution.resolved_eval_config,
            eval_role=expected_eval_role,
            provider_execution_policy_ref=(
                expected_provider_execution_policy_ref
            ),
            reward=reward.value,
            expected_reward_policy_hash=reward.reward_policy_hash,
            eval_result_ref=resolution.eval_result_ref,
            reward_evidence_refs=resolution.reward_evidence_refs,
            reward_ref=reward_ref,
            mutation_field=mutation_field,
        )

    def prompt_entry(self) -> ImmutableJsonObject:
        return ImmutableJsonObject(
            {
                "occurrence_ordinal": self.occurrence_ordinal,
                "candidate_id": self.candidate_id,
                "instruction": self.instruction,
                "reward": self.reward,
            }
        )


class CoproState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_candidate: Candidate
    mutation_field: StrictStr = "user_prompt_template"
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
    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: tuple[int, ...]
    max: tuple[float, ...]
    average: tuple[float, ...]
    min: tuple[float, ...]
    std: tuple[float, ...]


class CoproStatistics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_calls: StrictInt
    results_latest: CoproStatisticsSeries
    results_best: CoproStatisticsSeries


class CoproFinalization(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ranked_attempts: tuple[CoproAttempt, ...]
    total_calls: StrictInt
    statistics: CoproStatistics | None = None


def attempt_history_entries(
    request: OptimStepRequest,
) -> tuple[CoproAttempt, ...]:

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

    unique: dict[str, CoproAttempt] = {}
    for entry in entries:
        prior = unique.get(entry.instruction)
        if prior is None or entry.reward > prior.reward:
            unique[entry.instruction] = entry
    return tuple(unique.values())


def rank_attempt_history(
    entries: tuple[CoproAttempt, ...],
) -> tuple[CoproAttempt, ...]:

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
            instruction_history=tuple(
                attempt.prompt_entry()
                for attempt in reversed(selected_best_first)
            ),
        )

    @staticmethod
    def terminal_ranking(
        attempt_history: tuple[CoproAttempt, ...],
    ) -> tuple[CoproAttempt, ...]:

        return rank_attempt_history(attempt_history)

    def initial_state(
        self,
        initial_candidate: Candidate,
        *,
        mutation_field: str,
    ) -> CoproState:
        return CoproState(
            initial_candidate=initial_candidate,
            mutation_field=mutation_field,
        )

    def fold_round(
        self,
        state: CoproState,
        attempts: tuple[CoproAttempt, ...],
    ) -> CoproState:

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
        expected_eval_config_ref = (
            state.attempts[0].eval_config_ref
            if state.attempts
            else attempts[0].eval_config_ref
        )
        expected_eval_role = (
            state.attempts[0].eval_role
            if state.attempts
            else attempts[0].eval_role
        )
        expected_provider_execution_policy_ref = (
            state.attempts[0].provider_execution_policy_ref
            if state.attempts
            else attempts[0].provider_execution_policy_ref
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
            if attempt.eval_config_ref != expected_eval_config_ref:
                raise ValueError(
                    "COPRO attempts span multiple Eval Configs"
                )
            if attempt.eval_role != expected_eval_role:
                raise ValueError(
                    "COPRO attempts span multiple Evaluation Roles"
                )
            if (
                attempt.provider_execution_policy_ref
                != expected_provider_execution_policy_ref
            ):
                raise ValueError(
                    "COPRO attempts span multiple Provider Execution Policies"
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
            field = state.mutation_field
            initial_fixed = {
                key: value
                for key, value in state.initial_candidate.payload.items()
                if key != field
            }
            attempt_fixed = {
                key: value
                for key, value in attempt.candidate.record.payload.items()
                if key != field
            }
            if attempt_fixed != initial_fixed:
                raise ValueError(
                    "COPRO attempt changes a field outside the instruction"
                )
        return CoproState(
            initial_candidate=state.initial_candidate,
            mutation_field=state.mutation_field,
            completed_rounds=state.completed_rounds + 1,
            attempts=state.attempts + attempts,
            total_calls=state.total_calls + len(attempts),
        )

    def restore_state(
        self,
        *,
        initial_candidate: Candidate,
        attempts: tuple[CoproAttempt, ...],
        mutation_field: str,
    ) -> CoproState:

        if len(attempts) % self.config.breadth:
            raise ValueError(
                "COPRO history ends with a partial evaluation round"
            )
        state = self.initial_state(
            initial_candidate,
            mutation_field=mutation_field,
        )
        for start in range(0, len(attempts), self.config.breadth):
            state = self.fold_round(
                state,
                attempts[start : start + self.config.breadth],
            )
        return state

    def advance(self, state: CoproState) -> CoproRoundPlan:

        if (
            self.restore_state(
                initial_candidate=state.initial_candidate,
                attempts=state.attempts,
                mutation_field=state.mutation_field,
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

        if (
            self.restore_state(
                initial_candidate=state.initial_candidate,
                attempts=state.attempts,
                mutation_field=state.mutation_field,
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
    request: OptimStepRequest,
) -> Candidate:
    field = resolve_mutation_field(run=request.run)
    raw = candidate.payload.get(field)
    if not isinstance(raw, str):
        raise ValueError("COPRO initial candidate requires an instruction")
    request.run.record.template_render_contract.validate_template(raw)
    return candidate


def _validate_attempt_placeholders(
    attempts: tuple[CoproAttempt, ...],
    request: OptimStepRequest,
) -> None:
    for attempt in attempts:
        try:
            request.run.record.template_render_contract.validate_template(
                attempt.instruction
            )
        except ValueError as error:
            raise ValueError(
                "COPRO history violates the run Template Render Contract at "
                f"occurrence {attempt.occurrence_ordinal}: {error}"
            ) from error


class CoproAdapter:
    def __init__(
        self,
        *,
        control: CoproControl,
        transport: ProposerTransport,
        proposal_executor: DurableProposalExecutor,
    ) -> None:
        require_canonical_proposal_executor(
            proposal_executor,
            algorithm="COPRO",
            purpose="paid proposal call",
        )
        self._control = control
        self._transport = transport
        self._proposal_executor = proposal_executor
        self.invocations = 0

    @property
    def key(self) -> str:
        return COPRO_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return self._proposal_executor.recovery_policy

    @property
    def proposal_executor(self) -> DurableProposalExecutor:
        return self._proposal_executor

    @property
    def proposer_config(self) -> CoproProposerConfig:
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
        request: OptimStepRequest,
        handles: tuple[Any, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        if handles:
            raise ValueError("COPRO receives no Runtime Tool Handles")

        if request.run.record.optimizer_config != self._control.reference():
            raise ValueError(
                "COPRO run optimizer_config does not bind the exact control"
            )
        config = CoproConfig(
            breadth=self._control.breadth,
            depth=self._control.depth,
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
        eval_config_ref = self._control.eval_config_ref
        eval_role = self._control.eval_role
        provider_execution_policy_ref = self._control.provider_execution_policy_ref
        if len(request.candidates) != 1:
            raise ValueError(
                "single-prompt COPRO requires exactly one initial candidate"
            )
        initial = _normalize_initial_candidate(request.candidates[0], request)
        mutation_field = resolve_mutation_field(run=request.run)
        self._control.proposal_contract.validate_instruction(
            str(initial.payload[mutation_field])
        )
        history = attempt_history_entries(request)
        for attempt in history:
            if attempt.run_id != request.run_id:
                raise ValueError("COPRO history belongs to another run")
            if attempt.eval_config_ref != eval_config_ref:
                raise ValueError(
                    "COPRO history uses an unexpected Eval Config"
                )
            if attempt.eval_role != eval_role:
                raise ValueError(
                    "COPRO history uses an unexpected Evaluation Role"
                )
            if (
                attempt.provider_execution_policy_ref
                != provider_execution_policy_ref
            ):
                raise ValueError(
                    "COPRO history uses an unexpected Provider Execution Policy"
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
            mutation_field=mutation_field,
        )
        if state.completed_rounds >= config.depth:
            finalization = driver.finalize(state)
            contract = request.run.record.terminal_output_contract
            ranked = finalization.ranked_attempts
            if not ranked:
                return AdapterOutput(
                    proposed_status=StepStatus.FAILED,
                    terminal_failure=TerminalFailure(
                        code="copro_no_measured_history",
                        message="COPRO cannot finalize without measured history",
                    ),
                )
            if len(ranked) < contract.returned_proposal_count:
                return AdapterOutput(
                    proposed_status=StepStatus.FAILED,
                    terminal_failure=TerminalFailure(
                        code="copro_insufficient_ranked_candidates",
                        message=(
                            "COPRO finalization produced "
                            f"{len(ranked)} ranked candidates but terminal "
                            f"contract requires {contract.returned_proposal_count}"
                        ),
                    ),
                )
            selected = ranked[: contract.returned_proposal_count]
            accepted = tuple(entry.candidate.record for entry in selected)
            return AdapterOutput(
                proposed_candidates=accepted,
                accepted_candidates=accepted,
                proposed_status=StepStatus.COMPLETE,
                budget_delta=BudgetDelta(),
                state_delta={
                    "copro_finalization": finalization.model_dump(mode="json"),
                },
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
        if iteration != request.step_index:
            raise ValueError(
                "COPRO round_index must match the durable step index"
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
        contract_payload = {
            **self._control.proposal_contract.model_dump(mode="json"),
            "budget_mode": plan.proposal_mode,
        }
        proposal_request = ProposalRequest(
            proposal_mode=plan.proposal_mode,
            request_ordinal=iteration,
            proposal_authority_identity_hash=request.run.config_hash,
            mutation_field=mutation_field,
            base_candidate=candidate_reference(base),
            context=ImmutableJsonObject(
                {
                    COPRO_INSTRUCTION_CONTRACT_KEY: contract_payload,
                    COPRO_INSTRUCTION_HISTORY_KEY: [
                        item.to_json() for item in plan.instruction_history
                    ],
                }
            ),
        )
        prompt = copro_proposal_prompt(proposal_request)
        base_context = proposal_request.context.to_json()
        proposal_request = ProposalRequest(
            proposal_mode=proposal_request.proposal_mode,
            request_ordinal=proposal_request.request_ordinal,
            proposal_authority_identity_hash=(
                proposal_request.proposal_authority_identity_hash
            ),
            mutation_field=proposal_request.mutation_field,
            base_candidate=proposal_request.base_candidate,
            context=ImmutableJsonObject(
                {**base_context, "proposal_prompt": prompt}
            ),
        )
        drafts = self._proposal_executor.execute(
            config=self.proposer_config,
            request=proposal_request,
            transport=self._transport,
            count=plan.proposal_count,
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

            template = draft.template.strip('"').strip()
            disposition = "accepted"
            reason: str | None = None
            try:
                normalized_draft = (
                    draft
                    if draft.failed
                    else draft.model_copy(update={"template": template})
                )
                if not normalized_draft.failed:
                    self._control.proposal_contract.validate_instruction(
                        normalized_draft.template
                    )
                candidate = candidate_from_draft(
                    base=base,
                    candidate_id=candidate_id,
                    draft=normalized_draft,
                    run=request.run,
                )
            except (DiffCheckError, ValueError) as exc:
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

        optim_eval_requests: list[OptimEvalRequest] = []
        for occurrence_ordinal, candidate in occurrences:
            candidate_ref = candidate_reference(candidate)
            optim_eval_requests.append(
                OptimEvalRequest(
                    optim_run_id=request.run_id,
                    optim_step_index=request.step_index,
                    eval_request=EvalRequest(
                        request_id=(
                            f"{request.run_id}:{request.step_index}:"
                            f"{occurrence_ordinal}:{candidate_ref.identity_hash}"
                        ),
                        candidate=candidate,
                        metadata=metadata_with_purpose(plan.proposal_mode),
                    ),
                    expected_reward_policy_hash=expected_reward_policy_hash,
                )
            )
        return AdapterOutput(
            proposed_candidates=tuple(proposed),
            accepted_candidates=tuple(proposed),
            optim_eval_requests=tuple(optim_eval_requests),
            budget_delta=BudgetDelta(
                consumed={"proposal_calls": plan.proposal_count}
            ),
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
