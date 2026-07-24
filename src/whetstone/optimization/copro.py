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
    model_validator,
)

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.copro_control import COPRO_ALGORITHM_VERSION
from whetstone.optimization.identity import (
    TypedRef,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.optimization.mutation import (
    MUTATION_FIELD,
    DiffCheckError,
    candidate_from_draft,
    invalid_template_placeholders,
    template_placeholder_fields,
)
from whetstone.optimization.proposal_prompts import (
    COPRO_PROPOSAL_PROMPT_SCHEMA_TAG,
    copro_proposal_prompt,
)
from whetstone.optimization.proposer import (
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)
from whetstone.optimization.reward import REWARD_RECORD_SCHEMA, Reward
from whetstone.optimization.schema import (
    BudgetDelta,
    Candidate,
    CandidateRef,
    EvalConfigRef,
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
    :class:`ProposerConfig` and the request's exact :class:`EvalConfigRef`.
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
    prompt_history: tuple[dict[str, Any], ...] = Field(default_factory=tuple)


class CoproAttempt(BaseModel):
    """One measured candidate occurrence in the append-only COPRO history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    occurrence_ordinal: StrictInt
    round_index: StrictInt
    run_id: StrictStr
    step_index: StrictInt
    intent_id: StrictStr
    candidate: CandidateRef
    eval_config: EvalConfigRef
    reward: float
    reward_policy_hash: StrictStr
    evaluation_evidence_refs: tuple[TypedRef, ...]
    reward_ref: TypedRef

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
            self.reward_policy_hash,
            field="reward_policy_hash",
        )
        template = self.candidate.record.payload.get(MUTATION_FIELD)
        if not isinstance(template, str) or not template:
            raise ValueError(
                "COPRO attempt candidate requires a non-empty "
                "user_prompt_template"
            )
        if not self.evaluation_evidence_refs:
            raise ValueError("COPRO attempt requires evaluation evidence")
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
        reward: Reward,
        expected_run_id: str,
        expected_eval_config: EvalConfigRef,
        expected_reward_policy_hash: str,
    ) -> CoproAttempt:
        """Bind an externally loaded Reward to one measured resolution."""

        if resolution.outcome is not IntentOutcome.COMPLETED:
            raise ValueError("COPRO folds only completed measured resolutions")
        if resolution.detail.classification is not ResolutionClass.MEASURED:
            raise ValueError("COPRO folds only measured resolution details")
        if resolution.reward_ref is None:
            raise ValueError("COPRO measured resolution requires Reward ref")
        if resolution.intent.context_role is not EvaluationRole.INTERNAL:
            raise ValueError("COPRO folds only internal evaluation intents")
        if resolution.intent.run_id != expected_run_id:
            raise ValueError("COPRO resolution belongs to another run")
        if resolution.intent.step_index != round_index:
            raise ValueError("COPRO resolution belongs to another round")
        if resolution.resolved_eval_config != expected_eval_config:
            raise ValueError("COPRO resolution uses an unexpected Eval Config")
        if reward.reward_policy_hash != expected_reward_policy_hash:
            raise ValueError("COPRO Reward uses an unexpected Reward Policy")
        expected_reward_ref = typed_ref_for_record(
            REWARD_RECORD_SCHEMA,
            reward.record_content(),
        )
        if resolution.reward_ref != expected_reward_ref:
            raise ValueError(
                "COPRO supplied Reward does not match resolution reward_ref"
            )
        return cls(
            occurrence_ordinal=occurrence_ordinal,
            round_index=round_index,
            run_id=expected_run_id,
            step_index=resolution.intent.step_index,
            intent_id=resolution.intent.intent_id,
            candidate=resolution.intent.candidate,
            eval_config=resolution.resolved_eval_config,
            reward=reward.value,
            reward_policy_hash=reward.reward_policy_hash,
            evaluation_evidence_refs=resolution.evaluation_evidence_refs,
            reward_ref=resolution.reward_ref,
        )

    def prompt_entry(self) -> dict[str, Any]:
        return {
            "occurrence_ordinal": self.occurrence_ordinal,
            "candidate_id": self.candidate_id,
            "template": self.template,
            "reward": self.reward,
        }


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

    raw = request.pools.get("attempt_history", [])
    if not isinstance(raw, list):
        raise ValueError("attempt_history must be a JSON list")
    attempts: list[CoproAttempt] = []
    for ordinal, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"attempt_history[{ordinal}] must be a JSON record"
            )
        attempts.append(CoproAttempt.model_validate(item))
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
        expected_eval_config = (
            state.attempts[0].eval_config
            if state.attempts
            else attempts[0].eval_config
        )
        expected_reward_policy_hash = (
            state.attempts[0].reward_policy_hash
            if state.attempts
            else attempts[0].reward_policy_hash
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
            if attempt.eval_config != expected_eval_config:
                raise ValueError("COPRO attempts span multiple Eval Configs")
            if attempt.reward_policy_hash != expected_reward_policy_hash:
                raise ValueError(
                    "COPRO attempts span multiple Reward Policies"
                )
            if attempt.intent_id in prior_intent_ids:
                raise ValueError("COPRO attempt intent IDs must be unique")
            prior_intent_ids.add(attempt.intent_id)
            if (
                attempt.candidate.record.base_ref
                != state.initial_candidate.base_ref
            ):
                raise ValueError("COPRO attempt changes the initial base_ref")
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


def _eval_config(request: OptimizationStepRequest) -> EvalConfigRef:
    raw = request.hyperparameters.get("eval_config")
    if raw is None:
        raise ValueError("COPRO requires one exact eval_config record")
    return EvalConfigRef.model_validate(raw)


def _copro_config(request: OptimizationStepRequest) -> CoproConfig:
    expected = {
        "algorithm_version",
        "breadth",
        "depth",
        "eval_config",
        "init_temperature",
        "prompt_adapter_identity_hash",
        "proposal_prompt_schema_tag",
        "provider_execution_policy_hash",
        "reward_policy_hash",
        "round_index",
        "track_stats",
    }
    actual = set(request.hyperparameters)
    if actual != expected:
        raise ValueError(
            "COPRO requires exact hyperparameters; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    if request.hyperparameters["algorithm_version"] != COPRO_ALGORITHM_VERSION:
        raise ValueError("COPRO algorithm_version does not match adapter")
    if (
        request.hyperparameters["proposal_prompt_schema_tag"]
        != COPRO_PROPOSAL_PROMPT_SCHEMA_TAG
    ):
        raise ValueError(
            "COPRO proposal_prompt_schema_tag does not match adapter"
        )
    values = {
        name: request.hyperparameters[name]
        for name in (
            "breadth",
            "depth",
            "init_temperature",
            "track_stats",
        )
        if name in request.hyperparameters
    }
    return CoproConfig.model_validate(values)


def _history_base(
    initial: Candidate,
    best: CoproAttempt,
) -> Candidate:
    """Reconstitute the best prompt over the initial fixed payload."""

    if best.candidate.record.base_ref != initial.base_ref:
        raise ValueError("COPRO attempt history changes the initial base_ref")
    return Candidate(
        candidate_id=best.candidate_id,
        base_ref=initial.base_ref,
        payload={**initial.payload, MUTATION_FIELD: best.template},
    )


def _valid_template_keys(
    request: OptimizationStepRequest,
) -> tuple[str, ...]:
    raw = request.pools.get("valid_template_keys")
    if not isinstance(raw, list):
        raise ValueError(
            "COPRO requires explicit valid_template_keys authority"
        )
    if any(not isinstance(item, str) or not item for item in raw):
        raise ValueError("valid_template_keys must contain non-empty strings")
    if len(raw) != len(set(raw)):
        raise ValueError("valid_template_keys must not contain duplicates")
    return tuple(raw)


def _normalize_initial_candidate(
    candidate: Candidate,
    valid_template_keys: tuple[str, ...],
) -> Candidate:
    raw = candidate.payload.get(MUTATION_FIELD)
    if not isinstance(raw, str):
        raise ValueError(
            "COPRO initial candidate requires user_prompt_template"
        )
    template = raw.strip('"').strip()
    if not template:
        raise ValueError("COPRO initial template is empty after normalization")
    invalid = invalid_template_placeholders(template, valid_template_keys)
    if invalid:
        raise ValueError(
            "COPRO initial template contains unavailable placeholders: "
            + ", ".join(invalid)
        )
    return candidate.model_copy(
        update={
            "payload": {
                **candidate.payload,
                MUTATION_FIELD: template,
            }
        }
    )


def _validate_attempt_placeholders(
    attempts: tuple[CoproAttempt, ...],
    valid_template_keys: tuple[str, ...],
    required_template_keys: tuple[str, ...],
) -> None:
    for attempt in attempts:
        invalid = invalid_template_placeholders(
            attempt.template, valid_template_keys
        )
        if invalid:
            raise ValueError(
                "COPRO history contains unavailable placeholders at "
                f"occurrence {attempt.occurrence_ordinal}: "
                + ", ".join(invalid)
            )
        fields = template_placeholder_fields(attempt.template)
        missing = [
            required
            for required in required_template_keys
            if fields.count(required) < required_template_keys.count(required)
        ]
        if missing:
            raise ValueError(
                "COPRO history removes required placeholders at "
                f"occurrence {attempt.occurrence_ordinal}: "
                + ", ".join(dict.fromkeys(missing))
            )


class CoproAdapter:
    """Plan one COPRO round and emit an exact intent for each candidate."""

    def __init__(
        self,
        *,
        proposer_config: ProposerConfig,
        transport: ProposerTransport,
    ) -> None:
        self._proposer_config = proposer_config
        self._transport = transport
        self.invocations = 0

    @property
    def key(self) -> str:
        return COPRO_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def proposer_config(self) -> ProposerConfig:
        return self._proposer_config

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

        config = _copro_config(request)
        expected_policy_hash = request.hyperparameters[
            "provider_execution_policy_hash"
        ]
        require_full_hash(
            expected_policy_hash,
            field="provider_execution_policy_hash",
        )
        expected_adapter_hash = request.hyperparameters[
            "prompt_adapter_identity_hash"
        ]
        require_full_hash(
            expected_adapter_hash,
            field="prompt_adapter_identity_hash",
        )
        if self.provider_execution_policy_hash != expected_policy_hash:
            raise ValueError(
                "COPRO provider execution policy conflicts with request"
            )
        if self.prompt_adapter_identity_hash != expected_adapter_hash:
            raise ValueError(
                "COPRO prompt adapter identity conflicts with request"
            )
        expected_reward_policy_hash = request.hyperparameters[
            "reward_policy_hash"
        ]
        require_full_hash(
            expected_reward_policy_hash,
            field="reward_policy_hash",
        )
        eval_config = _eval_config(request)
        if self._proposer_config.temperature != config.init_temperature:
            raise ValueError(
                "COPRO Proposer Config temperature must equal init_temperature"
            )
        iteration = request.hyperparameters["round_index"]
        if type(iteration) is not int:
            raise ValueError("COPRO round_index must be an integer")
        valid_template_keys = _valid_template_keys(request)
        if len(request.candidates) != 1:
            raise ValueError(
                "single-prompt COPRO requires exactly one initial candidate"
            )
        initial = _normalize_initial_candidate(
            request.candidates[0], valid_template_keys
        )
        required_template_keys = template_placeholder_fields(
            str(initial.payload[MUTATION_FIELD])
        )
        history = attempt_history_entries(request)
        for attempt in history:
            if attempt.run_id != request.run_id:
                raise ValueError("COPRO history belongs to another run")
            if attempt.eval_config != eval_config:
                raise ValueError(
                    "COPRO history uses an unexpected Eval Config"
                )
            if attempt.reward_policy_hash != expected_reward_policy_hash:
                raise ValueError(
                    "COPRO history uses an unexpected Reward Policy"
                )
        _validate_attempt_placeholders(
            history,
            valid_template_keys,
            required_template_keys,
        )
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
        if remaining is not None and remaining < plan.proposal_count:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                state_delta={
                    "reason": "proposal budget exhausted",
                    "required": plan.proposal_count,
                    "remaining": remaining,
                },
            )

        ranked = driver.terminal_ranking(history)
        base = (
            initial
            if plan.proposal_mode == SEED_PROPOSAL
            else _history_base(initial, ranked[0])
        )
        context: dict[str, Any] = {
            "prompt_history": [dict(item) for item in plan.prompt_history],
        }
        proposal_request = ProposalRequest(
            proposal_mode=plan.proposal_mode,
            request_ordinal=iteration,
            base_ref=base.base_ref,
            base_template=str(base.payload.get(MUTATION_FIELD, "")),
            run_id=request.run_id,
            step_index=request.step_index,
            context=context,
        )
        prompt = copro_proposal_prompt(proposal_request)
        proposal_request = proposal_request.model_copy(
            update={"context": {**context, "proposal_prompt": prompt}}
        )
        drafts = self._transport.draft(
            self._proposer_config,
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
                    valid_template_keys=valid_template_keys,
                    required_template_keys=required_template_keys,
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
                    "request": draft.request_evidence,
                    "response": draft.response_evidence,
                    "usage": draft.usage,
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
        proposed = [candidate for _, candidate in occurrences]
        if (
            len(drafts) != plan.proposal_count
            or len(proposed) != config.breadth
        ):
            return AdapterOutput(
                proposed_candidates=tuple(proposed),
                proposed_status=StepStatus.FAILED,
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
                    target_eval_config=eval_config,
                    context_role=EvaluationRole.INTERNAL,
                    purpose=plan.proposal_mode,
                    run_id=request.run_id,
                    step_index=request.step_index,
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
