"""Canonical MIPROv2 adapter and production cadence/state-folding seam."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any

from dr_store import ObjectStore

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.miprov2_identity import (
    MIPROV2_DEMO_SET_SCHEMA,
    DemoPair,
    DemoSetArtifact,
    DemoSetArtifactRef,
    DemoSetIdentity,
    InstructionIdentity,
    TrialCombinationIdentity,
)
from whetstone.optimization.mutation import DiffCheckError, diff_check
from whetstone.optimization.proposal_prompts import miprov2_pool_prompt
from whetstone.optimization.proposer import (
    ProposalPromptBuilder,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)
from whetstone.optimization.reward import Reward
from whetstone.optimization.schema import (
    BudgetDelta,
    Candidate,
    CandidateRef,
    EvalConfigRef,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    OptimizationStepRequest,
    StepMode,
    StepStatus,
    candidate_reference,
)

MIPROV2_ADAPTER_KEY = "miprov2"
BOOTSTRAP = "bootstrap"
POOL_CONSTRUCTION = "pool_construction"
BASELINE_FULL = "baseline_full"
MINIBATCH = "minibatch"
PROMOTION_FULL = "promotion_full"
COMPLETION = "completion"
DEFAULT_NUM_DEMO_SET_CANDIDATES = 4
DEFAULT_MAX_BOOTSTRAPPED_DEMOS = 3
DEFAULT_NUM_INSTRUCTION_CANDIDATES = 6
DEFAULT_INSTRUCTION_ATTEMPT_CAP = 12
TPE_GOOD_QUANTILE = 0.25


@dataclass(frozen=True, slots=True)
class Miprov2Plan:
    """Next production step kind and its exact proposal cardinality."""

    kind: str
    returned_proposal_count: int


class Miprov2Driver:
    """Fold durable resolution evidence and select the next cadence step."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def fold(
        self,
        state: dict[str, Any],
        resolutions: tuple[IntentResolution, ...],
    ) -> dict[str, Any]:
        folded = {
            **state,
            "demo_set_pool": list(state.get("demo_set_pool", [])),
            "study_state": dict(state.get("study_state", {})),
        }
        study = folded["study_state"]
        means = dict(study.get("combination_means", {}))
        counts = dict(study.get("combination_counts", {}))
        full_scores = dict(study.get("full_scores", {}))
        templates = dict(study.get("combination_templates", {}))
        fully = set(study.get("fully_evaluated", []))
        observations = list(study.get("study_observations", []))
        trials_completed = int(study.get("trials_completed", 0))

        def record_observation(
            candidate_id: str, value: float, purpose: str
        ) -> None:
            prior_count = int(counts.get(candidate_id, 0))
            prior_mean = float(means.get(candidate_id, 0.0))
            counts[candidate_id] = prior_count + 1
            means[candidate_id] = (prior_mean * prior_count + value) / (
                prior_count + 1
            )
            observations.append(
                {
                    "candidate_id": candidate_id,
                    "value": value,
                    "purpose": purpose,
                }
            )

        for resolution in resolutions:
            if resolution.outcome is not IntentOutcome.COMPLETED:
                continue
            intent = resolution.intent
            reward = self._reward(resolution)
            candidate = intent.candidate.record
            if intent.purpose == BOOTSTRAP:
                source_ref = resolution.evaluation_evidence_refs[0]
                if source_ref.schema_name != "whetstone.evaluation_evidence":
                    raise ValueError(
                        "bootstrap must reference canonical evaluation "
                        "evidence"
                    )
                evidence = self._store.get(source_ref.reference)
                if not isinstance(evidence, dict):
                    raise ValueError(
                        "bootstrap evaluation evidence must be an object"
                    )
                evidence_candidate = CandidateRef.model_validate(
                    evidence.get("candidate")
                )
                if evidence_candidate != intent.candidate:
                    raise ValueError(
                        "bootstrap evidence belongs to another candidate"
                    )
                outputs_ref = TypedRef.model_validate(
                    evidence.get("outputs_ref")
                )
                output_record = self._store.get(outputs_ref.reference)
                if not isinstance(output_record, dict):
                    raise ValueError(
                        "bootstrap output artifact must be an object"
                    )
                rows = output_record.get("outputs")
                if not isinstance(rows, list):
                    raise ValueError(
                        "bootstrap output artifact has no output rows"
                    )
                pair_list: list[DemoPair] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    rendered_input = row.get("rendered_prompt")
                    observed_output = row.get("output_text")
                    if (
                        isinstance(rendered_input, str)
                        and isinstance(observed_output, str)
                        and row.get("failure_code") is None
                    ):
                        pair_list.append(
                            DemoPair(
                                rendered_input=rendered_input,
                                observed_output=observed_output,
                            )
                        )
                pairs = tuple(pair_list)
                if not pairs:
                    raise ValueError(
                        "bootstrap evidence contains no successful demo pairs"
                    )
                target = folded.get(
                    "num_demo_set_candidates",
                    DEFAULT_NUM_DEMO_SET_CANDIDATES,
                )
                max_demos = folded.get(
                    "max_bootstrapped_demos",
                    DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
                )
                if not isinstance(target, int):
                    raise ValueError(
                        "num_demo_set_candidates must be an integer"
                    )
                if not isinstance(max_demos, int):
                    raise ValueError(
                        "max_bootstrapped_demos must be an integer"
                    )
                variants = [()]
                variants.extend(
                    tuple(selected)
                    for size in range(1, min(max_demos, len(pairs)) + 1)
                    for selected in itertools.combinations(pairs, size)
                )
                for variant in variants:
                    if len(folded["demo_set_pool"]) >= target:
                        break
                    demo = DemoSetIdentity(pairs=variant)
                    artifact = DemoSetArtifact(
                        demo_set=demo,
                        source_evidence_ref=source_ref,
                    )
                    reference, _ = self._store.put(
                        MIPROV2_DEMO_SET_SCHEMA,
                        artifact.model_dump(mode="json"),
                    )
                    entry = DemoSetArtifactRef(
                        identity_hash=demo.identity_hash(),
                        artifact_ref=TypedRef(
                            schema_name=reference.schema,
                            content_hash=reference.content_hash,
                        ),
                    )
                    if not any(
                        DemoSetArtifactRef.model_validate(item).identity_hash
                        == entry.identity_hash
                        for item in folded["demo_set_pool"]
                    ):
                        folded["demo_set_pool"].append(
                            entry.model_dump(mode="json")
                        )
            elif intent.purpose == MINIBATCH:
                key = candidate.candidate_id
                record_observation(key, reward, intent.purpose)
                templates[key] = candidate.payload.get(
                    "user_prompt_template", ""
                )
                trials_completed += 1
            elif intent.purpose in {BASELINE_FULL, PROMOTION_FULL}:
                key = candidate.candidate_id
                full_scores[key] = reward
                fully.add(key)
                record_observation(key, reward, intent.purpose)
                templates[key] = candidate.payload.get(
                    "user_prompt_template", ""
                )
                if intent.purpose == BASELINE_FULL:
                    study["baseline_complete"] = True
                else:
                    study["last_promotion_trial"] = trials_completed
        study.update(
            {
                "combination_means": means,
                "combination_counts": counts,
                "full_scores": full_scores,
                "combination_templates": templates,
                "fully_evaluated": sorted(fully),
                "study_observations": observations,
                "trials_completed": trials_completed,
            }
        )
        return folded

    def advance(
        self,
        state: dict[str, Any],
        output: AdapterOutput,
        resolutions: tuple[IntentResolution, ...],
    ) -> dict[str, Any]:
        """Fold one real harness step without test-only state injection."""
        prior_demo_count = len(state.get("demo_set_pool", []))
        advanced = {**state, **output.state_delta}
        if output.state_delta.get("promotion") == "noop":
            study = dict(advanced.get("study_state", {}))
            study["last_promotion_trial"] = int(
                study.get("trials_completed", 0)
            )
            advanced["study_state"] = study
        folded = self.fold(advanced, resolutions)
        if output.state_delta.get("bootstrap_requested") is True:
            folded["bootstrap_stalled"] = (
                len(folded["demo_set_pool"]) <= prior_demo_count
            )
        return folded

    def next_plan(
        self, state: dict[str, Any], hyperparameters: dict[str, Any]
    ) -> Miprov2Plan:
        demos = state.get("demo_set_pool", [])
        required_demos = int(
            hyperparameters.get(
                "num_demo_set_candidates",
                DEFAULT_NUM_DEMO_SET_CANDIDATES,
            )
        )
        if required_demos < 1:
            raise ValueError("num_demo_set_candidates must be positive")
        if len(demos) < required_demos:
            if state.get("bootstrap_stalled") is True:
                raise ValueError(
                    "bootstrap evidence cannot materialize the required "
                    f"{required_demos} distinct demo sets"
                )
            return Miprov2Plan(BOOTSTRAP, 0)
        if not state.get("pool_frozen", False):
            return Miprov2Plan(POOL_CONSTRUCTION, 0)
        study = dict(state.get("study_state", {}))
        if not study.get("baseline_complete", False):
            return Miprov2Plan(BASELINE_FULL, 0)
        trials = int(study.get("trials_completed", 0))
        target = int(hyperparameters.get("num_trials", 0))
        cadence = int(hyperparameters.get("minibatch_full_eval_steps", 1))
        last_promotion = int(study.get("last_promotion_trial", -1))
        if trials < target:
            if (
                trials > 0
                and trials % cadence == 0
                and last_promotion != trials
            ):
                return Miprov2Plan(PROMOTION_FULL, 0)
            return Miprov2Plan(MINIBATCH, 0)
        if last_promotion != trials:
            return Miprov2Plan(PROMOTION_FULL, 0)
        return Miprov2Plan(
            COMPLETION,
            int(hyperparameters.get("returned_proposal_count", 1)),
        )

    def _reward(self, resolution: IntentResolution) -> float:
        if resolution.reward_ref is None:
            raise ValueError("completed MIPRO resolution has no Reward ref")
        reward = Reward.model_validate(
            self._store.get(resolution.reward_ref.reference)
        )
        return reward.value


def _eval_config(
    request: OptimizationStepRequest, field: str
) -> EvalConfigRef:
    raw = request.hyperparameters.get(field)
    if raw is None:
        raise ValueError(f"MIPROv2 requires exact {field}")
    return EvalConfigRef.model_validate(raw)


def _intent(
    request: OptimizationStepRequest,
    *,
    candidate: Candidate,
    purpose: str,
    eval_config: EvalConfigRef,
) -> EvaluationIntent:
    return EvaluationIntent(
        intent_id=(
            f"{request.run_id}:{request.step_index}:{purpose}:"
            f"{candidate.candidate_id}"
        ),
        candidate=candidate_reference(candidate),
        target_eval_config=eval_config,
        context_role=EvaluationRole.INTERNAL,
        purpose=purpose,
        run_id=request.run_id,
        step_index=request.step_index,
    )


def _pool_candidates(
    request: OptimizationStepRequest,
) -> tuple[Candidate, ...]:
    raw = request.pools.get("combination_candidates", [])
    if not isinstance(raw, list):
        raise ValueError("combination_candidates must be a list")
    return tuple(Candidate.model_validate(item) for item in raw)


def _materialize_demonstrations(
    instruction: str, demo_set: DemoSetIdentity
) -> str:
    """Compose observed examples into the prompt template evaluation runs."""
    if not demo_set.pairs:
        return instruction

    def literal(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

    examples = "\n\n".join(
        (
            f"Observed input:\n{literal(pair.rendered_input)}\n"
            f"Observed output:\n{literal(pair.observed_output)}"
        )
        for pair in demo_set.pairs
    )
    return f"{instruction}\n\nUse these observed demonstrations:\n{examples}"


def _seeded_tpe_choice(
    pool: tuple[Candidate, ...],
    observations: list[dict[str, Any]],
    *,
    seed: int,
    trial_number: int,
) -> tuple[Candidate, float]:
    """Choose a categorical arm by a seeded Parzen likelihood ratio."""
    candidate_ids = {candidate.candidate_id for candidate in pool}
    validated: list[tuple[int, str, float]] = []
    for ordinal, raw in enumerate(observations):
        candidate_id = raw.get("candidate_id")
        value = raw.get("value")
        if (
            isinstance(candidate_id, str)
            and candidate_id in candidate_ids
            and isinstance(value, int | float)
        ):
            validated.append((ordinal, candidate_id, float(value)))
    ranked = sorted(
        validated,
        key=lambda item: (
            -item[2],
            hashlib.sha256(f"{seed}:{item[0]}:{item[1]}".encode()).hexdigest(),
        ),
    )
    good_count = max(1, math.ceil(len(ranked) * TPE_GOOD_QUANTILE))
    good = ranked[:good_count]
    bad = ranked[good_count:]
    cardinality = len(pool)

    def likelihood_ratio(candidate_id: str) -> float:
        good_matches = sum(item[1] == candidate_id for item in good)
        bad_matches = sum(item[1] == candidate_id for item in bad)
        good_probability = (good_matches + 1.0) / (len(good) + cardinality)
        bad_probability = (bad_matches + 1.0) / (len(bad) + cardinality)
        return good_probability / bad_probability

    ratios = {
        candidate.candidate_id: likelihood_ratio(candidate.candidate_id)
        for candidate in pool
    }
    selected = min(
        pool,
        key=lambda candidate: (
            -ratios[candidate.candidate_id],
            hashlib.sha256(
                (f"{seed}:{trial_number}:{candidate.candidate_id}").encode()
            ).hexdigest(),
            candidate.candidate_id,
        ),
    )
    return selected, ratios[selected.candidate_id]


class Miprov2Adapter:
    def __init__(
        self,
        *,
        store: ObjectStore,
        proposer_config: ProposerConfig,
        transport: ProposerTransport,
        prompt_builder: ProposalPromptBuilder = miprov2_pool_prompt,
    ) -> None:
        self._store = store
        self._proposer_config = proposer_config
        self._transport = transport
        self._prompt_builder = prompt_builder
        self.invocations = 0

    def _demo_sets(
        self, request: OptimizationStepRequest
    ) -> tuple[tuple[DemoSetArtifactRef, DemoSetIdentity], ...]:
        raw = request.pools.get("demo_set_pool", [])
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                "pool construction requires persisted demo artifacts"
            )
        loaded: list[tuple[DemoSetArtifactRef, DemoSetIdentity]] = []
        for item in raw:
            entry = DemoSetArtifactRef.model_validate(item)
            artifact = DemoSetArtifact.model_validate(
                self._store.get(entry.artifact_ref.reference)
            )
            if artifact.demo_set.identity_hash() != entry.identity_hash:
                raise ValueError(
                    "persisted demo artifact identity does not match its pool "
                    "entry"
                )
            loaded.append((entry, artifact.demo_set))
        return tuple(loaded)

    @property
    def key(self) -> str:
        return MIPROV2_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def proposer_config(self) -> ProposerConfig:
        return self._proposer_config

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[Any, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        if handles:
            raise ValueError("MIPROv2 receives no Runtime Tool Handles")
        handlers = {
            BOOTSTRAP: self._bootstrap,
            POOL_CONSTRUCTION: self._pool_construction,
            BASELINE_FULL: self._baseline,
            MINIBATCH: self._minibatch,
            PROMOTION_FULL: self._promotion,
            COMPLETION: self._completion,
        }
        try:
            handler = handlers[request.kind_label or ""]
        except KeyError:
            raise ValueError(
                f"unknown MIPROv2 kind {request.kind_label!r}"
            ) from None
        return handler(request)

    def _bootstrap(self, request: OptimizationStepRequest) -> AdapterOutput:
        eval_config = _eval_config(request, "bootstrap_eval_config")
        candidates = request.candidates
        target = int(
            request.hyperparameters.get(
                "num_demo_set_candidates",
                DEFAULT_NUM_DEMO_SET_CANDIDATES,
            )
        )
        max_demos = int(
            request.hyperparameters.get(
                "max_bootstrapped_demos",
                DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
            )
        )
        if target < 1:
            raise ValueError("num_demo_set_candidates must be positive")
        if max_demos < 0:
            raise ValueError("max_bootstrapped_demos cannot be negative")
        return AdapterOutput(
            proposed_candidates=candidates,
            evaluation_intents=tuple(
                _intent(
                    request,
                    candidate=candidate,
                    purpose=BOOTSTRAP,
                    eval_config=eval_config,
                )
                for candidate in candidates
            ),
            proposed_status=StepStatus.CONTINUE,
            state_delta={
                "bootstrap_requested": True,
                "num_demo_set_candidates": target,
                "max_bootstrapped_demos": max_demos,
                "bootstrap_candidate_ids": [
                    candidate.candidate_id for candidate in candidates
                ],
            },
        )

    def _pool_construction(
        self, request: OptimizationStepRequest
    ) -> AdapterOutput:
        target = int(
            request.hyperparameters.get(
                "num_instruction_candidates",
                DEFAULT_NUM_INSTRUCTION_CANDIDATES,
            )
        )
        cap = int(
            request.hyperparameters.get(
                "instruction_attempt_cap",
                DEFAULT_INSTRUCTION_ATTEMPT_CAP,
            )
        )
        field = str(
            request.hyperparameters.get(
                "mutation_field", "user_prompt_template"
            )
        )
        if not request.candidates:
            raise ValueError("pool construction requires a base candidate")
        base = request.candidates[0]
        accepted: list[str] = []
        for candidate in request.candidates:
            instruction = candidate.payload.get(field)
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(
                    f"base instruction candidates require a non-empty {field}"
                )
            if instruction not in accepted:
                accepted.append(instruction)
        if len(accepted) > target:
            raise ValueError(
                "num_instruction_candidates is smaller than the base "
                "instruction pool"
            )
        evidence: list[dict[str, Any]] = []
        attempts = 0
        while len(accepted) < target and attempts < cap:
            proposal_request = ProposalRequest(
                proposal_mode=POOL_CONSTRUCTION,
                request_ordinal=attempts,
                base_ref=base.base_ref,
                base_template=str(base.payload.get(field, "")),
                context={"accepted": list(accepted)},
            )
            prompt = self._prompt_builder(proposal_request)
            proposal_request = proposal_request.model_copy(
                update={
                    "context": {
                        **proposal_request.context,
                        "proposal_prompt": prompt,
                    }
                }
            )
            draft = self._transport.draft(
                self._proposer_config, proposal_request, 1
            )[0]
            attempts += 1
            if draft.template in accepted:
                continue
            candidate = Candidate(
                candidate_id=f"mipro-instruction-{attempts}",
                base_ref=base.base_ref,
                payload={field: draft.template},
            )
            try:
                diff_check(
                    base=base,
                    proposed=candidate,
                    mutation_field=field,
                )
            except DiffCheckError:
                continue
            accepted.append(draft.template)
            evidence.append(draft.model_dump(mode="json"))
        if len(accepted) != target:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                budget_delta=BudgetDelta(
                    consumed={"proposal_calls": attempts}
                ),
                state_delta={"reason": "instruction pool cardinality"},
            )
        demo_sets = self._demo_sets(request)
        combinations: list[dict[str, Any]] = []
        default_combination_id: str | None = None
        empty_demo_hash = DemoSetIdentity().identity_hash()
        for text in accepted:
            instruction_hash = InstructionIdentity(
                instruction_text=text
            ).identity_hash()
            for demo_ref, demo_set in demo_sets:
                identity = TrialCombinationIdentity(
                    instruction_hash=instruction_hash,
                    demo_set_hash=demo_ref.identity_hash,
                )
                materialized = _materialize_demonstrations(text, demo_set)
                candidate = Candidate(
                    candidate_id=identity.identity_hash(),
                    base_ref=base.base_ref,
                    payload={
                        field: materialized,
                        "instruction_template": text,
                        "demo_set_identity_hash": demo_ref.identity_hash,
                        "demo_set_artifact_ref": (
                            demo_ref.artifact_ref.model_dump(mode="json")
                        ),
                    },
                )
                combinations.append(candidate.model_dump(mode="json"))
                if (
                    text == accepted[0]
                    and demo_ref.identity_hash == empty_demo_hash
                ):
                    default_combination_id = candidate.candidate_id
        if default_combination_id is None:
            raise ValueError(
                "frozen MIPROv2 pool requires the empty demonstration set"
            )
        return AdapterOutput(
            proposed_status=StepStatus.CONTINUE,
            budget_delta=BudgetDelta(consumed={"proposal_calls": attempts}),
            state_delta={
                "pool_frozen": True,
                "instruction_pool": accepted,
                "demo_set_pool": [
                    entry.model_dump(mode="json")
                    for entry, _demo_set in demo_sets
                ],
                "combination_candidates": combinations,
                "default_combination_id": default_combination_id,
                "proposer_evidence": evidence,
            },
        )

    def _baseline(self, request: OptimizationStepRequest) -> AdapterOutput:
        default_id = request.pools.get("default_combination_id")
        if not isinstance(default_id, str):
            raise ValueError(
                "baseline step requires the frozen default combination"
            )
        by_id = {
            candidate.candidate_id: candidate
            for candidate in _pool_candidates(request)
        }
        try:
            candidate = by_id[default_id]
        except KeyError:
            raise ValueError(
                "default combination is absent from the frozen pool"
            ) from None
        return AdapterOutput(
            proposed_candidates=(candidate,),
            evaluation_intents=(
                _intent(
                    request,
                    candidate=candidate,
                    purpose=BASELINE_FULL,
                    eval_config=_eval_config(request, "full_eval_config"),
                ),
            ),
            proposed_status=StepStatus.CONTINUE,
        )

    def _minibatch(self, request: OptimizationStepRequest) -> AdapterOutput:
        pool = _pool_candidates(request)
        if not pool:
            raise ValueError("minibatch requires a frozen combination pool")
        remaining = request.budget.remaining.get("search_rollouts")
        if remaining is not None and remaining < 1:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                state_delta={"reason": "search budget exhausted"},
            )
        study = dict(request.pools.get("study_state", {}))
        seed = int(request.hyperparameters.get("seed", 0))
        observations = list(study.get("study_observations", []))
        trials = int(study.get("trials_completed", 0))
        candidate, acquisition_value = _seeded_tpe_choice(
            pool,
            observations,
            seed=seed,
            trial_number=trials,
        )
        return AdapterOutput(
            proposed_candidates=(candidate,),
            evaluation_intents=(
                _intent(
                    request,
                    candidate=candidate,
                    purpose=MINIBATCH,
                    eval_config=_eval_config(request, "minibatch_eval_config"),
                ),
            ),
            budget_delta=BudgetDelta(consumed={"search_rollouts": 1}),
            proposed_status=StepStatus.CONTINUE,
            state_delta={
                "trial_combination_hash": candidate.candidate_id,
                "acquisition": {
                    "policy": "seeded_categorical_tpe/v1",
                    "value": acquisition_value,
                },
            },
        )

    def _promotion(self, request: OptimizationStepRequest) -> AdapterOutput:
        study = dict(request.pools.get("study_state", {}))
        means = dict(study.get("combination_means", {}))
        fully = set(study.get("fully_evaluated", []))
        by_id = {
            candidate.candidate_id: candidate
            for candidate in _pool_candidates(request)
        }
        eligible = sorted(
            (
                (candidate_id, float(score))
                for candidate_id, score in means.items()
                if candidate_id not in fully and candidate_id in by_id
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not eligible:
            return AdapterOutput(
                proposed_status=StepStatus.CONTINUE,
                state_delta={
                    "promotion": "noop",
                    "reason": "all observed combinations fully evaluated",
                },
            )
        candidate = by_id[eligible[0][0]]
        return AdapterOutput(
            proposed_candidates=(candidate,),
            evaluation_intents=(
                _intent(
                    request,
                    candidate=candidate,
                    purpose=PROMOTION_FULL,
                    eval_config=_eval_config(request, "full_eval_config"),
                ),
            ),
            proposed_status=StepStatus.CONTINUE,
            state_delta={"promoted_combination_hash": candidate.candidate_id},
        )

    def _completion(self, request: OptimizationStepRequest) -> AdapterOutput:
        target = request.output_contract.returned_proposal_count
        study = dict(request.pools.get("study_state", {}))
        full = dict(study.get("full_scores", {}))
        means = dict(study.get("combination_means", {}))
        by_id = {
            candidate.candidate_id: candidate
            for candidate in _pool_candidates(request)
        }
        ordered = [
            key
            for key, _ in sorted(
                full.items(), key=lambda item: (-float(item[1]), item[0])
            )
            if key in by_id
        ]
        ordered.extend(
            key
            for key, _ in sorted(
                means.items(), key=lambda item: (-float(item[1]), item[0])
            )
            if key in by_id and key not in ordered
        )
        if len(ordered) < target:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                state_delta={"reason": "measured proposal cardinality"},
            )
        proposals = tuple(by_id[key] for key in ordered[:target])
        return AdapterOutput(
            proposed_candidates=proposals,
            accepted_candidates=proposals,
            proposed_status=StepStatus.COMPLETE,
            state_delta={"ordered_combination_hashes": ordered[:target]},
        )


__all__ = [
    "BASELINE_FULL",
    "BOOTSTRAP",
    "COMPLETION",
    "MINIBATCH",
    "MIPROV2_ADAPTER_KEY",
    "POOL_CONSTRUCTION",
    "PROMOTION_FULL",
    "Miprov2Adapter",
    "Miprov2Driver",
    "Miprov2Plan",
]
