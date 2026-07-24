"""The MIPROv2 proposal-only adapter.

Implements the ordered, restartable proposal-only loop pinned by
``optimizer-briefs.md`` §3 (MIPROv2 Run) and ``miprov2-run.html``. The
optimizer never executes evaluation: it emits candidates + immutable Evaluation
Intents (``bootstrap`` / ``baseline_full`` / ``minibatch`` /
``promotion_full``) and a completion Step; Whetstone alone materializes, plans,
executes, aggregates and scores. Pools, TPE study state, and budgets advance
ONLY through immutable references in the Step Requests/Results.

Step kinds (carried on the Step Request ``kind_label``), in order:

1. ``bootstrap`` — register the default combination; emit bootstrap
   Intents until ``num_demo_set_candidates`` valid demo-set identities exist.
2. ``pool_construction`` — the proposal LM drafts instructions at
   ``proposal_temperature`` up to ``instruction_attempt_cap``; freeze the
   instruction pool + demo-set pool -> the ``combination_pool_size`` pool. A
   duplicate instruction text consumes an attempt but adds no pool member.
3. ``baseline_full`` — one ``baseline_full`` Evaluation Intent for the
   default program under the full internal Eval Config; seeds the TPE study.
4. ``minibatch`` (x ``num_trials``) — the seeded sampler picks any pool
   combination (repeats allowed); emit one ``minibatch`` Evaluation Intent.
5. ``promotion_full`` (every ``minibatch_full_eval_steps`` trials + a final
   off-cadence promotion) — choose the highest-mean combination not yet fully
   evaluated under the exact target full Eval Config (stable identity-hash
   tie-break); emit one ``promotion_full`` Evaluation Intent. If every
   trial-observed combination is already fully evaluated, persist a
   deterministic **no-op** Step Result (reason recorded, no Evaluation Intent).
6. ``completion`` — order combinations (full-scored first by full score, then
   remaining slots from distinct trial-observed combinations by mean minibatch
   score; deterministic identity-hash tie-breaks both tiers) and return exactly
   ``returned_proposal_count`` distinct **measured** proposals. If the required
   pool cardinality or ``returned_proposal_count`` cannot be met, return an
   explicit **failed** terminal Step Result (the cardinality-failure gate),
   which prevents official materialization.

The proposal LM is reached through the **proposer route** (a
:class:`ProposerConfig` — a Provider Call Config distinct from any
encoder/decoder route), which lives in the optimizer Config identity, never a
graph identity.
"""

from __future__ import annotations

import hashlib
from typing import Any

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.miprov2_identity import (
    DemoSetIdentity,
    InstructionIdentity,
    TrialCombinationIdentity,
)
from whetstone.optimization.mutation import DiffCheckError, diff_check
from whetstone.optimization.proposer import (
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)
from whetstone.optimization.schema import (
    Candidate,
    EvaluationIntent,
    OptimizationStepRequest,
    StepMode,
    StepStatus,
)

__all__ = [
    "BASELINE_FULL",
    "BOOTSTRAP",
    "COMPLETION",
    "MINIBATCH",
    "POOL_CONSTRUCTION",
    "PROMOTION_FULL",
    "Miprov2Adapter",
]

BOOTSTRAP = "bootstrap"
POOL_CONSTRUCTION = "pool_construction"
BASELINE_FULL = "baseline_full"
MINIBATCH = "minibatch"
PROMOTION_FULL = "promotion_full"
COMPLETION = "completion"

# Hyperparameter keys carried on the Step Request. ``num_trials`` and
# ``minibatch_full_eval_steps`` fix the trial/promotion *cadence*, which the
# driver orchestrates by choosing each Step's ``kind_label`` (minibatch vs.
# promotion_full vs. completion); the adapter reads only the per-step keys.
_NUM_DEMO_SETS = "num_demo_set_candidates"
_NUM_INSTRUCTIONS = "num_instruction_candidates"
_INSTRUCTION_ATTEMPT_CAP = "instruction_attempt_cap"
_RETURNED_PROPOSAL_COUNT = "returned_proposal_count"
_SEED = "seed"
_MINIBATCH_EVAL_REF = "minibatch_eval_config_ref"
_MINIBATCH_EVAL_HASH = "minibatch_eval_config_hash"
_FULL_EVAL_REF = "full_eval_config_ref"
_FULL_EVAL_HASH = "full_eval_config_hash"
_MUTATION_FIELD = "mutation_field"

# Budget label for the internal measurement units a trial Step consumes.
_BUDGET_KEY = "search_rollouts"

# Pool / study-state keys carried on the Step Request ``pools``.
_DEMO_SET_POOL = "demo_set_pool"
_COMBINATION_POOL = "combination_pool"
_STUDY_STATE = "study_state"


def _stable_pick(items: tuple[str, ...], seed: int, step_index: int) -> str:
    """Deterministically pick one pool member by a seeded hash.

    A seeded, reproducible selection standing in for the TPE sampler's choice:
    a retry with the same immutable inputs recovers the same combination. The
    real sampler would consult study state; determinism from ``(seed,
    step_index)`` is what the restart invariant requires.
    """
    digest = hashlib.sha256(
        f"{seed}:{step_index}".encode()
    ).hexdigest()
    return items[int(digest, 16) % len(items)]


class Miprov2Adapter:
    """The proposal-only MIPROv2 adapter.

    Constructed with the proposer route + a :class:`ProposerTransport` (real or
    the scripted fake) used only in the ``pool_construction`` step to draft
    instructions. All other steps are pure sampler/selection logic over the
    frozen pools + study state carried by immutable references.
    """

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
        if request.mode is not StepMode.PROPOSAL_ONLY:
            raise ValueError("MIPROv2 runs only proposal-only Steps")
        if handles:
            raise ValueError("MIPROv2 is proposal-only and needs no handles")

        kind = request.kind_label
        dispatch = {
            BOOTSTRAP: self._bootstrap,
            POOL_CONSTRUCTION: self._pool_construction,
            BASELINE_FULL: self._baseline_full,
            MINIBATCH: self._minibatch,
            PROMOTION_FULL: self._promotion_full,
            COMPLETION: self._completion,
        }
        handler = dispatch.get(kind or "")
        if handler is None:
            raise ValueError(f"unknown MIPROv2 step kind {kind!r}")
        return handler(request)

    # -- step 1: bootstrap ----------------------------------------------------

    def _bootstrap(self, request: OptimizationStepRequest) -> AdapterOutput:
        hyper = request.hyperparameters
        num_demo_sets = int(hyper.get(_NUM_DEMO_SETS, 4))
        seed = int(hyper.get(_SEED, 0))
        eval_ref = str(hyper.get(_MINIBATCH_EVAL_REF, ""))
        eval_hash = str(hyper.get(_MINIBATCH_EVAL_HASH, ""))
        if not eval_ref or not eval_hash:
            raise ValueError("bootstrap requires a minibatch Eval Config ref")

        # Emit bootstrap Evaluation Intents (data-dependent) until
        # num_demo_set_candidates valid demo-set identities exist (incl. the
        # empty set). The measurement count is recorded, not assumed.
        intents: list[EvaluationIntent] = []
        for index in range(num_demo_sets):
            intents.append(
                EvaluationIntent(
                    intent_id=f"{request.run_id}-bootstrap-{index}",
                    candidate_id=f"bootstrap-demo-set-{index}",
                    target_eval_config_ref=eval_ref,
                    target_eval_config_hash=eval_hash,
                    context_role=EvaluationRole.INTERNAL,
                    purpose=BOOTSTRAP,
                    run_id=request.run_id,
                    step_index=request.step_index,
                )
            )
        return AdapterOutput(
            evaluation_intents=tuple(intents),
            proposed_status=StepStatus.CONTINUE,
            state_delta={
                "sampler": "tpe",
                "seed": seed,
                "num_demo_set_candidates": num_demo_sets,
                "bootstrap_measurement_count": len(intents),
            },
        )

    # -- step 2: pool construction --------------------------------------------

    def _pool_construction(
        self, request: OptimizationStepRequest
    ) -> AdapterOutput:
        hyper = request.hyperparameters
        num_instructions = int(hyper.get(_NUM_INSTRUCTIONS, 6))
        attempt_cap = int(hyper.get(_INSTRUCTION_ATTEMPT_CAP, 12))
        num_demo_sets = int(hyper.get(_NUM_DEMO_SETS, 4))
        mutation_field = str(
            hyper.get(_MUTATION_FIELD, "user_prompt_template")
        )

        # Seed instructions A/B come in as the starting candidates; the
        # proposal LM drafts the remaining accepted distinct identities.
        seed_texts = [
            str(c.payload.get(mutation_field, "")) for c in request.candidates
        ]
        base = request.candidates[0] if request.candidates else None

        instruction_hashes: list[str] = []
        instruction_texts: list[str] = []
        seen: set[str] = set()
        for text in seed_texts:
            if text and text not in seen:
                identity = InstructionIdentity(instruction_text=text)
                seen.add(text)
                instruction_hashes.append(identity.identity_hash())
                instruction_texts.append(text)

        attempts = 0
        proposer_evidence: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        while (
            len(instruction_texts) < num_instructions
            and attempts < attempt_cap
        ):
            proposal_request = ProposalRequest(
                proposal_mode=POOL_CONSTRUCTION,
                request_ordinal=attempts,
                base_ref=base.base_ref if base is not None else request.run_id,
                base_template=seed_texts[0] if seed_texts else "",
                context={"accepted": list(instruction_texts)},
            )
            drafts = self._transport.draft(
                self._proposer_config, proposal_request, 1
            )
            attempts += 1
            for draft in drafts:
                text = draft.template
                # Diff-check the drafted instruction as a candidate mutation.
                if base is not None:
                    candidate = Candidate(
                        candidate_id=f"instr-{attempts}",
                        base_ref=base.base_ref,
                        payload={mutation_field: text},
                    )
                    try:
                        diff_check(
                            base=base,
                            proposed=candidate,
                            mutation_field=mutation_field,
                        )
                    except DiffCheckError as exc:
                        rejected.append(
                            {"attempt": attempts, "reason": str(exc)}
                        )
                        continue
                if not text or text in seen:
                    # A duplicate consumes an attempt but adds no pool member.
                    rejected.append(
                        {"attempt": attempts, "reason": "duplicate/empty"}
                    )
                    continue
                identity = InstructionIdentity(instruction_text=text)
                seen.add(text)
                instruction_hashes.append(identity.identity_hash())
                instruction_texts.append(text)
                proposer_evidence.append(
                    {
                        "instruction_hash": identity.identity_hash(),
                        "request": draft.request_evidence,
                        "usage": draft.usage,
                        "cost": draft.cost,
                    }
                )

        # Cardinality-failure gate: fewer than num_instruction_candidates
        # distinct valid instruction identities by instruction_attempt_cap.
        if len(instruction_texts) < num_instructions:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                state_delta={
                    "reason": (
                        "instruction pool cardinality "
                        f"{len(instruction_texts)} < required "
                        f"{num_instructions} within attempt cap {attempt_cap}"
                    ),
                    "rejected": rejected,
                    "attempts": attempts,
                },
            )

        # Demo-set pool: num_demo_set_candidates identities (incl. the empty
        # set). Built from the bootstrap demo pairs carried in ``pools``.
        raw_demo_sets = request.pools.get(_DEMO_SET_POOL, [])
        demo_set_hashes = _demo_set_hashes(raw_demo_sets, num_demo_sets)

        # Freeze the combination pool = instructions x demo sets.
        combination_hashes: list[str] = []
        for ih in instruction_hashes[:num_instructions]:
            for dh in demo_set_hashes:
                combination_hashes.append(
                    TrialCombinationIdentity(
                        instruction_hash=ih, demo_set_hash=dh
                    ).identity_hash()
                )
        return AdapterOutput(
            proposed_status=StepStatus.CONTINUE,
            state_delta={
                "instruction_pool": instruction_hashes[:num_instructions],
                "instruction_texts": instruction_texts[:num_instructions],
                "demo_set_pool": demo_set_hashes,
                "combination_pool": combination_hashes,
                "combination_pool_size": len(combination_hashes),
                "pool_frozen": True,
                "proposal_temperature": self._proposer_config.temperature,
                "proposer_evidence": proposer_evidence,
                "instruction_attempts": attempts,
            },
        )

    # -- step 3: baseline full ------------------------------------------------

    def _baseline_full(
        self, request: OptimizationStepRequest
    ) -> AdapterOutput:
        ref, chash = _full_eval(request)
        intent = EvaluationIntent(
            intent_id=f"{request.run_id}-baseline-full",
            candidate_id="default-combination",
            target_eval_config_ref=ref,
            target_eval_config_hash=chash,
            context_role=EvaluationRole.INTERNAL,
            purpose=BASELINE_FULL,
            run_id=request.run_id,
            step_index=request.step_index,
        )
        return AdapterOutput(
            evaluation_intents=(intent,),
            proposed_status=StepStatus.CONTINUE,
            state_delta={"baseline_full_requested": True},
        )

    # -- step 4: minibatch trial ----------------------------------------------

    def _minibatch(self, request: OptimizationStepRequest) -> AdapterOutput:
        pool = _combination_pool(request)
        if not pool:
            raise ValueError(
                "minibatch trial requires a frozen combination pool"
            )
        seed = int(request.hyperparameters.get(_SEED, 0))
        ref = str(request.hyperparameters.get(_MINIBATCH_EVAL_REF, ""))
        chash = str(request.hyperparameters.get(_MINIBATCH_EVAL_HASH, ""))
        if not ref or not chash:
            raise ValueError(
                "minibatch trial requires a minibatch Eval Config"
            )
        # Budget guard: a minibatch trial consumes one internal measurement
        # unit. If the search budget is exhausted mid-loop, the trial cannot
        # proceed -> a failed Step Result carrying the exact accounting, which
        # blocks official materialization. Read only from the request budget.
        remaining = request.budget.remaining.get(_BUDGET_KEY)
        if remaining is not None and remaining < 1:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                state_delta={
                    "reason": "search budget exhausted mid-loop",
                    "budget_label": _BUDGET_KEY,
                    "required": 1,
                    "remaining": remaining,
                    "consumed": request.budget.consumed.get(_BUDGET_KEY, 0),
                },
            )
        # The seeded sampler picks any pool combination (repeats allowed).
        combination = _stable_pick(pool, seed, request.step_index)
        intent = EvaluationIntent(
            intent_id=f"{request.run_id}-minibatch-{request.step_index}",
            candidate_id=combination,
            target_eval_config_ref=ref,
            target_eval_config_hash=chash,
            context_role=EvaluationRole.INTERNAL,
            purpose=MINIBATCH,
            run_id=request.run_id,
            step_index=request.step_index,
        )
        return AdapterOutput(
            evaluation_intents=(intent,),
            proposed_status=StepStatus.CONTINUE,
            state_delta={
                "trial_combination_hash": combination,
                "sampler": "tpe",
            },
        )

    # -- step 5: promotion full -----------------------------------------------

    def _promotion_full(
        self, request: OptimizationStepRequest
    ) -> AdapterOutput:
        ref, chash = _full_eval(request)
        study = _study_state(request)
        means: dict[str, float] = study.get("combination_means", {})
        fully: set[str] = set(study.get("fully_evaluated", []))
        # Highest-mean combination not yet fully evaluated under the exact
        # target full Eval Config; stable identity-hash tie-break.
        candidates = [
            (chash_key, mean)
            for chash_key, mean in means.items()
            if chash_key not in fully
        ]
        if not candidates:
            # No-op promotion: every trial-observed combination is already
            # fully evaluated. Deterministic no-op Step Result, no Intent.
            return AdapterOutput(
                proposed_status=StepStatus.CONTINUE,
                state_delta={
                    "promotion": "noop",
                    "reason": (
                        "all trial-observed combinations fully evaluated"
                    ),
                },
            )
        candidates.sort(key=lambda item: (-item[1], item[0]))
        chosen = candidates[0][0]
        intent = EvaluationIntent(
            intent_id=f"{request.run_id}-promotion-{request.step_index}",
            candidate_id=chosen,
            target_eval_config_ref=ref,
            target_eval_config_hash=chash,
            context_role=EvaluationRole.INTERNAL,
            purpose=PROMOTION_FULL,
            run_id=request.run_id,
            step_index=request.step_index,
        )
        return AdapterOutput(
            evaluation_intents=(intent,),
            proposed_status=StepStatus.CONTINUE,
            state_delta={"promoted_combination_hash": chosen},
        )

    # -- step 6: completion ---------------------------------------------------

    def _completion(self, request: OptimizationStepRequest) -> AdapterOutput:
        hyper = request.hyperparameters
        returned = int(hyper.get(_RETURNED_PROPOSAL_COUNT, 3))
        mutation_field = str(
            hyper.get(_MUTATION_FIELD, "user_prompt_template")
        )
        study = _study_state(request)
        full_scores: dict[str, float] = study.get("full_scores", {})
        means: dict[str, float] = study.get("combination_means", {})
        templates: dict[str, str] = study.get("combination_templates", {})

        # Tier 1: full-scored combinations by full score (identity-hash tie).
        tier1 = sorted(
            full_scores.items(), key=lambda item: (-item[1], item[0])
        )
        ordered: list[str] = [chash for chash, _ in tier1]
        # Tier 2: remaining distinct trial-observed combinations by mean
        # minibatch score (identity-hash tie), never re-adding a tier-1 member.
        tier2 = sorted(
            (
                (chash, mean)
                for chash, mean in means.items()
                if chash not in set(ordered)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        ordered.extend(chash for chash, _ in tier2)

        # Cardinality-failure gate: fewer than returned_proposal_count distinct
        # MEASURED combinations. Completion never substitutes an unmeasured
        # fallback; it returns an explicit failed terminal Step Result that
        # prevents official materialization.
        if len(ordered) < returned:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                state_delta={
                    "reason": (
                        f"only {len(ordered)} distinct measured combinations; "
                        f"returned_proposal_count requires {returned}"
                    )
                },
            )

        proposals: list[Candidate] = []
        for rank, chash in enumerate(ordered[:returned]):
            template = templates.get(chash, "")
            if not template:
                # A returned proposal must be measured AND have a template; a
                # missing template means the combination was never realized.
                return AdapterOutput(
                    proposed_status=StepStatus.FAILED,
                    state_delta={
                        "reason": (
                            f"combination {chash} has no realized template"
                        )
                    },
                )
            proposals.append(
                Candidate(
                    candidate_id=f"miprov2-proposal-{rank}",
                    base_ref=request.run_id,
                    payload={mutation_field: template},
                )
            )
        return AdapterOutput(
            proposed_candidates=tuple(proposals),
            accepted_candidates=tuple(proposals),
            proposed_status=StepStatus.COMPLETE,
            state_delta={
                "returned_proposal_count": returned,
                "ordered_combination_hashes": ordered[:returned],
            },
        )


def _full_eval(request: OptimizationStepRequest) -> tuple[str, str]:
    ref = str(request.hyperparameters.get(_FULL_EVAL_REF, ""))
    chash = str(request.hyperparameters.get(_FULL_EVAL_HASH, ""))
    if not ref or not chash:
        raise ValueError("a full-eval step requires a full Eval Config ref")
    return ref, chash


def _combination_pool(request: OptimizationStepRequest) -> tuple[str, ...]:
    pool = request.pools.get(_COMBINATION_POOL, [])
    if not isinstance(pool, list):
        raise ValueError("combination_pool must be a JSON list")
    return tuple(str(h) for h in pool)


def _study_state(request: OptimizationStepRequest) -> dict[str, Any]:
    raw = request.pools.get(_STUDY_STATE, {})
    if not isinstance(raw, dict):
        raise ValueError("study_state must be a JSON object")
    return raw


def _demo_set_hashes(
    raw_demo_sets: Any, num_demo_sets: int
) -> list[str]:
    """The frozen demo-set pool: the empty set plus the bootstrap-supplied
    distinct demo sets, capped at ``num_demo_set_candidates``.

    Demo sets are data-dependent (Whetstone's bootstrap produces the demo
    pairs); this reads exactly what bootstrap carried in ``pools`` and never
    fabricates a demo set. The empty set is always a member.
    """
    hashes: list[str] = [DemoSetIdentity(pairs=()).identity_hash()]
    if isinstance(raw_demo_sets, list):
        for entry in raw_demo_sets:
            if len(hashes) >= num_demo_sets:
                break
            if not isinstance(entry, dict):
                continue
            pairs = entry.get("pairs", [])
            demo_set = DemoSetIdentity.model_validate({"pairs": pairs})
            h = demo_set.identity_hash()
            if h not in hashes:
                hashes.append(h)
    return hashes[:num_demo_sets]
