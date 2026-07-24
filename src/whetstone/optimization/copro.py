"""The COPRO proposal-only adapter (``copro_variant=whetstone_multi_seed/v1``).

Implements the loop shape pinned by ``optimizer-briefs.md`` §2 (COPRO Run) and
``copro-run.html``:

* **Depth** proposal-only Steps, one per proposal round (``depth=2`` default).
* Round 1 is **Seed Proposal**: the Proposal LM receives the starting
  candidates and their externally measured H0 entries.
* Round >=2 is **History Proposal**: the Proposal LM receives the complete
  **Reward-ranked Attempt History** produced through the preceding step.
* Each Step returns exactly **breadth** new candidate mutations (``breadth=4``
  default; this variant does NOT count the original prompt inside the batch) +
  stable candidate IDs + one **Evaluation Intent** (the harness evaluates the
  batch externally under ``step_eval_config_ref``) + Proposal LM evidence.
* The Attempt History is carried across Steps only through immutable state
  references in the Step Request/Result; COPRO **reads** it and never mutates
  in place. Next-round proposals are conditioned on the Reward evidence
  resolved from prior Step Results.

The adapter owns only algorithm-specific proposal logic. It reaches the
Proposal LM through the **proposer route** (a :class:`ProposerConfig` — a
Provider Call Config distinct from any encoder/decoder route), which lives in
the optimizer Config identity, never a graph identity. Whetstone validates,
materializes, evaluates, aggregates, and derives Reward outside the invocation.
"""

from __future__ import annotations

from typing import Any

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.adapters import AdapterOutput
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
    "COPRO_VARIANT",
    "HISTORY_PROPOSAL",
    "SEED_PROPOSAL",
    "CoproAdapter",
    "attempt_history_entries",
    "rank_attempt_history",
]

COPRO_VARIANT = "whetstone_multi_seed/v1"
SEED_PROPOSAL = "seed_proposal"
HISTORY_PROPOSAL = "history_proposal"

# Hyperparameter keys the harness carries on the Step Request.
_BREADTH_KEY = "breadth"
_DEPTH_KEY = "depth"
_MUTATION_FIELD_KEY = "mutation_field"
_STEP_EVAL_CONFIG_REF_KEY = "step_eval_config_ref"
_STEP_EVAL_CONFIG_HASH_KEY = "step_eval_config_hash"

# Budget label for the proposer-draft units a proposal Step consumes.
_BUDGET_KEY = "proposal_calls"

# Pool keys the harness carries on the Step Request (built from prior Results).
_HISTORY_KEY = "attempt_history"


def attempt_history_entries(
    request: OptimizationStepRequest,
) -> tuple[dict[str, Any], ...]:
    """The immutable Attempt History entries the harness carried in ``pools``.

    Whetstone builds each history version after external evaluation and threads
    it forward through the Step Request's ``pools``. The adapter only reads it.
    """
    raw = request.pools.get(_HISTORY_KEY, [])
    if not isinstance(raw, list):
        raise ValueError("attempt_history must be a JSON list")
    return tuple(dict(entry) for entry in raw)


def rank_attempt_history(
    entries: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Return the Attempt History Reward-ranked (best first, stable ties).

    The Reward ordering the run pins is: higher pass reward first, and the
    ``step_reward_policy_ref`` tie-break (candidate_id ASC) is applied as a
    deterministic secondary key so the ranking is reproducible. Missing rewards
    sort last (they never displace a measured entry).
    """

    def sort_key(entry: dict[str, Any]) -> tuple[float, str]:
        reward = entry.get("reward")
        # Higher Reward first -> negate; missing -> -inf sorts last.
        value = float(reward) if isinstance(reward, (int, float)) else None
        primary = -value if value is not None else float("inf")
        return (primary, str(entry.get("candidate_id", "")))

    return tuple(sorted(entries, key=sort_key))


class CoproAdapter:
    """The proposal-only COPRO adapter.

    Constructed with the proposer route (:class:`ProposerConfig`) + a
    :class:`ProposerTransport` (real dr-providers or the scripted fake). These
    are process-side compute, NOT serialized into the Step Request — the
    request carries only identity hashes and hyperparameters.
    """

    def __init__(
        self,
        *,
        proposer_config: ProposerConfig,
        transport: ProposerTransport,
    ) -> None:
        self._proposer_config = proposer_config
        self._transport = transport
        # Invocation counter lets restart tests prove a completed proposal
        # invocation is reused from the durable checkpoint, never rerun.
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
        if request.mode is not StepMode.PROPOSAL_ONLY:
            raise ValueError("COPRO runs only proposal-only Steps")
        if handles:
            raise ValueError("COPRO is proposal-only and needs no handles")

        hyper = request.hyperparameters
        breadth = int(hyper.get(_BREADTH_KEY, 4))
        depth = int(hyper.get(_DEPTH_KEY, 2))
        if breadth < 1:
            raise ValueError("breadth must be >= 1")
        mutation_field = str(
            hyper.get(_MUTATION_FIELD_KEY, "user_prompt_template")
        )
        eval_ref = str(hyper.get(_STEP_EVAL_CONFIG_REF_KEY, ""))
        eval_hash = str(hyper.get(_STEP_EVAL_CONFIG_HASH_KEY, ""))
        if not eval_ref or not eval_hash:
            raise ValueError(
                "COPRO requires a pinned step_eval_config_ref + hash to emit "
                "its Evaluation Intent"
            )

        # Round index is the Step's ordered index (0-based). Round 1 = index 0.
        round_index = request.step_index
        if round_index >= depth:
            raise ValueError(
                f"COPRO Step index {round_index} exceeds depth {depth}"
            )
        proposal_mode = SEED_PROPOSAL if round_index == 0 else HISTORY_PROPOSAL

        # Budget guard: this Step's batch needs `breadth` proposer-draft units.
        # If the remaining proposal budget cannot cover them, the loop cannot
        # proceed -> a failed Step Result with the exact accounting, which
        # blocks official materialization. Budget is read only from the
        # immutable request budget, never process memory.
        remaining = request.budget.remaining.get(_BUDGET_KEY)
        if remaining is not None and remaining < breadth:
            return AdapterOutput(
                proposed_status=StepStatus.FAILED,
                state_delta={
                    "reason": "proposal budget exhausted mid-loop",
                    "budget_label": _BUDGET_KEY,
                    "required": breadth,
                    "remaining": remaining,
                    "consumed": request.budget.consumed.get(_BUDGET_KEY, 0),
                },
            )

        history = attempt_history_entries(request)
        ranked = rank_attempt_history(history)

        # Round 1 (seed): condition on the starting candidates + their H0
        # measured entries. Round >=2 (history): condition on the complete
        # Reward-ranked Attempt History through the preceding Step.
        if proposal_mode == SEED_PROPOSAL:
            if not request.candidates:
                raise ValueError(
                    "Seed Proposal requires the starting candidates"
                )
            base = request.candidates[0]
            context: dict[str, Any] = {
                "seed_entries": [
                    {
                        "candidate_id": c.candidate_id,
                        "template": c.payload.get(mutation_field, ""),
                    }
                    for c in request.candidates
                ],
                "measured": list(ranked),
            }
        else:
            if not ranked:
                raise ValueError(
                    "History Proposal requires a non-empty Attempt History"
                )
            # The base to diff against is the best-ranked prior candidate's
            # template, reconstructed as a Candidate.
            best = ranked[0]
            base = Candidate(
                candidate_id=str(best.get("candidate_id", "best")),
                base_ref=str(best.get("base_ref", request.run_id)),
                payload={mutation_field: str(best.get("template", ""))},
            )
            context = {"ranked_history": list(ranked)}

        proposal_request = ProposalRequest(
            proposal_mode=proposal_mode,
            request_ordinal=round_index,
            base_ref=base.base_ref,
            base_template=str(base.payload.get(mutation_field, "")),
            context=context,
        )
        drafts = self._transport.draft(
            self._proposer_config, proposal_request, breadth
        )

        # Build exactly `breadth` new candidates, diff-checking each draft
        # against the base. A draft that fails the Mutation-Surface diff check
        # is rejected (recorded as provenance) and never becomes a proposal.
        candidates: list[Candidate] = []
        rejected: list[dict[str, Any]] = []
        proposer_evidence: list[dict[str, Any]] = []
        for offset, draft in enumerate(drafts):
            cid = f"P{round_index}-{offset}"
            candidate = Candidate(
                candidate_id=cid,
                base_ref=base.base_ref,
                payload={mutation_field: draft.template},
            )
            try:
                diff_check(
                    base=base,
                    proposed=candidate,
                    mutation_field=mutation_field,
                )
            except DiffCheckError as exc:
                rejected.append(
                    {"candidate_id": cid, "template": draft.template,
                     "reason": str(exc)}
                )
                continue
            candidates.append(candidate)
            proposer_evidence.append(
                {
                    "candidate_id": cid,
                    "request": draft.request_evidence,
                    "response": draft.response_evidence,
                    "usage": draft.usage,
                    "cost": draft.cost,
                }
            )

        if len(candidates) < breadth:
            # Cardinality failure: this variant fixes exactly `breadth` new
            # candidates per Step; if valid drafts fall short, the Step fails
            # (blocking official materialization) rather than under-proposing.
            return AdapterOutput(
                proposed_candidates=tuple(candidates),
                accepted_candidates=(),
                proposed_status=StepStatus.FAILED,
                state_delta={
                    "proposal_mode": proposal_mode,
                    "rejected": rejected,
                    "reason": (
                        f"produced {len(candidates)} valid candidates, "
                        f"breadth requires {breadth}"
                    ),
                },
            )

        # One Evaluation Intent for the Step's batch. The harness resolves it
        # externally under the exact target Eval Config; the batch's stable
        # identity correlates the resolution back to this Step's candidates.
        batch_id = f"copro-batch-{request.run_id}-{round_index}"
        intent = EvaluationIntent(
            intent_id=f"{request.run_id}-{round_index}-batch",
            candidate_id=batch_id,
            target_eval_config_ref=eval_ref,
            target_eval_config_hash=eval_hash,
            context_role=EvaluationRole.INTERNAL,
            purpose=proposal_mode,
            run_id=request.run_id,
            step_index=round_index,
        )

        # Depth-bounded: the final Step (index depth-1) proposes `complete`.
        status = (
            StepStatus.COMPLETE
            if round_index == depth - 1
            else StepStatus.CONTINUE
        )
        return AdapterOutput(
            proposed_candidates=tuple(candidates),
            accepted_candidates=tuple(candidates),
            evaluation_intents=(intent,),
            proposed_status=status,
            state_delta={
                "proposal_mode": proposal_mode,
                "proposer_config_hash": self._proposer_config.identity_hash(),
                "proposer_evidence": proposer_evidence,
                "batch_id": batch_id,
                "new_candidate_ids": [c.candidate_id for c in candidates],
            },
        )
