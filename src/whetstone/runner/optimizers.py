"""Brief-documented optimizer hyperparameters + the internal-split loop.

The five optimizers (Eval identity, COPRO, MIPROv2, GEPA, Codex) share one
harness contract: *the optimizer proposes candidate ``user_prompt_template``
mutations; Whetstone measures them on the internal split (Reward) outside the
optimizer invocation; the best-Reward candidate is returned for official
evaluation.* This module owns:

1. the brief-documented hyperparameters per optimizer
   (:func:`hyperparameters_for`), pinned from ``reports/optimizer-briefs.md``;
2. **scaling to pool sizes** (:func:`scaled_hyperparameters`) -- the brief's
   task/repeat counts assume HumanEval+'s 164/35/20-task splits; the env pools
   here are far smaller, so the internal minibatch/full-eval task counts are
   clamped to the available internal-split size. The scaling is documented in
   the CLI ``--help`` and echoed into the cell report;
3. :func:`run_optimize` -- the internal-split proposal/measure loop that plays
   Whetstone's measurement role: it drafts ``breadth`` templates per round
   through the injected proposer transport, evaluates each on the internal
   split via :func:`whetstone.runner.eval_run.evaluate_split`, and greedily
   keeps the best-Reward candidate. This is a faithful reduction of the durable
   harness's proposal-only path (COPRO/MIPROv2) and the tool-using inner loop
   (GEPA/Codex): identical proposer-route-distinct-from-graph identity,
   internal Reward measurement, no official authority. Codex's proposer is the
   local ``codex exec`` bridge -- ``--lane`` applies only to inner rollouts.

Nothing here makes a live paid call by itself: both the proposer transport and
the rollout transport are injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dr_store import MemoryBackend, ObjectStore
from whetstone_envs.core import Instance

from whetstone.envs.factory import EnvExperiment
from whetstone.envs.registry import env_spec
from whetstone.envs.reward import CandidateEvaluationFailure
from whetstone.envs.rollout_definition import valid_prompt_input_keys
from whetstone.execution.fanout import FanoutConfig
from whetstone.optimization.mutation import (
    MUTATION_FIELD,
    invalid_template_placeholders,
)
from whetstone.optimization.proposer import (
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)
from whetstone.optimization.schema import Candidate
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.eval_run import SplitEvaluation, evaluate_split
from whetstone.runner.execution_mode import ExecutionMode

__all__ = [
    "INVALID_TEMPLATE_PLACEHOLDERS",
    "OPTIMIZATION_TRACE_SCHEMA",
    "OPTIMIZERS",
    "PROPOSER_DRAFT_FAILED",
    "UNSCORABLE_CANDIDATE",
    "OptimizeResult",
    "ProposalStep",
    "hyperparameters_for",
    "run_optimize",
    "scaled_hyperparameters",
    "scaling_help",
]

#: The five optimizers (spend order: eval baseline, COPRO, MIPROv2, GEPA,
#: Codex). "eval" is the identity optimizer (no proposals -- naive == best).
OPTIMIZERS: tuple[str, ...] = ("eval", "copro", "miprov2", "gepa", "codex")

#: Brief-pinned hyperparameters (reports/optimizer-briefs.md registries). Only
#: the search-controlling knobs the internal-split loop needs are carried here;
#: the internal task/repeat counts are the ones scaled to pool sizes.
_BRIEF_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "eval": {
        "kind": "identity",
        "breadth": 0,
        "depth": 0,
        "internal_task_count": 0,
        "internal_repeat_count": 1,
    },
    "copro": {
        "copro_variant": "whetstone_multi_seed/v1",
        "breadth": 4,
        "depth": 2,
        "internal_task_count": 20,  # step_task_count
        "internal_repeat_count": 1,  # step_repeat_count
    },
    "miprov2": {
        "sampler": "tpe",
        "seed": 9,
        "num_trials": 12,
        "minibatch_full_eval_steps": 5,
        "num_demo_set_candidates": 4,
        "num_instruction_candidates": 6,
        "proposal_temperature": 1.0,
        "internal_task_count": 8,  # minibatch_task_count
        "full_eval_task_count": 35,
        "internal_repeat_count": 1,
    },
    "gepa": {
        "gepa_variant": "whetstone_multi_objective/v1",
        "minibatch_size": 3,
        "max_reflection_lm_calls": 8,
        "max_reflection_attempts_per_step": 3,
        "max_optimization_rollouts": 400,
        "acceptance_policy": "same_minibatch_strict_pareto/v1",
        "archive_capacity": 32,
        "internal_task_count": 24,  # objective_task_count (subset)
        "internal_repeat_count": 1,
    },
    "codex": {
        "kind": "opaque",
        "agent_process": "codex_cli",
        "agent_model": "gpt-5.6",
        "max_evaluation_calls": 20,
        "returned_proposal_count": 4,
        "internal_task_count": 20,  # internal_task_count
        "internal_repeat_count": 1,
    },
}


def hyperparameters_for(optimizer: str) -> dict[str, Any]:
    """The brief-pinned hyperparameters for one optimizer (a fresh copy)."""
    if optimizer not in _BRIEF_HYPERPARAMETERS:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}"
        )
    return dict(_BRIEF_HYPERPARAMETERS[optimizer])


def scaled_hyperparameters(
    optimizer: str,
    *,
    internal_pool_size: int,
) -> dict[str, Any]:
    """Scale the brief hyperparameters to the env's internal-pool size.

    The brief's internal task counts (20 / 8 / 24 / 20) assume HumanEval+'s
    large splits. The env pools are small, so every internal task count is
    clamped to ``min(brief_value, internal_pool_size)`` (never zero unless the
    pool is empty). The breadth/depth search knobs and repeat counts are
    unchanged -- only the *measurement* task scope scales. The returned dict
    adds ``internal_task_count_scaled`` and ``scaling_note`` for the report.
    """
    hyper = hyperparameters_for(optimizer)
    brief_internal = int(hyper.get("internal_task_count", 0) or 0)
    scaled = min(brief_internal, internal_pool_size) if brief_internal else 0
    if brief_internal and internal_pool_size < brief_internal:
        note = (
            f"internal_task_count {brief_internal} (brief) clamped to "
            f"{scaled} (internal pool size {internal_pool_size})"
        )
    else:
        note = (
            f"internal_task_count {scaled} (within internal pool size "
            f"{internal_pool_size})"
        )
    hyper["internal_task_count_scaled"] = scaled
    hyper["internal_pool_size"] = internal_pool_size
    hyper["scaling_note"] = note
    if "full_eval_task_count" in hyper:
        full = int(hyper["full_eval_task_count"] or 0)
        hyper["full_eval_task_count_scaled"] = min(full, internal_pool_size)
    return hyper


def scaling_help() -> str:
    """The scaling documentation shown in the CLI ``--help``."""
    lines = [
        "Optimizer hyperparameter scaling (brief -> pool size):",
        "  The brief pins internal task counts against HumanEval+'s large",
        "  splits (COPRO 20, MIPROv2 8 minibatch/35 full, GEPA 24, Codex 20).",
        "  The env pools are far smaller, so every internal task count is",
        "  clamped to min(brief_value, internal_pool_size). Breadth/depth and",
        "  repeat counts are unchanged; only measurement task scope scales.",
    ]
    return "\n".join(lines)


#: The schema tag stamped on the per-cell optimizer-search trace artifact.
OPTIMIZATION_TRACE_SCHEMA = "whetstone.runner.optimization_trace/v1"

#: The typed rejection reason for a candidate whose template references a
#: placeholder the env's render cannot fill. Recorded on the rejected
#: :class:`ProposalStep` so the offending fields are visible in step evidence.
INVALID_TEMPLATE_PLACEHOLDERS = "invalid_template_placeholders"

#: The typed rejection reason for a DRAFT the proposer route could not produce
#: (timeout / nonzero exit / empty / model-rejected). It is recorded as a
#: failed slot with NO template (never a fabricated candidate echoing the
#: base), never scored, never eligible for best -- so a proposer outage is
#: impossible to confuse with a real candidate in the trace/ledger. When EVERY
#: draft in a run fails this way (no real candidate was ever evaluated), the
#: cell finalizes with the loud typed ``proposer-failure`` status, distinct
#: from an honest no-improvement.
PROPOSER_DRAFT_FAILED = "proposer_draft_failed"

#: The typed rejection reason for a PROPOSAL candidate whose internal eval
#: could not be scored into a Reward (its ``env_exact_match`` aggregate came
#: back missing/incomplete under the FAIL Reward policy -- e.g. a transient
#: internal rollout wipeout). This isolates the failure to THAT candidate: it
#: is recorded as a failed step and never selected as best, and the optimizer
#: keeps its best-so-far and continues. It is NOT allowed to abort the whole
#: optimize run (which would discard every already-scored step and finalize the
#: cell incomplete-arm with optimizer_steps=0). Only a BASELINE internal-eval
#: failure -- the anchor itself -- legitimately makes the internal arm
#: incomplete.
UNSCORABLE_CANDIDATE = "unscorable_candidate_internal_eval"


@dataclass(frozen=True, slots=True)
class ProposalStep:
    """One proposal round's evaluated candidate + its internal Reward.

    A candidate REJECTED at intake (its template references a placeholder the
    env's render cannot fill) carries ``evaluation is None``,
    ``internal_score is None``, a typed ``rejected_reason`` and the offending
    ``rejected_fields`` -- it spent NO eval calls and is never selectable as
    best, but is recorded here so the rejection is counted, not silently
    dropped.

    A candidate whose INTERNAL EVAL could not be scored (a transient rollout
    wipeout leaving a missing aggregate under the FAIL Reward policy) also
    carries ``evaluation is None`` with ``rejected_reason`` =
    :data:`UNSCORABLE_CANDIDATE` and a human-readable ``rejected_detail``; it
    is isolated to this candidate (never best) so the optimize run continues
    rather than aborting and discarding every prior step.
    """

    step_index: int
    candidate_id: str
    template: str
    internal_score: float | None
    evaluation: SplitEvaluation | None
    rejected_reason: str | None = None
    rejected_fields: tuple[str, ...] = ()
    rejected_detail: str | None = None

    @property
    def rejected(self) -> bool:
        return self.rejected_reason is not None

    def to_trace_dict(self) -> dict[str, Any]:
        """The per-step evidence for the on-disk optimizer-search trace.

        Carries the candidate identity + FULL prompt text, the internal score,
        the accepted/rejected disposition + typed reason, and (for an evaluated
        candidate) the per-task per-repeat scoring evidence from its
        ``SplitEvaluation`` (per-task means, observation counts, task/repeat
        scope, aggregate value + row accounting). A rejected candidate carries
        ``evaluation=None`` -- its ``internal_score`` is null and the reason
        explains why it never scored.
        """
        entry: dict[str, Any] = {
            "step_index": self.step_index,
            "candidate_id": self.candidate_id,
            "template": self.template,
            "internal_score": self.internal_score,
            "accepted": not self.rejected,
            "rejected": self.rejected,
            "rejected_reason": self.rejected_reason,
            "rejected_detail": self.rejected_detail,
            "rejected_fields": list(self.rejected_fields),
        }
        ev = self.evaluation
        if ev is not None:
            agg = ev.aggregate
            entry["evaluation"] = {
                "score": ev.score,
                "task_count": ev.task_count,
                "repeat_count": ev.repeat_count,
                "per_task_scores": list(ev.per_task_scores),
                "per_task_counts": list(ev.per_task_counts),
                "rows_present": agg.rows_present,
                "rows_missing": agg.rows_missing,
                "rows_failed": agg.rows_failed,
                "rows_invalid": agg.rows_invalid,
                "graph_hash": agg.graph_hash,
                "eval_config_hash": agg.eval_config_hash,
            }
        else:
            entry["evaluation"] = None
        return entry


@dataclass(slots=True)
class OptimizeResult:
    """The optimizer's terminal output over the internal split.

    ``best_candidate`` is the highest-internal-Reward candidate (the naive
    Initial Candidate when no proposal beats it). ``steps`` records every
    evaluated proposal; ``internal_evals_count`` and ``optimizer_steps`` feed
    the cell ledger line.
    """

    optimizer: str
    best_candidate: Candidate
    best_internal_score: float | None
    baseline_internal_score: float | None
    steps: list[ProposalStep] = field(default_factory=list)
    scaled_hyperparameters: dict[str, Any] = field(default_factory=dict)
    #: The internal-split NAIVE baseline eval's SplitEvaluation (for its output
    #: text; the internal baseline is a distinct eval from the official naive).
    baseline_evaluation: SplitEvaluation | None = None

    @property
    def internal_evals_count(self) -> int:
        # Baseline internal eval + one per EVALUATED (non-rejected) step. A
        # rejected candidate spent no eval calls, so it does not count here.
        return 1 + sum(1 for step in self.steps if not step.rejected)

    @property
    def optimizer_steps(self) -> int:
        return len(self.steps)

    @property
    def rejected_candidate_count(self) -> int:
        """How many drafted candidates were rejected at intake (visible)."""
        return sum(1 for step in self.steps if step.rejected)

    @property
    def scored_candidate_count(self) -> int:
        """How many drafted candidates actually scored (real proposals)."""
        return sum(1 for step in self.steps if not step.rejected)

    @property
    def failed_draft_count(self) -> int:
        """How many draft SLOTS the proposer route could not produce."""
        return sum(
            1 for step in self.steps
            if step.rejected_reason == PROPOSER_DRAFT_FAILED
        )

    @property
    def all_drafts_failed(self) -> bool:
        """Whether the proposer produced drafts but EVERY one failed to draft.

        True iff the optimizer attempted at least one proposal slot and NOT ONE
        yielded a real (scorable) candidate -- every slot was a typed
        proposer-draft failure. This is the loud ``proposer-failure`` cell
        condition, distinct from an honest no-improvement (where real
        candidates WERE scored but none beat the baseline). It is False for the
        identity optimizer (which drafts nothing) and whenever any real
        candidate scored or was rejected for a NON-proposer reason (a bad
        placeholder / unscorable candidate is a real draft the proposer DID
        produce).
        """
        if not self.steps:
            return False
        if self.scored_candidate_count > 0:
            return False
        # Every step is a rejection; it is a proposer failure only if EVERY one
        # is a proposer-draft failure (not a placeholder/unscorable rejection
        # of a template the proposer actually produced).
        return all(
            step.rejected_reason == PROPOSER_DRAFT_FAILED
            for step in self.steps
        )

    def to_trace(self, *, header: dict[str, Any]) -> dict[str, Any]:
        """The full on-disk optimizer-search trace for this result.

        ``header`` carries the cell-level identity/status the runner supplies
        (cell id, optimizer, env, attempt, terminal status). The trace pins the
        accepted-candidate prompt text (``best_candidate_template``) so reports
        can quote it directly, the baseline vs best internal scores that drove
        selection, the as-run internal repeat count, and every per-round step's
        candidate evidence (:meth:`ProposalStep.to_trace_dict`).
        """
        return {
            **header,
            "schema": OPTIMIZATION_TRACE_SCHEMA,
            "best_candidate_id": self.best_candidate.candidate_id,
            "best_candidate_template": str(
                self.best_candidate.payload.get(MUTATION_FIELD, "")
            ),
            "baseline_internal_score": self.baseline_internal_score,
            "best_internal_score": self.best_internal_score,
            "optimizer_steps": self.optimizer_steps,
            "internal_evals_count": self.internal_evals_count,
            "rejected_candidate_count": self.rejected_candidate_count,
            "scored_candidate_count": self.scored_candidate_count,
            "failed_draft_count": self.failed_draft_count,
            "all_drafts_failed": self.all_drafts_failed,
            "internal_task_count_scaled": int(
                self.scaled_hyperparameters.get(
                    "internal_task_count_scaled", 0
                )
                or 0
            ),
            # The internal-eval size provenance: "brief" (the brief-clamped
            # default) or "power_stage" (from the power recommendation). The
            # brief-clamped value is retained for the recommended-vs-used line.
            "internal_task_count_source": self.scaled_hyperparameters.get(
                "internal_task_count_source", "brief"
            ),
            "internal_task_count_brief": int(
                self.scaled_hyperparameters.get(
                    "internal_task_count_brief",
                    self.scaled_hyperparameters.get(
                        "internal_task_count_scaled", 0
                    ),
                )
                or 0
            ),
            "steps": [step.to_trace_dict() for step in self.steps],
        }


def _intake_valid_keys(
    experiment: EnvExperiment, instances: tuple[Instance, ...]
) -> frozenset[str]:
    """The keyword fields a candidate template may reference at intake.

    QA envs derive them from the env definition + a sample instance. ed1's
    encoder Mutation Surface is not a QA ``EnvSpec``, so its valid keys are the
    fixed encoder placeholders (``input_code`` / ``max_budget``).
    """
    from whetstone.envs.ed1 import ED1_ENV_NAME

    if experiment.env_name == ED1_ENV_NAME:
        return frozenset({"input_code", "max_budget"})
    env = env_spec(experiment.env_name)
    return valid_prompt_input_keys(env, instances[0])


def _proposal_rounds(hyper: dict[str, Any]) -> tuple[int, int]:
    """(breadth, depth) proposal shape from the scaled hyperparameters.

    Maps each optimizer's search knobs onto a uniform breadth x depth loop the
    internal-split reduction drives: COPRO uses its breadth/depth; MIPROv2 maps
    num_instruction_candidates over one depth; GEPA maps a bounded reflection
    budget; Codex maps returned_proposal_count over one depth.
    """
    kind = hyper.get("kind")
    if kind == "identity":
        return (0, 0)
    if "breadth" in hyper and "depth" in hyper:
        return (int(hyper["breadth"]), int(hyper["depth"]))
    if "num_instruction_candidates" in hyper:
        return (int(hyper["num_instruction_candidates"]), 1)
    if "minibatch_size" in hyper:
        return (int(hyper.get("max_reflection_attempts_per_step", 3)), 2)
    if "returned_proposal_count" in hyper:
        return (int(hyper["returned_proposal_count"]), 1)
    return (0, 0)


def run_optimize(
    experiment: EnvExperiment,
    *,
    optimizer: str,
    proposer_config: ProposerConfig,
    proposer_transport: ProposerTransport,
    rollout_transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    internal_instances: tuple[Instance, ...],
    repeats: int,
    store: ObjectStore | None = None,
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS,
    fanout: FanoutConfig | None = None,
    internal_task_count_override: int | None = None,
) -> OptimizeResult:
    """Run the optimizer on the internal split; return the best candidate.

    The optimizer *proposes* templates through ``proposer_transport`` (the
    proposer route, whose identity is distinct from any graph route); Whetstone
    *measures* each on the internal split via ``rollout_transport``
    and greedily keeps the best internal Reward. The naive Initial Candidate is
    the baseline the best must beat.
    """
    backing = store or ObjectStore(MemoryBackend())
    naive = experiment.initial_candidate
    hyper = scaled_hyperparameters(
        optimizer, internal_pool_size=len(internal_instances)
    )
    brief_task_scope = int(hyper.get("internal_task_count_scaled", 0) or 0)
    # The OPT-IN power stage may override the internal task count (clamped to
    # the pool by the caller). Absent the override, the brief-clamped value
    # drives -- byte-identical to a run without it. The trace records both.
    if internal_task_count_override is not None:
        task_scope = min(
            int(internal_task_count_override), len(internal_instances)
        )
        hyper["internal_task_count_brief"] = brief_task_scope
        hyper["internal_task_count_scaled"] = task_scope
        hyper["internal_task_count_source"] = "power_stage"
    else:
        task_scope = brief_task_scope
        hyper["internal_task_count_source"] = "brief"
    scoped = (
        internal_instances[:task_scope]
        if task_scope
        else internal_instances
    )

    # The identity (eval) optimizer performs NO search: its single internal
    # measurement is never compared against a proposal, so its Reward is
    # vestigial. Deriving one under the env's FAIL missing-data policy would
    # crash the whole cell the moment any internal rollout fails (the ling/c11
    # 429 defect). So the identity optimizer measures the internal split with
    # NO Reward (aggregate + score only); every searching optimizer keeps the
    # Reward it actually greedily selects on.
    needs_reward = hyper.get("kind") != "identity"

    # Baseline: internal-eval of the naive Initial Candidate.
    baseline_eval = evaluate_split(
        experiment,
        candidate=naive,
        instances=scoped,
        split_role="internal_eval",
        transport=rollout_transport,
        execution_policy=execution_policy,
        repeats=repeats,
        store=backing,
        execution_mode=execution_mode,
        fanout=fanout,
        apply_reward=needs_reward,
    )
    best_candidate = naive
    best_score = baseline_eval.score
    steps: list[ProposalStep] = []

    breadth, depth = _proposal_rounds(hyper)

    # The keyword fields a candidate template may safely reference, derived
    # (never hardcoded) from the env definition + a sample of the split we will
    # actually render against. PROPOSED templates are untrusted LLM output: a
    # candidate naming a field the render cannot fill (e.g. c22's {question})
    # is rejected at intake below, before it spends any eval calls -- the c22
    # crash was an unhandled render KeyError that killed the whole cell.
    # Derived ONLY when the optimizer actually drafts (identity draws nothing,
    # and ed1's encoder surface is not a QA EnvSpec so its keys come from ed1).
    valid_keys = _intake_valid_keys(
        experiment, scoped or internal_instances
    ) if (breadth and depth) else frozenset()

    ordinal = 0
    base_template = str(naive.payload[MUTATION_FIELD])
    for round_index in range(depth):
        # The proposer conditions on the current best template + its Reward.
        request = ProposalRequest(
            proposal_mode=("seed_proposal" if round_index == 0
                           else "history_proposal"),
            request_ordinal=round_index,
            base_ref=best_candidate.base_ref,
            base_template=str(best_candidate.payload[MUTATION_FIELD]),
            context={"best_internal_score": best_score},
        )
        drafts = proposer_transport.draft(
            proposer_config, request, breadth
        )
        for draft in drafts:
            ordinal += 1
            candidate_id = f"{optimizer}-p{ordinal}"

            # A TYPED FAILED draft (the proposer route could not produce a
            # template -- timeout / nonzero exit / empty / model-rejected) is
            # recorded as a failed slot with NO template and NO eval. The base
            # template is never echoed, so a proposer outage can never be
            # confused with a real candidate. An empty template on a non-failed
            # draft is treated the same way (it cannot fill the surface).
            if draft.failed or not draft.template.strip():
                steps.append(
                    ProposalStep(
                        step_index=ordinal,
                        candidate_id=candidate_id,
                        template="",
                        internal_score=None,
                        evaluation=None,
                        rejected_reason=PROPOSER_DRAFT_FAILED,
                        rejected_detail=(
                            draft.failure_detail
                            or "proposer returned an empty draft"
                        ),
                    )
                )
                continue

            template = draft.template

            # Intake validation: an untrusted proposed template that references
            # a placeholder the render cannot fill is REJECTED here without any
            # eval spend. It is recorded as a failed step (internal_score=None,
            # evaluation=None) with a typed reason + the offending fields, so
            # the optimizer continues, the candidate is never selected as best,
            # and the rejection is counted rather than silently dropped.
            offending = invalid_template_placeholders(template, valid_keys)
            if offending:
                steps.append(
                    ProposalStep(
                        step_index=ordinal,
                        candidate_id=candidate_id,
                        template=template,
                        internal_score=None,
                        evaluation=None,
                        rejected_reason=INVALID_TEMPLATE_PLACEHOLDERS,
                        rejected_fields=offending,
                    )
                )
                continue

            candidate = Candidate(
                candidate_id=candidate_id,
                base_ref=naive.base_ref,
                payload={MUTATION_FIELD: template},
            )
            try:
                evaluation = evaluate_split(
                    experiment,
                    candidate=candidate,
                    instances=scoped,
                    split_role="internal_eval",
                    transport=rollout_transport,
                    execution_policy=execution_policy,
                    repeats=repeats,
                    store=backing,
                    execution_mode=execution_mode,
                    fanout=fanout,
                    apply_reward=needs_reward,
                    render_guard=True,
                )
            except CandidateEvaluationFailure as exc:
                # This PROPOSAL candidate's internal aggregate came back
                # missing/incomplete (a transient internal-rollout wipeout
                # under the FAIL Reward policy). Isolate the failure to this
                # candidate: record it as a failed step (never selected as
                # best) and keep scoring the rest. Pre-fix this exception
                # aborted the whole optimize run, DISCARDING every
                # already-scored step, so the cell finalized incomplete-arm
                # with optimizer_steps=0 -- a silent loss of all prior work.
                # Only the BASELINE anchor eval (outside this loop) may
                # legitimately make the internal arm incomplete.
                steps.append(
                    ProposalStep(
                        step_index=ordinal,
                        candidate_id=candidate_id,
                        template=template,
                        internal_score=None,
                        evaluation=None,
                        rejected_reason=UNSCORABLE_CANDIDATE,
                        rejected_detail=str(exc),
                    )
                )
                continue
            steps.append(
                ProposalStep(
                    step_index=ordinal,
                    candidate_id=candidate.candidate_id,
                    template=template,
                    internal_score=evaluation.score,
                    evaluation=evaluation,
                )
            )
            if evaluation.score is not None and (
                best_score is None or evaluation.score > best_score
            ):
                best_candidate = candidate
                best_score = evaluation.score

    _ = base_template  # retained provenance for the seed condition
    return OptimizeResult(
        optimizer=optimizer,
        best_candidate=best_candidate,
        best_internal_score=best_score,
        baseline_internal_score=baseline_eval.score,
        steps=steps,
        scaled_hyperparameters=hyper,
        baseline_evaluation=baseline_eval,
    )
