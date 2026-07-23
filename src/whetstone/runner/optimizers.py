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
from whetstone.execution.fanout import FanoutConfig
from whetstone.optimization.mutation import MUTATION_FIELD
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
    "OPTIMIZERS",
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


@dataclass(frozen=True, slots=True)
class ProposalStep:
    """One proposal round's evaluated candidate + its internal Reward."""

    step_index: int
    candidate_id: str
    template: str
    internal_score: float | None
    evaluation: SplitEvaluation


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

    @property
    def internal_evals_count(self) -> int:
        # Baseline internal eval + one per proposal step.
        return 1 + len(self.steps)

    @property
    def optimizer_steps(self) -> int:
        return len(self.steps)


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
    task_scope = int(hyper.get("internal_task_count_scaled", 0) or 0)
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
            template = draft.template
            if not template.strip():
                continue  # an empty draft cannot fill the surface
            ordinal += 1
            candidate = Candidate(
                candidate_id=f"{optimizer}-p{ordinal}",
                base_ref=naive.base_ref,
                payload={MUTATION_FIELD: template},
            )
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
            )
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
    )
