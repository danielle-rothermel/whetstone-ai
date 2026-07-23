"""The ed1m env: behavioral-mutant encoder-decoder with attractor dual scoring.

ed1m is the behavioral-mutant variant of ed1 (task 18). It reuses the whole ed1
enc-dec machinery -- the immutable encoder frame + strategy-body Mutation
Surface, the decoder, compression scoring, the task-22 weighted-blend reward,
the no-budget frame, SKIP tolerance, telemetry, dual-score sidecars -- and it
ONLY the correctness scorer:

  * The encoder's INPUT_CODE is the MUTATED HumanEval+ program (a seeded bug).
  * The decoder reconstructs a program.
  * The reconstruction is scored per-input against the mutant's recorded oracle
    (:mod:`whetstone.envs.ed1m_oracle`): ``fidelity_to_mutant`` (the fraction
    of inputs matching the mutant) is the REWARD-bearing metric (blended with
    compression per task 22 -- ed1m is in the ed1 family, so the guard rail
    applies); ``attractor_pull`` (the fraction of DISCRIMINATING inputs that
    snapped to the CANONICAL behavior) is the REPORTED contamination
    measurement, NEVER a reward objective.

The mutant suite is loaded DIRECTLY from ``mutants.jsonl``
(:mod:`whetstone.envs.ed1m_dataset`) -- no ``dr_code.mutants`` import, so ed1m
builds/tests WITHOUT a dr-code checkout flip.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from whetstone_envs.core import Instance

from whetstone.envs.ed1 import (
    Ed1Experiment,
    build_ed1_procedure_config,
    build_ed1_reward_policy,
    ed1_ceiling_candidate,
    ed1_initial_candidate,
)
from whetstone.envs.ed1_blended import BoundedCompressionMetricConfig
from whetstone.envs.ed1_scoring import CodeScore
from whetstone.envs.ed1m_dataset import (
    Ed1mMutant,
    ed1m_manifest_identity,
    load_ed1m_mutants,
)
from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction
from whetstone.envs.encdec_rollout import build_encdec_rollout_definition
from whetstone.envs.factory import EnvEvalConfigs
from whetstone.envs.sampling import Completeness

ED1M_ENV_NAME = "ed1m"
#: ed1m uses the same task model as ed1 (deepseek), a distinct provider Config.
ED1M_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

#: The ed1m stratum tag (single stratum; the mutant families are recorded but
#: not stratified for the first pass).
_ED1M_STRATUM = "ed1m"


def _mutant_to_instance(mutant: Ed1mMutant) -> Instance:
    """Pack one mutant as a whetstone Instance (mutated source = INPUT_CODE).

    The encoder INPUT_CODE is the MUTATED program; the compression reference is
    the same bytes (definitional continuity with ed1). The oracle fields ride
    on the experiment's mutant map (keyed by Instance id), not the Instance,
    so the Instance stays a light string carrier.
    """
    return Instance(
        id=mutant.mutant_id,
        seed=mutant.seed,
        strata=(_ED1M_STRATUM,),
        prompt_inputs={
            "input_code": mutant.mutated_full_source,
            "task_id": mutant.task_id,
            "entry_point": mutant.entry_point,
            "operator_family": mutant.operator_family,
        },
        gold=mutant.canonical_full_source,
    )


@dataclass(frozen=True, slots=True)
class Ed1mExperiment(Ed1Experiment):
    """An ``Ed1Experiment`` whose correctness scorer is the mutant oracle.

    Carries the per-instance mutant map (``mutants`` keyed by Instance id) so
    :func:`score_ed1m_row` scores a reconstruction against the right mutant's
    dual oracle. Everything else (enc-dec rollout, blend config, budget frame,
    reward policy, completeness) is inherited from :class:`Ed1Experiment`, so
    the ed1 eval / cell / telemetry pipeline flows unchanged.
    """

    #: Per-instance mutant map (Instance id -> the mutant its oracle scores).
    mutants: dict[str, Ed1mMutant] = field(default_factory=dict)


def score_ed1m_row(
    experiment: Ed1mExperiment, instance: Instance, reconstruction: str
) -> CodeScore:
    """Score one ed1m reconstruction via the instance's mutant dual oracle.

    Returns a :class:`CodeScore` whose ``fidelity`` (fractional, rewarded)
    + ``attractor_pull`` (reported) come from the per-input oracle. An
    infrastructure-unknown oracle (subprocess crash/timeout on every input)
    fails the row (never scores 0), matching the ed1 invariant.
    """
    mutant = experiment.mutants.get(str(instance.id))
    if mutant is None:  # pragma: no cover - guarded by construction
        raise KeyError(
            f"ed1m instance {instance.id!r} has no mutant in the map"
        )
    scorer = experiment.scorer
    if scorer is not None:
        # A test/dry-run may inject a fake scorer taking the reconstruction +
        # mutant; the production path uses the local subprocess oracle.
        result = scorer(reconstruction=reconstruction, mutant=mutant)
        if isinstance(result, CodeScore):
            return result
    score = score_ed1m_reconstruction(
        reconstruction=reconstruction, mutant=mutant
    )
    if score.infrastructure_unknown or score.fidelity_to_mutant is None:
        return CodeScore(
            passed=False, infrastructure_unknown=True,
            outcome="ed1m_oracle_infrastructure_unknown",
        )
    return CodeScore(
        passed=score.fidelity_to_mutant >= 1.0,
        infrastructure_unknown=False,
        outcome="ed1m_scored",
        fidelity=score.fidelity_to_mutant,
        attractor_pull=score.attractor_pull,
    )


def build_ed1m_experiment(
    *,
    model: str = ED1M_CANONICAL_MODEL,
    budget_ratio: float | None = None,
    prefer_snapshot: bool = True,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    repeats: int = 3,
    mutants: tuple[Ed1mMutant, ...] | None = None,
    exclude_mutant_ids: frozenset[str] | None = None,
    blend_config: BoundedCompressionMetricConfig | None = None,
    scorer: Callable[..., CodeScore] | None = None,
) -> Ed1mExperiment:
    """Build the ed1m experiment (mutant enc-dec + dual scoring).

    Loads the behavioral-mutant suite (or uses injected ``mutants`` for tests),
    packs each mutant as an Instance, and builds the SAME enc-dec rollout +
    configs / blended reward as ed1 -- with the mutant oracle as the scorer.
    ``budget_ratio=None`` (the default) uses the no-budget frame. Mirrors
    ``build_ed1_experiment``'s split + identity handling.
    """
    pool = (
        mutants if mutants is not None
        else load_ed1m_mutants(limit=limit)
    )
    if exclude_mutant_ids:
        pool = tuple(m for m in pool if m.mutant_id not in exclude_mutant_ids)
    if not pool:
        raise ValueError("ed1m mutant pool is empty")

    procedure = build_ed1_procedure_config()
    rollout = build_encdec_rollout_definition(
        ED1M_ENV_NAME,
        model=model,
        procedure_config_hash=procedure.config_identity_hash,
        budget_ratio=budget_ratio,
    )
    all_instances = tuple(_mutant_to_instance(m) for m in pool)
    mutant_map = {m.mutant_id: m for m in pool}
    n = len(all_instances)
    i_n = internal_n if internal_n is not None else min(max(1, n // 2), n)
    internal_instances = all_instances[:i_n]
    rest = all_instances[i_n:]
    o_n = official_n if official_n is not None else len(rest)
    official_instances = rest[:o_n] if rest else internal_instances[:o_n or n]
    if not official_instances:
        official_instances = internal_instances

    from whetstone.envs.ed1 import _ed1_split

    internal_split = _ed1_split(
        split_role="internal_eval", instances=internal_instances,
        procedure=procedure, completeness=completeness,
        max_skip_fraction=max_skip_fraction, repeats=repeats,
    )
    official_split = _ed1_split(
        split_role="official", instances=official_instances,
        procedure=procedure, completeness=completeness,
        max_skip_fraction=max_skip_fraction, repeats=repeats,
    )
    eval_configs = EnvEvalConfigs(
        env_name=ED1M_ENV_NAME,
        procedure_config_hash=procedure.config_identity_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_identities=(),
    )
    experiment = Ed1mExperiment(
        env_name=ED1M_ENV_NAME,
        rollout_definition=rollout,  # type: ignore[arg-type]
        initial_candidate=ed1_initial_candidate(),
        ceiling_candidate=ed1_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=build_ed1_reward_policy(),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        encdec_rollout=rollout,
        budget_ratio=budget_ratio,
        dataset_revision=ed1m_manifest_identity() or "unknown",
        scorer=scorer,
        blend_config=blend_config,
        mutants=mutant_map,
    )
    return experiment


__all__ = [
    "ED1M_CANONICAL_MODEL",
    "ED1M_ENV_NAME",
    "Ed1mExperiment",
    "build_ed1m_experiment",
    "score_ed1m_row",
]
