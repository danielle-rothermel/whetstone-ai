from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_providers import ProviderCallConfig
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.modes.encdec import (
    EncDecExperiment,
    _ed1_split,
    ed1_ceiling_candidate,
    ed1_initial_candidate,
)
from whetstone.envs.code_comp.mutant.dataset import MutantRecord, load_dataset
from whetstone.envs.code_comp.mutant.oracle import MutantScore
from whetstone.envs.code_comp.procedure import build_code_eval_procedure_config
from whetstone.envs.code_comp.reward.blended import (
    BoundedCompressionMetricConfig,
    build_ed1_blended_reward_policy,
)
from whetstone.envs.code_comp.rollout.encdec import (
    build_encdec_rollout_definition,
    build_encoder_provider_call_config,
)
from whetstone.envs.code_comp.scoring import CodeScore
from whetstone.envs.factory import EnvEvalConfigs
from whetstone.envs.sampling import Completeness
from whetstone.experiment.reward import (
    MissingDataPolicy,
    RewardPolicy,
    RewardTerm,
)

MUTANT_ENV_NAME = "ed1m"
ED1M_ENV_NAME = MUTANT_ENV_NAME
#: ed1m uses the same task model as ed1 (deepseek), a distinct provider Config.
ED1M_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"
_ED1M_CANONICAL_PROVIDER_CALL_CONFIG = build_encoder_provider_call_config(
    ED1M_CANONICAL_MODEL
)
#: The per-row metric, aggregate, and Reward-term identity for ED1M fidelity.
ED1M_FIDELITY_NAME = "fidelity_to_mutant"

#: The ed1m stratum tag; mutant families are recorded but not stratified.
_ED1M_STRATUM = "ed1m"


def build_ed1m_procedure_config():
    """The canonical ED1M fidelity-to-mutant evaluation procedure."""
    return build_code_eval_procedure_config(
        env_name=ED1M_ENV_NAME,
        primary_metric_name=ED1M_FIDELITY_NAME,
        primary_metric_settings=(
            (
                "scorer",
                "whetstone.envs.ed1m_oracle.score_ed1m_reconstruction",
            ),
            ("reference", "authenticated_mutant_record"),
        ),
    )


def build_ed1m_reward_policy() -> RewardPolicy:
    """The ED1M Reward Policy: maximize fidelity to the mutant."""
    return RewardPolicy(
        policy_name=f"whetstone.env.{ED1M_ENV_NAME}.reward",
        reward_name="reward",
        terms=(
            RewardTerm(
                name=ED1M_FIDELITY_NAME,
                weight=1.0,
                maximize=True,
            ),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def _mutant_to_instance(mutant: MutantRecord) -> Instance:
    """Pack one mutant as a whetstone Instance (mutated source = INPUT_CODE).

    The encoder INPUT_CODE is the MUTATED program; the compression reference is
    the same bytes (definitional continuity with ed1). The oracle fields ride
    on the experiment's mutant map (keyed by Instance id), not the Instance,
    so the Instance stays a light string carrier.
    """
    return Instance(
        id=mutant.content_identity,
        seed=mutant.seed,
        strata=(_ED1M_STRATUM,),
        prompt_inputs={
            "input_code": mutant.mutated_full_source,
            "task_id": mutant.task_id,
            "entry_point": mutant.entry_point,
            "operator_family": mutant.operator_family.value,
        },
        gold=mutant.canonical_full_source,
    )


@dataclass(frozen=True, slots=True)
class MutantExperiment(EncDecExperiment):
    """An ``EncDecExperiment`` whose correctness scorer is the mutant oracle.

    Carries the per-instance mutant map (``mutants`` keyed by Instance id) so
    :func:`score_ed1m_row` scores a reconstruction against the right mutant's
    dual oracle. Everything else (enc-dec rollout, blend config, budget frame,
    reward policy, completeness) is inherited from
    :class:`EncDecExperiment`, so the ed1 eval / cell / telemetry pipeline
    flows unchanged.
    """

    #: Per-instance mutant map (Instance id -> the mutant its oracle scores).
    mutants: dict[str, MutantRecord] = field(default_factory=dict)


def score_ed1m_row(
    experiment: MutantExperiment,
    instance: Instance,
    reconstruction: str,
    scorer: Callable[..., object],
) -> CodeScore:
    """Score one ed1m reconstruction via the instance's mutant dual oracle.

    Returns a :class:`CodeScore` whose ``fidelity_to_mutant`` (fractional,
    rewarded) + ``attractor_pull`` (reported) come from the per-input oracle.
    An infrastructure-unknown oracle failure fails the row (never scores 0),
    matching the ed1 invariant.
    """
    mutant = experiment.mutants.get(str(instance.id))
    if mutant is None:  # pragma: no cover - guarded by construction
        raise KeyError(
            f"ed1m instance {instance.id!r} has no mutant in the map"
        )
    score = scorer(reconstruction=reconstruction, mutant=mutant)
    if isinstance(score, CodeScore):
        return score
    if not isinstance(score, MutantScore):
        raise TypeError("ED1M scorer returned an unsupported result")
    if score.infrastructure_unknown or score.fidelity_to_mutant is None:
        return CodeScore(
            passed=False,
            infrastructure_unknown=True,
            outcome="ed1m_oracle_infrastructure_unknown",
        )
    return CodeScore(
        passed=score.fidelity_to_mutant >= 1.0,
        infrastructure_unknown=False,
        outcome="ed1m_scored",
        fidelity_to_mutant=score.fidelity_to_mutant,
        attractor_pull=score.attractor_pull,
    )


def build_mutant_experiment(
    *,
    artifact_dir: Path,
    provider_call_config: ProviderCallConfig = (
        _ED1M_CANONICAL_PROVIDER_CALL_CONFIG
    ),
    budget_ratio: float | None = None,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    repeats: int = 3,
    exclude_mutant_ids: frozenset[str] | None = None,
    blend_config: BoundedCompressionMetricConfig | None = None,
    scorer: Callable[..., CodeScore] | None = None,
) -> MutantExperiment:
    """Build the ed1m experiment (mutant enc-dec + dual scoring).

    Verifies the retained artifact schemas, hashes, identities, ordering, and
    internal consistency, then packs its ``MutantRecord`` values as Instances
    and builds the same enc-dec rollout, configs, and blended reward as ed1
    with the mutant oracle as scorer. The manifest's
    ``canonical_suite_digest`` is opaque recorded provenance; the external
    canonical suite is not independently reauthenticated. ``budget_ratio=None``
    (the default) uses the no-budget frame. The manifest's dataset identity is
    carried as the experiment and split dataset revision.
    """
    if not isinstance(artifact_dir, Path):
        raise TypeError("artifact_dir must be pathlib.Path")
    loaded = load_dataset(artifact_dir)
    pool = loaded.records[:limit] if limit is not None else loaded.records
    if exclude_mutant_ids:
        pool = tuple(
            mutant
            for mutant in pool
            if mutant.content_identity not in exclude_mutant_ids
        )
    if not pool:
        raise ValueError("ed1m mutant pool is empty")

    procedure = build_ed1m_procedure_config()
    rollout = build_encdec_rollout_definition(
        ED1M_ENV_NAME,
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure.config_identity_hash,
        budget_ratio=budget_ratio,
    )
    all_instances = tuple(_mutant_to_instance(m) for m in pool)
    mutant_map = {m.content_identity: m for m in pool}
    n = len(all_instances)
    i_n = internal_n if internal_n is not None else min(max(1, n // 2), n)
    internal_instances = all_instances[:i_n]
    rest = all_instances[i_n:]
    o_n = official_n if official_n is not None else len(rest)
    official_instances = rest[:o_n] if rest else internal_instances[: o_n or n]
    if not official_instances:
        official_instances = internal_instances

    internal_split = _ed1_split(
        env_name=ED1M_ENV_NAME,
        dataset_revision=loaded.manifest.dataset_identity,
        split_role="internal_eval",
        instances=internal_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
    )
    official_split = _ed1_split(
        env_name=ED1M_ENV_NAME,
        dataset_revision=loaded.manifest.dataset_identity,
        split_role="official",
        instances=official_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
    )
    eval_configs = EnvEvalConfigs(
        env_name=ED1M_ENV_NAME,
        procedure_config_hash=procedure.config_identity_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_identities=(),
    )
    experiment = MutantExperiment(
        env_name=ED1M_ENV_NAME,
        rollout_definition=rollout,  # type: ignore[arg-type]
        initial_candidate=ed1_initial_candidate(),
        ceiling_candidate=ed1_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=(
            build_ed1m_reward_policy()
            if blend_config is None
            else build_ed1_blended_reward_policy(
                blend_config, env_name=ED1M_ENV_NAME
            )
        ),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        encdec_rollout=rollout,
        budget_ratio=budget_ratio,
        dataset_revision=loaded.manifest.dataset_identity,
        scorer=scorer,
        blend_config=blend_config,
        mutants=mutant_map,
    )
    return experiment


__all__ = [
    "ED1M_CANONICAL_MODEL",
    "ED1M_ENV_NAME",
    "ED1M_FIDELITY_NAME",
    "MutantExperiment",
    "build_ed1m_procedure_config",
    "build_ed1m_reward_policy",
    "build_mutant_experiment",
    "score_ed1m_row",
]
