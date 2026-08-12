from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from dr_exec import Executor
from dr_providers import ProviderCallConfig
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.config import (
    CodeCompModelRouteConfig,
    CodeCompModelRoutesConfig,
    default_code_comp_config,
)
from whetstone.envs.code_comp.constants import CODE_COMP_ENV_NAME
from whetstone.envs.code_comp.experiment import MutantExperiment
from whetstone.envs.code_comp.generation_graph.encdec import (
    build_encoder_provider_call_config,
)
from whetstone.envs.code_comp.mode import CodeCompMode
from whetstone.envs.code_comp.mutant.dataset import MutantRecord
from whetstone.envs.code_comp.mutant.oracle import (
    score_mutant_reconstruction_with_outcomes,
)
from whetstone.envs.code_comp.procedure import build_code_eval_procedure_config
from whetstone.envs.code_comp.reward.blended import (
    BoundedCompressionMetricConfig,
)
from whetstone.envs.code_comp.submission_result import (
    CodeSubmissionResult,
    MutantSubmissionResult,
    project_mutant_submission_result,
)
from whetstone.envs.sampling import Completeness
from whetstone.experiment.reward import (
    MissingDataPolicy,
    RewardPolicy,
    RewardTerm,
)

CODE_COMP_MUTANT_MODE = CodeCompMode.ENCDEC_MUTANT

#: ed1m uses the same task model as ed1 (deepseek), a distinct provider Config.
CODE_COMP_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"
_MUTANT_CANONICAL_PROVIDER_CALL_CONFIG = build_encoder_provider_call_config(
    CODE_COMP_CANONICAL_MODEL
)
#: The per-row metric, aggregate, and Reward-term identity for ED1M fidelity.
CODE_COMP_MUTANT_FIDELITY_NAME = "fidelity_to_mutant"

#: The encdec_mutant stratum tag; mutant families are recorded but not
#: stratified.
_MUTANT_STRATUM = "encdec_mutant"


def build_mutant_procedure_config():
    """The canonical ED1M fidelity-to-mutant evaluation procedure."""
    return build_code_eval_procedure_config(
        env_name=CODE_COMP_ENV_NAME,
        primary_metric_name=CODE_COMP_MUTANT_FIDELITY_NAME,
        primary_metric_settings=(
            (
                "scorer",
                "whetstone.envs.code_comp.mutant.oracle.score_mutant_reconstruction",
            ),
            ("reference", "authenticated_mutant_record"),
        ),
    )


def build_mutant_reward_policy() -> RewardPolicy:
    """The ED1M Reward Policy: maximize fidelity to the mutant."""
    return RewardPolicy(
        policy_name=f"whetstone.env.{CODE_COMP_ENV_NAME}.reward",
        reward_name="reward",
        terms=(
            RewardTerm(
                name=CODE_COMP_MUTANT_FIDELITY_NAME,
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
        id=mutant.content_hash,
        seed=mutant.seed,
        strata=(_MUTANT_STRATUM,),
        prompt_inputs={
            "input_code": mutant.mutated_full_source,
            "task_id": mutant.task_id,
            "entry_point": mutant.entry_point,
            "operator_family": mutant.operator_family.value,
        },
        gold=mutant.canonical_full_source,
    )


def build_mutant_experiment(
    *,
    artifact_dir: Path,
    provider_call_config: ProviderCallConfig = (
        _MUTANT_CANONICAL_PROVIDER_CALL_CONFIG
    ),
    budget_ratio: float | None = None,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    num_samples: int = 3,
    exclude_mutant_ids: frozenset[str] | None = None,
    blend_config: BoundedCompressionMetricConfig | None = None,
    scorer: Callable[..., CodeSubmissionResult] | None = None,
) -> MutantExperiment:
    """Build the ed1m experiment (mutant enc-dec + dual scoring)."""
    if not isinstance(artifact_dir, Path):
        raise TypeError("artifact_dir must be pathlib.Path")
    config = default_code_comp_config(
        CodeCompMode.ENCDEC_MUTANT,
        artifact_dir=artifact_dir,
        models=CodeCompModelRoutesConfig(
            encoder=CodeCompModelRouteConfig(
                provider_call_config=provider_call_config
            )
        ),
        mutant={
            "budget_ratio": budget_ratio,
            "exclude_mutant_ids": exclude_mutant_ids or frozenset(),
            "blend_config": blend_config,
        },
        pool={"limit": limit},
        split={"internal_n": internal_n, "official_n": official_n},
        sampling={
            "completeness": completeness,
            "max_skip_fraction": max_skip_fraction,
            "num_samples": num_samples,
        },
    )
    return cast(MutantExperiment, config.build_experiment(scorer=scorer))


def score_mutant_submission(
    *,
    reconstruction: str,
    mutant: MutantRecord,
    executor: object,
) -> MutantSubmissionResult:
    """Score one reconstruction and retain per-input oracle outcomes."""

    score, outcomes = score_mutant_reconstruction_with_outcomes(
        reconstruction=reconstruction,
        mutant=mutant,
        executor=cast(Executor, executor),
    )
    return project_mutant_submission_result(
        score=score,
        mutant=mutant,
        outcomes=outcomes,
    )


def score_mutant_row(
    experiment: MutantExperiment,
    instance: Instance,
    reconstruction: str,
    scorer: Callable[..., object],
) -> MutantSubmissionResult:
    """Score one ed1m reconstruction via the instance's mutant dual oracle."""

    mutant = experiment.mutants.get(str(instance.id))
    if mutant is None:  # pragma: no cover - guarded by construction
        raise KeyError(
            f"ed1m instance {instance.id!r} has no mutant in the map"
        )
    result = scorer(reconstruction=reconstruction, mutant=mutant)
    if not isinstance(result, MutantSubmissionResult):
        raise TypeError("ED1M scorer returned an unsupported result")
    return result


__all__ = [
    "CODE_COMP_CANONICAL_MODEL",
    "CODE_COMP_ENV_NAME",
    "CODE_COMP_MUTANT_FIDELITY_NAME",
    "MutantExperiment",
    "build_mutant_experiment",
    "build_mutant_procedure_config",
    "build_mutant_reward_policy",
    "score_mutant_row",
    "score_mutant_submission",
]
