from __future__ import annotations

import keyword
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_code.humaneval import HumanEvalTask
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.constants import (
    ED1_DATASET_REVISION,
    ED1_SUBMISSION_SCORE_NAME,
    MUTATION_FIELD,
)
from whetstone.envs.code_comp.dataset import CodeCompTaskInstance, load_tasks
from whetstone.envs.code_comp.procedure import build_ed1_procedure_config
from whetstone.envs.code_comp.rollout.direct import (
    D1_DEFAULT_RENAME_TOKEN,
    D1_ENV_NAME,
    D1_INPUT_ARMS,
    D1_RENAMED_ARM,
    D1RolloutDefinition,
    build_d1_rollout_definition,
    d1_graph_definition,
    render_d1_frame,
)
from whetstone.envs.code_comp.scoring import CodeScore
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.rollout_definition import env_candidate_base_ref
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    EnvSplitSampling,
    derive_split_sampling,
)
from whetstone.evaluation.aggregate import aggregation_definition
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.reward import (
    MissingDataPolicy,
    RewardPolicy,
    RewardTerm,
)
from whetstone.experiment.task_selection import (
    TaskSplitRoles,
    resolve_manifest_split,
)

D1_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

D1_SUBMISSION_SCORE_NAME = ED1_SUBMISSION_SCORE_NAME

D1_WRAPPER_BODY_NAIVE = (
    "Write a complete, correct Python implementation for the following. "
    "Output only Python code."
)

D1_WRAPPER_BODY_CEILING = (
    "You are an expert Python engineer. Implement the following completely "
    "and correctly, handling all edge cases. Output only the Python function."
)


def _d1_candidate(*, candidate_id: str, body: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        base_ref=env_candidate_base_ref(D1_ENV_NAME),
        payload={MUTATION_FIELD: body},
    )


def d1_initial_candidate() -> Candidate:
    return _d1_candidate(
        candidate_id=f"{D1_ENV_NAME}-naive", body=D1_WRAPPER_BODY_NAIVE
    )


def d1_ceiling_candidate() -> Candidate:
    """The ceiling reference: the explicit-instruction wrapper body."""
    return _d1_candidate(
        candidate_id=f"{D1_ENV_NAME}-ceiling", body=D1_WRAPPER_BODY_CEILING
    )


def build_d1_reward_policy() -> RewardPolicy:
    """The D1 Reward Policy: maximize HumanEval Submission Score only."""
    return RewardPolicy(
        policy_name=f"whetstone.env.{D1_ENV_NAME}.reward",
        reward_name="reward",
        terms=(
            RewardTerm(
                name=D1_SUBMISSION_SCORE_NAME,
                weight=1.0,
                maximize=True,
            ),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def _d1_split(
    *,
    split_role: str,
    instances: tuple[Instance, ...],
    procedure,
    completeness: Completeness,
    max_skip_fraction: float,
    repeats: int,
    input_arm: str,
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
    manifest_tag: str | None = None,
) -> EnvSplitSampling:
    """A d1 split whose Task Set + sampling fold in the FROZEN input arm."""
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    aggregation = aggregation_definition(
        "whetstone.d1.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )
    namespace = f"whetstone.d1.{input_arm}"
    if input_arm == D1_RENAMED_ARM:
        namespace = f"{namespace}.{rename_token}"
    if manifest_tag is not None:
        namespace = f"{namespace}.{manifest_tag}"
    return derive_split_sampling(
        namespace=namespace,
        dataset_revision=ED1_DATASET_REVISION,
        split_role=split_role,
        instances=instances,
        task_identity_of=lambda instance: str(instance.id),
        repeats=repeats,
        procedure=procedure,
        aggregation=aggregation,
    )


@dataclass(frozen=True, slots=True)
class DirectExperiment(EnvExperiment):
    """An ``EnvExperiment`` for the d1 direct-generation env."""

    input_arm: str = "original"
    rename_token: str = D1_DEFAULT_RENAME_TOKEN
    dataset_revision: str = ""
    scorer: Callable[..., CodeScore] | None = None
    humaneval_by_id: dict[str, HumanEvalTask] = field(default_factory=dict)

    def humaneval_for(self, instance: Instance) -> HumanEvalTask:
        """The parsed HumanEval task for one d1 Instance."""
        return self.humaneval_by_id[str(instance.id)]


def build_direct_experiment(
    *,
    model: str = D1_CANONICAL_MODEL,
    input_arm: str = "original",
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
    scorer: Callable[..., CodeScore] | None = None,
    snapshot_path: Path | None = None,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    repeats: int = 3,
    tasks: tuple[CodeCompTaskInstance, ...] | None = None,
    exclude_task_ids: frozenset[str] | None = None,
    split_manifest: TaskSplitRoles | None = None,
) -> DirectExperiment:
    """Build the d1 direct-generation experiment the runner cell consumes."""
    if input_arm not in D1_INPUT_ARMS:
        raise ValueError(
            f"unknown d1 input arm {input_arm!r} "
            f"(choose one of {D1_INPUT_ARMS})"
        )
    if not rename_token.isidentifier() or keyword.iskeyword(rename_token):
        raise ValueError(
            f"d1 rename_token {rename_token!r} is not a valid, "
            "non-keyword Python identifier"
        )
    pool = (
        tasks
        if tasks is not None
        else load_tasks(snapshot_path=snapshot_path, limit=limit)
    )
    if exclude_task_ids:
        pool = tuple(
            t for t in pool if str(t.instance.id) not in exclude_task_ids
        )
    if not pool:
        raise ValueError("d1 task pool is empty")
    procedure = build_d1_procedure_config()
    rollout = build_d1_rollout_definition(
        model=model,
        procedure_config_hash=procedure.config_identity_hash,
        input_arm=input_arm,
        rename_token=rename_token,
    )
    humaneval_by_id = {str(t.instance.id): t.humaneval_task for t in pool}
    manifest_tag: str | None = None
    if split_manifest is not None:
        resolved = resolve_manifest_split(
            roles=split_manifest,
            items=pool,
            id_of=lambda t: str(t.instance.id),
            official_n=official_n,
        )
        internal_instances = tuple(t.instance for t in resolved.internal)
        official_instances = tuple(t.instance for t in resolved.official)
        manifest_tag = resolved.manifest_tag
        if resolved.official_capped:
            print(f"[d1] {resolved.official_capped}")
    else:
        all_instances = tuple(t.instance for t in pool)
        n = len(all_instances)
        i_n = internal_n if internal_n is not None else min(max(1, n // 2), n)
        internal_instances = all_instances[:i_n]
        rest = all_instances[i_n:]
        o_n = official_n if official_n is not None else len(rest)
        official_instances = (
            rest[:o_n] if rest else internal_instances[: o_n or n]
        )
        if not official_instances:
            official_instances = internal_instances
    internal_split = _d1_split(
        split_role="internal_eval",
        instances=internal_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        input_arm=input_arm,
        rename_token=rename_token,
        manifest_tag=manifest_tag,
    )
    official_split = _d1_split(
        split_role="official",
        instances=official_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        input_arm=input_arm,
        rename_token=rename_token,
        manifest_tag=manifest_tag,
    )
    eval_configs = EnvEvalConfigs(
        env_name=D1_ENV_NAME,
        procedure_config_hash=procedure.config_identity_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_identities=(),
    )
    return DirectExperiment(
        env_name=D1_ENV_NAME,
        rollout_definition=rollout,  # type: ignore[arg-type]
        initial_candidate=d1_initial_candidate(),
        ceiling_candidate=d1_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=build_d1_reward_policy(),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        input_arm=input_arm,
        rename_token=rename_token,
        dataset_revision=ED1_DATASET_REVISION,
        scorer=scorer,
        humaneval_by_id=humaneval_by_id,
    )


def build_d1_procedure_config():
    """The d1 direct code-eval Evaluation Procedure Config."""
    return build_ed1_procedure_config()


__all__ = [
    "D1_CANONICAL_MODEL",
    "D1_ENV_NAME",
    "D1_INPUT_ARMS",
    "D1_RENAMED_ARM",
    "D1_SUBMISSION_SCORE_NAME",
    "D1_WRAPPER_BODY_CEILING",
    "D1_WRAPPER_BODY_NAIVE",
    "D1RolloutDefinition",
    "DirectExperiment",
    "build_d1_procedure_config",
    "build_d1_reward_policy",
    "build_d1_rollout_definition",
    "build_direct_experiment",
    "d1_ceiling_candidate",
    "d1_graph_definition",
    "d1_initial_candidate",
    "render_d1_frame",
]
