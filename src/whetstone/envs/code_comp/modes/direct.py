from __future__ import annotations

import keyword
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_code.humaneval import HumanEvalTask
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.constants import (
    CODE_COMP_DATASET_REVISION,
    CODE_COMP_ENV_NAME,
    CODE_COMP_SUBMISSION_SCORE_NAME,
    MUTATION_FIELD,
)
from whetstone.envs.code_comp.dataset import CodeCompTaskInstance, load_tasks
from whetstone.envs.code_comp.procedure import build_encdec_procedure_config
from whetstone.envs.code_comp.registry import (
    CodeCompMode,
    code_comp_identity_prefix,
)
from whetstone.envs.code_comp.rollout.direct import (
    DIRECT_DEFAULT_RENAME_TOKEN,
    DIRECT_INPUT_ARMS,
    DIRECT_RENAMED_ARM,
    DirectRolloutDefinition,
    build_direct_rollout_definition,
    direct_graph_definition,
    render_direct_frame,
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

CODE_COMP_CANONICAL_MODEL = "deepseek/deepseek-v4-flash"

DIRECT_WRAPPER_BODY_NAIVE = (
    "Write a complete, correct Python implementation for the following. "
    "Output only Python code."
)

DIRECT_WRAPPER_BODY_CEILING = (
    "You are an expert Python engineer. Implement the following completely "
    "and correctly, handling all edge cases. Output only the Python function."
)


def _direct_candidate(*, candidate_id: str, body: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        base_ref=env_candidate_base_ref(CODE_COMP_ENV_NAME),
        payload={MUTATION_FIELD: body},
    )


def direct_initial_candidate() -> Candidate:
    prefix = code_comp_identity_prefix(CodeCompMode.DIRECT)
    return _direct_candidate(
        candidate_id=f"{prefix}-naive", body=DIRECT_WRAPPER_BODY_NAIVE
    )


def direct_ceiling_candidate() -> Candidate:
    """The ceiling reference: the explicit-instruction wrapper body."""
    prefix = code_comp_identity_prefix(CodeCompMode.DIRECT)
    return _direct_candidate(
        candidate_id=f"{prefix}-ceiling", body=DIRECT_WRAPPER_BODY_CEILING
    )


def build_direct_reward_policy() -> RewardPolicy:
    """The D1 Reward Policy: maximize HumanEval Submission Score only."""
    return RewardPolicy(
        policy_name=f"whetstone.env.{CODE_COMP_ENV_NAME}.reward",
        reward_name="reward",
        terms=(
            RewardTerm(
                name=CODE_COMP_SUBMISSION_SCORE_NAME,
                weight=1.0,
                maximize=True,
            ),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def _direct_split(
    *,
    split_role: str,
    tasks: tuple[Instance, ...],
    procedure,
    completeness: Completeness,
    max_skip_fraction: float,
    num_samples: int,
    input_arm: str,
    rename_token: str = DIRECT_DEFAULT_RENAME_TOKEN,
    manifest_tag: str | None = None,
) -> EnvSplitSampling:
    """A d1 split whose Task Set + sampling fold in the FROZEN input arm."""
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    aggregation = aggregation_definition(
        "whetstone.code_comp.direct.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )
    namespace = f"{code_comp_identity_prefix(CodeCompMode.DIRECT)}.{input_arm}"
    if input_arm == DIRECT_RENAMED_ARM:
        namespace = f"{namespace}.{rename_token}"
    if manifest_tag is not None:
        namespace = f"{namespace}.{manifest_tag}"
    return derive_split_sampling(
        namespace=namespace,
        dataset_revision=CODE_COMP_DATASET_REVISION,
        split_role=split_role,
        tasks=tasks,
        task_hash_of=lambda instance: str(instance.id),
        num_samples=num_samples,
        procedure=procedure,
        aggregation=aggregation,
    )


@dataclass(frozen=True, slots=True)
class DirectExperiment(EnvExperiment):
    """An ``EnvExperiment`` for the d1 direct-generation env."""

    input_arm: str = "original"
    rename_token: str = DIRECT_DEFAULT_RENAME_TOKEN
    dataset_revision: str = ""
    scorer: Callable[..., CodeScore] | None = None
    humaneval_by_id: dict[str, HumanEvalTask] = field(default_factory=dict)

    def humaneval_for(self, instance: Instance) -> HumanEvalTask:
        """The parsed HumanEval task for one d1 Instance."""
        return self.humaneval_by_id[str(instance.id)]


def build_direct_experiment(
    *,
    model: str = CODE_COMP_CANONICAL_MODEL,
    input_arm: str = "original",
    rename_token: str = DIRECT_DEFAULT_RENAME_TOKEN,
    scorer: Callable[..., CodeScore] | None = None,
    snapshot_path: Path | None = None,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    num_samples: int = 3,
    tasks: tuple[CodeCompTaskInstance, ...] | None = None,
    exclude_task_ids: frozenset[str] | None = None,
    split_manifest: TaskSplitRoles | None = None,
) -> DirectExperiment:
    """Build the d1 direct-generation experiment the runner cell consumes."""
    if input_arm not in DIRECT_INPUT_ARMS:
        raise ValueError(
            f"unknown d1 input arm {input_arm!r} "
            f"(choose one of {DIRECT_INPUT_ARMS})"
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
    procedure = build_direct_procedure_config()
    rollout = build_direct_rollout_definition(
        model=model,
        procedure_config_hash=procedure.config_hash,
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
        official_tasks = tuple(t.instance for t in resolved.official)
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
        official_tasks = rest[:o_n] if rest else internal_instances[: o_n or n]
        if not official_tasks:
            official_tasks = internal_instances
    internal_split = _direct_split(
        split_role="internal_eval",
        tasks=internal_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        num_samples=num_samples,
        input_arm=input_arm,
        rename_token=rename_token,
        manifest_tag=manifest_tag,
    )
    official_split = _direct_split(
        split_role="official",
        tasks=official_tasks,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        num_samples=num_samples,
        input_arm=input_arm,
        rename_token=rename_token,
        manifest_tag=manifest_tag,
    )
    eval_configs = EnvEvalConfigs(
        env_name=CODE_COMP_ENV_NAME,
        procedure_config_hash=procedure.config_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_hashes=(),
    )
    return DirectExperiment(
        env_name=CODE_COMP_ENV_NAME,
        rollout_definition=rollout,  # type: ignore[arg-type]
        initial_candidate=direct_initial_candidate(),
        ceiling_candidate=direct_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=build_direct_reward_policy(),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        input_arm=input_arm,
        rename_token=rename_token,
        dataset_revision=CODE_COMP_DATASET_REVISION,
        scorer=scorer,
        humaneval_by_id=humaneval_by_id,
    )


def build_direct_procedure_config():
    """The d1 direct code-eval Evaluation Procedure Config."""
    return build_encdec_procedure_config()


__all__ = [
    "CODE_COMP_CANONICAL_MODEL",
    "CODE_COMP_ENV_NAME",
    "CODE_COMP_SUBMISSION_SCORE_NAME",
    "DIRECT_INPUT_ARMS",
    "DIRECT_RENAMED_ARM",
    "DIRECT_WRAPPER_BODY_CEILING",
    "DIRECT_WRAPPER_BODY_NAIVE",
    "DirectExperiment",
    "DirectRolloutDefinition",
    "build_direct_experiment",
    "build_direct_procedure_config",
    "build_direct_reward_policy",
    "build_direct_rollout_definition",
    "direct_ceiling_candidate",
    "direct_graph_definition",
    "direct_initial_candidate",
    "render_direct_frame",
]
