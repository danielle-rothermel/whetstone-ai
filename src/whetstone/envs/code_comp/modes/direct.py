from __future__ import annotations

import keyword
from collections.abc import Callable
from pathlib import Path
from typing import cast

from whetstone_envs.core import Instance

from whetstone.envs.code_comp.candidates import env_candidate_base_ref
from whetstone.envs.code_comp.config import default_code_comp_config
from whetstone.envs.code_comp.constants import (
    CODE_COMP_DATASET_REVISION,
    CODE_COMP_ENV_NAME,
    CODE_COMP_SUBMISSION_SCORE_NAME,
    MUTATION_FIELD,
)
from whetstone.envs.code_comp.dataset import CodeCompTaskInstance
from whetstone.envs.code_comp.experiment import DirectExperiment
from whetstone.envs.code_comp.generation_graph.direct import (
    DIRECT_DEFAULT_RENAME_TOKEN,
    DIRECT_INPUT_ARMS,
    DIRECT_RENAMED_ARM,
    DirectGenerationGraph,
    build_direct_generation_graph,
    direct_graph_definition,
    render_direct_frame,
)
from whetstone.envs.code_comp.mode import (
    CodeCompMode,
    code_comp_identity_prefix,
)
from whetstone.envs.code_comp.procedure import build_encdec_procedure_config
from whetstone.envs.code_comp.submission_result import CodeSubmissionResult
from whetstone.envs.sampling import (
    Completeness,
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
from whetstone.experiment.task_selection import TaskSplitRoles

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


def build_direct_experiment(
    *,
    model: str = CODE_COMP_CANONICAL_MODEL,
    input_arm: str = "original",
    rename_token: str = DIRECT_DEFAULT_RENAME_TOKEN,
    scorer: Callable[..., CodeSubmissionResult] | None = None,
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
    config = default_code_comp_config(
        CodeCompMode.DIRECT,
        direct={
            "model": model,
            "input_arm": input_arm,
            "rename_token": rename_token,
        },
        pool={
            "tasks": tasks,
            "snapshot_path": snapshot_path,
            "limit": limit,
        },
        split={
            "internal_n": internal_n,
            "official_n": official_n,
            "split_manifest": split_manifest,
            "exclude_task_ids": exclude_task_ids or frozenset(),
        },
        sampling={
            "completeness": completeness,
            "max_skip_fraction": max_skip_fraction,
            "num_samples": num_samples,
        },
    )
    return cast(DirectExperiment, config.build_experiment(scorer=scorer))


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
    "DirectGenerationGraph",
    "build_direct_experiment",
    "build_direct_generation_graph",
    "build_direct_procedure_config",
    "build_direct_reward_policy",
    "direct_ceiling_candidate",
    "direct_graph_definition",
    "direct_initial_candidate",
    "render_direct_frame",
]
