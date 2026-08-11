from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import UNIQUE, StrEnum, verify
from pathlib import Path

from dr_code.humaneval import HumanEvalTask
from dr_providers import ProviderCallConfig
from pydantic import BaseModel, ConfigDict
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.constants import (
    ED1_CANONICAL_MODEL,
    ED1_DATASET_REVISION,
    ED1_DEFAULT_BUDGET_RATIO,
    ED1_ENV_NAME,
    ENCODER_BODY_A,
    ENCODER_BODY_B,
    MUTATION_FIELD,
)
from whetstone.envs.code_comp.dataset import Ed1Instance, load_ed1_tasks
from whetstone.envs.code_comp.procedure import build_ed1_procedure_config
from whetstone.envs.code_comp.reward.blended import (
    ED1_DEFAULT_BLEND_CONFIG,
    BoundedCompressionMetricConfig,
    build_ed1_blended_reward_policy,
)
from whetstone.envs.code_comp.rollout.encdec import (
    EncDecRolloutDefinition,
    build_encdec_rollout_definition,
    build_encoder_provider_call_config,
)
from whetstone.envs.code_comp.runtime import Ed1ScoringRuntimeSummary
from whetstone.envs.code_comp.scoring import CodeScore
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.rollout_definition import env_candidate_base_ref
from whetstone.envs.sampling import (
    Completeness,
    EnvEvalConfigs,
    EnvSplitSampling,
    derive_split_sampling,
)
from whetstone.evaluation import EvaluationProcedureConfig
from whetstone.evaluation.aggregate import aggregation_definition
from whetstone.evaluation.preview.preflight import PreviewMetadata
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.task_selection import (
    TaskSplitRoles,
    resolve_manifest_split,
)
from whetstone.provider.policy import ProviderExecutionPolicy


@verify(UNIQUE)
class Ed1TaskModelKind(StrEnum):
    """Execution route for ED1 encoder and decoder generations."""

    DUMMY = "dummy"
    PROVIDER = "provider"


class Ed1TaskModelConfig(BaseModel):
    """Exact task-model mode, provider request, and execution policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Ed1TaskModelKind
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy

    @property
    def model(self) -> str:
        """The exact provider request's model slug for display."""
        return self.provider_call_config.definition.route.model


#: The canonical enc/dec task model (same route plays both encoder + decoder).
#: ``--task-model`` overrides and folds into ``graph_hash``.
_ED1_CANONICAL_PROVIDER_CALL_CONFIG = build_encoder_provider_call_config(
    ED1_CANONICAL_MODEL
)


def _ed1_split(
    *,
    env_name: str = ED1_ENV_NAME,
    dataset_revision: str,
    split_role: str,
    instances: tuple[Instance, ...],
    procedure: EvaluationProcedureConfig,
    completeness: Completeness,
    max_skip_fraction: float,
    repeats: int,
    manifest_tag: str | None = None,
) -> EnvSplitSampling:
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    aggregation = aggregation_definition(
        f"whetstone.{env_name}.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )
    namespace = f"whetstone.{env_name}"
    if manifest_tag is not None:
        namespace = f"{namespace}.{manifest_tag}"
    return derive_split_sampling(
        namespace=namespace,
        dataset_revision=dataset_revision,
        split_role=split_role,
        instances=instances,
        task_identity_of=lambda instance: str(instance.id),
        repeats=repeats,
        procedure=procedure,
        aggregation=aggregation,
    )


def _ed1_candidate(*, candidate_id: str, body: str) -> Candidate:
    # The Mutation Surface payload is the INSTRUCTION BODY only; the code,
    # budget suffix, and punctuation are composed by the immutable frame.
    return Candidate(
        candidate_id=candidate_id,
        base_ref=env_candidate_base_ref(ED1_ENV_NAME),
        payload={MUTATION_FIELD: body},
    )


def ed1_initial_candidate() -> Candidate:
    return _ed1_candidate(
        candidate_id=f"{ED1_ENV_NAME}-naive", body=ENCODER_BODY_A
    )


def ed1_ceiling_candidate() -> Candidate:
    return _ed1_candidate(
        candidate_id=f"{ED1_ENV_NAME}-ceiling", body=ENCODER_BODY_B
    )


@dataclass(frozen=True, slots=True)
class Ed1Experiment(EnvExperiment):
    """An ``EnvExperiment`` carrying the ed1-specific enc-dec rollout + tasks.

    Adds the enc-dec :class:`EncDecRolloutDefinition` (a 3-node graph, with the
    ``budget_ratio`` folded into ``graph_hash``) on top of the base experiment
    shape the runner reads. ``rollout_definition`` (the base field) is set to
    the
    same enc-dec rollout so ``experiment.rollout_definition.graph_hash`` etc.
    resolve for the runner.
    """

    encdec_rollout: EncDecRolloutDefinition | None = None
    #: The per-task Character Budget ratio, or ``None`` for the no-budget frame
    #: without a "Use at most N characters" line or MAX_BUDGET.
    #: ``None`` is the default for ed1 optimizer cells to optimize compression
    #: without listing a budget at all; the reward's
    #: compression term carries the pressure instead.
    budget_ratio: float | None = ED1_DEFAULT_BUDGET_RATIO
    dataset_revision: str = ED1_DATASET_REVISION
    #: The injectable code scorer (raw_submission, task) -> CodeScore. The
    #: scorer is INJECTED by the caller that drives rows; the production
    #: injection is :func:`whetstone.envs.ed1_scoring.score_ed1_submission`,
    #: which runs candidate code through the caller's explicit dr-exec
    #: executor.
    scorer: Callable[..., CodeScore] | None = None
    #: ED1 always uses this per-task blend for internal selection and the
    #: official comparison vector; primary score + compression are still
    #: reported separately. The optional type is required only because ED1M
    #: shares this runtime model and retains its independent reward behavior.
    blend_config: BoundedCompressionMetricConfig | None = field(
        default_factory=BoundedCompressionMetricConfig
    )

    def __post_init__(self) -> None:
        if self.env_name == ED1_ENV_NAME and self.blend_config is None:
            raise ValueError("ED1 requires a bounded compression blend config")


def build_ed1_experiment(
    *,
    provider_call_config: ProviderCallConfig = (
        _ED1_CANONICAL_PROVIDER_CALL_CONFIG
    ),
    budget_ratio: float | None = ED1_DEFAULT_BUDGET_RATIO,
    scorer: Callable[..., CodeScore] | None = None,
    snapshot_path: Path | None = None,
    limit: int | None = None,
    internal_n: int | None = None,
    official_n: int | None = None,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    repeats: int = 3,
    tasks: tuple[Ed1Instance, ...] | None = None,
    exclude_task_ids: frozenset[str] | None = None,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    split_manifest: TaskSplitRoles | None = None,
) -> Ed1Experiment:
    """Build the ed1 enc-dec experiment the runner cell consumes.

    Loads the pinned HumanEval+ pool (or uses injected ``tasks`` for tests),
    splits it into internal/official (first-N ordered), builds the 3-node
    enc-dec rollout at ``budget_ratio`` (folded into ``graph_hash``), the naive
    (A) + ceiling (B) encoder candidates, and the two Eval Configs sharing the
    code-eval Procedure identity. ED1 always advertises and applies the
    per-task bounded-compression blend; callers may configure its weight and
    bounds but cannot disable it.

    ``exclude_task_ids`` drops those task ids from the pool before the split:
    excluded tasks are removed from the train / eval / test (internal /
    official / held-out) pools. The exclusion applies to the
    ordered pool, so the filtered Task Set is deterministic; because each
    split's Task Set identity folds its task ids, a filtered pool yields a
    DISTINCT ``eval_config_hash`` per split -- the exclusion folds into the id
    by construction. The caller passes the exclusion list for the model the
    cell actually runs.

    ``split_manifest`` overrides the first-N slice with role-true
    train/val/test semantics: the internal split = the manifest's
    ``train + val`` ids (by MEMBERSHIP, in manifest order -- the internal
    machinery has no val sub-split, so val folds into internal alongside
    train); the official split = the manifest's ``test`` ids EXACTLY
    (membership, NOT a first-N slice).
    ``official_n`` then caps WITHIN the test set. Mutually exclusive with
    ``exclude_task_ids`` (the caller enforces the CLI refusal). The manifest's
    content hash + pool folds into each split's Task Set identity.
    """
    if not isinstance(blend_config, BoundedCompressionMetricConfig):
        raise TypeError("ED1 requires a bounded compression blend config")
    pool = (
        tasks
        if tasks is not None
        else load_ed1_tasks(snapshot_path=snapshot_path, limit=limit)
    )
    if exclude_task_ids:
        pool = tuple(
            t for t in pool if str(t.instance.id) not in exclude_task_ids
        )
    if not pool:
        raise ValueError("ed1 task pool is empty")
    procedure = build_ed1_procedure_config()
    rollout = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure.config_identity_hash,
        budget_ratio=budget_ratio,
    )
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
            print(f"[ed1] {resolved.official_capped}")
    else:
        all_instances = tuple(t.instance for t in pool)
        n = len(all_instances)
        # First-N ordered split: internal then official (disjoint, contiguous).
        # A small pool may put all tasks in the official split.
        i_n = internal_n if internal_n is not None else min(max(1, n // 2), n)
        internal_instances = all_instances[:i_n]
        rest = all_instances[i_n:]
        o_n = official_n if official_n is not None else len(rest)
        official_instances = (
            rest[:o_n] if rest else internal_instances[: o_n or n]
        )
        if not official_instances:
            official_instances = internal_instances
    internal_split = _ed1_split(
        dataset_revision=ED1_DATASET_REVISION,
        split_role="internal_eval",
        instances=internal_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        manifest_tag=manifest_tag,
    )
    official_split = _ed1_split(
        dataset_revision=ED1_DATASET_REVISION,
        split_role="official",
        instances=official_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        repeats=repeats,
        manifest_tag=manifest_tag,
    )
    eval_configs = EnvEvalConfigs(
        env_name=ED1_ENV_NAME,
        procedure_config_hash=procedure.config_identity_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_identities=(),
    )
    return Ed1Experiment(
        env_name=ED1_ENV_NAME,
        rollout_definition=rollout,  # type: ignore[arg-type]
        initial_candidate=ed1_initial_candidate(),
        ceiling_candidate=ed1_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=build_ed1_blended_reward_policy(
            blend_config, env_name=ED1_ENV_NAME
        ),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        encdec_rollout=rollout,
        budget_ratio=budget_ratio,
        dataset_revision=ED1_DATASET_REVISION,
        scorer=scorer,
        blend_config=blend_config,
    )


def ed1_preview_metadata(
    *,
    task_model: Ed1TaskModelConfig,
    runtime: Ed1ScoringRuntimeSummary,
    blend_config: BoundedCompressionMetricConfig,
) -> PreviewMetadata:
    """Persist ED1-specific preview fields alongside generic transcripts."""

    return PreviewMetadata.model_validate(
        {
            "task_model": task_model.model_dump(mode="json"),
            "runtime": runtime.model_dump(mode="json"),
            "blend_config": blend_config.model_dump(mode="json"),
        }
    )


def ed1_task_model_from_metadata(
    metadata: PreviewMetadata,
) -> Ed1TaskModelConfig:
    return Ed1TaskModelConfig.model_validate(
        metadata.model_dump(mode="python")["task_model"]
    )


def ed1_runtime_from_metadata(
    metadata: PreviewMetadata,
) -> Ed1ScoringRuntimeSummary:
    return Ed1ScoringRuntimeSummary.model_validate(
        metadata.model_dump(mode="python")["runtime"]
    )


def ed1_blend_config_from_metadata(
    metadata: PreviewMetadata,
) -> BoundedCompressionMetricConfig:
    return BoundedCompressionMetricConfig.model_validate(
        metadata.model_dump(mode="python")["blend_config"]
    )


#: Callable type for reconstructing a HumanEvalTask (test injection point).
HumanEvalTaskFromInstance = Callable[[Instance], HumanEvalTask]

__all__ = [
    "Ed1Experiment",
    "Ed1TaskModelConfig",
    "Ed1TaskModelKind",
    "HumanEvalTaskFromInstance",
    "_ed1_split",
    "build_ed1_experiment",
    "ed1_blend_config_from_metadata",
    "ed1_ceiling_candidate",
    "ed1_initial_candidate",
    "ed1_preview_metadata",
    "ed1_runtime_from_metadata",
    "ed1_task_model_from_metadata",
]
