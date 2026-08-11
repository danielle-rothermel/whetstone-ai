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
    CODE_COMP_CANONICAL_MODEL,
    CODE_COMP_DATASET_REVISION,
    CODE_COMP_DEFAULT_BUDGET_RATIO,
    CODE_COMP_ENV_NAME,
    ENCODER_BODY_A,
    ENCODER_BODY_B,
    MUTATION_FIELD,
)
from whetstone.envs.code_comp.dataset import CodeCompTaskInstance, load_tasks
from whetstone.envs.code_comp.generation_graph.encdec import (
    EncDecGenerationGraph,
    build_encdec_generation_graph,
    build_encoder_provider_call_config,
)
from whetstone.envs.code_comp.procedure import build_encdec_procedure_config
from whetstone.envs.code_comp.registry import (
    CodeCompMode,
    code_comp_identity_prefix,
)
from whetstone.envs.code_comp.reward.blended import (
    CODE_COMP_DEFAULT_BLEND_CONFIG,
    BoundedCompressionMetricConfig,
    build_code_comp_blended_reward_policy,
)
from whetstone.envs.code_comp.runtime import EncDecScoringRuntimeSummary
from whetstone.envs.code_comp.scoring import CodeScore
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.generation_graph import env_candidate_base_ref
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
class EncDecTaskModelKind(StrEnum):
    """Execution route for encdec encoder and decoder generations."""

    DUMMY = "dummy"
    PROVIDER = "provider"


class EncDecTaskModelConfig(BaseModel):
    """Exact task-model mode, provider request, and execution policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EncDecTaskModelKind
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy

    @property
    def model(self) -> str:
        """The exact provider request's model slug for display."""
        return self.provider_call_config.definition.route.model


#: The canonical enc/dec task model (same route plays both encoder + decoder).
#: ``--task-model`` overrides and folds into ``graph_hash``.
_ED1_CANONICAL_PROVIDER_CALL_CONFIG = build_encoder_provider_call_config(
    CODE_COMP_CANONICAL_MODEL
)


def _code_comp_split(
    *,
    env_name: str = CODE_COMP_ENV_NAME,
    dataset_revision: str,
    split_role: str,
    tasks: tuple[Instance, ...],
    procedure: EvaluationProcedureConfig,
    completeness: Completeness,
    max_skip_fraction: float,
    num_samples: int,
    manifest_tag: str | None = None,
) -> EnvSplitSampling:
    policy = completeness.to_policy(max_skip_fraction=max_skip_fraction)
    aggregation = aggregation_definition(
        f"{code_comp_identity_prefix(CodeCompMode.ENCDEC)}.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )
    namespace = code_comp_identity_prefix(CodeCompMode.ENCDEC)
    if manifest_tag is not None:
        namespace = f"{namespace}.{manifest_tag}"
    return derive_split_sampling(
        namespace=namespace,
        dataset_revision=dataset_revision,
        split_role=split_role,
        tasks=tasks,
        task_hash_of=lambda instance: str(instance.id),
        num_samples=num_samples,
        procedure=procedure,
        aggregation=aggregation,
    )


def _encdec_candidate(*, candidate_id: str, body: str) -> Candidate:
    # The Mutation Surface payload is the INSTRUCTION BODY only; the code,
    # budget suffix, and punctuation are composed by the immutable frame.
    return Candidate(
        candidate_id=candidate_id,
        base_ref=env_candidate_base_ref(CODE_COMP_ENV_NAME),
        payload={MUTATION_FIELD: body},
    )


def encdec_initial_candidate() -> Candidate:
    prefix = code_comp_identity_prefix(CodeCompMode.ENCDEC)
    return _encdec_candidate(
        candidate_id=f"{prefix}-naive", body=ENCODER_BODY_A
    )


def encdec_ceiling_candidate() -> Candidate:
    prefix = code_comp_identity_prefix(CodeCompMode.ENCDEC)
    return _encdec_candidate(
        candidate_id=f"{prefix}-ceiling", body=ENCODER_BODY_B
    )


@dataclass(frozen=True, slots=True)
class EncDecExperiment(EnvExperiment):
    """An ``EnvExperiment`` with the ed1 enc-dec generation graph and tasks.

    Adds the enc-dec :class:`EncDecGenerationGraph` (a 3-node graph, with the
    ``budget_ratio`` folded into ``graph_hash``) on top of the base experiment
    shape the runner reads. ``generation_graph`` (the base field) is set to
    the
    same enc-dec generation graph so ``experiment.generation_graph.graph_hash``
    etc.
    resolve for the runner.
    """

    encdec_generation_graph: EncDecGenerationGraph | None = None
    #: The per-task Character Budget ratio, or ``None`` for the no-budget frame
    #: without a "Use at most N characters" line or MAX_BUDGET.
    #: ``None`` is the default for ed1 optimizer cells to optimize compression
    #: without listing a budget at all; the reward's
    #: compression term carries the pressure instead.
    budget_ratio: float | None = CODE_COMP_DEFAULT_BUDGET_RATIO
    dataset_revision: str = CODE_COMP_DATASET_REVISION
    #: The injectable code scorer (raw_submission, task) -> CodeScore. The
    #: scorer is INJECTED by the caller that drives rows; the production
    #: injection is
    #: :func:`whetstone.envs.code_comp.scoring.score_code_comp_submission`,
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
        from whetstone.envs.code_comp.modes.mutant import MutantExperiment

        if isinstance(self, MutantExperiment):
            return
        if self.env_name == CODE_COMP_ENV_NAME and self.blend_config is None:
            raise ValueError(
                "encdec requires a bounded compression blend config"
            )


def build_encdec_experiment(
    *,
    provider_call_config: ProviderCallConfig = (
        _ED1_CANONICAL_PROVIDER_CALL_CONFIG
    ),
    budget_ratio: float | None = CODE_COMP_DEFAULT_BUDGET_RATIO,
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
    blend_config: BoundedCompressionMetricConfig = (
        CODE_COMP_DEFAULT_BLEND_CONFIG
    ),
    split_manifest: TaskSplitRoles | None = None,
) -> EncDecExperiment:
    """Build the ed1 enc-dec experiment the runner cell consumes.

    Loads the pinned HumanEval+ pool (or uses injected ``tasks`` for tests),
    splits it into internal/official (first-N ordered), builds the 3-node
    enc-dec generation graph at ``budget_ratio`` (folded into ``graph_hash``),
    the naive
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
        else load_tasks(snapshot_path=snapshot_path, limit=limit)
    )
    if exclude_task_ids:
        pool = tuple(
            t for t in pool if str(t.instance.id) not in exclude_task_ids
        )
    if not pool:
        raise ValueError("ed1 task pool is empty")
    procedure = build_encdec_procedure_config()
    generation_graph = build_encdec_generation_graph(
        CODE_COMP_ENV_NAME,
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure.config_hash,
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
        official_tasks = tuple(t.instance for t in resolved.official)
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
        official_tasks = rest[:o_n] if rest else internal_instances[: o_n or n]
        if not official_tasks:
            official_tasks = internal_instances
    internal_split = _code_comp_split(
        dataset_revision=CODE_COMP_DATASET_REVISION,
        split_role="internal_eval",
        tasks=internal_instances,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        num_samples=num_samples,
        manifest_tag=manifest_tag,
    )
    official_split = _code_comp_split(
        dataset_revision=CODE_COMP_DATASET_REVISION,
        split_role="official",
        tasks=official_tasks,
        procedure=procedure,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        num_samples=num_samples,
        manifest_tag=manifest_tag,
    )
    eval_configs = EnvEvalConfigs(
        env_name=CODE_COMP_ENV_NAME,
        procedure_config_hash=procedure.config_hash,
        internal=internal_split,
        official=official_split,
        held_out_task_hashes=(),
    )
    return EncDecExperiment(
        env_name=CODE_COMP_ENV_NAME,
        generation_graph=generation_graph,  # type: ignore[arg-type]
        initial_candidate=encdec_initial_candidate(),
        ceiling_candidate=encdec_ceiling_candidate(),
        eval_configs=eval_configs,
        reward_policy=build_code_comp_blended_reward_policy(
            blend_config, env_name=CODE_COMP_ENV_NAME
        ),
        completeness_policy=completeness.to_policy(
            max_skip_fraction=max_skip_fraction
        ),
        encdec_generation_graph=generation_graph,
        budget_ratio=budget_ratio,
        dataset_revision=CODE_COMP_DATASET_REVISION,
        scorer=scorer,
        blend_config=blend_config,
    )


def encdec_preview_metadata(
    *,
    task_model: EncDecTaskModelConfig,
    runtime: EncDecScoringRuntimeSummary,
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


def encdec_task_model_from_metadata(
    metadata: PreviewMetadata,
) -> EncDecTaskModelConfig:
    return EncDecTaskModelConfig.model_validate(
        metadata.model_dump(mode="python")["task_model"]
    )


def encdec_runtime_from_metadata(
    metadata: PreviewMetadata,
) -> EncDecScoringRuntimeSummary:
    return EncDecScoringRuntimeSummary.model_validate(
        metadata.model_dump(mode="python")["runtime"]
    )


def encdec_blend_config_from_metadata(
    metadata: PreviewMetadata,
) -> BoundedCompressionMetricConfig:
    return BoundedCompressionMetricConfig.model_validate(
        metadata.model_dump(mode="python")["blend_config"]
    )


#: Callable type for reconstructing a HumanEvalTask (test injection point).
HumanEvalTaskFromInstance = Callable[[Instance], HumanEvalTask]

__all__ = [
    "EncDecExperiment",
    "EncDecTaskModelConfig",
    "EncDecTaskModelKind",
    "HumanEvalTaskFromInstance",
    "_code_comp_split",
    "build_encdec_experiment",
    "encdec_blend_config_from_metadata",
    "encdec_ceiling_candidate",
    "encdec_initial_candidate",
    "encdec_preview_metadata",
    "encdec_runtime_from_metadata",
    "encdec_task_model_from_metadata",
]
