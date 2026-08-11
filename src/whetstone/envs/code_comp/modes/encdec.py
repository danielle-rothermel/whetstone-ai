from __future__ import annotations

from collections.abc import Callable
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import cast

from dr_code.humaneval import HumanEvalTask
from dr_providers import ProviderCallConfig
from pydantic import BaseModel, ConfigDict
from whetstone_envs.core import Instance

from whetstone.envs.code_comp.candidates import env_candidate_base_ref
from whetstone.envs.code_comp.config import default_code_comp_config
from whetstone.envs.code_comp.constants import (
    CODE_COMP_CANONICAL_MODEL,
    CODE_COMP_DEFAULT_BUDGET_RATIO,
    CODE_COMP_ENV_NAME,
    ENCODER_BODY_A,
    ENCODER_BODY_B,
    MUTATION_FIELD,
)
from whetstone.envs.code_comp.dataset import CodeCompTaskInstance
from whetstone.envs.code_comp.experiment import EncDecExperiment
from whetstone.envs.code_comp.generation_graph.encdec import (
    build_encoder_provider_call_config,
)
from whetstone.envs.code_comp.mode import (
    CodeCompMode,
    code_comp_identity_prefix,
)
from whetstone.envs.code_comp.reward.blended import (
    CODE_COMP_DEFAULT_BLEND_CONFIG,
    BoundedCompressionMetricConfig,
)
from whetstone.envs.code_comp.runtime import EncDecScoringRuntimeSummary
from whetstone.envs.code_comp.submission_result import CodeSubmissionResult
from whetstone.envs.sampling import (
    Completeness,
    EnvSplitSampling,
    derive_split_sampling,
)
from whetstone.evaluation import EvaluationProcedureConfig
from whetstone.evaluation.aggregate import aggregation_definition
from whetstone.evaluation.preview.preflight import PreviewMetadata
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.task_selection import (
    TaskSplitRoles,
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


def build_encdec_experiment(
    *,
    provider_call_config: ProviderCallConfig = (
        _ED1_CANONICAL_PROVIDER_CALL_CONFIG
    ),
    budget_ratio: float | None = CODE_COMP_DEFAULT_BUDGET_RATIO,
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
    blend_config: BoundedCompressionMetricConfig = (
        CODE_COMP_DEFAULT_BLEND_CONFIG
    ),
    split_manifest: TaskSplitRoles | None = None,
) -> EncDecExperiment:
    """Build the ed1 enc-dec experiment the runner cell consumes."""
    if blend_config is None:
        raise TypeError("ED1 requires a bounded compression blend config")
    if not isinstance(blend_config, BoundedCompressionMetricConfig):
        raise TypeError("ED1 requires a bounded compression blend config")
    from whetstone.envs.code_comp.config import (
        CodeCompModelRouteConfig,
        CodeCompModelRoutesConfig,
    )
    from whetstone.envs.code_comp.mode import CodeCompMode

    config = default_code_comp_config(
        CodeCompMode.ENCDEC,
        models=CodeCompModelRoutesConfig(
            encoder=CodeCompModelRouteConfig(
                provider_call_config=provider_call_config
            )
        ),
        encdec={
            "budget_ratio": budget_ratio,
            "blend_config": blend_config,
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
    return cast(EncDecExperiment, config.build_experiment(scorer=scorer))


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
