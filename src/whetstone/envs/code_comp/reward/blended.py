from __future__ import annotations

from typing import Literal

from dr_store import ObjectStore

from whetstone.core.identity import TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.envs.code_comp.constants import (
    BLENDED_METRIC_ID,
    CODE_COMP_BLENDED_REWARD_NAME,
    CODE_COMP_ENV_NAME,
)
from whetstone.evaluation.metrics.blended import BoundedCompressionBlendConfig
from whetstone.evaluation.preview.persisted import load_aggregate_value
from whetstone.experiment.reward import (
    MissingDataPolicy,
    Reward,
    RewardPolicy,
    RewardRef,
    RewardTerm,
    apply_reward_policy,
)


class BoundedCompressionMetricConfig(BoundedCompressionBlendConfig):
    """ED1 identity-bearing blended-reward configuration."""

    metric_id: Literal["primary_score_with_bounded_compression_penalty"] = (
        BLENDED_METRIC_ID
    )

    def identity_key(self) -> str:
        """Fold ED1 metric identity with the shared blend parameters."""
        return f"{self.metric_id}|{self.blend_identity_key()}"


CODE_COMP_DEFAULT_BLEND_CONFIG = BoundedCompressionMetricConfig()


def code_comp_blended_aggregate_values(
    store: ObjectStore,
    reward_ref: RewardRef,
) -> tuple[float | None, float | None]:
    """Load primary and compression aggregate values from a blended reward."""
    if len(reward_ref.record.evidence_refs) != 2:
        raise RuntimeError(
            "ED1 blended Reward must cite primary and compression aggregates"
        )
    return (
        load_aggregate_value(store, reward_ref.record.evidence_refs[0]),
        load_aggregate_value(store, reward_ref.record.evidence_refs[1]),
    )


def reward_from_primary_score(
    policy: RewardPolicy,
    *,
    primary_score: float | None,
    evidence_refs: tuple[TypedRef, ...],
) -> Reward:
    """Apply a one-term environment policy to its internal primary score."""
    from whetstone.envs.reward import CandidateEvaluationFailure

    if len(policy.terms) != 1:
        raise ValueError(
            "primary-score Reward Policy must have exactly one term"
        )
    metric_name = policy.terms[0].name
    try:
        return apply_reward_policy(
            policy,
            aggregates={metric_name: primary_score},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=evidence_refs,
        )
    except ValueError as exc:
        raise CandidateEvaluationFailure(
            "internal candidate has no computable Reward: the "
            f"{metric_name!r} aggregate is missing/incomplete under the "
            f"FAIL missing-data policy (primary_score={primary_score!r})"
        ) from exc


def build_code_comp_blended_reward_policy(
    blend_config: BoundedCompressionMetricConfig,
    *,
    env_name: str = CODE_COMP_ENV_NAME,
) -> RewardPolicy:
    """An ED1-family blended Reward Policy with one blended-reward term."""
    return RewardPolicy(
        policy_name=(
            f"whetstone.env.{env_name}.blended_reward"
            f"|{blend_config.identity_key()}"
        ),
        reward_name="reward",
        terms=(
            RewardTerm(
                name=CODE_COMP_BLENDED_REWARD_NAME, weight=1.0, maximize=True
            ),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def code_comp_reward_from_blended(
    blend_config: BoundedCompressionMetricConfig,
    *,
    env_name: str,
    blended: float | None,
    evidence_refs: tuple[TypedRef, ...],
) -> Reward:
    """Apply the blended Reward Policy to the mean per-task blended reward."""
    from whetstone.envs.reward import CandidateEvaluationFailure

    policy = build_code_comp_blended_reward_policy(
        blend_config,
        env_name=env_name,
    )
    try:
        return apply_reward_policy(
            policy,
            aggregates={CODE_COMP_BLENDED_REWARD_NAME: blended},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=evidence_refs,
        )
    except ValueError as exc:
        raise CandidateEvaluationFailure(
            "code_comp internal candidate has no computable blended Reward: "
            f"the {CODE_COMP_BLENDED_REWARD_NAME!r} aggregate is missing "
            f"under FAIL (blended={blended!r})"
        ) from exc


__all__ = [
    "CODE_COMP_DEFAULT_BLEND_CONFIG",
    "BoundedCompressionMetricConfig",
    "build_code_comp_blended_reward_policy",
    "code_comp_blended_aggregate_values",
    "code_comp_reward_from_blended",
    "reward_from_primary_score",
]
