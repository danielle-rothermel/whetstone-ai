"""The per-env Reward Policy over the internal ``env_exact_match`` aggregate.

The optimizer sees only internal-split evaluation. This module builds the
reusable :class:`~whetstone.optimization.reward.RewardPolicy` that maps the
env's internal ``env_exact_match`` Rollout Aggregate to a single scalar
Reward, higher-is-better (``maximize=True``, unit weight). Applying it is
delegated to :func:`~whetstone.optimization.reward.apply_reward_policy`, the
sole Reward constructor, which refuses official-role evidence -- so the
Reward can only ever be computed from internal evidence, and every produced
Reward cites its policy Identity Hash and the exact ``env_exact_match`` input
value it scalarized.

The aggregate name the policy term selects is exactly the aggregate name the
internal Rollout Aggregate carries (:data:`ENV_EXACT_MATCH_AGGREGATE_NAME`),
so the mapping is unambiguous.
"""

from __future__ import annotations

from whetstone.envs.oracle_operator import ENV_EXACT_MATCH_NAME
from whetstone.envs.registry import EnvSpec
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.reward import (
    MissingDataPolicy,
    Reward,
    RewardPolicy,
    RewardTerm,
    apply_reward_policy,
)

#: The name the internal Rollout Aggregate carries and the Reward term
#: selects: the mean ``env_exact_match`` over the internal split.
ENV_EXACT_MATCH_AGGREGATE_NAME = ENV_EXACT_MATCH_NAME


class CandidateEvaluationFailure(RuntimeError):
    """An internal-path candidate could not be scored into a Reward.

    Raised when the internal ``env_exact_match`` aggregate is
    missing/incomplete (e.g. every internal observation timed out) and the
    Reward Policy's ``missing_data`` rule is ``FAIL``: the candidate has no
    computable internal Reward. This is a TYPED, optimizer-handleable failure
    (the candidate is marked failed per the optimizer loop's policy), never a
    bare ``ValueError`` surfacing as an unhandled crash. Official evaluations
    never reach this path: they compute aggregates + per-task vectors only and
    derive NO Reward.
    """


def build_reward_policy(env: EnvSpec) -> RewardPolicy:
    """The env Reward Policy: maximize the internal ``env_exact_match`` mean.

    One unit-weight, direction-``maximize`` term over the
    ``env_exact_match`` internal aggregate. ``missing_data=FAIL``: a missing
    internal aggregate makes the Reward not computable rather than silently
    substituting a value (the optimizer must see real internal evidence).
    """
    return RewardPolicy(
        policy_name=f"whetstone.env.{env.name}.reward",
        reward_name="reward",
        terms=(
            RewardTerm(
                name=ENV_EXACT_MATCH_AGGREGATE_NAME,
                weight=1.0,
                maximize=True,
            ),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def reward_from_internal_aggregate(
    policy: RewardPolicy,
    *,
    env_exact_match_value: float | None,
    evidence_ref_content_hash: str | None = None,
) -> Reward:
    """Apply ``policy`` to one internal ``env_exact_match`` aggregate value.

    A thin, correctly-typed wrapper over
    :func:`~whetstone.optimization.reward.apply_reward_policy`: it names the
    single internal aggregate the policy consumes and pins the evidence role
    to ``internal`` so the refusal-of-official invariant holds by
    construction.

    A missing required term under ``missing_data=FAIL`` surfaces as a typed
    :class:`CandidateEvaluationFailure` (the internal-path candidate is not
    scorable), never a bare ``ValueError``: the optimizer loop marks the
    candidate failed per policy and continues.
    """
    try:
        return apply_reward_policy(
            policy,
            aggregates={
                ENV_EXACT_MATCH_AGGREGATE_NAME: env_exact_match_value,
            },
            evidence_role=EvaluationRole.INTERNAL,
            evidence_ref_content_hash=evidence_ref_content_hash,
        )
    except ValueError as exc:
        raise CandidateEvaluationFailure(
            "internal candidate has no computable Reward: the "
            f"{ENV_EXACT_MATCH_AGGREGATE_NAME!r} aggregate is missing/"
            "incomplete under the FAIL missing-data policy "
            f"(env_exact_match_value={env_exact_match_value!r})"
        ) from exc


__all__ = [
    "ENV_EXACT_MATCH_AGGREGATE_NAME",
    "CandidateEvaluationFailure",
    "build_reward_policy",
    "reward_from_internal_aggregate",
]
