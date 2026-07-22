"""The per-env Reward Policy over the internal env_exact_match aggregate."""

from __future__ import annotations

import pytest

from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.envs.reward import (
    ENV_EXACT_MATCH_AGGREGATE_NAME,
    CandidateEvaluationFailure,
    build_reward_policy,
    reward_from_internal_aggregate,
)
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.reward import (
    MissingDataPolicy,
    OfficialRewardError,
    apply_reward_policy,
)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_policy_maps_env_exact_match_higher_better(env_name: str) -> None:
    policy = build_reward_policy(env_spec(env_name))
    assert len(policy.terms) == 1
    term = policy.terms[0]
    assert term.name == ENV_EXACT_MATCH_AGGREGATE_NAME
    assert term.maximize is True
    assert term.weight == 1.0
    # A higher internal mean produces a strictly higher Reward.
    low = reward_from_internal_aggregate(policy, env_exact_match_value=0.25)
    high = reward_from_internal_aggregate(policy, env_exact_match_value=0.75)
    assert high.value > low.value


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_reward_cites_policy_and_inputs(env_name: str) -> None:
    policy = build_reward_policy(env_spec(env_name))
    reward = reward_from_internal_aggregate(
        policy, env_exact_match_value=0.5
    )
    assert reward.reward_policy_hash == policy.identity_hash()
    assert reward.evidence_role is EvaluationRole.INTERNAL
    cited = {c.name for c in reward.input_citations}
    assert ENV_EXACT_MATCH_AGGREGATE_NAME in cited


def test_reward_refuses_official_role() -> None:
    policy = build_reward_policy(env_spec("c18"))
    with pytest.raises(OfficialRewardError):
        apply_reward_policy(
            policy,
            aggregates={ENV_EXACT_MATCH_AGGREGATE_NAME: 0.5},
            evidence_role=EvaluationRole.OFFICIAL,
        )


def test_missing_internal_aggregate_fails_the_reward() -> None:
    # A missing required internal term under FAIL surfaces as the TYPED
    # CandidateEvaluationFailure (the optimizer loop marks the candidate failed
    # per policy), not a bare ValueError crash.
    policy = build_reward_policy(env_spec("c18"))
    assert policy.missing_data is MissingDataPolicy.FAIL
    with pytest.raises(
        CandidateEvaluationFailure, match="no computable Reward"
    ):
        reward_from_internal_aggregate(policy, env_exact_match_value=None)
