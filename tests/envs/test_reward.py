from __future__ import annotations

import pytest

from whetstone.core.identity import typed_ref_for_record
from whetstone.core.roles import EvaluationRole
from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.envs.reward import (
    ENV_EXACT_MATCH_AGGREGATE_NAME,
    CandidateEvaluationFailure,
    build_reward_policy,
    reward_from_internal_aggregate,
)
from whetstone.experiment.reward import (
    MissingDataPolicy,
    OfficialRewardError,
    apply_reward_policy,
)

_EVIDENCE_REFS = (
    typed_ref_for_record("whetstone.test.aggregate", {"value": 1}),
)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_policy_maps_env_exact_match_higher_better(env_name: str) -> None:
    policy = build_reward_policy(env_spec(env_name))
    assert len(policy.terms) == 1
    term = policy.terms[0]
    assert term.name == ENV_EXACT_MATCH_AGGREGATE_NAME
    assert term.maximize is True
    assert term.weight == 1.0
    low = reward_from_internal_aggregate(
        policy, env_exact_match_value=0.25, evidence_refs=_EVIDENCE_REFS
    )
    high = reward_from_internal_aggregate(
        policy, env_exact_match_value=0.75, evidence_refs=_EVIDENCE_REFS
    )
    assert high.value > low.value


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_reward_cites_policy_and_inputs(env_name: str) -> None:
    policy = build_reward_policy(env_spec(env_name))
    reward = reward_from_internal_aggregate(
        policy, env_exact_match_value=0.5, evidence_refs=_EVIDENCE_REFS
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
            evidence_refs=_EVIDENCE_REFS,
        )


def test_missing_internal_aggregate_fails_the_reward() -> None:
    policy = build_reward_policy(env_spec("c18"))
    assert policy.missing_data is MissingDataPolicy.FAIL
    with pytest.raises(
        CandidateEvaluationFailure, match="no computable Reward"
    ):
        reward_from_internal_aggregate(
            policy,
            env_exact_match_value=None,
            evidence_refs=_EVIDENCE_REFS,
        )
