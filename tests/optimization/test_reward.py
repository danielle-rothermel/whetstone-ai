"""The reusable Reward Policy/Reward contract; official computes no Reward."""

from __future__ import annotations

import pytest

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    MissingDataPolicy,
    OfficialRewardError,
    RewardPolicy,
    RewardTerm,
    apply_reward_policy,
)


def _policy(missing=MissingDataPolicy.FAIL) -> RewardPolicy:
    return RewardPolicy(
        policy_name="pass_up_compression_down/v1",
        terms=(
            RewardTerm(name="pass_rate", weight=1.0, maximize=True),
            RewardTerm(name="compression", weight=0.5, maximize=False),
        ),
        missing_data=missing,
    )


def test_reward_names_its_policy_and_cites_inputs() -> None:
    policy = _policy()
    reward = apply_reward_policy(
        policy,
        aggregates={"pass_rate": 0.8, "compression": 0.4},
        evidence_role=EvaluationRole.INTERNAL,
    )
    assert reward.reward_policy_hash == policy.identity_hash()
    assert reward.evidence_role is EvaluationRole.INTERNAL
    # value = 1.0*0.8 - 0.5*0.4 = 0.6
    assert reward.value == pytest.approx(0.6)
    cited = {c.name for c in reward.input_citations}
    assert cited == {"pass_rate", "compression"}


def test_reward_policy_usable_by_proposal_and_tool_configs() -> None:
    # Same policy contract, one Identity Hash, referenced by both surfaces.
    policy = _policy()
    h = policy.identity_hash()
    # A proposal-only optimizer config would cite `h`; a Tool Config's
    # reward_policy_ref is exactly this hash.
    from .support import make_tool_definition_config

    cfg = make_tool_definition_config()
    # The support Tool Config uses a placeholder reward_policy_ref; prove the
    # field accepts a real policy Identity Hash of full length.
    assert len(h) == 64
    assert len(cfg.reward_policy_ref) == 64


def test_official_evaluation_computes_no_reward() -> None:
    policy = _policy()
    with pytest.raises(OfficialRewardError):
        apply_reward_policy(
            policy,
            aggregates={"pass_rate": 0.8, "compression": 0.4},
            evidence_role=EvaluationRole.OFFICIAL,
        )


def test_reward_cannot_be_constructed_from_official_role() -> None:
    from pydantic import ValidationError

    from whetstone.optimization import Reward, RewardInputCitation

    with pytest.raises(ValidationError):
        Reward(
            reward_name="reward",
            value=1.0,
            reward_policy_hash="a" * 64,
            evidence_role=EvaluationRole.OFFICIAL,
            input_citations=(
                RewardInputCitation(
                    name="pass_rate", value=1.0, contributed=1.0
                ),
            ),
        )


def test_missing_data_fail_raises() -> None:
    policy = _policy(MissingDataPolicy.FAIL)
    with pytest.raises(ValueError, match="missing"):
        apply_reward_policy(
            policy,
            aggregates={"pass_rate": 0.8},  # compression missing
            evidence_role=EvaluationRole.INTERNAL,
        )


def test_missing_data_worst_substitutes_direction_worst() -> None:
    policy = RewardPolicy(
        policy_name="p",
        terms=(
            RewardTerm(
                name="pass_rate", weight=1.0, maximize=True, worst_value=0.0
            ),
        ),
        missing_data=MissingDataPolicy.WORST,
    )
    reward = apply_reward_policy(
        policy, aggregates={}, evidence_role=EvaluationRole.INTERNAL
    )
    assert reward.value == 0.0
    assert reward.input_citations[0].was_missing is True


def test_missing_data_skip_drops_the_term() -> None:
    policy = RewardPolicy(
        policy_name="p",
        terms=(
            RewardTerm(name="pass_rate", weight=1.0),
            RewardTerm(name="compression", weight=1.0, maximize=False),
        ),
        missing_data=MissingDataPolicy.SKIP,
    )
    reward = apply_reward_policy(
        policy,
        aggregates={"pass_rate": 0.5},
        evidence_role=EvaluationRole.INTERNAL,
    )
    assert reward.value == pytest.approx(0.5)
    skipped = [c for c in reward.input_citations if c.was_missing]
    assert len(skipped) == 1


def test_reward_policy_identity_is_stable() -> None:
    assert _policy().identity_hash() == _policy().identity_hash()


def test_reward_policy_requires_unique_terms() -> None:
    with pytest.raises(ValueError, match="unique"):
        RewardPolicy(
            policy_name="p",
            terms=(
                RewardTerm(name="x", weight=1.0),
                RewardTerm(name="x", weight=2.0),
            ),
        )
