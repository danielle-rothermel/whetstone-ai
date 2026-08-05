"""The reusable Reward Policy/Reward contract; official computes no Reward."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from whetstone.core.identity import typed_ref_for_record
from whetstone.core.roles import EvaluationRole
from whetstone.experiment.reward import (
    REWARD_POLICY_SCHEMA,
    REWARD_POLICY_SCHEMA_VERSION,
    MissingDataPolicy,
    OfficialRewardError,
    Reward,
    RewardInputCitation,
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


def _evidence_refs():
    return (
        typed_ref_for_record(
            "whetstone.test.evaluation_evidence",
            {"rollout_id": "rollout-1"},
        ),
    )


def test_reward_names_its_policy_and_cites_inputs() -> None:
    policy = _policy()
    reward = apply_reward_policy(
        policy,
        aggregates={"pass_rate": 0.8, "compression": 0.4},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=_evidence_refs(),
    )
    assert reward.reward_policy_hash == policy.identity_hash()
    assert reward.evidence_role is EvaluationRole.INTERNAL
    # value = 1.0*0.8 - 0.5*0.4 = 0.6
    assert reward.value == pytest.approx(0.6)
    cited = {c.name for c in reward.input_citations}
    assert cited == {"pass_rate", "compression"}


def test_reward_policy_identity_contract_is_exact() -> None:
    policy = _policy()

    assert REWARD_POLICY_SCHEMA == "whetstone.reward_policy"
    assert REWARD_POLICY_SCHEMA_VERSION == 1
    assert policy.identity_payload() == {
        "policy_name": "pass_up_compression_down/v1",
        "reward_name": "reward",
        "terms": [
            {
                "name": "pass_rate",
                "weight": 1.0,
                "maximize": True,
                "worst_value": 0.0,
            },
            {
                "name": "compression",
                "weight": 0.5,
                "maximize": False,
                "worst_value": 0.0,
            },
        ],
        "missing_data": "fail",
    }
    assert (
        policy.identity_hash()
        == "68a949e0ed2b767918e567319f9533f24c894392fcd351bdb73327024a8a0d36"
    )
    assert (
        RewardPolicy.model_validate(policy.model_dump(mode="json")) == policy
    )
    assert RewardPolicy.model_validate_json(policy.model_dump_json()) == policy


@pytest.mark.parametrize(
    "mutation",
    ["term-order", "direction", "missing-data", "reward-name"],
)
def test_reward_policy_identity_changes_with_exact_content(
    mutation: str,
) -> None:
    policy = _policy()
    payload = policy.model_dump(mode="json")
    if mutation == "term-order":
        payload["terms"].reverse()
    elif mutation == "direction":
        payload["terms"][0]["maximize"] = False
    elif mutation == "missing-data":
        payload["missing_data"] = MissingDataPolicy.SKIP.value
    else:
        payload["reward_name"] = "proposal_reward"

    assert RewardPolicy.model_validate(payload).identity_hash() != (
        policy.identity_hash()
    )


@pytest.mark.parametrize("unordered", [set(), frozenset(), {}])
def test_reward_policy_terms_reject_unordered_containers(
    unordered: object,
) -> None:
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        RewardPolicy.model_validate(
            {
                **_policy().model_dump(mode="json"),
                "terms": unordered,
            }
        )


def test_official_evaluation_computes_no_reward() -> None:
    policy = _policy()
    with pytest.raises(OfficialRewardError):
        apply_reward_policy(
            policy,
            aggregates={"pass_rate": 0.8, "compression": 0.4},
            evidence_role=EvaluationRole.OFFICIAL,
            evidence_refs=_evidence_refs(),
        )


def test_reward_cannot_be_constructed_from_official_role() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Reward(
            reward_name="reward",
            value=1.0,
            reward_policy=RewardPolicy(
                policy_name="official",
                terms=(RewardTerm(name="pass_rate", weight=1.0),),
            ),
            evidence_role=EvaluationRole.OFFICIAL,
            input_citations=(
                RewardInputCitation(
                    name="pass_rate", value=1.0, contributed=1.0
                ),
            ),
            evidence_refs=_evidence_refs(),
        )


def test_missing_data_fail_raises() -> None:
    policy = _policy(MissingDataPolicy.FAIL)
    with pytest.raises(ValueError, match="missing"):
        apply_reward_policy(
            policy,
            aggregates={"pass_rate": 0.8},  # compression missing
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=_evidence_refs(),
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
        policy,
        aggregates={},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=_evidence_refs(),
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
        evidence_refs=_evidence_refs(),
    )
    assert reward.value == pytest.approx(0.5)
    skipped = [c for c in reward.input_citations if c.was_missing]
    assert len(skipped) == 1


def test_reward_policy_requires_unique_terms() -> None:
    with pytest.raises(ValueError, match="unique"):
        RewardPolicy(
            policy_name="p",
            terms=(
                RewardTerm(name="x", weight=1.0),
                RewardTerm(name="x", weight=2.0),
            ),
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_aggregate_obeys_missing_data_policy(invalid) -> None:
    with pytest.raises(ValueError, match="missing or invalid"):
        apply_reward_policy(
            _policy(MissingDataPolicy.FAIL),
            aggregates={"pass_rate": invalid, "compression": 0.1},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=_evidence_refs(),
        )
    skipped = apply_reward_policy(
        _policy(MissingDataPolicy.SKIP),
        aggregates={"pass_rate": invalid, "compression": 0.1},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=_evidence_refs(),
    )
    citation = next(
        item for item in skipped.input_citations if item.name == "pass_rate"
    )
    assert citation.was_missing is True
    assert citation.value is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight", float("nan")),
        ("worst_value", float("inf")),
    ],
)
def test_reward_terms_reject_nonfinite_identity_values(field, value) -> None:
    with pytest.raises(ValueError, match="finite"):
        if field == "weight":
            RewardTerm(name="score", weight=value)
        else:
            RewardTerm(
                name="score",
                weight=1.0,
                worst_value=value,
            )


def test_reward_rejects_wrong_name_and_nonfinite_value() -> None:
    from pydantic import ValidationError

    policy = RewardPolicy(
        policy_name="score/v1",
        terms=(RewardTerm(name="score", weight=1.0),),
    )
    citation = RewardInputCitation(
        name="score",
        value=1.0,
        contributed=1.0,
    )
    with pytest.raises(ValidationError, match="name must match"):
        Reward(
            reward_name="different",
            value=1.0,
            reward_policy=policy,
            evidence_role=EvaluationRole.INTERNAL,
            input_citations=(citation,),
            evidence_refs=_evidence_refs(),
        )
    with pytest.raises(ValidationError, match="finite"):
        Reward(
            reward_name="reward",
            value=float("nan"),
            reward_policy=policy,
            evidence_role=EvaluationRole.INTERNAL,
            input_citations=(citation,),
            evidence_refs=_evidence_refs(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("value", 99.0, "scalarized citation total"),
        ("contributed", 99.0, "exactly apply"),
    ],
)
def test_hostile_reward_cannot_forge_scalarized_totals(
    field: str,
    value: float,
    message: str,
) -> None:
    from pydantic import ValidationError

    reward = apply_reward_policy(
        _policy(),
        aggregates={"pass_rate": 0.8, "compression": 0.4},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=_evidence_refs(),
    )
    payload = reward.model_dump(mode="json")
    if field == "value":
        payload["value"] = value
    else:
        payload["input_citations"][0]["contributed"] = value

    with pytest.raises(ValidationError, match=message):
        Reward.model_validate(payload)


def test_reward_rejects_citations_that_diverge_from_exact_policy() -> None:
    from pydantic import ValidationError

    reward = apply_reward_policy(
        _policy(),
        aggregates={"pass_rate": 0.8, "compression": 0.4},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=_evidence_refs(),
    )
    payload = reward.model_dump(mode="json")
    payload["input_citations"].reverse()

    with pytest.raises(ValidationError, match="term names and order"):
        Reward.model_validate(payload)


@pytest.mark.parametrize(
    ("value", "was_missing"),
    [(None, False), (1.0, True)],
    ids=["missing-value-without-flag", "present-value-with-missing-flag"],
)
def test_reward_input_citation_rejects_value_missingness_disagreement(
    value: float | None,
    was_missing: bool,
) -> None:
    with pytest.raises(ValidationError, match="missingness must match"):
        RewardInputCitation(
            name="score",
            value=value,
            contributed=0.0,
            was_missing=was_missing,
        )


def test_internal_reward_requires_ordered_nonempty_exact_evidence() -> None:
    from pydantic import ValidationError

    reward = apply_reward_policy(
        _policy(),
        aggregates={"pass_rate": 0.8, "compression": 0.4},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=_evidence_refs(),
    )
    payload = reward.model_dump(mode="json")
    payload["evidence_refs"] = []
    with pytest.raises(ValidationError, match="at least one exact evidence"):
        Reward.model_validate(payload)

    payload["evidence_refs"] = {_evidence_refs()[0]}
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        Reward.model_validate(payload)

    exact_ref = _evidence_refs()[0].model_dump(mode="json")
    payload["evidence_refs"] = [exact_ref, exact_ref]
    with pytest.raises(ValidationError, match="evidence_refs must be unique"):
        Reward.model_validate(payload)


def test_finite_negative_zero_canonicalizes_to_positive_zero() -> None:
    import math

    positive = RewardTerm(name="score", weight=0.0, worst_value=0.0)
    hostile = RewardTerm.model_validate(
        {"name": "score", "weight": -0.0, "worst_value": -0.0}
    )

    assert hostile == positive
    assert hostile.identity_payload() == positive.identity_payload()
    assert math.copysign(1.0, hostile.weight) == 1.0
    assert math.copysign(1.0, hostile.worst_value) == 1.0
    assert (
        RewardPolicy(
            policy_name="zero/v1",
            terms=(hostile,),
        ).identity_hash()
        == RewardPolicy(
            policy_name="zero/v1",
            terms=(positive,),
        ).identity_hash()
    )
