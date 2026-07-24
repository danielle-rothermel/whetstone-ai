"""Objectives, Objective Vectors, and Pareto Fronts.

Proves the direction-bearing Objective derivation, the ordered Objective
Vector, deterministic Pareto Front construction (stable ordering, explicit tie
behavior, direction per objective), and the type-level refusal that Reward is
never accepted as an Objective.
"""

from __future__ import annotations

import pytest

from whetstone.objectives import (
    Direction,
    Objective,
    ObjectiveDerivation,
    ObjectiveDerivationSource,
    ObjectiveVector,
    RewardIsNotAnObjectiveError,
    TieBehavior,
    dominates,
    objective_from_aggregate_value,
    objective_from_score_value,
    pareto_front,
)


def _obj(name: str, value: float, direction: Direction) -> Objective:
    return objective_from_aggregate_value(
        name=name,
        value=value,
        direction=direction,
        source_name=f"agg::{name}",
    )


def _vector(pass_rate: float, compression: float) -> ObjectiveVector:
    return ObjectiveVector(
        objectives=(
            _obj("pass_rate", pass_rate, Direction.MAXIMIZE),
            _obj("compression", compression, Direction.MINIMIZE),
        )
    )


# ---------------------------------------------------------------------------
# Reward is never an Objective (type-level)
# ---------------------------------------------------------------------------


def test_reward_name_rejected_at_construction() -> None:
    with pytest.raises(RewardIsNotAnObjectiveError):
        _obj("reward", 1.0, Direction.MAXIMIZE)


def test_reward_name_rejected_case_insensitively() -> None:
    with pytest.raises(RewardIsNotAnObjectiveError):
        _obj("Reward", 1.0, Direction.MAXIMIZE)


def test_direct_objective_construction_also_refuses_reward() -> None:
    # The raw model validator (defense in depth) refuses the reserved name
    # too; pydantic surfaces it as a ValidationError whose message names it.
    with pytest.raises(ValueError, match="reserved Reward name"):
        Objective(
            name="reward",
            value=1.0,
            direction=Direction.MAXIMIZE,
            derivation=ObjectiveDerivation(
                source=ObjectiveDerivationSource.SCORE,
                source_name="x",
            ),
        )


def test_no_reward_derivation_source_exists() -> None:
    # The closed derivation source enum has exactly Score and Rollout
    # Aggregate; there is no Reward member, so an Objective cannot cite Reward.
    members = {m.value for m in ObjectiveDerivationSource}
    assert members == {"score", "rollout_aggregate"}
    assert "reward" not in members


def test_score_derived_objective_records_lineage() -> None:
    obj = objective_from_score_value(
        name="pass_rate",
        value=0.9,
        direction=Direction.MAXIMIZE,
        source_name="binary_test_pass_score",
    )
    assert obj.derivation.source is ObjectiveDerivationSource.SCORE
    assert obj.derivation.source_name == "binary_test_pass_score"


def test_derivation_is_deterministic() -> None:
    a = _obj("pass_rate", 0.75, Direction.MAXIMIZE)
    b = _obj("pass_rate", 0.75, Direction.MAXIMIZE)
    assert a == b


# ---------------------------------------------------------------------------
# Objective Vector ordering and uniqueness
# ---------------------------------------------------------------------------


def test_objective_vector_preserves_order() -> None:
    vector = _vector(0.8, 3.0)
    assert vector.names == ("pass_rate", "compression")
    assert vector.directions == (Direction.MAXIMIZE, Direction.MINIMIZE)
    assert vector.values() == (0.8, 3.0)


def test_objective_vector_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        ObjectiveVector(
            objectives=(
                _obj("x", 1.0, Direction.MAXIMIZE),
                _obj("x", 2.0, Direction.MAXIMIZE),
            )
        )


def test_objective_vector_requires_at_least_one() -> None:
    with pytest.raises(ValueError, match=">=1"):
        ObjectiveVector(objectives=())


# ---------------------------------------------------------------------------
# Dominance is direction-aware
# ---------------------------------------------------------------------------


def test_dominance_respects_directions() -> None:
    # a: higher pass rate AND lower compression -> dominates b.
    a = _vector(0.9, 2.0)
    b = _vector(0.8, 3.0)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_no_dominance_on_tradeoff() -> None:
    # a better on pass rate, b better on compression -> neither dominates.
    a = _vector(0.9, 3.0)
    b = _vector(0.8, 2.0)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_equal_vectors_do_not_dominate() -> None:
    a = _vector(0.8, 3.0)
    b = _vector(0.8, 3.0)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominance_requires_matching_shape() -> None:
    a = _vector(0.8, 3.0)
    other = ObjectiveVector(
        objectives=(_obj("pass_rate", 0.8, Direction.MAXIMIZE),)
    )
    with pytest.raises(ValueError, match="identical objective names"):
        dominates(a, other)


# ---------------------------------------------------------------------------
# Pareto Front: determinism, stable order, explicit ties, direction
# ---------------------------------------------------------------------------


def test_pareto_front_keeps_non_dominated_only() -> None:
    candidates = [
        ("c0", _vector(0.9, 2.0)),  # dominates c1
        ("c1", _vector(0.8, 3.0)),  # dominated by c0
        ("c2", _vector(0.7, 1.0)),  # tradeoff (best compression)
    ]
    front = pareto_front(candidates)
    ids = [m.candidate_id for m in front.members]
    assert ids == ["c0", "c2"]


def test_pareto_front_is_stable_ordered() -> None:
    candidates = [
        ("c0", _vector(0.9, 3.0)),
        ("c1", _vector(0.7, 1.0)),
        ("c2", _vector(0.8, 2.0)),
    ]
    front = pareto_front(candidates)
    # Members are in strictly ascending original-index order.
    indices = [m.original_index for m in front.members]
    assert indices == sorted(indices)
    assert indices == [0, 1, 2]


def test_pareto_front_ties_keep_both_members() -> None:
    # Two identical vectors never dominate each other: both stay on the front,
    # in stable input order, under the explicit STABLE_INDEX tie behavior.
    candidates = [
        ("c0", _vector(0.8, 3.0)),
        ("c1", _vector(0.8, 3.0)),
    ]
    front = pareto_front(candidates)
    assert front.tie_behavior is TieBehavior.STABLE_INDEX
    assert [m.candidate_id for m in front.members] == ["c0", "c1"]


def test_pareto_front_is_deterministic() -> None:
    candidates = [
        ("c0", _vector(0.9, 2.0)),
        ("c1", _vector(0.7, 1.0)),
        ("c2", _vector(0.8, 2.5)),
    ]
    a = pareto_front(candidates)
    b = pareto_front(candidates)
    assert a == b


def test_pareto_front_records_direction_per_objective() -> None:
    front = pareto_front([("c0", _vector(0.8, 3.0))])
    assert front.objective_names == ("pass_rate", "compression")
    assert front.objective_directions == (
        Direction.MAXIMIZE,
        Direction.MINIMIZE,
    )


def test_pareto_front_requires_matching_shapes() -> None:
    candidates = [
        ("c0", _vector(0.8, 3.0)),
        (
            "c1",
            ObjectiveVector(
                objectives=(_obj("pass_rate", 0.8, Direction.MAXIMIZE),)
            ),
        ),
    ]
    with pytest.raises(ValueError, match="same ordered objective names"):
        pareto_front(candidates)


def test_direction_bearing_objective_has_direction() -> None:
    obj = _obj("compression", 2.0, Direction.MINIMIZE)
    assert obj.direction is Direction.MINIMIZE
    assert obj.is_better_than(3.0)  # lower is better
    assert not obj.is_better_than(1.0)


def test_objective_derivation_requires_source_name() -> None:
    with pytest.raises(ValueError, match="source_name"):
        ObjectiveDerivation(
            source=ObjectiveDerivationSource.SCORE,
            source_name="",
        )
