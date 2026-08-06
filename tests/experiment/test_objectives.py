from __future__ import annotations

from typing import cast

import pytest

from whetstone.experiment.objectives import (
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


def _vector(quality: float, compression: float) -> ObjectiveVector:
    return ObjectiveVector(
        objectives=(
            _obj("quality", quality, Direction.MAXIMIZE),
            _obj("compression", compression, Direction.MINIMIZE),
        )
    )


def test_reward_name_rejected_at_construction() -> None:
    with pytest.raises(RewardIsNotAnObjectiveError):
        _obj("reward", 1.0, Direction.MAXIMIZE)


def test_reward_name_rejected_case_insensitively() -> None:
    with pytest.raises(RewardIsNotAnObjectiveError):
        _obj("Reward", 1.0, Direction.MAXIMIZE)


def test_direct_objective_construction_also_refuses_reward() -> None:
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
    members = {m.value for m in ObjectiveDerivationSource}
    assert members == {"score", "rollout_aggregate"}
    assert "reward" not in members


def test_score_derived_objective_records_lineage() -> None:
    obj = objective_from_score_value(
        name="quality",
        value=0.9,
        direction=Direction.MAXIMIZE,
        source_name="quality_score",
    )
    assert obj.derivation.source is ObjectiveDerivationSource.SCORE
    assert obj.derivation.source_name == "quality_score"


@pytest.mark.parametrize("value", [True, False, "1.0", None])
def test_objective_rejects_bool_and_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        _obj("quality", cast(float, value), Direction.MAXIMIZE)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_objective_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _obj("quality", value, Direction.MAXIMIZE)


def test_objective_vector_preserves_order() -> None:
    vector = _vector(0.8, 3.0)
    assert vector.names == ("quality", "compression")
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


def test_dominance_respects_directions() -> None:
    a = _vector(0.9, 2.0)
    b = _vector(0.8, 3.0)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_no_dominance_on_tradeoff() -> None:
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
        objectives=(_obj("quality", 0.8, Direction.MAXIMIZE),)
    )
    with pytest.raises(ValueError, match="identical objective names"):
        dominates(a, other)


def test_dominance_rejects_opposite_directions() -> None:
    maximize = ObjectiveVector(
        objectives=(_obj("quality", 0.8, Direction.MAXIMIZE),)
    )
    minimize = ObjectiveVector(
        objectives=(_obj("quality", 0.8, Direction.MINIMIZE),)
    )

    with pytest.raises(ValueError, match="identical objective directions"):
        dominates(maximize, minimize)


def test_pareto_front_keeps_non_dominated_only() -> None:
    candidates = [
        ("c0", _vector(0.9, 2.0)),
        ("c1", _vector(0.8, 3.0)),
        ("c2", _vector(0.7, 1.0)),
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
    indices = [m.original_index for m in front.members]
    assert indices == sorted(indices)
    assert indices == [0, 1, 2]


def test_pareto_front_ties_keep_both_members() -> None:
    candidates = [
        ("c0", _vector(0.8, 3.0)),
        ("c1", _vector(0.8, 3.0)),
    ]
    front = pareto_front(candidates)
    assert front.tie_behavior is TieBehavior.STABLE_INDEX
    assert [m.candidate_id for m in front.members] == ["c0", "c1"]


def test_pareto_front_records_direction_per_objective() -> None:
    front = pareto_front([("c0", _vector(0.8, 3.0))])
    assert front.objective_names == ("quality", "compression")
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
                objectives=(_obj("quality", 0.8, Direction.MAXIMIZE),)
            ),
        ),
    ]
    with pytest.raises(ValueError, match="same ordered objective names"):
        pareto_front(candidates)


def test_pareto_front_rejects_opposite_directions() -> None:
    candidates = [
        (
            "maximize",
            ObjectiveVector(
                objectives=(_obj("quality", 0.8, Direction.MAXIMIZE),)
            ),
        ),
        (
            "minimize",
            ObjectiveVector(
                objectives=(_obj("quality", 0.8, Direction.MINIMIZE),)
            ),
        ),
    ]

    with pytest.raises(ValueError, match="same objective directions"):
        pareto_front(candidates)


def test_pareto_front_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        pareto_front([])


def test_direction_bearing_objective_has_direction() -> None:
    obj = _obj("compression", 2.0, Direction.MINIMIZE)
    assert obj.direction is Direction.MINIMIZE
    assert obj.is_better_than(3.0)
    assert not obj.is_better_than(1.0)


def test_objective_derivation_requires_source_name() -> None:
    with pytest.raises(ValueError, match="source_name"):
        ObjectiveDerivation(
            source=ObjectiveDerivationSource.SCORE,
            source_name="",
        )
