"""Direction-bearing objectives and deterministic Pareto fronts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "RESERVED_OBJECTIVE_NAMES",
    "Direction",
    "Objective",
    "ObjectiveDerivation",
    "ObjectiveDerivationSource",
    "ObjectiveVector",
    "ParetoFront",
    "ParetoMember",
    "RewardIsNotAnObjectiveError",
    "TieBehavior",
    "dominates",
    "objective_from_aggregate_value",
    "objective_from_score_value",
    "pareto_front",
    "reject_reward_name",
]

RESERVED_OBJECTIVE_NAMES = frozenset({"reward"})


class Direction(StrEnum):
    """The optimization direction carried by an objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveDerivationSource(StrEnum):
    """The complete set of evidence from which an objective may derive."""

    SCORE = "score"
    ROLLOUT_AGGREGATE = "rollout_aggregate"


class RewardIsNotAnObjectiveError(ValueError):
    """A Reward name was offered as an official-selection objective."""


def reject_reward_name(name: str) -> None:
    """Reject names reserved for optimizer-facing Reward values."""

    if name.lower() in RESERVED_OBJECTIVE_NAMES:
        raise RewardIsNotAnObjectiveError(
            f"{name!r} is a reserved Reward name and can never be an "
            "Objective; Reward is optimizer-facing, not an official "
            "selection criterion"
        )


@dataclass(frozen=True, slots=True)
class ObjectiveDerivation:
    """Deterministic lineage from one eligible evidence value."""

    source: ObjectiveDerivationSource
    source_name: str
    graph_hash: str | None = None
    eval_config_hash: str | None = None
    derivation_id: str = "identity"

    def __post_init__(self) -> None:
        if not isinstance(self.source, ObjectiveDerivationSource):
            raise TypeError("source must be an ObjectiveDerivationSource")
        if not self.source_name:
            raise ValueError("ObjectiveDerivation source_name must be set")
        if not self.derivation_id:
            raise ValueError("ObjectiveDerivation derivation_id must be set")


@dataclass(frozen=True, slots=True)
class Objective:
    """One named, direction-bearing, deterministically derived criterion."""

    name: str
    value: float
    direction: Direction
    derivation: ObjectiveDerivation

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Objective name must be non-empty")
        if isinstance(self.value, bool) or not isinstance(
            self.value, (int, float)
        ):
            raise TypeError("Objective value must be numeric")
        if not isfinite(self.value):
            raise ValueError("Objective value must be finite")
        if not isinstance(self.direction, Direction):
            raise TypeError("Objective direction must be a Direction")
        if not isinstance(self.derivation, ObjectiveDerivation):
            raise TypeError(
                "Objective derivation must be an ObjectiveDerivation"
            )
        reject_reward_name(self.name)

    def is_better_than(self, other_value: float) -> bool:
        """Whether this objective value beats ``other_value``."""

        if self.direction is Direction.MAXIMIZE:
            return self.value > other_value
        return self.value < other_value


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    """An ordered, non-empty tuple of uniquely named objectives."""

    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("an Objective Vector must have >=1 Objective")
        if any(
            not isinstance(objective, Objective)
            for objective in self.objectives
        ):
            raise TypeError("objectives must contain only Objective values")
        names = tuple(objective.name for objective in self.objectives)
        if len(set(names)) != len(names):
            raise ValueError("Objective Vector names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(objective.name for objective in self.objectives)

    @property
    def directions(self) -> tuple[Direction, ...]:
        return tuple(objective.direction for objective in self.objectives)

    def values(self) -> tuple[float, ...]:
        return tuple(objective.value for objective in self.objectives)


class TieBehavior(StrEnum):
    """Declared tie behavior for front ordering and official selection."""

    STABLE_INDEX = "stable_index"


def dominates(a: ObjectiveVector, b: ObjectiveVector) -> bool:
    """Whether ``a`` is no worse everywhere and better somewhere than ``b``."""

    if a.names != b.names:
        raise ValueError(
            "dominance requires identical objective names in the same order: "
            f"{a.names} != {b.names}"
        )
    if a.directions != b.directions:
        raise ValueError(
            "dominance requires identical objective directions: "
            f"{a.directions} != {b.directions}"
        )

    no_worse_everywhere = True
    strictly_better_somewhere = False
    for objective_a, objective_b in zip(
        a.objectives, b.objectives, strict=True
    ):
        if objective_a.direction is Direction.MAXIMIZE:
            if objective_a.value < objective_b.value:
                no_worse_everywhere = False
            elif objective_a.value > objective_b.value:
                strictly_better_somewhere = True
        else:
            if objective_a.value > objective_b.value:
                no_worse_everywhere = False
            elif objective_a.value < objective_b.value:
                strictly_better_somewhere = True
    return no_worse_everywhere and strictly_better_somewhere


@dataclass(frozen=True, slots=True)
class ParetoMember:
    """One non-dominated candidate in stable input order."""

    original_index: int
    candidate_id: str
    vector: ObjectiveVector

    def __post_init__(self) -> None:
        if self.original_index < 0:
            raise ValueError("original_index cannot be negative")
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ParetoFront:
    """A stable, direction-aware set of non-dominated candidates."""

    objective_names: tuple[str, ...]
    objective_directions: tuple[Direction, ...]
    tie_behavior: TieBehavior
    members: tuple[ParetoMember, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tie_behavior, TieBehavior):
            raise TypeError("tie_behavior must be a TieBehavior")
        if len(self.objective_names) != len(self.objective_directions):
            raise ValueError(
                "objective_names and objective_directions must align"
            )
        prior = -1
        for member in self.members:
            if member.original_index <= prior:
                raise ValueError(
                    "Pareto Front members must be in strictly ascending "
                    "original-index (stable) order"
                )
            prior = member.original_index
            if member.vector.names != self.objective_names:
                raise ValueError(
                    "Pareto member vector names must match the front's "
                    "declared objective names"
                )
            if member.vector.directions != self.objective_directions:
                raise ValueError(
                    "Pareto member vector directions must match the front's "
                    "declared objective directions"
                )


def pareto_front(
    candidates: Sequence[tuple[str, ObjectiveVector]],
    *,
    tie_behavior: TieBehavior = TieBehavior.STABLE_INDEX,
) -> ParetoFront:
    """Construct the stable, deterministic Pareto front over ``candidates``."""

    if not candidates:
        raise ValueError("pareto_front requires at least one candidate")

    names = candidates[0][1].names
    directions = candidates[0][1].directions
    for _candidate_id, vector in candidates:
        if vector.names != names:
            raise ValueError(
                "all candidate vectors must share the same ordered objective "
                f"names: {vector.names} != {names}"
            )
        if vector.directions != directions:
            raise ValueError(
                "all candidate vectors must share the same objective "
                f"directions: {vector.directions} != {directions}"
            )

    members = tuple(
        ParetoMember(
            original_index=index,
            candidate_id=candidate_id,
            vector=vector,
        )
        for index, (candidate_id, vector) in enumerate(candidates)
        if not any(
            dominates(other_vector, vector)
            for other_index, (_other_id, other_vector) in enumerate(candidates)
            if other_index != index
        )
    )
    return ParetoFront(
        objective_names=names,
        objective_directions=directions,
        tie_behavior=tie_behavior,
        members=members,
    )


def objective_from_aggregate_value(
    *,
    name: str,
    value: float,
    direction: Direction,
    source_name: str,
    graph_hash: str | None = None,
    eval_config_hash: str | None = None,
    derivation_id: str = "identity",
) -> Objective:
    """Derive an objective from one Rollout Aggregate value."""

    reject_reward_name(name)
    return Objective(
        name=name,
        value=value,
        direction=direction,
        derivation=ObjectiveDerivation(
            source=ObjectiveDerivationSource.ROLLOUT_AGGREGATE,
            source_name=source_name,
            graph_hash=graph_hash,
            eval_config_hash=eval_config_hash,
            derivation_id=derivation_id,
        ),
    )


def objective_from_score_value(
    *,
    name: str,
    value: float,
    direction: Direction,
    source_name: str,
    graph_hash: str | None = None,
    eval_config_hash: str | None = None,
    derivation_id: str = "identity",
) -> Objective:
    """Derive an objective from one named Score value."""

    reject_reward_name(name)
    return Objective(
        name=name,
        value=value,
        direction=direction,
        derivation=ObjectiveDerivation(
            source=ObjectiveDerivationSource.SCORE,
            source_name=source_name,
            graph_hash=graph_hash,
            eval_config_hash=eval_config_hash,
            derivation_id=derivation_id,
        ),
    )
