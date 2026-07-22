"""Direction-bearing Objectives, Objective Vectors, and Pareto Fronts.

Per the vocabulary (``design/vocab_and_defs.html`` — *Objective*, *Objective
Vector*, *Pareto Front*) and Workstream 8 of ``design/concrete-changes.html``:

* An :class:`Objective` is a *named direction-bearing official-selection
  criterion deterministically derived from a Score or Rollout Aggregate*. It
  carries its name, its direction (maximize / minimize), the numeric value,
  and the derivation lineage naming exactly which Score/Aggregate it came from
  and how. The derivation is deterministic: identical evidence produces an
  identical Objective.

* An :class:`ObjectiveVector` is an *ordered tuple of named Objectives*
  preserving their values, directions, and multi-objective semantics. Order is
  significant and preserved end to end; every Objective name is unique within a
  vector.

* A :class:`ParetoFront` is the *deterministically ordered non-dominated*
  subset of candidates under one declared Objective Vector, with a direction
  for every Objective. Ordering is stable and explicit, ties are handled by a
  declared rule, and dominance respects each Objective's direction.

Two load-bearing invariants are enforced at the *type level* (not only by
test):

* **Reward is never an Objective.** Deriving an Objective from Reward
  evidence, or naming an Objective ``reward``/an existing Reward name, is
  refused by construction: :class:`Objective` rejects the Reward derivation
  source and the reserved ``reward`` name, so a Reward can never masquerade as
  an official selection criterion. This is proven by :mod:`tests.objectives`.

* **Objectives come only from official-eligible evidence.** The derivation
  source is a closed enum of ``score`` / ``rollout_aggregate`` — the two the
  vocabulary permits — and explicitly *not* ``reward``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    model_validator,
)

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
    "RewardIsNotAnObjectiveError",
    "TieBehavior",
    "dominates",
    "pareto_front",
    "reject_reward_name",
]


# A Reward is an optimizer-facing value, never an official selection criterion.
# Its canonical name (and any alias) is reserved so an Objective can never be
# named to impersonate one.
RESERVED_OBJECTIVE_NAMES = frozenset({"reward"})


class Direction(StrEnum):
    """The optimization direction a single Objective bears.

    ``MAXIMIZE`` — larger values are better (e.g. Average Binary Test Pass
    Rate). ``MINIMIZE`` — smaller values are better (e.g. Mean Compression
    Ratio). Every Objective and every Pareto dominance comparison is
    direction-aware; there is no implicit "bigger is better" default at the
    dominance layer.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveDerivationSource(StrEnum):
    """The closed set of evidence an Objective may be derived from.

    Exactly the two the vocabulary permits: a named :class:`Score` or a
    :class:`~whetstone.code_eval.RolloutAggregate`. There is deliberately
    **no** ``REWARD`` member: Reward can never be a derivation source, so an
    Objective cannot be built from Reward evidence at the type level.
    """

    SCORE = "score"
    ROLLOUT_AGGREGATE = "rollout_aggregate"


class RewardIsNotAnObjectiveError(ValueError):
    """Reward was offered as an Objective name or derivation source.

    Raised at construction so the "Reward is never accepted as an Objective"
    rule is enforced by type, not merely by convention or downstream check.
    """


def reject_reward_name(name: str) -> None:
    """Refuse the reserved Reward name for an Objective.

    Raised as :class:`RewardIsNotAnObjectiveError` *before* pydantic validation
    in the derivation constructors, so the specific type surfaces directly
    (pydantic would otherwise wrap it in a ``ValidationError``). The
    :class:`Objective` model validator re-checks it as defense in depth for the
    direct-construction path.
    """
    if name.lower() in RESERVED_OBJECTIVE_NAMES:
        raise RewardIsNotAnObjectiveError(
            f"{name!r} is a reserved Reward name and can never be an "
            "Objective; Reward is optimizer-facing, not an official "
            "selection criterion"
        )


class ObjectiveDerivation(BaseModel):
    """Deterministic derivation lineage for one Objective.

    Names the source kind (Score / Rollout Aggregate), the exact source name it
    was read from, and the derivation identity so an Objective is reproducible
    from its evidence. It carries no Reward source:
    :class:`ObjectiveDerivation` can only cite a Score or a Rollout Aggregate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: ObjectiveDerivationSource
    #: The exact Score / Rollout Aggregate name the value was read from.
    source_name: StrictStr
    #: The measurement-cell identity the source belongs to (attribution).
    graph_hash: StrictStr | None = None
    eval_config_hash: StrictStr | None = None
    #: A stable derivation identifier (e.g. "identity", "per_task_mean").
    derivation_id: StrictStr = "identity"

    @model_validator(mode="after")
    def _validate(self) -> ObjectiveDerivation:
        if not self.source_name:
            raise ValueError("ObjectiveDerivation source_name must be set")
        return self


class Objective(BaseModel):
    """A named, direction-bearing, deterministically derived Objective.

    Refuses — at construction — any attempt to be a Reward: the reserved name
    ``reward`` is rejected, and because :class:`ObjectiveDerivationSource` has
    no Reward member the derivation can never cite Reward evidence. The value
    is the concrete measured number; ``direction`` is its optimization sense.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    value: float
    direction: Direction
    derivation: ObjectiveDerivation

    @model_validator(mode="after")
    def _validate(self) -> Objective:
        if not self.name:
            raise ValueError("Objective name must be non-empty")
        reject_reward_name(self.name)
        return self

    def is_better_than(self, other_value: float) -> bool:
        """Whether this value beats ``other_value`` under its direction."""
        if self.direction is Direction.MAXIMIZE:
            return self.value > other_value
        return self.value < other_value


class ObjectiveVector(BaseModel):
    """An ordered tuple of named Objectives with unique names.

    Order is significant: it fixes the objective ordering used for stable
    Pareto ordering and for the persisted selection evidence. Names are unique
    so a vector can be addressed positionally or by name without ambiguity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objectives: tuple[Objective, ...]

    @model_validator(mode="after")
    def _validate(self) -> ObjectiveVector:
        if not self.objectives:
            raise ValueError("an Objective Vector must have >=1 Objective")
        names = [obj.name for obj in self.objectives]
        if len(set(names)) != len(names):
            raise ValueError("Objective Vector names must be unique")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objectives)

    @property
    def directions(self) -> tuple[Direction, ...]:
        return tuple(obj.direction for obj in self.objectives)

    def values(self) -> tuple[float, ...]:
        return tuple(obj.value for obj in self.objectives)


class TieBehavior(StrEnum):
    """Explicit tie behavior for Pareto Front ordering and selection.

    ``STABLE_INDEX`` — equal candidates keep their original supplied order
    (the front's stable ordering is the input order restricted to the
    non-dominated set); selection among Pareto-equivalent candidates picks the
    lowest original index. This is the only supported behavior today, but the
    field is explicit and persisted so a future behavior is a visible change,
    never a silent one.
    """

    STABLE_INDEX = "stable_index"


def dominates(
    a: ObjectiveVector,
    b: ObjectiveVector,
) -> bool:
    """Whether Objective Vector ``a`` Pareto-dominates ``b``.

    Direction-aware multi-objective dominance: ``a`` dominates ``b`` iff ``a``
    is no worse than ``b`` on every Objective and strictly better on at least
    one, each compared under that Objective's own direction. Both vectors MUST
    share the same ordered objective names and directions.
    """
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
    for obj_a, obj_b in zip(a.objectives, b.objectives, strict=True):
        if obj_a.direction is Direction.MAXIMIZE:
            if obj_a.value < obj_b.value:
                no_worse_everywhere = False
            elif obj_a.value > obj_b.value:
                strictly_better_somewhere = True
        else:
            if obj_a.value > obj_b.value:
                no_worse_everywhere = False
            elif obj_a.value < obj_b.value:
                strictly_better_somewhere = True
    return no_worse_everywhere and strictly_better_somewhere


class ParetoMember(BaseModel):
    """One non-dominated member of a Pareto Front.

    Carries its stable original index (for reproducible ordering and tie
    resolution), an opaque candidate identifier the caller supplies (e.g. a
    Materialization Record reference or graph_hash), and its Objective Vector.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_index: int
    candidate_id: StrictStr
    vector: ObjectiveVector


class ParetoFront(BaseModel):
    """A deterministically ordered non-dominated set under one Vector shape.

    The front is the subset of supplied candidates that no other candidate
    dominates, in *stable input order* (each member keeps its original index).
    ``objective_names`` and ``objective_directions`` declare the shared vector
    shape a direction for every Objective. ``tie_behavior`` states the explicit
    tie rule used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_names: tuple[str, ...]
    objective_directions: tuple[Direction, ...]
    tie_behavior: TieBehavior
    members: tuple[ParetoMember, ...]

    @model_validator(mode="after")
    def _validate(self) -> ParetoFront:
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
        return self


def pareto_front(
    candidates: Sequence[tuple[str, ObjectiveVector]],
    *,
    tie_behavior: TieBehavior = TieBehavior.STABLE_INDEX,
) -> ParetoFront:
    """Deterministically construct the Pareto Front over ``candidates``.

    ``candidates`` is an ordered sequence of ``(candidate_id, vector)`` pairs;
    the input order fixes the stable ordering and tie resolution. A candidate
    is on the front iff no *other* candidate dominates it. Two candidates with
    identical Objective values never dominate each other, so both remain on the
    front, ordered by original index (``STABLE_INDEX`` tie behavior).

    All vectors MUST share the same ordered objective names and directions;
    this is the "direction per objective" requirement made explicit.
    """
    if not candidates:
        raise ValueError("pareto_front requires at least one candidate")

    first_vector = candidates[0][1]
    names = first_vector.names
    directions = first_vector.directions
    for _cid, vector in candidates:
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

    members: list[ParetoMember] = []
    for index, (candidate_id, vector) in enumerate(candidates):
        dominated = any(
            dominates(other_vector, vector)
            for other_index, (_ocid, other_vector) in enumerate(candidates)
            if other_index != index
        )
        if not dominated:
            members.append(
                ParetoMember(
                    original_index=index,
                    candidate_id=candidate_id,
                    vector=vector,
                )
            )
    return ParetoFront(
        objective_names=names,
        objective_directions=directions,
        tie_behavior=tie_behavior,
        members=tuple(members),
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
    """Deterministically derive an Objective from a Rollout Aggregate value.

    A thin, named constructor that fixes the derivation source to
    ``ROLLOUT_AGGREGATE`` and records the source name and cell identity as
    lineage. It cannot produce a Reward-sourced Objective (there is no such
    source), and the reserved-name check on :class:`Objective` still applies.
    """
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
    """Deterministically derive an Objective from a named Score value."""
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


# Extend the public surface with the concrete derivation constructors and the
# Pareto member type.
__all__ += [
    "ParetoMember",
    "objective_from_aggregate_value",
    "objective_from_score_value",
]
