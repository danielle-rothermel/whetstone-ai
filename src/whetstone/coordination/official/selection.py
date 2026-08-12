from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import require_full_hash
from whetstone.evaluation import AggregationStatus
from whetstone.experiment.objectives import (
    Direction,
    Objective,
    ObjectiveVector,
    ParetoFront,
    TieBehavior,
    objective_from_aggregate_value,
    pareto_front,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone.evaluation.aggregate import Aggregate

__all__ = [
    "IncompleteEvidenceError",
    "ObjectiveSpec",
    "SelectionCandidate",
    "SelectionEvidence",
    "select_official",
]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    objective_name: str
    aggregate_name: str
    direction: Direction

    def __post_init__(self) -> None:
        if not self.objective_name:
            raise ValueError("objective_name must be non-empty")
        if not self.aggregate_name:
            raise ValueError("aggregate_name must be non-empty")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    candidate_id: str
    graph_hash: str
    aggregates: Mapping[str, Aggregate]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        require_full_hash(self.graph_hash, field="graph_hash")
        if not self.aggregates:
            raise ValueError("aggregates must be non-empty")
        object.__setattr__(
            self, "aggregates", MappingProxyType(dict(self.aggregates))
        )


class IncompleteEvidenceError(ValueError):
    pass


class SelectionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_specs: tuple[ObjectiveSpec, ...]

    candidate_order: tuple[str, ...]

    candidate_vectors: tuple[ObjectiveVector, ...]

    tie_behavior: TieBehavior

    front: ParetoFront

    selected_candidate_id: StrictStr

    selected_index: int

    selected_by_tie_rule: StrictBool

    @model_validator(mode="after")
    def _validate(self) -> SelectionEvidence:
        if len(self.candidate_order) != len(self.candidate_vectors):
            raise ValueError(
                "candidate_order and candidate_vectors must align 1:1"
            )
        if not self.objective_specs:
            raise ValueError("selection needs >=1 objective spec")
        if self.selected_candidate_id not in self.candidate_order:
            raise ValueError(
                "selected_candidate_id must be one of the candidates"
            )
        front_ids = {m.candidate_id for m in self.front.members}
        if self.selected_candidate_id not in front_ids:
            raise ValueError(
                "the official selection must be a Pareto Front member"
            )
        if not 0 <= self.selected_index < len(self.candidate_order):
            raise ValueError("selected_index must index candidate_order")
        if self.candidate_order[self.selected_index] != (
            self.selected_candidate_id
        ):
            raise ValueError(
                "selected_index must identify selected_candidate_id in "
                "candidate_order"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _certified_value(
    aggregate: Aggregate,
) -> float:
    output = aggregate.aggregation_output
    if output.status is not AggregationStatus.OK or output.value is None:
        raise IncompleteEvidenceError(
            f"aggregate {aggregate.name!r} is not OK "
            f"(status={output.status}); official selection runs only over "
            "complete certified aggregate evidence"
        )
    return float(output.value)


def select_official(
    candidates: Sequence[SelectionCandidate],
    *,
    objective_specs: Sequence[ObjectiveSpec],
    tie_behavior: TieBehavior = TieBehavior.STABLE_INDEX,
) -> SelectionEvidence:
    if not candidates:
        raise ValueError("select_official requires at least one candidate")
    specs = tuple(objective_specs)
    if not specs:
        raise ValueError("select_official requires at least one ObjectiveSpec")
    spec_names = [s.objective_name for s in specs]
    if len(set(spec_names)) != len(spec_names):
        raise ValueError("objective spec objective_name values must be unique")

    candidate_order: list[str] = []
    candidate_vectors: list[ObjectiveVector] = []
    front_input: list[tuple[str, ObjectiveVector]] = []
    seen_ids: set[str] = set()

    eval_config_hashes: set[str] = set()

    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            raise ValueError(
                f"duplicate candidate_id {candidate.candidate_id!r}"
            )
        seen_ids.add(candidate.candidate_id)

        objectives: list[Objective] = []
        for spec in specs:
            aggregate = candidate.aggregates.get(spec.aggregate_name)
            if aggregate is None:
                raise IncompleteEvidenceError(
                    f"candidate {candidate.candidate_id!r} is missing "
                    f"aggregate {spec.aggregate_name!r}; official selection "
                    "runs only over complete certified aggregate evidence"
                )
            if aggregate.graph_hash != candidate.graph_hash:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} graph_hash does "
                    f"not match aggregate {spec.aggregate_name!r}"
                )
            eval_config_hashes.add(aggregate.eval_config_hash)
            if len(eval_config_hashes) > 1:
                raise ValueError(
                    "official selection requires every candidate aggregate to "
                    "share one eval_config_hash; saw "
                    f"{sorted(eval_config_hashes)}"
                )
            value = _certified_value(aggregate)
            objectives.append(
                objective_from_aggregate_value(
                    name=spec.objective_name,
                    value=value,
                    direction=spec.direction,
                    source_name=spec.aggregate_name,
                    graph_hash=aggregate.graph_hash,
                    eval_config_hash=aggregate.eval_config_hash,
                )
            )
        vector = ObjectiveVector(objectives=tuple(objectives))
        candidate_order.append(candidate.candidate_id)
        candidate_vectors.append(vector)
        front_input.append((candidate.candidate_id, vector))

    front = pareto_front(front_input, tie_behavior=tie_behavior)

    selected = front.members[0]
    selected_by_tie_rule = len(front.members) > 1

    return SelectionEvidence(
        objective_specs=specs,
        candidate_order=tuple(candidate_order),
        candidate_vectors=tuple(candidate_vectors),
        tie_behavior=tie_behavior,
        front=front,
        selected_candidate_id=selected.candidate_id,
        selected_index=selected.original_index,
        selected_by_tie_rule=selected_by_tie_rule,
    )
