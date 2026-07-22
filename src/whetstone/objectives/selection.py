"""Official selection over complete certified aggregate evidence.

Workstream 8 of ``design/concrete-changes.html`` and the *Official Evaluation*
Running Example row require an official-selection procedure that:

* runs only over **complete, certified** Rollout Aggregate evidence — an
  incomplete or non-certified aggregate is refused, never silently skipped;
* derives a per-candidate :class:`~whetstone.objectives.ObjectiveVector`
  deterministically from that evidence;
* constructs a deterministic :class:`~whetstone.objectives.ParetoFront` with
  stable ordering, explicit tie behavior, and a direction per objective;
* persists the *derivation*, the candidate *order*, the *tie* behavior, and the
  *selection* result as :class:`SelectionEvidence` — a complete, canonical,
  content-addressable record; and
* never computes or accepts a Reward as an Objective.

The selection input is a per-candidate bundle of certified aggregates. "One
candidate" here is one admitted Graph Hash / curve slot; its aggregates are
the Whetstone :class:`~whetstone.code_eval.RolloutAggregate` values already
bound to ``(graph_hash, eval_config_hash)`` and a stated Evaluation Context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_code.eval import AggregationStatus
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictStr,
    model_validator,
)

from whetstone.objectives.objective import (
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

    from whetstone.code_eval.aggregate import RolloutAggregate

__all__ = [
    "IncompleteEvidenceError",
    "ObjectiveSpec",
    "SelectionCandidate",
    "SelectionEvidence",
    "select_official",
]


class ObjectiveSpec(BaseModel):
    """A frozen declaration of one Objective to derive during selection.

    Names the aggregate to read (``aggregate_name``), the Objective name to
    publish it under, and the direction. The spec fixes the deterministic
    derivation; it carries no Reward source (there is no such option), and the
    Objective name is validated against the reserved Reward name when the
    Objective is built.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_name: StrictStr
    aggregate_name: StrictStr
    direction: Direction

    @model_validator(mode="after")
    def _validate(self) -> ObjectiveSpec:
        if not self.objective_name:
            raise ValueError("objective_name must be non-empty")
        if not self.aggregate_name:
            raise ValueError("aggregate_name must be non-empty")
        return self


class SelectionCandidate:
    """One candidate's certified aggregate evidence for selection.

    A candidate is one admitted Graph Hash / curve slot. ``aggregates`` maps
    aggregate name -> :class:`RolloutAggregate`; ``candidate_id`` is the opaque
    identifier the front carries (typically the selected Materialization Record
    reference or the ``graph_hash``). This is a plain value holder — not a
    persisted record — so it is a lightweight class, not a pydantic model, and
    is validated when consumed by :func:`select_official`.
    """

    __slots__ = ("aggregates", "candidate_id", "graph_hash")

    def __init__(
        self,
        *,
        candidate_id: str,
        graph_hash: str,
        aggregates: dict[str, RolloutAggregate],
    ) -> None:
        self.candidate_id = candidate_id
        self.graph_hash = graph_hash
        self.aggregates = aggregates


class IncompleteEvidenceError(ValueError):
    """A candidate's aggregate evidence is missing or not complete/certified.

    Selection runs only over COMPLETE certified evidence; an aggregate that is
    absent, does not account for the complete planned matrix, or whose pure
    reduction is not ``OK`` is refused here rather than silently dropped.
    """


class SelectionEvidence(BaseModel):
    """Persisted, canonical evidence of one official-selection run.

    A complete, self-describing record capturing everything needed to reproduce
    and audit the selection: the ordered objective specs (derivation +
    direction), the ordered candidate list and each candidate's derived
    Objective Vector, the tie behavior, the constructed Pareto Front, and the
    single official selection. It carries a Content Hash (via
    :meth:`record_content`), never an Identity Hash, and computes no Reward.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Ordered objective specs (the deterministic derivation + direction).
    objective_specs: tuple[ObjectiveSpec, ...]
    #: Ordered candidate ids, in the exact input order (stable ordering).
    candidate_order: tuple[str, ...]
    #: Each candidate's derived Objective Vector, aligned to candidate_order.
    candidate_vectors: tuple[ObjectiveVector, ...]
    #: The declared, explicit tie behavior.
    tie_behavior: TieBehavior
    #: The deterministic Pareto Front over the derived vectors.
    front: ParetoFront
    #: The single officially selected candidate id.
    selected_candidate_id: StrictStr
    #: The selected candidate's original index (tie resolution is explicit).
    selected_index: int
    #: True when the front had >1 member and the tie rule broke the choice.
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
        return self

    def record_content(self) -> dict[str, Any]:
        """The complete canonical persisted content (for a Content Hash)."""
        return self.model_dump(mode="json")


def _certified_value(
    aggregate: RolloutAggregate,
) -> float:
    """Return the certified numeric value of a complete aggregate.

    Refuses any aggregate whose pure reduction is not ``OK``: a non-OK status
    (missing data, zero denominator, not applicable) is exactly the incomplete
    evidence official selection must never select over.
    """
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
    """Run official selection over complete certified aggregate evidence.

    Deterministic end to end:

    1. For every candidate, derive one Objective per spec from that candidate's
       named certified aggregate (refusing missing / non-OK / incomplete
       evidence), forming an ordered :class:`ObjectiveVector`.
    2. Construct the deterministic :class:`ParetoFront` over those vectors in
       stable input order with the explicit ``tie_behavior``.
    3. Officially select the front's first member (lowest original index) under
       the ``STABLE_INDEX`` tie rule, recording whether a tie rule broke it.
    4. Persist the derivation, order, tie behavior, front, and selection as
       :class:`SelectionEvidence`.

    No Reward is computed: Objectives derive only from Scores / Rollout
    Aggregates, and the reserved Reward name is refused by :class:`Objective`.
    """
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
    # Stable tie rule: the official selection is the front's first member
    # (lowest original index). A tie broke the choice when the front holds
    # more than one non-dominated candidate.
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
