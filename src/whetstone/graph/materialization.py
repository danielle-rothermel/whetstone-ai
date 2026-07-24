"""Candidate materialization, selection, and execution planning.

This module implements the Whetstone harness pipeline that turns an
Optimization Proposal (or Experiment Candidate) into an execution plan:

1. **Proposal validation + Diff Check** — every proposal is validated and
   Diff-Checked against its explicit base: every changed path MUST lie
   within the closed Mutation Surface.
2. **Fixed-axis / Variance Source expansion** — every valid proposal is
   expanded across all fixed axes, producing one complete slot per axis
   selection.
3. **Materialization** — each complete slot yields exactly one Graph Config
   (via the dr-graph Graph Definition in the Rollout Definition role) plus
   exactly one immutable :class:`MaterializationRecord`. Records have NO
   Identity Hash and are stored only by typed Object Reference / Content
   Hash through dr-store. Multiple records MAY converge on one
   ``graph_hash``.
4. **Selection Policy** — a frozen ordered-subset rule selects an ordered
   subset of record slots for evaluation, without rewriting them.
5. **Execution planning** — selected records are deduplicated by full
   ``graph_hash`` before task/repeat planning. For ``D`` distinct admitted
   Graph Hashes the plan has exactly ``D x task_count x repeat_count`` work
   items and ``D`` aggregates, with no duplicate Rollout Execution Keys.

Record references never enter Rollout Keys, Work Requests, Results, or
aggregate identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dr_graph import GraphConfig, graph_hash
from dr_store import ObjectReference, ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_validator,
)

from whetstone.graph.rollout import (
    RolloutExecutionKey,
    RolloutKey,
    rollout_execution_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dr_graph import GraphDefinition

    from whetstone.graph.rollout import EvaluationContext

# Schema under which a Materialization Record is stored in dr-store. This is
# a *record* schema for Content Hash addressing, NOT an Identity Document
# schema: a Materialization Record has no Identity Hash.
MATERIALIZATION_RECORD_SCHEMA = "whetstone.materialization_record"


# ---------------------------------------------------------------------------
# Proposal + Diff Check
# ---------------------------------------------------------------------------


class ProposalValidationError(ValueError):
    """A proposal assigns a path outside the closed Mutation Surface, or is
    otherwise malformed."""


class OptimizationProposal(BaseModel):
    """An explicit base plus assignment only to Mutation Surface paths.

    ``base_assignment`` is the explicit base the proposal is diffed against;
    ``assignment`` is the proposed set of variable-path values. Diff Check
    proves every changed path lies within the Mutation Surface.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_assignment: Mapping[str, Any] = Field(default_factory=dict)
    assignment: Mapping[str, Any] = Field(default_factory=dict)


def diff_check(
    proposal: OptimizationProposal,
    *,
    mutation_surface: frozenset[str],
) -> frozenset[str]:
    """Deterministic Diff Check of a proposal against its explicit base.

    Returns the set of changed paths. Raises
    :class:`ProposalValidationError` if any changed path is outside the
    closed Mutation Surface.
    """
    changed = {
        path
        for path in set(proposal.base_assignment) | set(proposal.assignment)
        if proposal.base_assignment.get(path) != proposal.assignment.get(path)
    }
    outside = sorted(changed - mutation_surface)
    if outside:
        joined = ", ".join(repr(path) for path in outside)
        raise ProposalValidationError(
            f"proposal changes path(s) outside the Mutation Surface: {joined}"
        )
    return frozenset(changed)


def validate_proposal(
    proposal: OptimizationProposal,
    *,
    mutation_surface: frozenset[str],
) -> frozenset[str]:
    """Validate a proposal and Diff Check it; returns the changed paths."""
    for path in proposal.assignment:
        if path not in mutation_surface:
            raise ProposalValidationError(
                f"proposal assigns path {path!r} not in the Mutation Surface"
            )
    return diff_check(proposal, mutation_surface=mutation_surface)


# ---------------------------------------------------------------------------
# Fixed-axis / Variance Source expansion
# ---------------------------------------------------------------------------


class VarianceAxis(BaseModel):
    """One fixed experiment axis swept after proposal assignment.

    ``name`` is the axis name; ``values`` are the ordered selections. Each
    combination of axis selections produces one complete slot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    values: tuple[Any, ...]

    @model_validator(mode="after")
    def _validate(self) -> VarianceAxis:
        if not self.values:
            raise ValueError(f"variance axis {self.name!r} has no values")
        return self


def expand_axes(
    axes: Sequence[VarianceAxis],
) -> list[dict[str, Any]]:
    """Expand fixed axes into every ordered combination of axis selections.

    Returns one dict of ``{axis_name: value}`` per complete slot. With no
    axes there is exactly one (empty) slot.
    """
    combinations: list[dict[str, Any]] = [{}]
    for axis in axes:
        combinations = [
            {**partial, axis.name: value}
            for partial in combinations
            for value in axis.values
        ]
    return combinations


# ---------------------------------------------------------------------------
# Materialization Record
# ---------------------------------------------------------------------------


class MaterializationRecord(BaseModel):
    """Immutable provenance record for one successful materialization.

    It is NOT a Config, Identity Document, or variant identity: it has no
    Identity Hash and is stored only by typed Object Reference / Content
    Hash. Multiple records MAY reference one ``graph_hash``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Definition ref / hash (the Rollout Definition's underlying Graph
    # Definition identity).
    definition_ref: StrictStr
    definition_schema_version: int
    # Source lineage: candidate / proposal / base references as applicable.
    source_candidate_ref: StrictStr | None = None
    source_proposal_base: Mapping[str, Any] = Field(default_factory=dict)
    source_proposal_assignment: Mapping[str, Any] = Field(default_factory=dict)
    # Complete validated assignments + fixed-axis selections.
    assignments: Mapping[str, Any] = Field(default_factory=dict)
    axis_selections: Mapping[str, Any] = Field(default_factory=dict)
    # Materializer identity (name, version, implementation reference).
    materializer_name: StrictStr
    materializer_version: StrictStr
    materializer_impl_ref: StrictStr | None = None
    # Resulting Graph Config reference and full graph_hash.
    graph_config_ref: StrictStr
    graph_hash: StrictStr
    # Validation + Diff Check evidence.
    changed_paths: tuple[str, ...] = ()
    diff_check_passed: bool = True
    validation_passed: bool = True
    # Provenance (record-local typed fields).
    provenance_note: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> MaterializationRecord:
        if len(self.graph_hash) != 64:
            raise ValueError("graph_hash must be a full 64-char hash")
        return self

    def record_content(self) -> dict[str, Any]:
        """The complete canonical persisted content (for Content Hash)."""
        return self.model_dump(mode="json")


def materialization_record_reference(
    record: MaterializationRecord,
) -> ObjectReference:
    """The typed Object Reference a Materialization Record resolves under.

    Addressed by Content Hash only; no Identity Hash is ever computed.
    """
    return ObjectReference.for_record(
        MATERIALIZATION_RECORD_SCHEMA, record.record_content()
    )


def store_materialization_record(
    store: ObjectStore,
    record: MaterializationRecord,
) -> ObjectReference:
    """Persist a Materialization Record by typed Object Reference only."""
    reference, _status = store.put(
        MATERIALIZATION_RECORD_SCHEMA, record.record_content()
    )
    return reference


# ---------------------------------------------------------------------------
# Proposal materialization
# ---------------------------------------------------------------------------


class MaterializedSlot(BaseModel):
    """One complete materialized slot: a Graph Config plus its record.

    The Graph Config is transient here (it is addressed by ``graph_hash``);
    the durable artifact is the :class:`MaterializationRecord`.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    graph_config: GraphConfig
    record: MaterializationRecord


def materialize_proposal(
    *,
    definition: GraphDefinition,
    proposal: OptimizationProposal,
    mutation_surface: frozenset[str],
    axes: Sequence[VarianceAxis],
    build_assignments: Any,
    materializer_name: str,
    materializer_version: str,
    definition_ref: str,
    source_candidate_ref: str | None = None,
    materializer_impl_ref: str | None = None,
) -> list[MaterializedSlot]:
    """Validate + Diff Check a proposal, expand every fixed axis, and emit
    one Graph Config plus one Materialization Record per complete slot.

    ``build_assignments`` is a callable ``(proposal_assignment, axis_sel) ->
    Mapping[node_id, Mapping[var, value]]`` producing the per-node Variable
    assignments the Graph Definition needs to ``materialize`` a Graph Config.
    It lets the caller own the mapping from abstract proposal paths to
    concrete Node Variables without this module knowing app-specific paths.
    """
    changed = validate_proposal(proposal, mutation_surface=mutation_surface)
    slots: list[MaterializedSlot] = []
    for axis_selection in expand_axes(axes):
        assignments = build_assignments(proposal.assignment, axis_selection)
        graph_config = definition.materialize(assignments)
        gh = graph_hash(graph_config)
        record = MaterializationRecord(
            definition_ref=definition_ref,
            definition_schema_version=definition.schema_version,
            source_candidate_ref=source_candidate_ref,
            source_proposal_base=dict(proposal.base_assignment),
            source_proposal_assignment=dict(proposal.assignment),
            assignments=dict(assignments),
            axis_selections=dict(axis_selection),
            materializer_name=materializer_name,
            materializer_version=materializer_version,
            materializer_impl_ref=materializer_impl_ref,
            graph_config_ref=gh,
            graph_hash=gh,
            changed_paths=tuple(sorted(changed)),
            diff_check_passed=True,
            validation_passed=True,
        )
        slots.append(
            MaterializedSlot(graph_config=graph_config, record=record)
        )
    return slots


# ---------------------------------------------------------------------------
# Selection Policy
# ---------------------------------------------------------------------------


def select_ordered_subset(
    slots: Sequence[MaterializedSlot],
    *,
    selected_indices: Sequence[int],
) -> list[MaterializedSlot]:
    """Frozen Selection Policy: choose an ordered subset of record slots.

    ``selected_indices`` is an ordered subset of ``range(len(slots))``. The
    policy does not rewrite records; unselected valid slots simply keep their
    records with no selected-lineage mapping. Indices must be a strictly
    ordered subset (ascending, no repeats) so selection is deterministic.
    """
    previous = -1
    for index in selected_indices:
        if not (0 <= index < len(slots)):
            raise ValueError(f"selection index {index} out of range")
        if index <= previous:
            raise ValueError(
                "selection indices must be a strictly ascending ordered subset"
            )
        previous = index
    return [slots[index] for index in selected_indices]


# ---------------------------------------------------------------------------
# Execution planning (dedup by graph_hash)
# ---------------------------------------------------------------------------


class WorkItem(BaseModel):
    """One Rollout Work Item: exactly one Rollout Execution Key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_key: RolloutExecutionKey


class AggregatePlan(BaseModel):
    """One Rollout Aggregate identity: ``(graph_hash, eval_config_hash)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    graph_hash: StrictStr
    eval_config_hash: StrictStr


class ExecutionPlan(BaseModel):
    """The deduplicated execution plan.

    For ``D`` distinct admitted Graph Hashes and one Eval Config, the plan
    has exactly ``D x task_count x repeat_count`` work items and ``D``
    aggregates. Each selected Materialization Record reference maps to its
    ``graph_hash`` (converged records share one work matrix / aggregate) so
    candidate/curve-slot attribution is preserved without duplicate keys.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    admitted_graph_hashes: tuple[str, ...]
    work_items: tuple[WorkItem, ...]
    aggregates: tuple[AggregatePlan, ...]
    # Ordered mapping: selected record reference -> its graph_hash. Preserves
    # separate candidate/curve-slot attribution across convergence.
    record_ref_to_graph_hash: tuple[tuple[str, str], ...]

    @property
    def distinct_admitted_count(self) -> int:
        return len(self.admitted_graph_hashes)


def plan_execution(
    *,
    selected: Sequence[MaterializedSlot],
    selected_refs: Sequence[str],
    eval_config_hash: str,
    context: EvaluationContext,
    task_identities: Sequence[str],
    repeat_ids: Sequence[str],
) -> ExecutionPlan:
    """Deduplicate selected records by full ``graph_hash`` then plan the
    task/repeat matrix.

    ``selected_refs`` are the stored Materialization Record Object Reference
    strings aligned with ``selected`` (one per selected slot); they preserve
    per-record attribution while sharing execution/aggregate evidence.
    """
    if len(selected_refs) != len(selected):
        raise ValueError("selected_refs must align 1:1 with selected slots")

    # Preserve ordered record-ref -> graph_hash mapping (attribution).
    ref_map = tuple(
        (ref, slot.record.graph_hash)
        for ref, slot in zip(selected_refs, selected, strict=True)
    )

    # Deduplicate by full graph_hash, preserving first-seen order.
    admitted: list[str] = []
    seen: set[str] = set()
    for slot in selected:
        gh = slot.record.graph_hash
        if gh not in seen:
            seen.add(gh)
            admitted.append(gh)

    work_items: list[WorkItem] = []
    execution_keys: set[RolloutExecutionKey] = set()
    for gh in admitted:
        for task_identity in task_identities:
            for repeat_id in repeat_ids:
                rollout_key = RolloutKey(
                    graph_hash=gh,
                    eval_config_hash=eval_config_hash,
                    task_identity=task_identity,
                    repeat_id=repeat_id,
                )
                execution_key = rollout_execution_key(
                    rollout_key=rollout_key,
                    context=context,
                )
                if execution_key in execution_keys:
                    raise ValueError(
                        "duplicate Rollout Execution Key produced during "
                        "execution planning"
                    )
                execution_keys.add(execution_key)
                work_items.append(WorkItem(execution_key=execution_key))

    aggregates = tuple(
        AggregatePlan(graph_hash=gh, eval_config_hash=eval_config_hash)
        for gh in admitted
    )

    return ExecutionPlan(
        admitted_graph_hashes=tuple(admitted),
        work_items=tuple(work_items),
        aggregates=aggregates,
        record_ref_to_graph_hash=ref_map,
    )


def distinct_graph_hashes(
    records: Iterable[MaterializationRecord],
) -> list[str]:
    """First-seen-ordered distinct ``graph_hash`` values over records."""
    seen: set[str] = set()
    out: list[str] = []
    for record in records:
        if record.graph_hash not in seen:
            seen.add(record.graph_hash)
            out.append(record.graph_hash)
    return out


__all__ = [
    "MATERIALIZATION_RECORD_SCHEMA",
    "AggregatePlan",
    "ExecutionPlan",
    "MaterializationRecord",
    "MaterializedSlot",
    "OptimizationProposal",
    "ProposalValidationError",
    "VarianceAxis",
    "WorkItem",
    "diff_check",
    "distinct_graph_hashes",
    "expand_axes",
    "materialization_record_reference",
    "materialize_proposal",
    "plan_execution",
    "select_ordered_subset",
    "store_materialization_record",
    "validate_proposal",
]
