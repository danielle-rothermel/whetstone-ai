"""The mandatory ordered selected-record -> graph -> keys -> aggregate mapping.

Both the Official Evaluation Record and the Official Plot Manifest MUST
preserve, per the vocabulary, *the ordered mapping from every selected
Materialization Record Object Reference plus Content Hash to its*
``graph_hash`` *, shared planned/result-key set, and aggregate Object
Reference plus Content Hash*.

The load-bearing property is **separate attributability under convergence**:
two selected Materialization Records that share one ``graph_hash`` (converged
assignments) also share one planned/result-key set and one aggregate reference,
yet each keeps its own ordered entry so the two selected records stay
separately attributable to their curve slots / candidates. This module owns
the entry type and the validated ordered container both records embed.
"""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    model_validator,
)

from whetstone.optimization.identity import TypedRef, require_full_hash

__all__ = [
    "SelectedRecordMapping",
    "SelectedRecordMappingEntry",
]


class SelectedRecordMappingEntry(BaseModel):
    """One ordered entry: selected record -> graph -> keys -> aggregate.

    Maps exactly one selected Materialization Record (by typed Object Reference
    plus Content Hash) through its ``graph_hash`` to the shared planned/result
    Rollout Execution Key set and the aggregate reference (typed Object
    Reference plus Content Hash). Two entries that converged on one
    ``graph_hash`` carry the *same* ``graph_hash``, ``planned_key_set``,
    ``result_key_set`` and ``aggregate_ref`` but remain distinct entries keyed
    by their own ``record_ref`` — that is what keeps them separately
    attributable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The selected Materialization Record (typed ref + Content Hash).
    record_ref: TypedRef
    #: The Graph Hash this record materialized to.
    graph_hash: StrictStr
    #: The shared planned Rollout Execution Key set (canonical strings).
    planned_key_set: tuple[str, ...]
    #: The shared result-bound Rollout Execution Key set (subset of planned).
    result_key_set: tuple[str, ...]
    #: The aggregate this graph produced (typed ref + Content Hash).
    aggregate_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> SelectedRecordMappingEntry:
        require_full_hash(self.graph_hash, field="graph_hash")
        if not self.planned_key_set:
            raise ValueError("planned_key_set must be non-empty")
        planned = set(self.planned_key_set)
        if len(planned) != len(self.planned_key_set):
            raise ValueError("planned_key_set must have no duplicates")
        results = set(self.result_key_set)
        if len(results) != len(self.result_key_set):
            raise ValueError("result_key_set must have no duplicates")
        # Every result key MUST be a planned key: an official record can never
        # attribute a result to a graph it did not plan.
        extra = results - planned
        if extra:
            raise ValueError(
                "result_key_set contains keys not in planned_key_set: "
                f"{sorted(extra)}"
            )
        return self


class SelectedRecordMapping(BaseModel):
    """The mandatory ordered mapping preserved by official records/manifests.

    An ordered tuple of :class:`SelectedRecordMappingEntry`, one per selected
    Materialization Record, in selection order. Record references are unique
    (each selected record appears exactly once); ``graph_hash`` /
    ``aggregate_ref`` MAY repeat across entries (convergence), and when they do
    the shared entries MUST agree on the planned/result key sets and aggregate
    reference so a shared graph is never given two conflicting attributions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[SelectedRecordMappingEntry, ...]

    @model_validator(mode="after")
    def _validate(self) -> SelectedRecordMapping:
        if not self.entries:
            raise ValueError("the ordered mapping must have >=1 entry")

        record_refs = [e.record_ref.content_hash for e in self.entries]
        if len(set(record_refs)) != len(record_refs):
            raise ValueError(
                "each selected Materialization Record must appear exactly "
                "once in the ordered mapping"
            )

        # Convergence consistency: entries sharing a graph_hash MUST share the
        # same planned key set, result key set, and aggregate reference.
        by_graph: dict[str, SelectedRecordMappingEntry] = {}
        for entry in self.entries:
            prior = by_graph.get(entry.graph_hash)
            if prior is None:
                by_graph[entry.graph_hash] = entry
                continue
            if prior.planned_key_set != entry.planned_key_set:
                raise ValueError(
                    f"entries sharing graph_hash {entry.graph_hash} disagree "
                    "on planned_key_set"
                )
            if prior.result_key_set != entry.result_key_set:
                raise ValueError(
                    f"entries sharing graph_hash {entry.graph_hash} disagree "
                    "on result_key_set"
                )
            if prior.aggregate_ref != entry.aggregate_ref:
                raise ValueError(
                    f"entries sharing graph_hash {entry.graph_hash} disagree "
                    "on aggregate_ref"
                )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def distinct_graph_hashes(self) -> tuple[str, ...]:
        """First-seen-ordered distinct graph hashes across the mapping."""
        seen: set[str] = set()
        out: list[str] = []
        for entry in self.entries:
            if entry.graph_hash not in seen:
                seen.add(entry.graph_hash)
                out.append(entry.graph_hash)
        return tuple(out)

    def entries_for_graph(
        self, graph_hash: str
    ) -> tuple[SelectedRecordMappingEntry, ...]:
        """Every ordered entry attributed to one ``graph_hash``.

        More than one entry here means multiple selected records converged on
        that graph and stay separately attributable through their own
        ``record_ref``.
        """
        return tuple(e for e in self.entries if e.graph_hash == graph_hash)
