from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import TypedRef, require_full_hash

__all__ = [
    "SelectedRecordMapping",
    "SelectedRecordMappingEntry",
]


class SelectedRecordMappingEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_ref: TypedRef

    graph_hash: StrictStr

    planned_key_set: tuple[str, ...]

    result_key_set: tuple[str, ...]

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

        extra = results - planned
        if extra:
            raise ValueError(
                "result_key_set contains keys not in planned_key_set: "
                f"{sorted(extra)}"
            )
        return self


class SelectedRecordMapping(BaseModel):
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
        return tuple(e for e in self.entries if e.graph_hash == graph_hash)
