from __future__ import annotations

from typing import Any

from dr_serialize import sha256_json_digest

from whetstone.graph.models import GraphSpec

GRAPH_DIGEST_LENGTH = 16


def canonical_graph_payload(graph: GraphSpec) -> dict[str, Any]:
    return {"graph": graph.model_dump(mode="json")}


def graph_digest(
    graph: GraphSpec,
    *,
    length: int = GRAPH_DIGEST_LENGTH,
) -> str:
    return sha256_json_digest(
        canonical_graph_payload(graph),
        length=length,
    )
