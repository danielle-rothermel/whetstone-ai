from __future__ import annotations

import hashlib
import json
from typing import Any

from dr_dspy.graph.models import GraphSpec

GRAPH_DIGEST_LENGTH = 16
TEXT_ENCODING = "utf-8"


def canonical_graph_payload(graph: GraphSpec) -> dict[str, Any]:
    return {"graph": graph.model_dump(mode="json")}


def graph_digest(
    graph: GraphSpec,
    *,
    length: int = GRAPH_DIGEST_LENGTH,
) -> str:
    raw = json.dumps(
        canonical_graph_payload(graph),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode(TEXT_ENCODING)).hexdigest()
    return digest[:length]
