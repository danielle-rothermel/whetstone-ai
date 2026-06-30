from __future__ import annotations

import hashlib
import json
from typing import Any

from dr_dspy.graph.models import GraphSpec

GRAPH_DIGEST_LENGTH = 16
SHA256_HEX_DIGEST_LENGTH = 64
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
    if length < 1 or length > SHA256_HEX_DIGEST_LENGTH:
        raise ValueError(
            f"graph digest length must be between 1 and "
            f"{SHA256_HEX_DIGEST_LENGTH}, got {length}"
        )
    digest = hashlib.sha256(raw.encode(TEXT_ENCODING)).hexdigest()
    return digest[:length]
