from __future__ import annotations

import hashlib
import json
from typing import Any

SHA256_HEX_DIGEST_LENGTH = 64
TEXT_ENCODING = "utf-8"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json_digest(
    value: Any,
    *,
    length: int | None = None,
) -> str:
    digest = hashlib.sha256(
        canonical_json(value).encode(TEXT_ENCODING)
    ).hexdigest()
    if length is None:
        return digest
    if length < 1 or length > SHA256_HEX_DIGEST_LENGTH:
        raise ValueError(
            f"digest length must be between 1 and "
            f"{SHA256_HEX_DIGEST_LENGTH}, got {length}"
        )
    return digest[:length]
