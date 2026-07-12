"""Fail-closed operator configuration for published bundle integrity."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dr_platform.publication import OpenSslEd25519Signer

INTEGRITY_KEY_ID_ENV = "WHETSTONE_BUNDLE_INTEGRITY_KEY_ID"
INTEGRITY_PRIVATE_KEY_PATH_ENV = "WHETSTONE_BUNDLE_INTEGRITY_PRIVATE_KEY_PATH"
INTEGRITY_PUBLIC_KEY_RING_ENV = "WHETSTONE_BUNDLE_INTEGRITY_PUBLIC_KEY_RING"


@dataclass(frozen=True)
class BundleIntegrityConfiguration:
    signer: OpenSslEd25519Signer
    public_key_ring: Mapping[str, str]


def required_bundle_integrity_configuration() -> BundleIntegrityConfiguration:
    """Load the operator-owned signing boundary without secret fallbacks."""

    key_id = _required_env(INTEGRITY_KEY_ID_ENV)
    private_key_path = Path(_required_env(INTEGRITY_PRIVATE_KEY_PATH_ENV))
    if not private_key_path.is_file():
        raise ValueError(f"{INTEGRITY_PRIVATE_KEY_PATH_ENV} must name a file")
    encoded_ring = _required_env(INTEGRITY_PUBLIC_KEY_RING_ENV)
    try:
        decoded_ring = json.loads(encoded_ring)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{INTEGRITY_PUBLIC_KEY_RING_ENV} must be a JSON object"
        ) from exc
    if not isinstance(decoded_ring, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in decoded_ring.items()
    ):
        raise ValueError(
            f"{INTEGRITY_PUBLIC_KEY_RING_ENV} must map non-empty key IDs "
            "to public keys"
        )
    if key_id not in decoded_ring:
        raise ValueError(
            f"{INTEGRITY_PUBLIC_KEY_RING_ENV} must include "
            f"{INTEGRITY_KEY_ID_ENV}"
        )
    return BundleIntegrityConfiguration(
        signer=OpenSslEd25519Signer(
            key_id=key_id, private_key_path=private_key_path
        ),
        public_key_ring=decoded_ring,
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value
