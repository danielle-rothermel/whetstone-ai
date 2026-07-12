from __future__ import annotations

# ruff: noqa: E501
import os
from pathlib import Path

import pytest

from whetstone.platform.release_parity_fixture import (
    cleanup,
    prepare,
    verify_evidence,
)


@pytest.mark.integration
def test_release_parity_fixture_prepare_resolve_and_cleanup(
    tmp_path: Path,
) -> None:
    """Exercise the real MotherDuck/Neon boundary only with explicit secrets."""

    required = ("DATABASE_URL", "MOTHERDUCK_DATABASE_URL", "NEON_DATABASE_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip("release-parity credentials are not configured")
    descriptor = tmp_path / "descriptor.json"
    proof = tmp_path / "cleanup-proof.json"
    try:
        prepared = prepare(descriptor)
        prepared.validate_contract()
    finally:
        cleanup(descriptor, proof)
    verify_evidence(descriptor, proof)
