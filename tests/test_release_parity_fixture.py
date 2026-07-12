from __future__ import annotations

import pytest

from whetstone.platform.release_parity_fixture import (
    CleanupProof,
    LocalPlane,
    PinIdentity,
    PlaneDestination,
    ReleaseParityDescriptor,
)
from whetstone.publication import (
    ANALYSIS_BUNDLE_KEY,
    ANALYSIS_MEMBERS,
    DETAIL_BUNDLE_KEY,
    DETAIL_MEMBERS,
)


def _plane(
    members: tuple[str, ...], bundle: str
) -> tuple[LocalPlane, PlaneDestination]:
    counts = {member: 1 for member in members}
    checksums = {member: "a" * 64 for member in members}
    pin = PinIdentity(pin_id="pin", bundle_id="bundle", expires_at_ms=1)
    local = LocalPlane(
        path="fixture.duckdb",
        bundle=bundle,
        pin=pin,
        snapshot_seq=1,
        members={member: f"local_{member}" for member in members},
        member_counts=counts,
        member_checksums=checksums,
    )
    remote = PlaneDestination(
        destination_id=f"{bundle}-destination",
        bundle_key=bundle,
        pin=pin,
        snapshot_seq=1,
        members={member: f"main.remote_{member}" for member in members},
        member_counts=counts,
        member_checksums=checksums,
    )
    return local, remote


def _descriptor() -> ReleaseParityDescriptor:
    analysis_local, analysis_remote = _plane(
        ANALYSIS_MEMBERS, ANALYSIS_BUNDLE_KEY
    )
    detail_local, detail_remote = _plane(DETAIL_MEMBERS, DETAIL_BUNDLE_KEY)
    return ReleaseParityDescriptor(
        schema_version=1,
        run_id="run",
        fixture_sha256="b" * 64,
        source_schema="fixture_schema",
        analysis={"local": analysis_local, "remote": analysis_remote},
        detail={"local": detail_local, "remote": detail_remote},
    )


def test_descriptor_requires_frozen_complete_nonempty_planes() -> None:
    descriptor = _descriptor()
    descriptor.validate_contract()

    broken = descriptor.model_copy(
        update={
            "analysis": {
                "local": descriptor.analysis["local"],
                "remote": descriptor.analysis["remote"].model_copy(
                    update={
                        "member_counts": {
                            member: 0 for member in ANALYSIS_MEMBERS
                        }
                    }
                ),
            }
        }
    )
    with pytest.raises(ValueError, match="empty"):
        broken.validate_contract()


def test_descriptor_rejects_secret_shaped_data() -> None:
    payload = _descriptor().model_dump(mode="json")
    payload["analysis"]["remote"]["destination_id"] = (
        "postgresql://not-allowed"
    )
    with pytest.raises(ValueError, match="URLs"):
        ReleaseParityDescriptor.model_validate(payload).validate_contract()


def test_cleanup_proof_requires_independent_zero_state() -> None:
    descriptor = _descriptor()
    proof = CleanupProof(
        schema_version=1,
        run_id="run",
        source_schema_absent=True,
        local_files_absent=True,
        destinations={
            "whetstone-analysis-destination": {
                "state_rows": 0,
                "bundle_rows": 0,
                "pin_rows": 0,
                "physical_candidates": 0,
            },
            "whetstone-detail-destination": {
                "state_rows": 0,
                "bundle_rows": 0,
                "pin_rows": 0,
                "physical_candidates": 0,
            },
        },
    )
    proof.validate_against(descriptor)
    with pytest.raises(ValueError, match="remaining"):
        proof.model_copy(
            update={
                "destinations": {
                    **proof.destinations,
                    "whetstone-analysis-destination": {"state_rows": 1},
                }
            }
        ).validate_against(descriptor)
