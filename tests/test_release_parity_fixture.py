from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from whetstone.platform import release_parity_fixture
from whetstone.platform.connections import DatabaseBoundary
from whetstone.platform.release_parity_fixture import (
    CleanupProof,
    LocalPlane,
    PinIdentity,
    PlaneDestination,
    ReleaseParityDescriptor,
    RunJournal,
    _cleanup_descriptor_or_journal,
    _journal_path,
    _source_url,
    _trace,
    _write_journal,
    verify_evidence,
)
from whetstone.publication import (
    ANALYSIS_BUNDLE_KEY,
    ANALYSIS_MEMBERS,
    DETAIL_BUNDLE_KEY,
    DETAIL_MEMBERS,
)


def _plane(
    members: tuple[str, ...],
    bundle: Literal["whetstone-analysis", "whetstone-detail"],
    name: str,
) -> tuple[LocalPlane, PlaneDestination]:
    counts = {member: 1 for member in members}
    checksums = {member: "a" * 64 for member in members}
    run_id = "a" * 32
    pin = PinIdentity(
        pin_id=f"{run_id}-{name}-local", bundle_id="bundle", expires_at_ms=1
    )
    local = LocalPlane(
        path=f"{run_id}-{name}.duckdb",
        bundle=bundle,
        pin=pin,
        snapshot_seq=1,
        members={member: f"local_{member}" for member in members},
        member_counts=counts,
        member_checksums=checksums,
    )
    remote = PlaneDestination(
        destination_id=f"whetstone-v6-{name}-{run_id}",
        bundle_key=bundle,
        pin=PinIdentity(
            pin_id=f"{run_id}-{name}-remote",
            bundle_id="bundle",
            expires_at_ms=1,
        ),
        snapshot_seq=1,
        members={member: f"main.remote_{member}" for member in members},
        member_counts=counts,
        member_checksums=checksums,
    )
    return local, remote


def _descriptor() -> ReleaseParityDescriptor:
    analysis_local, analysis_remote = _plane(
        ANALYSIS_MEMBERS, ANALYSIS_BUNDLE_KEY, "analysis"
    )
    detail_local, detail_remote = _plane(
        DETAIL_MEMBERS, DETAIL_BUNDLE_KEY, "detail"
    )
    return ReleaseParityDescriptor(
        schema_version=1,
        run_id="a" * 32,
        fixture_sha256="b" * 64,
        fixture_prediction_id=(
            f"release_parity_{'a' * 32}_prediction_small_positive"
        ),
        source_schema=f"whetstone_v6_release_{'a' * 32}",
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


@pytest.mark.parametrize(
    "field, value",
    [
        ("member_checksums", {}),
        ("member_checksums", {"predictions": "not-a-checksum"}),
    ],
)
def test_descriptor_rejects_missing_or_invalid_checksums(
    field: str, value: object
) -> None:
    descriptor = _descriptor()
    broken = descriptor.model_copy(
        update={
            "analysis": {
                "local": descriptor.analysis["local"],
                "remote": descriptor.analysis["remote"].model_copy(
                    update={field: value}
                ),
            }
        }
    )
    with pytest.raises(ValueError):
        broken.validate_contract()


def test_descriptor_rejects_substituted_cleanup_identity() -> None:
    descriptor = _descriptor()
    broken = descriptor.model_copy(
        update={"source_schema": "unrelated_schema"}
    )
    with pytest.raises(ValueError, match="owned"):
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
        run_id="a" * 32,
        source_schema_absent=True,
        local_files_absent=True,
        destinations={
            f"whetstone-v6-analysis-{'a' * 32}": {
                "state_rows": 0,
                "bundle_rows": 0,
                "pin_rows": 0,
                "physical_candidates": 0,
            },
            f"whetstone-v6-detail-{'a' * 32}": {
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
                    f"whetstone-v6-analysis-{'a' * 32}": {
                        "state_rows": 1,
                        "bundle_rows": 0,
                        "pin_rows": 0,
                        "physical_candidates": 0,
                    },
                }
            }
        ).validate_against(descriptor)


def test_missing_descriptor_retains_journal_recovery_authority(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    journal = RunJournal(
        schema_version=1,
        run_id="a" * 32,
        source_schema=f"whetstone_v6_release_{'a' * 32}",
        analysis_path=f"{'a' * 32}-analysis.duckdb",
        detail_path=f"{'a' * 32}-detail.duckdb",
        analysis_destination_id=f"whetstone-v6-analysis-{'a' * 32}",
        detail_destination_id=f"whetstone-v6-detail-{'a' * 32}",
    )
    _write_journal(_journal_path(descriptor_path), journal)
    assert _cleanup_descriptor_or_journal(descriptor_path, journal) is None


def test_recovery_evidence_accepts_a_missing_descriptor_with_zero_proof(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    journal = RunJournal(
        schema_version=1,
        run_id="a" * 32,
        source_schema=f"whetstone_v6_release_{'a' * 32}",
        analysis_path=f"{'a' * 32}-analysis.duckdb",
        detail_path=f"{'a' * 32}-detail.duckdb",
        analysis_destination_id=f"whetstone-v6-analysis-{'a' * 32}",
        detail_destination_id=f"whetstone-v6-detail-{'a' * 32}",
    )
    journal_path = _journal_path(descriptor_path)
    _write_journal(journal_path, journal)
    proof_path = tmp_path / "proof.json"
    zero = {
        "state_rows": 0,
        "bundle_rows": 0,
        "pin_rows": 0,
        "physical_candidates": 0,
    }
    proof = CleanupProof(
        schema_version=1,
        run_id=journal.run_id,
        source_schema_absent=True,
        local_files_absent=True,
        destinations={
            journal.analysis_destination_id: zero,
            journal.detail_destination_id: zero,
        },
    )
    proof_path.write_text(proof.model_dump_json())
    verify_evidence(descriptor_path, proof_path, journal_path)


def test_trace_is_opt_in_and_uses_a_test_owned_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "evidence" / "release-parity.jsonl"
    monkeypatch.setenv("WHETSTONE_RELEASE_PARITY_TRACE_PATH", str(trace_path))

    _trace("fixture_test", run_id="a" * 32, database_url="secret")

    assert json.loads(trace_path.read_text()) == {
        "event": "fixture_test",
        "run_id": "a" * 32,
    }


def test_source_url_preserves_credentials_at_connection_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://fixture:encoded%2Fpassword@db.example/source",
    )

    source_url = _source_url("run_owned")

    assert "fixture:encoded%2Fpassword@db.example" in source_url
    assert "***" not in source_url
    assert "search_path%3Drun_owned%2Cpublic" in source_url


def test_new_source_uses_admin_and_schema_connection_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, DatabaseBoundary, str]] = []
    admin = _EngineStub()
    source = _EngineStub()
    monkeypatch.setenv("DATABASE_URL", "postgresql://operator:p%2Fss@db/test")
    monkeypatch.setattr(
        release_parity_fixture,
        "create_whetstone_engine",
        lambda url, *, boundary: calls.append((str(url), boundary, "default"))
        or (admin if len(calls) == 1 else source),
    )
    monkeypatch.setattr(
        release_parity_fixture.MigrationContext,
        "configure",
        lambda connection: object(),
    )
    monkeypatch.setattr(
        release_parity_fixture, "Operations", lambda _: object()
    )
    migration = type(
        "Migration", (), {"upgrade": staticmethod(lambda: None)}
    )()
    monkeypatch.setattr(
        release_parity_fixture.importlib,
        "import_module",
        lambda _: migration,
    )
    monkeypatch.setattr(
        release_parity_fixture, "ensure_platform_schema", lambda _: None
    )

    assert release_parity_fixture._new_source("run_owned") is source
    assert [boundary for _, boundary, _ in calls] == [
        DatabaseBoundary.SOURCE_ADMIN,
        DatabaseBoundary.SOURCE_SCHEMA,
    ]
    assert calls[1][0].startswith("postgresql+psycopg://")
    assert "search_path%3Drun_owned%2Cpublic" in calls[1][0]


@pytest.mark.parametrize(
    ("kind", "boundary"),
    [
        ("motherduck", DatabaseBoundary.MOTHERDUCK_POSTGRES),
        ("neon", DatabaseBoundary.NEON_POSTGRES),
    ],
)
def test_fence_uses_ephemeral_boundary_engine(
    kind: Literal["motherduck", "neon"],
    boundary: DatabaseBoundary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, DatabaseBoundary, str]] = []
    marker = object()
    monkeypatch.setattr(
        release_parity_fixture,
        "create_whetstone_engine",
        lambda url, *, boundary, pool_mode: calls.append(
            (url, boundary, pool_mode)
        )
        or marker,
    )

    fence = release_parity_fixture._fence(
        "postgresql://operator:p%2Fss@db/test", "destination", kind
    )

    assert fence.engine is marker
    assert calls == [
        ("postgresql://operator:p%2Fss@db/test", boundary, "ephemeral")
    ]


class _ConnectionStub:
    def __enter__(self) -> _ConnectionStub:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, *args: object, **kwargs: object) -> None:
        return None


class _EngineStub:
    def begin(self) -> _ConnectionStub:
        return _ConnectionStub()

    def dispose(self) -> None:
        return None


def test_release_parity_workflow_scopes_credentials_and_pins_actions() -> None:
    workflow = Path(".github/workflows/release-parity.yml").read_text()
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in workflow
    analysis_url = (
        "ANALYSIS_DATABASE_URL: ${{ secrets.MOTHERDUCK_DATABASE_URL }}"
    )
    assert analysis_url in workflow
    assert "DATABASE_URL: ${{ secrets.NEON_DATABASE_URL }}" in workflow
    assert "POSTGRES_USER: whetstone" in Path(
        ".github/workflows/whetstone_tests.yml"
    ).read_text()
    journal = (
        '--journal "$RUNNER_TEMP/release-parity/descriptor.json.journal.json"'
    )
    assert journal in workflow
    action = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    assert action in workflow
    assert workflow.count("actions/upload-artifact@") == 1
    retired_v3_action = (
        "actions/upload-artifact@0b7f8abb1508181956e8e162db84b466c27e18ce"
    )
    assert retired_v3_action not in workflow
    assert "actions/upload-artifact@v3" not in workflow
    assert "WHETSTONE_BUNDLE_INTEGRITY_PRIVATE_KEY:" in workflow
    assert "Materialize ephemeral bundle integrity key" in workflow
    assert "Remove ephemeral bundle integrity key" in workflow
    assert (
        "WHETSTONE_BUNDLE_INTEGRITY_PRIVATE_KEY_PATH: "
        "${{ runner.temp }}/whetstone-integrity/private.pem" in workflow
    )


def test_release_parity_maps_public_integrity_keys_to_unitbench() -> (
    None
):
    workflow = Path(".github/workflows/release-parity.yml").read_text()
    consumer_step = workflow.split(
        "      - name: Unitbench live delivery-parity evidence\n", 1
    )[1].split("      - name: Always clean run-owned fixture\n", 1)[0]

    assert (
        "UNITBENCH_BUNDLE_INTEGRITY_PUBLIC_KEYS: "
        "${{ secrets.WHETSTONE_BUNDLE_INTEGRITY_PUBLIC_KEY_RING }}"
        in consumer_step
    )
    assert "WHETSTONE_BUNDLE_INTEGRITY_PRIVATE_KEY" not in consumer_step
    assert "WHETSTONE_BUNDLE_INTEGRITY_PRIVATE_KEY_PATH" not in consumer_step
    assert "GIT_CONFIG_COUNT" not in consumer_step
    assert "GH_DR_ORG_REPOS_READ_TOKEN" not in consumer_step
    install_step = workflow.split(
        "      - name: Install Unitbench dependencies\n", 1
    )[1].split(
        "      - name: Unitbench live delivery-parity evidence\n", 1
    )[0]
    assert "GIT_CONFIG_COUNT: 1" in install_step
    assert (
        "GIT_CONFIG_KEY_0: url.https://x-access-token:"
        "${{ secrets.GH_DR_ORG_REPOS_READ_TOKEN }}"
        "@github.com/danielle-rothermel/.insteadOf" in install_step
    )
    assert (
        "GIT_CONFIG_VALUE_0: https://github.com/danielle-rothermel/"
        in install_step
    )
