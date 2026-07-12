"""Disposable, credential-free evidence for the v6 release parity gate.

The descriptor is deliberately a reader contract: it identifies a pinned
publication but never serializes a connection string, credential, or fixture
payload.  The command obtains all connection strings from its environment.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from urllib.parse import quote

from alembic.migration import MigrationContext
from alembic.operations import Operations
from dr_platform import (
    BundlePin,
    ExportReconciliationDependencies,
    PinnedBundle,
    PostgresPublicationFence,
    pin_local_bundle,
    resolve_local_pin,
)
from dr_platform.reconciliation_runtime import ReconcileOptions
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.pool import NullPool

from whetstone.platform.integrity import (
    BundleIntegrityConfiguration,
    required_bundle_integrity_configuration,
)
from whetstone.platform.platform_db import ensure_platform_schema
from whetstone.platform.targets import target_registry
from whetstone.publication import (
    ANALYSIS_BUNDLE_KEY,
    ANALYSIS_MEMBERS,
    DETAIL_BUNDLE_KEY,
    DETAIL_MEMBERS,
    export_whetstone,
)

SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET = re.compile(r"(token|password|secret|dsn|url|authorization)", re.I)
_TRACE = Path(
    "/private/tmp/platform-v6-headless-whetstone-parity-v2-99.trace.txt"
)

NonEmpty = Annotated[StrictStr, Field(min_length=1)]


def _trace(event: str, **facts: object) -> None:
    """Write only stable, non-sensitive operational facts."""
    safe = {
        key: value for key, value in facts.items() if not _SECRET.search(key)
    }
    with _TRACE.open("a") as stream:
        stream.write(
            json.dumps({"event": event, **safe}, sort_keys=True) + "\n"
        )


class PinIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pin_id: NonEmpty
    bundle_id: NonEmpty
    expires_at_ms: StrictInt = Field(ge=0)


class PlaneDestination(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    destination_id: NonEmpty
    bundle_key: Literal["whetstone-analysis", "whetstone-detail"]
    pin: PinIdentity
    snapshot_seq: StrictInt = Field(ge=0)
    members: Mapping[NonEmpty, NonEmpty]
    member_counts: Mapping[NonEmpty, StrictInt]
    member_checksums: Mapping[NonEmpty, NonEmpty]


class LocalPlane(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: NonEmpty
    bundle: NonEmpty
    pin: PinIdentity
    snapshot_seq: StrictInt = Field(ge=0)
    members: Mapping[NonEmpty, NonEmpty]
    member_counts: Mapping[NonEmpty, StrictInt]
    member_checksums: Mapping[NonEmpty, NonEmpty]


PlaneMap = Mapping[Literal["local", "remote"], LocalPlane | PlaneDestination]


class ReleaseParityDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    run_id: NonEmpty
    fixture_sha256: NonEmpty
    source_schema: NonEmpty
    analysis: PlaneMap
    detail: PlaneMap

    def validate_contract(self) -> None:
        _reject_secrets(self.model_dump(mode="json"))
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id is not a generated run identity")
        if _SHA256.fullmatch(self.fixture_sha256) is None:
            raise ValueError("fixture_sha256 must be a SHA-256 checksum")
        if self.source_schema != f"whetstone_v6_release_{self.run_id}":
            raise ValueError("source_schema is not owned by this run")
        for name, expected, bundle in (
            ("analysis", ANALYSIS_MEMBERS, ANALYSIS_BUNDLE_KEY),
            ("detail", DETAIL_MEMBERS, DETAIL_BUNDLE_KEY),
        ):
            planes = getattr(self, name)
            if set(planes) != {"local", "remote"}:
                raise ValueError(
                    f"{name} must contain local and remote planes"
                )
            local, remote = planes["local"], planes["remote"]
            for field in ("members", "member_counts", "member_checksums"):
                if set(getattr(local, field)) != set(expected) or set(
                    getattr(remote, field)
                ) != set(expected):
                    raise ValueError(f"{name} {field} inventory is not frozen")
            if local.bundle != bundle or remote.bundle_key != bundle:
                raise ValueError(f"{name} bundle key is not frozen")
            if local.path != f"{self.run_id}-{name}.duckdb":
                raise ValueError(f"{name} local path is not owned by this run")
            if remote.destination_id != f"whetstone-v6-{name}-{self.run_id}":
                raise ValueError(
                    f"{name} destination is not owned by this run"
                )
            for pin, suffix in ((local.pin, "local"), (remote.pin, "remote")):
                if pin.pin_id != f"{self.run_id}-{name}-{suffix}":
                    raise ValueError(f"{name} pin is not owned by this run")
            if local.pin.bundle_id != remote.pin.bundle_id:
                raise ValueError(
                    f"{name} local and remote bundle identities differ"
                )
            if local.snapshot_seq != remote.snapshot_seq:
                raise ValueError(f"{name} local and remote snapshots differ")
            if any(
                value <= 0
                for value in (
                    *local.member_counts.values(),
                    *remote.member_counts.values(),
                )
            ):
                raise ValueError(f"{name} contains an empty member")
            if any(
                _SHA256.fullmatch(value) is None
                for value in (
                    *local.member_checksums.values(),
                    *remote.member_checksums.values(),
                )
            ):
                raise ValueError(f"{name} contains a non-SHA-256 checksum")


class CleanupProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    run_id: NonEmpty
    source_schema_absent: bool
    local_files_absent: bool
    destinations: Mapping[NonEmpty, Mapping[NonEmpty, StrictInt]]

    def validate_against(self, descriptor: ReleaseParityDescriptor) -> None:
        _reject_secrets(self.model_dump(mode="json"))
        if self.run_id != descriptor.run_id:
            raise ValueError("cleanup proof belongs to another run")
        if not self.source_schema_absent or not self.local_files_absent:
            raise ValueError("cleanup proof is not zero-state")
        expected = {
            _remote(descriptor.analysis).destination_id,
            _remote(descriptor.detail).destination_id,
        }
        if set(self.destinations) != expected:
            raise ValueError("cleanup proof misses a remote destination")
        expected_facts = {
            "state_rows",
            "bundle_rows",
            "pin_rows",
            "physical_candidates",
        }
        if any(
            set(facts) != expected_facts
            for facts in self.destinations.values()
        ):
            raise ValueError(
                "cleanup proof has an incomplete catalog observation"
            )
        if any(
            value != 0
            for facts in self.destinations.values()
            for value in facts.values()
        ):
            raise ValueError("cleanup proof reports remaining remote state")


def _reject_secrets(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET.search(str(key)):
                raise ValueError(
                    "descriptor/proof may not contain secret-shaped fields"
                )
            _reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item)
    elif isinstance(value, str) and "://" in value:
        raise ValueError("descriptor/proof may not contain URLs")


def load_descriptor(path: Path) -> ReleaseParityDescriptor:
    descriptor = ReleaseParityDescriptor.model_validate_json(path.read_text())
    descriptor.validate_contract()
    return descriptor


class RunJournal(BaseModel):
    """Durable, secret-free recovery authority written before any resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    run_id: NonEmpty
    source_schema: NonEmpty
    analysis_path: NonEmpty
    detail_path: NonEmpty
    analysis_destination_id: NonEmpty
    detail_destination_id: NonEmpty
    analysis_bundle_id: str = ""
    detail_bundle_id: str = ""

    def validate_contract(self) -> None:
        _reject_secrets(self.model_dump(mode="json"))
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("journal run_id is invalid")
        expected = {
            "source_schema": f"whetstone_v6_release_{self.run_id}",
            "analysis_path": f"{self.run_id}-analysis.duckdb",
            "detail_path": f"{self.run_id}-detail.duckdb",
            "analysis_destination_id": f"whetstone-v6-analysis-{self.run_id}",
            "detail_destination_id": f"whetstone-v6-detail-{self.run_id}",
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("journal contains an unrelated cleanup identity")


def _journal_path(descriptor_path: Path) -> Path:
    return descriptor_path.with_name(descriptor_path.name + ".journal.json")


def _write_journal(path: Path, journal: RunJournal) -> None:
    journal.validate_contract()
    path.write_text(journal.model_dump_json(indent=2))


def _load_journal(descriptor_path: Path) -> RunJournal:
    journal = RunJournal.model_validate_json(
        _journal_path(descriptor_path).read_text()
    )
    journal.validate_contract()
    return journal


def _descriptor_matches_journal(
    descriptor: ReleaseParityDescriptor, journal: RunJournal
) -> None:
    if (
        descriptor.run_id != journal.run_id
        or descriptor.source_schema != journal.source_schema
        or _local(descriptor.analysis).path != journal.analysis_path
        or _local(descriptor.detail).path != journal.detail_path
        or _remote(descriptor.analysis).destination_id
        != journal.analysis_destination_id
        or _remote(descriptor.detail).destination_id
        != journal.detail_destination_id
        or (
            journal.analysis_bundle_id
            and _remote(descriptor.analysis).pin.bundle_id
            != journal.analysis_bundle_id
        )
        or (
            journal.detail_bundle_id
            and _remote(descriptor.detail).pin.bundle_id
            != journal.detail_bundle_id
        )
    ):
        raise ValueError("descriptor does not match its run journal")


def _remote(descriptor_plane: PlaneMap) -> PlaneDestination:
    value = descriptor_plane["remote"]
    if not isinstance(value, PlaneDestination):
        raise ValueError("remote plane has an invalid shape")
    return value


def _local(descriptor_plane: PlaneMap) -> LocalPlane:
    value = descriptor_plane["local"]
    if not isinstance(value, LocalPlane):
        raise ValueError("local plane has an invalid shape")
    return value


class _UnusedQueueLookup:
    def retrieve_queue(self, name: str) -> object | None:
        raise AssertionError(f"unexpected queue lookup: {name}")


class _UnusedLifecycleReader:
    def observe(self, *, workflow_id: str) -> object:
        raise AssertionError(f"unexpected lifecycle read: {workflow_id}")

    def read_step_history(
        self, *, workflow_id: str, limit: int = 100
    ) -> object:
        raise AssertionError(f"unexpected step history: {workflow_id}")


def _reconciliation(source: Engine) -> ExportReconciliationDependencies:
    return ExportReconciliationDependencies(
        resolver=target_registry(),
        queue_lookup=_UnusedQueueLookup(),
        reader=cast(Any, _UnusedLifecycleReader()),
        dbos_engine=source,
        options=ReconcileOptions(page_size=100),
        max_cycles=2,
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _source_url(schema: str) -> str:
    base = make_url(_required_env("DATABASE_URL"))
    return str(
        base.update_query_dict(
            {"options": quote(f"-csearch_path={schema},public", safe="")}
        )
    )


def _new_source(schema: str) -> Engine:
    admin = create_engine(_required_env("DATABASE_URL"))
    try:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            migration = cast(
                Any,
                importlib.import_module(
                    "whetstone.db.migrations.versions.20260712_0001_whetstone_baseline"
                ),
            )
            migration.op = Operations(MigrationContext.configure(connection))
            migration.upgrade()
    finally:
        admin.dispose()
    source_url = _source_url(schema)
    ensure_platform_schema(source_url)
    return create_engine(source_url)


def _seed_accepted_fixture(source: Engine, run_id: str) -> str:
    """Seed one accepted, non-empty graph result; values never leave the DB."""
    now = datetime.now(UTC)
    fixture = f"release_parity_{run_id}"
    values = {"fixture": fixture, "now": now}
    with source.begin() as connection:
        connection.execute(
            text(
                """
            INSERT INTO whetstone_operations (
                operation_key, group_key, workflow_role, status, requested_count,
                manifest_version, manifest_digest, manifest_page_size,
                manifest_page_count, operation_execution_recipe_digest,
                target_key, target_version, target_contract_digest,
                platform_cut_version, registration_cursor, retry_policy,
                inserted_count, already_present_count, enqueued_count,
                workflow_already_present_count, enqueue_failed_count,
                active_count, succeeded_count, terminal_failed_count,
                cancelled_count, spec, metadata, created_at,
                registration_completed_at, updated_at, completed_at, change_seq
            ) VALUES (
                :fixture || '_operation', :fixture, 'generation', 'succeeded',
                0, 3, 'manifest', 1, 0, 'recipe', 'target', 1, 'contract',
                1, 0, '{}'::jsonb, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                '{}'::jsonb, '{}'::jsonb, :now, :now, :now, :now, 1
            )
        """
            ),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_experiments (experiment_name, config_metadata, acceptance_source_version, current_acceptance_id, acceptance_updated_at, created_at)
            VALUES (:fixture, '{"experiment_kind":"humaneval_encdec"}'::jsonb, 1, :fixture || '_acceptance', :now, :now)
        """),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_prediction_specs (prediction_id, experiment_name, task_id, repetition_seed, graph_digest, dimensions_digest, graph_layout, provider_kind, endpoint_kind, model, throttle_key, task_snapshot, graph_snapshot, dimensions, provider_configs, created_at)
            VALUES (:fixture || '_prediction', :fixture, 'HumanEval/0', 0, 'graph', 'dimensions', 'layout', 'openai', 'responses', 'fixture-model', 'fixture-throttle', '{"kind":"code"}'::jsonb, '{"prompt":"complete"}'::jsonb, '{}'::jsonb, '{}'::jsonb, :now)
        """),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_generation_runs (generation_run_id, prediction_id, attempt_index, execution_recipe_digest, platform_item_id, platform_attempt, status, terminal_node_id, terminal_output_node_id, summary, started_at, completed_at)
            VALUES (:fixture || '_generation', :fixture || '_prediction', 0, 'recipe', :fixture || '_item', 0, 'success', 'terminal', 'terminal', '{"terminal_output":"return a+b"}'::jsonb, :now, :now)
        """),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_node_attempts (node_attempt_id, generation_run_id, prediction_id, node_id, attempt_index, status, provider_kind, endpoint_kind, model, throttle_key, provider_config, output, usage_cost, response_metadata, started_at, completed_at)
            VALUES (:fixture || '_node', :fixture || '_generation', :fixture || '_prediction', 'terminal', 0, 'success', 'openai', 'responses', 'fixture-model', 'fixture-throttle', '{}'::jsonb, '{"value":"return a+b"}'::jsonb, '{"provider_cost":0.125}'::jsonb, '{}'::jsonb, :now, :now)
        """),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_score_attempts (score_attempt_id, prediction_id, generation_run_id, attempt_index, execution_recipe_digest, platform_item_id, platform_attempt, scoring_profile_id, scoring_profile_version, parser_profile_id, parser_version, dataset_name, dataset_split, dataset_snapshot, status, submission_outcome, score, extracted_submission, metrics, per_test_results, started_at, completed_at)
            VALUES (:fixture || '_score', :fixture || '_prediction', :fixture || '_generation', 0, 'score', :fixture || '_score_item', 0, 'humaneval', '1', 'python', '1', 'humaneval', 'test', '{}'::jsonb, 'success', 'passed', 1, '{"code":"return a+b"}'::jsonb, '{"realized_compression_ratio":0.5}'::jsonb, '[]'::jsonb, :now, :now)
        """),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_experiment_acceptance_evaluations (acceptance_id, experiment_name, acceptance_source_version, status, generation_operation_key, generation_manifest_digest, scoring_relationships, scoring_relationships_digest, selected_scoring_candidates, selected_scoring_candidates_digest, domain_cut, domain_cut_digest, platform_cut, platform_cut_digest, required_profiles, required_profiles_digest, policy, policy_digest, observed_matrix, observed_matrix_digest, expected_count, accepted_count, missing_count, rejected_count, created_at)
            VALUES (:fixture || '_acceptance', :fixture, 1, 'ACCEPTED', 'operation', 'digest', '[]'::jsonb, 'digest', '[]'::jsonb, 'digest', '{}'::jsonb, 'digest', jsonb_build_array(jsonb_build_object('operation_key', :fixture || '_operation', 'platform_cut_version', 1)), 'digest', '[]'::jsonb, 'digest', '{}'::jsonb, 'digest', '{}'::jsonb, 'digest', 1, 1, 0, 0, :now)
        """),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_experiment_acceptance_generation_members (acceptance_id, prediction_id, disposition, generation_run_id, generation_operation_key, platform_item_id, platform_attempt)
            VALUES (:fixture || '_acceptance', :fixture || '_prediction', 'selected_success', :fixture || '_generation', 'operation', :fixture || '_item', 0)
        """),
            values,
        )
        connection.execute(
            text("""
            INSERT INTO whetstone_experiment_acceptance_scoring_members (acceptance_id, prediction_id, scoring_profile_id, scoring_profile_version, parser_profile_id, parser_version, dataset_name, dataset_split, disposition, generation_run_id, score_attempt_id, accepted_scoring_ordinal)
            VALUES (:fixture || '_acceptance', :fixture || '_prediction', 'humaneval', '1', 'python', '1', 'humaneval', 'test', 'accepted', :fixture || '_generation', :fixture || '_score', 1)
        """),
            values,
        )
    return hashlib.sha256(fixture.encode()).hexdigest()


def _fence(
    url: str,
    destination_id: str,
    kind: Literal["motherduck", "neon"],
    *,
    integrity: BundleIntegrityConfiguration | None = None,
) -> PostgresPublicationFence:
    return PostgresPublicationFence(
        create_engine(url, poolclass=NullPool),
        destination_id=destination_id,
        kind=kind,
        signer=integrity.signer if integrity else None,
        public_key_ring=integrity.public_key_ring if integrity else {},
    )


def _pin_identity(pin: BundlePin) -> PinIdentity:
    return PinIdentity(
        pin_id=pin.pin_id,
        bundle_id=pin.bundle_id,
        expires_at_ms=pin.expires_at_ms,
    )


def _plane(
    result: Any,
    pin: BundlePin,
    pinned: PinnedBundle,
    destination_id: str,
    bundle_key: Literal["whetstone-analysis", "whetstone-detail"],
) -> PlaneDestination:
    if (
        pinned.bundle_id != pin.bundle_id
        or pinned.snapshot_seq != result.snapshot_seq
    ):
        raise ValueError("fresh pin resolution disagrees with publication")
    return PlaneDestination(
        destination_id=destination_id,
        bundle_key=bundle_key,
        pin=_pin_identity(pin),
        snapshot_seq=pinned.snapshot_seq,
        members=pinned.members,
        member_counts=result.member_counts,
        member_checksums=result.member_checksums,
    )


def prepare(descriptor_path: Path) -> ReleaseParityDescriptor:
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    schema = f"whetstone_v6_release_{run_id}"
    analysis_path = descriptor_path.parent / f"{run_id}-analysis.duckdb"
    detail_path = descriptor_path.parent / f"{run_id}-detail.duckdb"
    analysis_id, detail_id = (
        f"whetstone-v6-analysis-{run_id}",
        f"whetstone-v6-detail-{run_id}",
    )
    journal_path = _journal_path(descriptor_path)
    journal = RunJournal(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        source_schema=schema,
        analysis_path=analysis_path.name,
        detail_path=detail_path.name,
        analysis_destination_id=analysis_id,
        detail_destination_id=detail_id,
    )
    # This is deliberately first: every later boundary has durable recovery
    # authority even if descriptor serialization never happens.
    _write_journal(journal_path, journal)
    source: Engine | None = None
    analysis_fence: PostgresPublicationFence | None = None
    detail_fence: PostgresPublicationFence | None = None
    try:
        integrity = required_bundle_integrity_configuration()
        source = _new_source(schema)
        analysis_fence = _fence(
            _required_env("MOTHERDUCK_DATABASE_URL"),
            analysis_id,
            "motherduck",
            integrity=integrity,
        )
        detail_fence = _fence(
            _required_env("NEON_DATABASE_URL"),
            detail_id,
            "neon",
            integrity=integrity,
        )
        fixture_sha256 = _seed_accepted_fixture(source, run_id)
        analysis_fence.ensure_schema()
        detail_fence.ensure_schema()
        analysis, detail = export_whetstone(
            source,
            reconciliation=_reconciliation(source),
            integrity_signer=integrity.signer,
            destination_path=analysis_path,
            detail_destination_path=detail_path,
            analysis_remote_destinations=(analysis_fence,),
            detail_remote_destinations=(detail_fence,),
        )
        journal = journal.model_copy(
            update={
                "analysis_bundle_id": str(getattr(analysis, "bundle_id", "")),
                "detail_bundle_id": str(getattr(detail, "bundle_id", "")),
            }
        )
        _write_journal(journal_path, journal)
        if [item.status for item in analysis.destinations] != [
            "PROMOTED",
            "PROMOTED",
        ] or [item.status for item in detail.destinations] != [
            "PROMOTED",
            "PROMOTED",
        ]:
            raise RuntimeError("publication did not promote every destination")
        analysis_local_pin = pin_local_bundle(
            analysis_path,
            bundle_key=ANALYSIS_BUNDLE_KEY,
            pin_id=f"{run_id}-analysis-local",
        )
        detail_local_pin = pin_local_bundle(
            detail_path,
            bundle_key=DETAIL_BUNDLE_KEY,
            pin_id=f"{run_id}-detail-local",
        )
        analysis_remote_pin = analysis_fence.pin_bundle(
            bundle_key=ANALYSIS_BUNDLE_KEY, pin_id=f"{run_id}-analysis-remote"
        )
        detail_remote_pin = detail_fence.pin_bundle(
            bundle_key=DETAIL_BUNDLE_KEY, pin_id=f"{run_id}-detail-remote"
        )
        descriptor = ReleaseParityDescriptor(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            fixture_sha256=fixture_sha256,
            source_schema=schema,
            analysis={
                "local": LocalPlane(
                    path=analysis_path.name,
                    bundle=ANALYSIS_BUNDLE_KEY,
                    pin=_pin_identity(analysis_local_pin),
                    snapshot_seq=resolve_local_pin(
                        analysis_path,
                        analysis_local_pin,
                        public_key_ring=integrity.public_key_ring,
                    ).snapshot_seq,
                    members=resolve_local_pin(
                        analysis_path,
                        analysis_local_pin,
                        public_key_ring=integrity.public_key_ring,
                    ).members,
                    member_counts=analysis.member_counts,
                    member_checksums=analysis.member_checksums,
                ),
                "remote": _plane(
                    analysis,
                    analysis_remote_pin,
                    _fence(
                        _required_env("MOTHERDUCK_DATABASE_URL"),
                        analysis_id,
                        "motherduck",
                        integrity=integrity,
                    ).resolve_pin(analysis_remote_pin),
                    analysis_id,
                    ANALYSIS_BUNDLE_KEY,
                ),
            },
            detail={
                "local": LocalPlane(
                    path=detail_path.name,
                    bundle=DETAIL_BUNDLE_KEY,
                    pin=_pin_identity(detail_local_pin),
                    snapshot_seq=resolve_local_pin(
                        detail_path,
                        detail_local_pin,
                        public_key_ring=integrity.public_key_ring,
                    ).snapshot_seq,
                    members=resolve_local_pin(
                        detail_path,
                        detail_local_pin,
                        public_key_ring=integrity.public_key_ring,
                    ).members,
                    member_counts=detail.member_counts,
                    member_checksums=detail.member_checksums,
                ),
                "remote": _plane(
                    detail,
                    detail_remote_pin,
                    _fence(
                        _required_env("NEON_DATABASE_URL"),
                        detail_id,
                        "neon",
                        integrity=integrity,
                    ).resolve_pin(detail_remote_pin),
                    detail_id,
                    DETAIL_BUNDLE_KEY,
                ),
            },
        )
        descriptor.validate_contract()
        descriptor_path.write_text(descriptor.model_dump_json(indent=2))
        journal = journal.model_copy(
            update={
                "analysis_bundle_id": _remote(
                    descriptor.analysis
                ).pin.bundle_id,
                "detail_bundle_id": _remote(descriptor.detail).pin.bundle_id,
            }
        )
        _write_journal(journal_path, journal)
        _trace(
            "prepare_succeeded",
            run_id=run_id,
            analysis_members=len(ANALYSIS_MEMBERS),
            detail_members=len(DETAIL_MEMBERS),
        )
        return descriptor
    except Exception:
        _rollback_prepare(journal, descriptor_path.parent)
        raise
    finally:
        if source is not None:
            source.dispose()
        if analysis_fence is not None:
            analysis_fence.engine.dispose()
        if detail_fence is not None:
            detail_fence.engine.dispose()


def _rollback_prepare(journal: RunJournal, directory: Path) -> None:
    """Best-effort rollback retains the journal for a later always-cleanup."""
    for name in (journal.analysis_path, journal.detail_path):
        path = directory / name
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".lock").unlink(missing_ok=True)
    for url_name, destination, bundle, bundle_id, kind in (
        (
            "MOTHERDUCK_DATABASE_URL",
            journal.analysis_destination_id,
            ANALYSIS_BUNDLE_KEY,
            journal.analysis_bundle_id,
            "motherduck",
        ),
        (
            "NEON_DATABASE_URL",
            journal.detail_destination_id,
            DETAIL_BUNDLE_KEY,
            journal.detail_bundle_id,
            "neon",
        ),
    ):
        if bundle_id and os.environ.get(url_name):
            try:
                fence = _fence(_required_env(url_name), destination, kind)
                try:
                    with fence.engine.begin() as connection:
                        manifest = connection.execute(
                            text(
                                f"SELECT manifest_json FROM {fence._bundles_table} WHERE destination_id=:destination AND bundle_key=:bundle AND bundle_id=:bundle_id"
                            ),
                            {
                                "destination": destination,
                                "bundle": bundle,
                                "bundle_id": bundle_id,
                            },
                        ).scalar_one_or_none()
                        if manifest is not None:
                            for facts in json.loads(str(manifest)).get(
                                "members", []
                            ):
                                schema, table = (
                                    facts["schema_name"],
                                    facts["table_name"],
                                )
                                if _IDENTIFIER.fullmatch(
                                    schema
                                ) and _IDENTIFIER.fullmatch(table):
                                    connection.execute(
                                        text(
                                            f'DROP TABLE IF EXISTS "{schema}"."{table}"'
                                        )
                                    )
                        params = {
                            "destination": destination,
                            "bundle": bundle,
                            "bundle_id": bundle_id,
                        }
                        connection.execute(
                            text(
                                f"DELETE FROM {fence._pins_table} WHERE destination_id=:destination AND bundle_key=:bundle AND bundle_id=:bundle_id"
                            ),
                            params,
                        )
                        connection.execute(
                            text(
                                f"DELETE FROM {fence._bundles_table} WHERE destination_id=:destination AND bundle_key=:bundle AND bundle_id=:bundle_id"
                            ),
                            params,
                        )
                        connection.execute(
                            text(
                                f"DELETE FROM {fence._table} WHERE destination_id=:destination AND bundle_key=:bundle AND bundle_id=:bundle_id"
                            ),
                            params,
                        )
                finally:
                    fence.engine.dispose()
            except Exception:
                _trace(
                    "prepare_remote_rollback_deferred", run_id=journal.run_id
                )
    try:
        admin = create_engine(_required_env("DATABASE_URL"))
        try:
            with admin.begin() as connection:
                connection.execute(
                    text(
                        f'DROP SCHEMA IF EXISTS "{journal.source_schema}" CASCADE'
                    )
                )
        finally:
            admin.dispose()
    except Exception:
        # The journal is intentionally retained as evidence and recovery input.
        _trace("prepare_rollback_deferred", run_id=journal.run_id)


def _delete_remote(
    fence: PostgresPublicationFence, plane: PlaneDestination
) -> Mapping[str, int]:
    members = plane.members
    if not members:
        with fence.engine.connect() as connection:
            manifest = connection.execute(
                text(
                    f"SELECT manifest_json FROM {fence._bundles_table} WHERE destination_id=:destination AND bundle_key=:bundle AND bundle_id=:bundle_id"
                ),
                {
                    "destination": plane.destination_id,
                    "bundle": plane.bundle_key,
                    "bundle_id": plane.pin.bundle_id,
                },
            ).scalar_one_or_none()
        members = (
            {
                str(
                    item["table_name"]
                ): f"{item['schema_name']}.{item['table_name']}"
                for item in json.loads(str(manifest)).get("members", [])
            }
            if manifest is not None
            else {}
        )
    for member in members.values():
        schema, table = member.split(".", 1)
        if not _IDENTIFIER.fullmatch(schema) or not _IDENTIFIER.fullmatch(
            table
        ):
            raise ValueError("unsafe descriptor member")
    with fence.engine.begin() as connection:
        for member in members.values():
            schema, table = member.split(".", 1)
            connection.execute(
                text(f'DROP TABLE IF EXISTS "{schema}"."{table}"')
            )
        params = {
            "destination": plane.destination_id,
            "bundle": plane.bundle_key,
            "bundle_id": plane.pin.bundle_id,
            "pin": plane.pin.pin_id,
        }
        connection.execute(
            text(
                f"DELETE FROM {fence._pins_table} WHERE destination_id=:destination AND bundle_key=:bundle AND pin_id=:pin"
            ),
            params,
        )
        connection.execute(
            text(
                f"DELETE FROM {fence._bundles_table} WHERE destination_id=:destination AND bundle_key=:bundle AND bundle_id=:bundle_id"
            ),
            params,
        )
        connection.execute(
            text(
                f"DELETE FROM {fence._table} WHERE destination_id=:destination AND bundle_key=:bundle AND bundle_id=:bundle_id"
            ),
            params,
        )
    with fence.engine.connect() as connection:
        physical = 0
        for member in members.values():
            schema, table = member.split(".", 1)
            physical += int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_schema=:schema AND table_name=:table"
                    ),
                    {"schema": schema, "table": table},
                ).scalar_one()
            )
        return {
            name: int(
                connection.execute(
                    text(
                        f"SELECT count(*) FROM {table} WHERE destination_id=:destination "
                        "AND bundle_key=:bundle "
                        + (
                            "AND bundle_id=:bundle_id"
                            if name != "pin_rows"
                            else "AND pin_id=:pin AND bundle_id=:bundle_id"
                        )
                    ),
                    params,
                ).scalar_one()
            )
            for name, table in {
                "state_rows": fence._table,
                "bundle_rows": fence._bundles_table,
                "pin_rows": fence._pins_table,
            }.items()
        } | {"physical_candidates": physical}


def _journal_plane(
    journal: RunJournal, name: Literal["analysis", "detail"]
) -> PlaneDestination | None:
    """Reconstruct only deterministic, journal-owned remote cleanup authority."""
    bundle_id = getattr(journal, f"{name}_bundle_id")
    if not bundle_id:
        return None
    destination_id = getattr(journal, f"{name}_destination_id")
    bundle_key = (
        ANALYSIS_BUNDLE_KEY if name == "analysis" else DETAIL_BUNDLE_KEY
    )
    return PlaneDestination(
        destination_id=destination_id,
        bundle_key=bundle_key,
        pin=PinIdentity(
            pin_id=f"{journal.run_id}-{name}-recovery",
            bundle_id=bundle_id,
            expires_at_ms=0,
        ),
        snapshot_seq=0,
        members={},
        member_counts={},
        member_checksums={},
    )


def _cleanup_descriptor_or_journal(
    descriptor_path: Path, journal: RunJournal
) -> ReleaseParityDescriptor | None:
    try:
        descriptor = load_descriptor(descriptor_path)
    except (FileNotFoundError, ValueError):
        return None
    _descriptor_matches_journal(descriptor, journal)
    return descriptor


def _delete_journal_remote(
    fence: PostgresPublicationFence,
    journal: RunJournal,
    name: Literal["analysis", "detail"],
) -> Mapping[str, int]:
    """Delete only bundle identities under this run's unique destination."""
    destination = getattr(journal, f"{name}_destination_id")
    bundle = ANALYSIS_BUNDLE_KEY if name == "analysis" else DETAIL_BUNDLE_KEY
    with fence.engine.connect() as connection:
        values = connection.execute(
            text(
                f"SELECT DISTINCT bundle_id FROM {fence._bundles_table} "
                "WHERE destination_id=:destination AND bundle_key=:bundle"
            ),
            {"destination": destination, "bundle": bundle},
        ).scalars()
        bundle_ids = [str(value) for value in values]
    expected_id = getattr(journal, f"{name}_bundle_id")
    if expected_id and expected_id not in bundle_ids:
        bundle_ids.append(expected_id)
    observations = [
        _delete_remote(
            fence,
            PlaneDestination(
                destination_id=destination,
                bundle_key=bundle,
                pin=PinIdentity(
                    pin_id=f"{journal.run_id}-{name}-recovery",
                    bundle_id=bundle_id,
                    expires_at_ms=0,
                ),
                snapshot_seq=0,
                members={},
                member_counts={},
                member_checksums={},
            ),
        )
        for bundle_id in bundle_ids
    ]
    if not observations:
        return dict.fromkeys(
            ("state_rows", "bundle_rows", "pin_rows", "physical_candidates"),
            0,
        )
    return {
        key: sum(item[key] for item in observations) for key in observations[0]
    }


def cleanup(
    descriptor_path: Path, proof_path: Path, journal_path: Path | None = None
) -> CleanupProof:
    if journal_path is not None and journal_path != _journal_path(
        descriptor_path
    ):
        journal = RunJournal.model_validate_json(journal_path.read_text())
        journal.validate_contract()
    else:
        journal = _load_journal(descriptor_path)
    descriptor = _cleanup_descriptor_or_journal(descriptor_path, journal)
    analysis = _remote(descriptor.analysis) if descriptor else None
    detail = _remote(descriptor.detail) if descriptor else None
    analysis_fence = _fence(
        _required_env("MOTHERDUCK_DATABASE_URL"),
        journal.analysis_destination_id,
        "motherduck",
    )
    detail_fence = _fence(
        _required_env("NEON_DATABASE_URL"),
        journal.detail_destination_id,
        "neon",
    )
    try:
        destinations = {
            journal.analysis_destination_id: _delete_remote(
                analysis_fence, analysis
            )
            if analysis
            else _delete_journal_remote(analysis_fence, journal, "analysis"),
            journal.detail_destination_id: _delete_remote(detail_fence, detail)
            if detail
            else _delete_journal_remote(detail_fence, journal, "detail"),
        }
    finally:
        analysis_fence.engine.dispose()
        detail_fence.engine.dispose()
    admin = create_engine(_required_env("DATABASE_URL"))
    try:
        with admin.begin() as connection:
            connection.execute(
                text(
                    f'DROP SCHEMA IF EXISTS "{journal.source_schema}" CASCADE'
                )
            )
        with admin.connect() as connection:
            absent = (
                connection.execute(
                    text(
                        "SELECT 1 FROM information_schema.schemata WHERE schema_name=:schema"
                    ),
                    {"schema": journal.source_schema},
                ).one_or_none()
                is None
            )
    finally:
        admin.dispose()
    files = [journal.analysis_path, journal.detail_path]
    for name in files:
        path = descriptor_path.parent / name
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".lock").unlink(missing_ok=True)
    proof = CleanupProof(
        schema_version=SCHEMA_VERSION,
        run_id=journal.run_id,
        source_schema_absent=absent,
        local_files_absent=all(
            not (descriptor_path.parent / name).exists() for name in files
        ),
        destinations=destinations,
    )
    if descriptor is not None:
        proof.validate_against(descriptor)
    elif any(
        value != 0
        for facts in destinations.values()
        for value in facts.values()
    ):
        raise ValueError("journal recovery cleanup proof is not zero-state")
    proof_path.write_text(proof.model_dump_json(indent=2))
    _trace(
        "cleanup_succeeded", run_id=journal.run_id, recovery=descriptor is None
    )
    return proof


def verify_evidence(
    descriptor_path: Path, proof_path: Path, journal_path: Path | None = None
) -> None:
    journal = (
        RunJournal.model_validate_json(journal_path.read_text())
        if journal_path is not None
        else _load_journal(descriptor_path)
    )
    journal.validate_contract()
    proof = CleanupProof.model_validate_json(proof_path.read_text())
    descriptor = _cleanup_descriptor_or_journal(descriptor_path, journal)
    if descriptor is not None:
        proof.validate_against(descriptor)
    elif (
        proof.run_id != journal.run_id
        or set(proof.destinations)
        != {journal.analysis_destination_id, journal.detail_destination_id}
        or not proof.source_schema_absent
        or not proof.local_files_absent
        or any(
            value != 0
            for facts in proof.destinations.values()
            for value in facts.values()
        )
    ):
        raise ValueError("journal recovery cleanup proof is not zero-state")
    _trace(
        "evidence_verified", run_id=journal.run_id, recovery=descriptor is None
    )
