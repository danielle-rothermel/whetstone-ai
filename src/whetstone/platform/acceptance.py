"""Immutable relationships and deterministic acceptance reduction."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from dr_platform import PlatformOperationCut, PlatformSchema
from dr_serialize import Jsonable, sha256_json_digest
from pydantic import BaseModel, ConfigDict, StrictStr
from sqlalchemy import func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.engine import Connection

from whetstone.db import schema
from whetstone.records import GenerationRunStatus, ScoreAttemptStatus

_TERMINAL_OPERATION_STATUSES = frozenset(
    {"succeeded", "partial", "failed", "cancelled"}
)
_DBOS_SUCCESS = "SUCCESS"


class ManifestRelationshipResult(StrEnum):
    ACCEPTED = "accepted"
    ALREADY_ACCEPTED = "already_accepted"
    GENERATION_MEMBERSHIP_CONFLICT = "generation_membership_conflict"


class GenerationMembershipConflictError(ValueError):
    """An unequal second Generation Manifest cannot enter an Experiment."""


class AcceptanceStatus(StrEnum):
    PARTIAL = "partial"
    ACCEPTED = "accepted"


class AcceptanceDisposition(StrEnum):
    PROMOTED = "promoted"
    HISTORICAL_PARTIAL = "historical_partial"
    EXECUTION_NOT_TERMINAL = "execution_not_terminal"
    EXECUTION_INCOMPATIBLE = "execution_incompatible"
    SOURCE_ADVANCED = "source_advanced"
    PLATFORM_CUT_ADVANCED = "platform_cut_advanced"


class CurrentAcceptanceDisposition(StrEnum):
    CURRENT = "current"
    NOT_ACCEPTED = "not_accepted"
    SOURCE_ADVANCED = "source_advanced"
    STALE_PLATFORM_CUT = "stale_platform_cut"


class GenerationDisposition(StrEnum):
    MISSING = "missing"
    REJECTED = "rejected"
    SELECTED_SUCCESS = "selected_success"
    TYPED_FAILURE = "typed_failure"


class GenerationCandidateDisposition(StrEnum):
    SELECTED = "selected"
    SUPERSEDED_SUCCESS = "superseded_success"
    REJECTED = "rejected"


class ScoringDisposition(StrEnum):
    MISSING_GENERATION = "missing_generation"
    MISSING_SCORE = "missing_score"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


class ScoringCandidateDisposition(StrEnum):
    SELECTED = "selected"
    SUPERSEDED_GENERATION = "superseded_generation"
    SUPERSEDED_RELATIONSHIP = "superseded_relationship"
    SUPERSEDED_ATTEMPT = "superseded_attempt"
    REJECTED = "rejected"


class RequiredScoringProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scoring_profile_id: StrictStr
    scoring_profile_version: StrictStr
    parser_profile_id: StrictStr
    parser_version: StrictStr
    dataset_name: StrictStr
    dataset_split: StrictStr


class AcceptanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    acceptance_id: StrictStr | None
    status: AcceptanceStatus
    disposition: AcceptanceDisposition
    expected_count: int
    accepted_count: int
    missing_count: int
    rejected_count: int


class AcceptanceEvaluation(BaseModel):
    """Typed immutable public view of one historical evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    acceptance_id: StrictStr
    experiment_name: StrictStr
    acceptance_source_version: int
    status: AcceptanceStatus
    platform_cut: tuple[PlatformOperationCut, ...]
    required_profiles: tuple[RequiredScoringProfile, ...]
    expected_count: int
    accepted_count: int
    missing_count: int
    rejected_count: int
    created_at: datetime


class CurrentAcceptanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: CurrentAcceptanceDisposition
    evaluation: AcceptanceEvaluation | None = None


def _digest(value: object) -> str:
    return sha256_json_digest(cast(Jsonable, value))


def _profile_payloads(
    profiles: tuple[RequiredScoringProfile, ...],
) -> list[dict[str, Any]]:
    return [profile.model_dump(mode="json") for profile in profiles]


def accept_operation_manifest(
    connection: Connection,
    *,
    experiment_name: str,
    workflow_role: str,
    operation_key: str,
    manifest_digest: str,
    target_ref: dict[str, Any],
    selection_digest: str | None = None,
    generation_member_keys: tuple[str, ...] | None = None,
) -> ManifestRelationshipResult:
    """Accept one Generation Manifest and ordered Scoring Manifests."""
    experiment = (
        connection.execute(
            select(schema.experiments)
            .where(schema.experiments.c.experiment_name == experiment_name)
            .with_for_update()
        )
        .mappings()
        .one()
    )
    existing = (
        connection.execute(
            select(schema.experiment_operation_manifests).where(
                schema.experiment_operation_manifests.c.experiment_name
                == experiment_name,
                schema.experiment_operation_manifests.c.workflow_role
                == workflow_role,
            )
        )
        .mappings()
        .all()
    )
    exact = next(
        (
            row
            for row in existing
            if row["operation_key"] == operation_key
            and row["manifest_digest"] == manifest_digest
            and row["selection_digest"] == selection_digest
            and dict(row["target_ref"]) == target_ref
        ),
        None,
    )
    if exact is not None:
        return ManifestRelationshipResult.ALREADY_ACCEPTED
    generation_ordinal: int | None = None
    scoring_ordinal: int | None = None
    if workflow_role == "generation":
        platform = PlatformSchema(prefix="whetstone")
        existing_operation_keys = [
            str(row["operation_key"])
            for row in existing
            if row["workflow_role"] == "generation"
        ]
        if existing_operation_keys:
            existing_members = set(
                connection.execute(
                    select(platform.items.c.item_key).where(
                        platform.items.c.operation_key.in_(
                            existing_operation_keys
                        )
                    )
                ).scalars()
            )
            candidate_members = set(generation_member_keys or ())
            if not candidate_members:
                candidate_members = set(
                    connection.execute(
                        select(platform.items.c.item_key).where(
                            platform.items.c.operation_key == operation_key
                        )
                    ).scalars()
                )
            if existing_members & candidate_members:
                return (
                    ManifestRelationshipResult.GENERATION_MEMBERSHIP_CONFLICT
                )
        generation_ordinal = (
            int(
                connection.execute(
                    select(
                        func.coalesce(
                            func.max(
                                schema.experiment_operation_manifests.c.accepted_generation_ordinal
                            ),
                            0,
                        )
                    ).where(
                        schema.experiment_operation_manifests.c.experiment_name
                        == experiment_name,
                        schema.experiment_operation_manifests.c.workflow_role
                        == "generation",
                    )
                ).scalar_one()
            )
            + 1
        )
    if workflow_role == "scoring":
        scoring_ordinal = (
            int(
                connection.execute(
                    select(
                        func.coalesce(
                            func.max(
                                schema.experiment_operation_manifests.c.accepted_scoring_ordinal
                            ),
                            0,
                        )
                    ).where(
                        schema.experiment_operation_manifests.c.experiment_name
                        == experiment_name,
                        schema.experiment_operation_manifests.c.workflow_role
                        == "scoring",
                    )
                ).scalar_one()
            )
            + 1
        )
    connection.execute(
        insert(schema.experiment_operation_manifests).values(
            experiment_name=experiment_name,
            workflow_role=workflow_role,
            operation_key=operation_key,
            manifest_digest=manifest_digest,
            selection_digest=selection_digest,
            target_ref=target_ref,
            accepted_at=datetime.now(UTC),
            accepted_generation_ordinal=generation_ordinal,
            accepted_scoring_ordinal=scoring_ordinal,
        )
    )
    connection.execute(
        update(schema.experiments)
        .where(schema.experiments.c.experiment_name == experiment_name)
        .values(
            acceptance_source_version=experiment["acceptance_source_version"]
            + 1,
            current_acceptance_id=None,
            acceptance_updated_at=datetime.now(UTC),
        )
    )
    return ManifestRelationshipResult.ACCEPTED


def _load_relationships(
    connection: Connection, experiment_name: str
) -> tuple[list[Any], list[Any]]:
    relationships = (
        connection.execute(
            select(schema.experiment_operation_manifests)
            .where(
                schema.experiment_operation_manifests.c.experiment_name
                == experiment_name
            )
            .order_by(
                schema.experiment_operation_manifests.c.accepted_generation_ordinal,
                schema.experiment_operation_manifests.c.accepted_scoring_ordinal,
            )
        )
        .mappings()
        .all()
    )
    generations = [
        row for row in relationships if row["workflow_role"] == "generation"
    ]
    if not generations:
        raise ValueError("acceptance requires a Generation Manifest")
    return generations, [
        row for row in relationships if row["workflow_role"] == "scoring"
    ]


def _lock_platform_cut(
    connection: Connection,
    *,
    operation_keys: list[str],
    platform: PlatformSchema,
) -> list[dict[str, Any]]:
    keys = sorted(set(operation_keys))
    rows = (
        connection.execute(
            select(
                platform.operations.c.operation_key,
                platform.operations.c.platform_cut_version,
                platform.operations.c.status,
            )
            .where(platform.operations.c.operation_key.in_(keys))
            .order_by(platform.operations.c.operation_key)
            .with_for_update()
        )
        .mappings()
        .all()
    )
    if [row["operation_key"] for row in rows] != keys:
        raise ValueError(
            "an accepted Manifest relationship Operation is missing"
        )
    return [
        {
            "operation_key": str(row["operation_key"]),
            "platform_cut_version": int(row["platform_cut_version"]),
            "status": str(row["status"]),
        }
        for row in rows
    ]


def _generation_inputs(
    connection: Connection,
    *,
    relationships: list[Any],
    platform: PlatformSchema,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    members: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_predictions: set[str] = set()
    for relationship in relationships:
        predictions = list(
            connection.execute(
                select(platform.items.c.item_key)
                .where(
                    platform.items.c.operation_key
                    == relationship["operation_key"]
                )
                .order_by(platform.items.c.item_index)
            ).scalars()
        )
        overlap = seen_predictions.intersection(
            str(value) for value in predictions
        )
        if overlap:
            raise ValueError(
                "a prediction belongs to multiple Generation Manifests"
            )
        seen_predictions.update(str(value) for value in predictions)
        for prediction_id in predictions:
            rows = (
                connection.execute(
                    select(
                        schema.generation_runs,
                        platform.item_attempts.c.execution_state.label(
                            "platform_execution_state"
                        ),
                        platform.item_attempts.c.dbos_status.label(
                            "platform_dbos_status"
                        ),
                    )
                    .join(
                        platform.items,
                        platform.items.c.item_id
                        == schema.generation_runs.c.platform_item_id,
                    )
                    .join(
                        platform.item_attempts,
                        (
                            platform.item_attempts.c.item_id
                            == schema.generation_runs.c.platform_item_id
                        )
                        & (
                            platform.item_attempts.c.attempt
                            == schema.generation_runs.c.platform_attempt
                        ),
                    )
                    .where(
                        schema.generation_runs.c.prediction_id
                        == prediction_id,
                        platform.items.c.operation_key
                        == relationship["operation_key"],
                    )
                    .order_by(
                        schema.generation_runs.c.platform_attempt.desc(),
                        schema.generation_runs.c.generation_run_id,
                    )
                )
                .mappings()
                .all()
            )
            selected = next(
                (
                    row
                    for row in rows
                    if row["status"] == GenerationRunStatus.SUCCESS.value
                ),
                None,
            )
            typed_failure = next(
                (
                    row
                    for row in rows
                    if row["status"]
                    in {
                        GenerationRunStatus.ERROR.value,
                        GenerationRunStatus.BLOCKED.value,
                    }
                ),
                None,
            )
            member_run = selected or typed_failure
            members.append(
                {
                    "prediction_id": prediction_id,
                    "disposition": (
                        GenerationDisposition.SELECTED_SUCCESS.value
                        if selected is not None
                        else GenerationDisposition.TYPED_FAILURE.value
                        if typed_failure is not None
                        else GenerationDisposition.MISSING.value
                        if not rows
                        else GenerationDisposition.REJECTED.value
                    ),
                    "generation_run_id": (
                        member_run["generation_run_id"] if member_run else None
                    ),
                    "generation_operation_key": relationship["operation_key"],
                    "platform_item_id": (
                        member_run["platform_item_id"] if member_run else None
                    ),
                    "platform_attempt": (
                        member_run["platform_attempt"] if member_run else None
                    ),
                }
            )
            for row in rows:
                if (
                    member_run is not None
                    and row["generation_run_id"]
                    == member_run["generation_run_id"]
                ):
                    disposition = GenerationCandidateDisposition.SELECTED
                elif row["status"] == GenerationRunStatus.SUCCESS.value:
                    disposition = (
                        GenerationCandidateDisposition.SUPERSEDED_SUCCESS
                    )
                else:
                    disposition = GenerationCandidateDisposition.REJECTED
                candidates.append(
                    {
                        "prediction_id": prediction_id,
                        "generation_run_id": row["generation_run_id"],
                        "disposition": disposition.value,
                        "generation_operation_key": relationship[
                            "operation_key"
                        ],
                        "platform_item_id": row["platform_item_id"],
                        "platform_attempt": row["platform_attempt"],
                        "status": row["status"],
                        "platform_execution_state": row[
                            "platform_execution_state"
                        ],
                        "platform_dbos_status": row["platform_dbos_status"],
                    }
                )
    return members, candidates


def _score_rows_for_relationship(
    connection: Connection,
    *,
    relationship: Any,
    prediction_id: str,
    profile: dict[str, Any],
    platform: PlatformSchema,
) -> list[dict[str, Any]]:
    filters = (
        schema.score_attempts.c.prediction_id == prediction_id,
        schema.score_attempts.c.scoring_profile_id
        == profile["scoring_profile_id"],
        schema.score_attempts.c.scoring_profile_version
        == profile["scoring_profile_version"],
        schema.score_attempts.c.parser_profile_id
        == profile["parser_profile_id"],
        schema.score_attempts.c.parser_version == profile["parser_version"],
        schema.score_attempts.c.dataset_name == profile["dataset_name"],
        schema.score_attempts.c.dataset_split == profile["dataset_split"],
        platform.items.c.operation_key == relationship["operation_key"],
    )
    score_rows = (
        connection.execute(
            select(
                schema.score_attempts,
                platform.item_attempts.c.execution_state.label(
                    "platform_execution_state"
                ),
                platform.item_attempts.c.dbos_status.label(
                    "platform_dbos_status"
                ),
            )
            .join(
                platform.items,
                platform.items.c.item_id
                == schema.score_attempts.c.platform_item_id,
            )
            .join(
                platform.item_attempts,
                (
                    platform.item_attempts.c.item_id
                    == schema.score_attempts.c.platform_item_id
                )
                & (
                    platform.item_attempts.c.attempt
                    == schema.score_attempts.c.platform_attempt
                ),
            )
            .where(*filters)
        )
        .mappings()
        .all()
    )
    failure_filters = (
        schema.score_harness_failures.c.prediction_id == prediction_id,
        schema.score_harness_failures.c.scoring_profile_id
        == profile["scoring_profile_id"],
        schema.score_harness_failures.c.scoring_profile_version
        == profile["scoring_profile_version"],
        schema.score_harness_failures.c.parser_profile_id
        == profile["parser_profile_id"],
        schema.score_harness_failures.c.parser_version
        == profile["parser_version"],
        schema.score_harness_failures.c.dataset_name
        == profile["dataset_name"],
        schema.score_harness_failures.c.dataset_split
        == profile["dataset_split"],
        platform.items.c.operation_key == relationship["operation_key"],
    )
    failure_rows = (
        connection.execute(
            select(
                schema.score_harness_failures,
                platform.item_attempts.c.execution_state.label(
                    "platform_execution_state"
                ),
                platform.item_attempts.c.dbos_status.label(
                    "platform_dbos_status"
                ),
            )
            .join(
                platform.items,
                platform.items.c.item_id
                == schema.score_harness_failures.c.platform_item_id,
            )
            .join(
                platform.item_attempts,
                (
                    platform.item_attempts.c.item_id
                    == schema.score_harness_failures.c.platform_item_id
                )
                & (
                    platform.item_attempts.c.attempt
                    == schema.score_harness_failures.c.platform_attempt
                ),
            )
            .where(*failure_filters)
        )
        .mappings()
        .all()
    )
    common = {
        "accepted_scoring_ordinal": relationship["accepted_scoring_ordinal"],
        "operation_key": relationship["operation_key"],
        "manifest_digest": relationship["manifest_digest"],
        "selection_digest": relationship["selection_digest"],
    }
    normalized = [
        {**dict(row), **common, "candidate_kind": "score_attempt"}
        for row in score_rows
    ]
    normalized.extend(
        {
            **dict(row),
            **common,
            "status": "harness_failure",
            "candidate_kind": "score_harness_failure",
        }
        for row in failure_rows
    )
    return sorted(
        normalized,
        key=lambda row: (
            -int(row["accepted_scoring_ordinal"]),
            -int(row["platform_attempt"]),
            str(row["score_attempt_id"]),
        ),
    )


def _scoring_inputs(
    connection: Connection,
    *,
    generation_members: list[dict[str, Any]],
    relationships: list[Any],
    profiles: list[dict[str, Any]],
    platform: PlatformSchema,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    members: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for generation in generation_members:
        for profile in profiles:
            rows: list[dict[str, Any]] = []
            for relationship in relationships:
                rows.extend(
                    _score_rows_for_relationship(
                        connection,
                        relationship=relationship,
                        prediction_id=generation["prediction_id"],
                        profile=profile,
                        platform=platform,
                    )
                )
            rows.sort(
                key=lambda row: (
                    -int(row["accepted_scoring_ordinal"]),
                    -int(row["platform_attempt"]),
                    str(row["score_attempt_id"]),
                )
            )
            selected_run_id = generation["generation_run_id"]
            winner = next(
                (
                    row
                    for row in rows
                    if row["generation_run_id"] == selected_run_id
                    and row["status"] == ScoreAttemptStatus.SUCCESS.value
                ),
                None,
            )
            if selected_run_id is None:
                member_disposition = ScoringDisposition.MISSING_GENERATION
            elif winner is not None:
                member_disposition = ScoringDisposition.ACCEPTED
            elif rows:
                member_disposition = ScoringDisposition.REJECTED
            else:
                member_disposition = ScoringDisposition.MISSING_SCORE
            members.append(
                {
                    **profile,
                    "prediction_id": generation["prediction_id"],
                    "disposition": member_disposition.value,
                    "generation_run_id": selected_run_id,
                    "score_attempt_id": winner["score_attempt_id"]
                    if winner
                    else None,
                    "accepted_scoring_ordinal": winner[
                        "accepted_scoring_ordinal"
                    ]
                    if winner
                    else None,
                    "scoring_operation_key": winner["operation_key"]
                    if winner
                    else None,
                    "platform_item_id": winner["platform_item_id"]
                    if winner
                    else None,
                    "platform_attempt": winner["platform_attempt"]
                    if winner
                    else None,
                    "manifest_digest": winner["manifest_digest"]
                    if winner
                    else None,
                }
            )
            for row in rows:
                if row["generation_run_id"] != selected_run_id:
                    disposition = (
                        ScoringCandidateDisposition.SUPERSEDED_GENERATION
                    )
                elif (
                    winner is not None
                    and row["score_attempt_id"] == winner["score_attempt_id"]
                ):
                    disposition = ScoringCandidateDisposition.SELECTED
                elif row["status"] != ScoreAttemptStatus.SUCCESS.value:
                    disposition = ScoringCandidateDisposition.REJECTED
                elif winner is not None and (
                    row["accepted_scoring_ordinal"]
                    != winner["accepted_scoring_ordinal"]
                ):
                    disposition = (
                        ScoringCandidateDisposition.SUPERSEDED_RELATIONSHIP
                    )
                else:
                    disposition = (
                        ScoringCandidateDisposition.SUPERSEDED_ATTEMPT
                    )
                candidates.append(
                    {
                        **profile,
                        "prediction_id": generation["prediction_id"],
                        "accepted_scoring_ordinal": row[
                            "accepted_scoring_ordinal"
                        ],
                        "score_attempt_id": row["score_attempt_id"],
                        "generation_run_id": row["generation_run_id"],
                        "disposition": disposition.value,
                        "operation_key": row["operation_key"],
                        "manifest_digest": row["manifest_digest"],
                        "selection_digest": row["selection_digest"],
                        "platform_item_id": row["platform_item_id"],
                        "platform_attempt": row["platform_attempt"],
                        "status": row["status"],
                        "candidate_kind": row["candidate_kind"],
                        "platform_execution_state": row[
                            "platform_execution_state"
                        ],
                        "platform_dbos_status": row["platform_dbos_status"],
                    }
                )
    return members, candidates


def _execution_disposition(
    *,
    platform_cut: list[dict[str, Any]],
    generation_candidates: list[dict[str, Any]],
    scoring_candidates: list[dict[str, Any]],
) -> AcceptanceDisposition | None:
    if any(
        row["status"] not in _TERMINAL_OPERATION_STATUSES
        for row in platform_cut
    ):
        return AcceptanceDisposition.EXECUTION_NOT_TERMINAL
    selected = [
        row
        for row in (*generation_candidates, *scoring_candidates)
        if row["disposition"] == "selected"
    ]
    if any(
        row["platform_execution_state"] != "succeeded"
        or row["platform_dbos_status"] != _DBOS_SUCCESS
        for row in selected
    ):
        return AcceptanceDisposition.EXECUTION_INCOMPATIBLE
    return None


def evaluate_strict_acceptance(
    connection: Connection,
    *,
    experiment_name: str,
    required_profiles: tuple[RequiredScoringProfile, ...],
) -> AcceptanceResult:
    """Append one complete acceptance cut and promote it only when current."""
    observed_source_version = connection.execute(
        select(schema.experiments.c.acceptance_source_version).where(
            schema.experiments.c.experiment_name == experiment_name
        )
    ).scalar_one()
    generation_relationships, scoring_relationships = _load_relationships(
        connection, experiment_name
    )
    platform = PlatformSchema(prefix="whetstone")
    platform_cut = _lock_platform_cut(
        connection,
        operation_keys=[
            *(row["operation_key"] for row in generation_relationships),
            *(row["operation_key"] for row in scoring_relationships),
        ],
        platform=platform,
    )
    experiment = (
        connection.execute(
            select(schema.experiments)
            .where(schema.experiments.c.experiment_name == experiment_name)
            .with_for_update()
        )
        .mappings()
        .one()
    )
    source_version = int(experiment["acceptance_source_version"])
    if source_version != observed_source_version:
        return AcceptanceResult(
            acceptance_id=None,
            status=AcceptanceStatus.PARTIAL,
            disposition=AcceptanceDisposition.SOURCE_ADVANCED,
            expected_count=0,
            accepted_count=0,
            missing_count=0,
            rejected_count=0,
        )
    generation_members, generation_candidates = _generation_inputs(
        connection,
        relationships=generation_relationships,
        platform=platform,
    )
    profiles = _profile_payloads(required_profiles)
    scoring_members, scoring_candidates = _scoring_inputs(
        connection,
        generation_members=generation_members,
        relationships=scoring_relationships,
        profiles=profiles,
        platform=platform,
    )
    expected = len(scoring_members)
    accounted_generation_failures = {
        row["prediction_id"]
        for row in generation_members
        if row["disposition"] == GenerationDisposition.TYPED_FAILURE.value
    }
    accounted_scoring_failures = {
        row["prediction_id"]
        for row in scoring_members
        if row["disposition"] == ScoringDisposition.REJECTED.value
        and any(
            candidate["prediction_id"] == row["prediction_id"]
            and candidate["generation_run_id"] == row["generation_run_id"]
            and candidate["disposition"]
            == ScoringCandidateDisposition.REJECTED.value
            and candidate["candidate_kind"] == "score_harness_failure"
            for candidate in scoring_candidates
        )
    }
    accounted_prediction_ids = (
        accounted_generation_failures | accounted_scoring_failures
    )
    accepted = sum(
        row["disposition"] == "accepted"
        or row["prediction_id"] in accounted_prediction_ids
        for row in scoring_members
    )
    missing = sum(
        row["disposition"].startswith("missing_")
        and row["prediction_id"] not in accounted_prediction_ids
        for row in scoring_members
    )
    rejected = sum(
        row["disposition"] == "rejected"
        and row["prediction_id"] not in accounted_prediction_ids
        for row in scoring_members
    )
    status = (
        AcceptanceStatus.ACCEPTED
        if expected > 0 and accepted == expected
        else AcceptanceStatus.PARTIAL
    )
    generation_relationship_payload = [
        {
            "accepted_generation_ordinal": row["accepted_generation_ordinal"],
            "operation_key": row["operation_key"],
            "manifest_digest": row["manifest_digest"],
        }
        for row in generation_relationships
    ]
    scoring_relationship_payload = [
        {
            "accepted_scoring_ordinal": row["accepted_scoring_ordinal"],
            "operation_key": row["operation_key"],
            "manifest_digest": row["manifest_digest"],
            "selection_digest": row["selection_digest"],
        }
        for row in scoring_relationships
    ]
    selected_scoring_candidates = [
        row for row in scoring_candidates if row["disposition"] == "selected"
    ]
    domain_cut = {
        "generation_members": generation_members,
        "generation_candidates": generation_candidates,
        "scoring_members": scoring_members,
        "scoring_candidates": scoring_candidates,
    }
    identity = {
        "experiment_name": experiment_name,
        "acceptance_source_version": source_version,
        "generation_relationships_digest": _digest(
            generation_relationship_payload
        ),
        "scoring_relationships_digest": _digest(scoring_relationship_payload),
        "selected_scoring_candidates_digest": _digest(
            selected_scoring_candidates
        ),
        "domain_cut_digest": _digest(domain_cut),
        "platform_cut_digest": _digest(platform_cut),
        "policy_digest": _digest({"name": "strict", "version": 1}),
        "observed_matrix_digest": _digest(scoring_members),
    }
    acceptance_id = _digest(identity)
    execution_disposition = _execution_disposition(
        platform_cut=platform_cut,
        generation_candidates=generation_candidates,
        scoring_candidates=scoring_candidates,
    )
    connection.execute(
        postgres_insert(schema.experiment_acceptance_evaluations)
        .values(
            acceptance_id=acceptance_id,
            experiment_name=experiment_name,
            acceptance_source_version=source_version,
            status=status.value,
            generation_relationships=generation_relationship_payload,
            generation_relationships_digest=_digest(
                generation_relationship_payload
            ),
            # Retained for readers on the v6 baseline; the vector is authority.
            generation_operation_key=generation_relationships[0][
                "operation_key"
            ],
            generation_manifest_digest=generation_relationships[0][
                "manifest_digest"
            ],
            scoring_relationships=scoring_relationship_payload,
            scoring_relationships_digest=_digest(scoring_relationship_payload),
            selected_scoring_candidates=selected_scoring_candidates,
            selected_scoring_candidates_digest=_digest(
                selected_scoring_candidates
            ),
            domain_cut=domain_cut,
            domain_cut_digest=_digest(domain_cut),
            platform_cut=platform_cut,
            platform_cut_digest=_digest(platform_cut),
            required_profiles=profiles,
            required_profiles_digest=_digest(profiles),
            policy={"name": "strict", "version": 1},
            policy_digest=_digest({"name": "strict", "version": 1}),
            observed_matrix=scoring_members,
            observed_matrix_digest=_digest(scoring_members),
            expected_count=expected,
            accepted_count=accepted,
            missing_count=missing,
            rejected_count=rejected,
            created_at=datetime.now(UTC),
        )
        .on_conflict_do_nothing(index_elements=["acceptance_id"])
    )
    for member in generation_members:
        connection.execute(
            postgres_insert(schema.experiment_acceptance_generation_members)
            .values(acceptance_id=acceptance_id, **member)
            .on_conflict_do_nothing()
        )
    for candidate in generation_candidates:
        persisted = {
            key: value
            for key, value in candidate.items()
            if not key.startswith("platform_")
            or key in {"platform_item_id", "platform_attempt"}
        }
        connection.execute(
            postgres_insert(schema.experiment_acceptance_generation_candidates)
            .values(acceptance_id=acceptance_id, **persisted)
            .on_conflict_do_nothing()
        )
    for member in scoring_members:
        connection.execute(
            postgres_insert(schema.experiment_acceptance_scoring_members)
            .values(acceptance_id=acceptance_id, **member)
            .on_conflict_do_nothing()
        )
    for candidate in scoring_candidates:
        persisted = {
            key: value
            for key, value in candidate.items()
            if not key.startswith("platform_")
            or key in {"platform_item_id", "platform_attempt"}
        }
        connection.execute(
            postgres_insert(schema.experiment_acceptance_scoring_candidates)
            .values(acceptance_id=acceptance_id, **persisted)
            .on_conflict_do_nothing()
        )
    disposition = execution_disposition
    if disposition is None and status is AcceptanceStatus.ACCEPTED:
        current_cut = _lock_platform_cut(
            connection,
            operation_keys=[row["operation_key"] for row in platform_cut],
            platform=platform,
        )
        if current_cut != platform_cut:
            disposition = AcceptanceDisposition.PLATFORM_CUT_ADVANCED
        else:
            promoted = connection.execute(
                update(schema.experiments)
                .where(
                    schema.experiments.c.experiment_name == experiment_name,
                    schema.experiments.c.acceptance_source_version
                    == source_version,
                )
                .values(
                    current_acceptance_id=acceptance_id,
                    acceptance_updated_at=datetime.now(UTC),
                )
            )
            disposition = (
                AcceptanceDisposition.PROMOTED
                if promoted.rowcount == 1
                else AcceptanceDisposition.SOURCE_ADVANCED
            )
    if disposition is None:
        disposition = AcceptanceDisposition.HISTORICAL_PARTIAL
    return AcceptanceResult(
        acceptance_id=acceptance_id,
        status=status,
        disposition=disposition,
        expected_count=expected,
        accepted_count=accepted,
        missing_count=missing,
        rejected_count=rejected,
    )


def load_acceptance(
    connection: Connection, *, acceptance_id: str
) -> AcceptanceEvaluation:
    row = (
        connection.execute(
            select(schema.experiment_acceptance_evaluations).where(
                schema.experiment_acceptance_evaluations.c.acceptance_id
                == acceptance_id
            )
        )
        .mappings()
        .one()
    )
    return AcceptanceEvaluation(
        acceptance_id=row["acceptance_id"],
        experiment_name=row["experiment_name"],
        acceptance_source_version=row["acceptance_source_version"],
        status=row["status"],
        platform_cut=tuple(
            PlatformOperationCut.model_validate(
                {
                    "operation_key": cut["operation_key"],
                    "platform_cut_version": cut["platform_cut_version"],
                }
            )
            for cut in row["platform_cut"]
        ),
        required_profiles=tuple(
            RequiredScoringProfile.model_validate(profile)
            for profile in row["required_profiles"]
        ),
        expected_count=row["expected_count"],
        accepted_count=row["accepted_count"],
        missing_count=row["missing_count"],
        rejected_count=row["rejected_count"],
        created_at=row["created_at"],
    )


def load_current_acceptance(
    connection: Connection, *, experiment_name: str
) -> CurrentAcceptanceResult:
    observed_experiment = (
        connection.execute(
            select(schema.experiments).where(
                schema.experiments.c.experiment_name == experiment_name
            )
        )
        .mappings()
        .one()
    )
    acceptance_id = observed_experiment["current_acceptance_id"]
    if acceptance_id is None:
        return CurrentAcceptanceResult(
            disposition=CurrentAcceptanceDisposition.NOT_ACCEPTED
        )
    evaluation = load_acceptance(connection, acceptance_id=acceptance_id)
    platform = PlatformSchema(prefix="whetstone")
    current_cut = _lock_platform_cut(
        connection,
        operation_keys=[cut.operation_key for cut in evaluation.platform_cut],
        platform=platform,
    )
    experiment = (
        connection.execute(
            select(schema.experiments)
            .where(schema.experiments.c.experiment_name == experiment_name)
            .with_for_update()
        )
        .mappings()
        .one()
    )
    if experiment["current_acceptance_id"] != acceptance_id:
        return CurrentAcceptanceResult(
            disposition=CurrentAcceptanceDisposition.SOURCE_ADVANCED
        )
    if (
        evaluation.acceptance_source_version
        != experiment["acceptance_source_version"]
    ):
        return CurrentAcceptanceResult(
            disposition=CurrentAcceptanceDisposition.SOURCE_ADVANCED
        )
    expected_versions = {
        cut.operation_key: cut.platform_cut_version
        for cut in evaluation.platform_cut
    }
    if {
        row["operation_key"]: row["platform_cut_version"]
        for row in current_cut
    } != expected_versions:
        return CurrentAcceptanceResult(
            disposition=CurrentAcceptanceDisposition.STALE_PLATFORM_CUT
        )
    return CurrentAcceptanceResult(
        disposition=CurrentAcceptanceDisposition.CURRENT,
        evaluation=evaluation,
    )
