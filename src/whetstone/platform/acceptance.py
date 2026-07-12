"""Immutable experiment relationship and deterministic acceptance reduction.

This module owns domain selection.  It deliberately does not infer a DAG from
platform operations: callers accept completed immutable manifests explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from dr_serialize import Jsonable, sha256_json_digest
from sqlalchemy import func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.engine import Connection

from whetstone.db import schema
from whetstone.records import GenerationRunStatus, ScoreAttemptStatus


class ManifestRelationshipResult(StrEnum):
    ACCEPTED = "accepted"
    ALREADY_ACCEPTED = "already_accepted"
    GENERATION_MEMBERSHIP_CONFLICT = "generation_membership_conflict"


class AcceptanceStatus(StrEnum):
    PARTIAL = "partial"
    ACCEPTED = "accepted"


class GenerationDisposition(StrEnum):
    MISSING = "missing"
    REJECTED = "rejected"
    SELECTED_SUCCESS = "selected_success"


class ScoringDisposition(StrEnum):
    MISSING_GENERATION = "missing_generation"
    MISSING_SCORE = "missing_score"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


@dataclass(frozen=True)
class AcceptanceResult:
    acceptance_id: str
    status: AcceptanceStatus
    expected_count: int
    accepted_count: int
    missing_count: int
    rejected_count: int


def _digest(value: object) -> str:
    return sha256_json_digest(cast(Jsonable, value))


def accept_operation_manifest(
    connection: Connection,
    *,
    experiment_name: str,
    workflow_role: str,
    operation_key: str,
    manifest_digest: str,
    target_ref: dict[str, Any],
    selection_digest: str | None = None,
) -> ManifestRelationshipResult:
    """Accept exactly one generation manifest and ordered scoring manifests.

    The Experiment row lock makes source invalidation and scoring ordinal
    allocation one transaction.  Exact replays do not change either value.
    """
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
        ),
        None,
    )
    if exact is not None:
        return ManifestRelationshipResult.ALREADY_ACCEPTED
    if workflow_role == "generation" and existing:
        return ManifestRelationshipResult.GENERATION_MEMBERSHIP_CONFLICT
    ordinal: int | None = None
    if workflow_role == "scoring":
        ordinal = (
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
            accepted_scoring_ordinal=ordinal,
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


def evaluate_strict_acceptance(
    connection: Connection,
    *,
    experiment_name: str,
    required_profiles: tuple[dict[str, str], ...],
) -> AcceptanceResult:
    """Persist a complete, populated matrix from append-only domain rows.

    Selection is run-pinned: the highest successful generation ordinal wins;
    a score belonging to any other run is never eligible for the cell.
    """
    experiment = (
        connection.execute(
            select(schema.experiments).where(
                schema.experiments.c.experiment_name == experiment_name
            )
        )
        .mappings()
        .one()
    )
    specs = (
        connection.execute(
            select(schema.prediction_specs.c.prediction_id).where(
                schema.prediction_specs.c.experiment_name == experiment_name
            )
        )
        .scalars()
        .all()
    )
    generation_members: list[dict[str, Any]] = []
    scoring_members: list[dict[str, Any]] = []
    expected = accepted = missing = rejected = 0
    for prediction_id in specs:
        run = (
            connection.execute(
                select(schema.generation_runs)
                .where(
                    schema.generation_runs.c.prediction_id == prediction_id,
                    schema.generation_runs.c.status
                    == GenerationRunStatus.SUCCESS.value,
                )
                .order_by(schema.generation_runs.c.platform_attempt.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )
        generation_members.append(
            {
                "prediction_id": prediction_id,
                "disposition": GenerationDisposition.SELECTED_SUCCESS.value
                if run
                else GenerationDisposition.MISSING.value,
                "generation_run_id": run["generation_run_id"] if run else None,
                "platform_item_id": run["platform_item_id"] if run else None,
                "platform_attempt": run["platform_attempt"] if run else None,
            }
        )
        for profile in required_profiles:
            expected += 1
            if run is None:
                missing += 1
                scoring_members.append(
                    {
                        **profile,
                        "prediction_id": prediction_id,
                        "disposition": (
                            ScoringDisposition.MISSING_GENERATION.value
                        ),
                        "generation_run_id": None,
                        "score_attempt_id": None,
                    }
                )
                continue
            score = (
                connection.execute(
                    select(schema.score_attempts)
                    .where(
                        schema.score_attempts.c.generation_run_id
                        == run["generation_run_id"],
                        schema.score_attempts.c.scoring_profile_id
                        == profile["scoring_profile_id"],
                        schema.score_attempts.c.scoring_profile_version
                        == profile["scoring_profile_version"],
                        schema.score_attempts.c.parser_profile_id
                        == profile["parser_profile_id"],
                        schema.score_attempts.c.parser_version
                        == profile["parser_version"],
                        schema.score_attempts.c.dataset_name
                        == profile["dataset_name"],
                        schema.score_attempts.c.dataset_split
                        == profile["dataset_split"],
                        schema.score_attempts.c.status
                        == ScoreAttemptStatus.SUCCESS.value,
                    )
                    .order_by(schema.score_attempts.c.platform_attempt.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if score is None:
                missing += 1
                disposition = ScoringDisposition.MISSING_SCORE.value
            else:
                accepted += 1
                disposition = ScoringDisposition.ACCEPTED.value
            scoring_members.append(
                {
                    **profile,
                    "prediction_id": prediction_id,
                    "disposition": disposition,
                    "generation_run_id": run["generation_run_id"],
                    "score_attempt_id": score["score_attempt_id"]
                    if score
                    else None,
                }
            )
    payload = {
        "experiment_name": experiment_name,
        "source_version": experiment["acceptance_source_version"],
        "generation_members": generation_members,
        "scoring_members": scoring_members,
        "required_profiles": required_profiles,
    }
    acceptance_id = _digest(payload)
    status = (
        AcceptanceStatus.ACCEPTED
        if expected and accepted == expected
        else AcceptanceStatus.PARTIAL
    )
    connection.execute(
        postgres_insert(schema.experiment_acceptance_evaluations)
        .values(
            acceptance_id=acceptance_id,
            experiment_name=experiment_name,
            acceptance_source_version=experiment["acceptance_source_version"],
            status=status.value,
            generation_operation_key="",
            generation_manifest_digest="",
            scoring_relationships=[],
            scoring_relationships_digest=_digest([]),
            selected_scoring_candidates=[],
            selected_scoring_candidates_digest=_digest([]),
            domain_cut=payload,
            domain_cut_digest=_digest(payload),
            platform_cut=[],
            platform_cut_digest=_digest([]),
            required_profiles=list(required_profiles),
            required_profiles_digest=_digest(list(required_profiles)),
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
    return AcceptanceResult(
        acceptance_id, status, expected, accepted, missing, rejected
    )
