"""Generation Manifest preparation and kernel submission."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dr_code.humaneval import resolve_humaneval_scoring_profile
from dr_platform import OperationManifest, SubmitOptions, SubmitResult, submit
from dr_platform.submission import prepare_manifest
from dr_serialize import sha256_json_digest
from pydantic import BaseModel, ConfigDict, StrictStr
from sqlalchemy import select
from sqlalchemy.engine import Connection, Engine

from whetstone.db import io as db_io
from whetstone.db import schema
from whetstone.platform.targets import (
    generation_target,
    scoring_target,
    target_registry,
)
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
    DatasetSnapshotIdentityPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    PredictionSpecRecord,
)


@dataclass(frozen=True)
class PredictionSpecManifestSource:
    specs: tuple[PredictionSpecRecord, ...]

    @property
    def item_count(self) -> int:
        return len(self.specs)

    def read_items(
        self, *, start_index: int, end_index: int
    ) -> tuple[PredictionSpecRecord, ...]:
        return self.specs[start_index:end_index]


class ScoringTargetSpec(BaseModel):
    """A concrete, immutable scoring Item recipe input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: StrictStr
    generation_run_id: StrictStr
    scoring_profile_id: StrictStr
    scoring_profile_version: StrictStr
    parser_profile_id: StrictStr
    parser_version: StrictStr
    dataset_name: StrictStr
    dataset_split: StrictStr
    dataset_snapshot: DatasetSnapshotIdentityPayload

    @property
    def item_key(self) -> str:
        return sha256_json_digest(
            {
                "generation_run_id": self.generation_run_id,
                "scoring_profile_id": self.scoring_profile_id,
                "scoring_profile_version": self.scoring_profile_version,
                "parser_profile_id": self.parser_profile_id,
                "parser_version": self.parser_version,
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
            }
        )

    @property
    def spec(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def service_class(self):
        from dr_platform import ServiceClass

        return ServiceClass.STANDARD


@dataclass(frozen=True)
class ScoringManifestSource:
    targets: tuple[ScoringTargetSpec, ...]

    @property
    def item_count(self) -> int:
        return len(self.targets)

    def read_items(
        self, *, start_index: int, end_index: int
    ) -> tuple[ScoringTargetSpec, ...]:
        return self.targets[start_index:end_index]


def scoring_target_for_generation_run(
    *,
    spec: PredictionSpecRecord,
    generation_run: GenerationRunRecord,
    scoring_profile_id: str,
    scoring_profile_version: str,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
) -> ScoringTargetSpec:
    """Freeze only scoreable Generation outcomes into a scoring Item.

    The populated-only rule is intentionally applied before item identity and
    manifest construction.  A PARTIAL outcome remains scoreable, but is not a
    strict-acceptance success.
    """
    if generation_run.status is GenerationRunStatus.PARTIAL and not (
        generation_run.summary.terminal_submission_text
        and generation_run.summary.terminal_submission_text.strip()
    ):
        raise ValueError("partial generation run has no populated submission")
    if generation_run.status not in {
        GenerationRunStatus.SUCCESS,
        GenerationRunStatus.PARTIAL,
    }:
        raise ValueError(
            "only successful or populated partial runs are scoreable"
        )
    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
    snapshot = spec.task.metadata.get("dataset_snapshot")
    if snapshot is None:
        raise ValueError(
            "prediction spec is missing dataset snapshot identity"
        )
    return ScoringTargetSpec(
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run.generation_run_id,
        scoring_profile_id=profile.profile_id,
        scoring_profile_version=profile.version,
        parser_profile_id=profile.parser_profile.profile_id,
        parser_version=profile.parser_profile.version,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_snapshot=DatasetSnapshotIdentityPayload.model_validate(
            snapshot
        ),
    )


def select_populated_scoring_generation_runs(
    connection: Connection,
    *,
    experiment_name: str,
) -> tuple[GenerationRunRecord, ...]:
    """Return the one persisted eligibility boundary used before scoring IDs.

    PostgreSQL's POSIX class deliberately handles spaces, tabs, and newlines;
    Python ``strip`` is not permitted to stand in for this source-of-truth
    query.
    """
    rows = connection.execute(
        select(schema.generation_runs)
        .join(
            schema.prediction_specs,
            schema.prediction_specs.c.prediction_id
            == schema.generation_runs.c.prediction_id,
        )
        .where(
            schema.prediction_specs.c.experiment_name == experiment_name,
            schema.generation_runs.c.status.in_(
                (
                    GenerationRunStatus.SUCCESS.value,
                    GenerationRunStatus.PARTIAL.value,
                )
            ),
            schema.generation_runs.c.summary[
                "terminal_submission_text"
            ].astext.is_not(None),
            schema.generation_runs.c.summary["terminal_submission_text"].astext.op("~")(
                "[^[:space:]]"
            ),
        )
        .order_by(
            schema.generation_runs.c.prediction_id,
            schema.generation_runs.c.platform_attempt.desc(),
        )
    ).mappings()
    return tuple(
        db_io.generation_run_record_from_row(dict(row)) for row in rows
    )


def prepare_generation_manifest(
    *,
    operation_key: str,
    experiment_name: str,
    specs: Iterable[PredictionSpecRecord],
    options: SubmitOptions | None = None,
) -> tuple[OperationManifest, PredictionSpecManifestSource]:
    source = PredictionSpecManifestSource(tuple(specs))
    manifest = prepare_manifest(
        operation_key=operation_key,
        workflow_role="generation",
        group_key=experiment_name,
        target=generation_target(),
        source=source,
        options=options,
    )
    return manifest, source


def submit_prediction_specs(
    engine: Engine,
    *,
    operation_key: str,
    experiment_name: str,
    specs: Iterable[PredictionSpecRecord],
    submit_spec: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    options: SubmitOptions | None = None,
) -> SubmitResult:
    manifest, source = prepare_generation_manifest(
        operation_key=operation_key,
        experiment_name=experiment_name,
        specs=specs,
        options=options,
    )
    return submit(
        manifest,
        source,
        engine=engine,
        resolver=target_registry(),
        spec=submit_spec,
        metadata=metadata,
        options=options,
    )


def submit_prediction_specs_jsonl(
    engine: Engine,
    *,
    operation_key: str,
    experiment_name: str,
    specs_file: Path,
    submit_spec: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **_: Any,
) -> SubmitResult:
    specs = (
        PredictionSpecRecord.model_validate_json(line)
        for line in specs_file.read_text().splitlines()
        if line.strip()
    )
    return submit_prediction_specs(
        engine,
        operation_key=operation_key,
        experiment_name=experiment_name,
        specs=specs,
        submit_spec=submit_spec,
        metadata=metadata,
    )


def prepare_scoring_manifest(
    *,
    operation_key: str,
    experiment_name: str,
    targets: Iterable[ScoringTargetSpec],
    options: SubmitOptions | None = None,
) -> tuple[OperationManifest, ScoringManifestSource, str]:
    source = ScoringManifestSource(tuple(targets))
    selection_digest = sha256_json_digest(
        [target.model_dump(mode="json") for target in source.targets]
    )
    manifest = prepare_manifest(
        operation_key=operation_key,
        workflow_role="scoring",
        group_key=experiment_name,
        target=scoring_target(),
        source=source,
        options=options,
    )
    return manifest, source, selection_digest


def submit_scoring_targets(
    engine: Engine,
    *,
    operation_key: str,
    experiment_name: str,
    targets: Iterable[ScoringTargetSpec],
    source_generation_operation_key: str,
    metadata: dict[str, Any] | None = None,
    options: SubmitOptions | None = None,
) -> SubmitResult:
    manifest, source, selection_digest = prepare_scoring_manifest(
        operation_key=operation_key,
        experiment_name=experiment_name,
        targets=targets,
        options=options,
    )
    return submit(
        manifest,
        source,
        engine=engine,
        resolver=target_registry(),
        spec={
            "experiment_name": experiment_name,
            "source_generation_operation_key": source_generation_operation_key,
            "selection_digest": selection_digest,
        },
        metadata=metadata,
        options=options,
    )
