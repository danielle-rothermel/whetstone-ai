"""Generation Manifest preparation and kernel submission."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dr_platform import OperationManifest, SubmitOptions, SubmitResult, submit
from dr_platform.submission import prepare_manifest
from sqlalchemy.engine import Engine

from whetstone.platform.targets import generation_target, target_registry
from whetstone.records import PredictionSpecRecord


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
