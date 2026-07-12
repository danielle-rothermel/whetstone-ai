"""HumanEval raw-row snapshot loading for platform specs and scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from dr_code.humaneval import parse_human_eval_dataset
from dr_code.humaneval.sampling import (
    DEFAULT_HUMAN_EVAL_HF_REVISION,
    HumanEvalRawRowsSnapshot,
    HumanEvalRawRowsSnapshotHeader,
    SampledHumanEvalTask,
    sample_human_eval_tasks_from_rows,
    validate_snapshot_header,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.records import (
    DatasetSnapshotHeaderPayload,
    DatasetSnapshotIdentityPayload,
)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
SHA256_ALGORITHM = "sha256"


class HumanEvalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: DatasetSnapshotIdentityPayload
    rows: tuple[dict[str, Any], ...]
    snapshot_bytes: bytes = Field(exclude=True, repr=False)

    def validate_content_coupling(self) -> HumanEvalSnapshot:
        parsed = HumanEvalRawRowsSnapshot.model_validate_json(
            self.snapshot_bytes
        )
        validate_snapshot_header(
            parsed.header,
            dataset_name=parsed.header.dataset_id,
            hf_revision=DEFAULT_HUMAN_EVAL_HF_REVISION,
        )
        observed_identity = _snapshot_identity(
            self.snapshot_bytes,
            parsed.header,
        )
        if self.identity != observed_identity:
            raise ValueError("snapshot identity must match snapshot bytes")
        observed_rows = tuple(
            row.model_dump(mode="json") for row in parsed.rows
        )
        if self.rows != observed_rows:
            raise ValueError("snapshot rows must match snapshot bytes")
        return self

    @model_validator(mode="after")
    def validate_initial_content_coupling(self) -> HumanEvalSnapshot:
        return self.validate_content_coupling()


def resolve_snapshot_path(snapshot_path: str | Path) -> Path:
    path = Path(snapshot_path)
    if path.is_absolute():
        return path
    return (DEFAULT_REPO_ROOT / path).resolve()


def load_humaneval_snapshot(
    *,
    dataset_name: str,
    dataset_split: str,
    snapshot_path: str | Path,
    expected_identity: DatasetSnapshotIdentityPayload | None = None,
) -> HumanEvalSnapshot:
    resolved_path = resolve_snapshot_path(snapshot_path)
    snapshot_bytes = resolved_path.read_bytes()
    return snapshot_from_bytes(
        snapshot_bytes,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        expected_identity=expected_identity,
    )


def snapshot_from_bytes(
    snapshot_bytes: bytes,
    *,
    dataset_name: str,
    dataset_split: str,
    expected_identity: DatasetSnapshotIdentityPayload | None = None,
) -> HumanEvalSnapshot:
    snapshot = HumanEvalRawRowsSnapshot.model_validate_json(snapshot_bytes)
    validate_snapshot_header(
        snapshot.header,
        dataset_name=dataset_name,
        hf_revision=DEFAULT_HUMAN_EVAL_HF_REVISION,
    )
    identity = _snapshot_identity(snapshot_bytes, snapshot.header)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(
            "dataset snapshot identity does not match registration"
        )
    rows = tuple(row.model_dump(mode="json") for row in snapshot.rows)
    parse_human_eval_dataset(rows)
    return HumanEvalSnapshot(
        identity=identity,
        rows=rows,
        snapshot_bytes=snapshot_bytes,
    )


def sample_humaneval_snapshot_tasks(
    *,
    dataset_name: str,
    dataset_split: str,
    snapshot_path: str | Path,
    seed: int,
    sample_count: int,
) -> tuple[SampledHumanEvalTask, ...]:
    snapshot = load_humaneval_snapshot(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        snapshot_path=snapshot_path,
    )
    return tuple(
        sample_human_eval_tasks_from_rows(
            snapshot.rows,
            seed=seed,
            sample_count=sample_count,
        )
    )


def _header_payload(
    header: HumanEvalRawRowsSnapshotHeader,
) -> DatasetSnapshotHeaderPayload:
    return DatasetSnapshotHeaderPayload.model_validate(
        header.model_dump(mode="json")
    )


def _snapshot_identity(
    snapshot_bytes: bytes,
    header: HumanEvalRawRowsSnapshotHeader,
) -> DatasetSnapshotIdentityPayload:
    return DatasetSnapshotIdentityPayload(
        sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
        header=_header_payload(header),
    )
