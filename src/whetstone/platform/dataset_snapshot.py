"""HumanEval raw-row snapshot loading for platform specs and scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from dr_code.humaneval import parse_human_eval_dataset
from dr_code.humaneval.sampling import (
    HumanEvalRawRowsSnapshot,
    HumanEvalRawRowsSnapshotHeader,
    SampledHumanEvalTask,
    load_human_eval_rows,
    sample_human_eval_tasks_from_rows,
)
from pydantic import BaseModel, ConfigDict

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


def resolve_snapshot_path(snapshot_path: str | Path) -> Path:
    path = Path(snapshot_path)
    if path.is_absolute():
        return path
    return (DEFAULT_REPO_ROOT / path).resolve()


def read_snapshot_identity(
    snapshot_path: str | Path,
) -> DatasetSnapshotIdentityPayload:
    resolved_path = resolve_snapshot_path(snapshot_path)
    snapshot_bytes = resolved_path.read_bytes()
    snapshot = HumanEvalRawRowsSnapshot.model_validate_json(snapshot_bytes)
    return DatasetSnapshotIdentityPayload(
        source_path=str(resolved_path),
        sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
        header=_header_payload(snapshot.header),
    )


def load_humaneval_snapshot(
    *,
    dataset_name: str,
    dataset_split: str,
    snapshot_path: str | Path,
) -> HumanEvalSnapshot:
    resolved_path = resolve_snapshot_path(snapshot_path)
    identity = read_snapshot_identity(resolved_path)
    rows = tuple(
        dict(row)
        for row in load_human_eval_rows(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            snapshot_path=resolved_path,
        )
    )
    parse_human_eval_dataset(rows)
    return HumanEvalSnapshot(identity=identity, rows=rows)


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
