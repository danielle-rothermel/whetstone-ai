"""Shared JSONL helpers for platform submit tests."""

from __future__ import annotations

from pathlib import Path

from whetstone.records import PredictionSpecRecord


def write_prediction_specs_jsonl(
    path: Path,
    specs: tuple[PredictionSpecRecord, ...],
) -> None:
    lines = [spec.model_dump_json() for spec in specs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
