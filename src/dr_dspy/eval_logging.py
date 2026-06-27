from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from rich.console import Console

from dr_dspy import dbos_runtime
from dr_dspy.lm_utils import stable_json

DEFAULT_OPERATOR_TIMESTAMP_FORMAT = "%H:%M:%S"


class PredictionLogContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    repetition_seed: StrictInt
    dimensions: dict[str, Any] = Field(default_factory=dict)


def sanitize_log_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-.")
    return sanitized or "experiment"


def hashed_experiment_log_name(
    experiment_name: str, *, hash_length: int
) -> str:
    experiment_hash = dbos_runtime.experiment_hash(
        experiment_name, hash_length=hash_length
    )
    return (
        f"{sanitize_log_name(experiment_name)}-"
        f"{experiment_hash}"
    )


def default_worker_log_path(
    *,
    log_root: Path,
    experiment_name: str,
    queue: dbos_runtime.QueueSelection,
    hash_length: int,
    now: datetime | None = None,
    pid: int | None = None,
) -> Path:
    resolved_now = now or datetime.now()
    resolved_pid = pid if pid is not None else os.getpid()
    filename = (
        f"{resolved_now:%Y%m%d-%H%M%S}-{queue.value}-pid"
        f"{resolved_pid}.log"
    )
    return (
        log_root
        / hashed_experiment_log_name(
            experiment_name, hash_length=hash_length
        )
        / filename
    )


def resolve_worker_log_path(
    *,
    log_root: Path,
    experiment_name: str,
    queue: dbos_runtime.QueueSelection,
    log_file: Path | None,
    hash_length: int,
) -> Path:
    if log_file is not None:
        return log_file
    return default_worker_log_path(
        log_root=log_root,
        experiment_name=experiment_name,
        queue=queue,
        hash_length=hash_length,
    )


def configure_worker_file_logging(
    log_file: Path, *, logger_name: str
) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def emit_worker_detail_log(
    event: str, payload: Mapping[str, Any], *, logger_name: str
) -> None:
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        return
    logger.info(stable_json({"event": event, **payload}))


def emit_prediction_log_event(
    event: str,
    context: PredictionLogContext,
    *,
    logger_name: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload = context.model_dump(mode="json")
    if extra is not None:
        payload.update(extra)
    emit_worker_detail_log(event, payload, logger_name=logger_name)


def operator_timestamp(
    now: datetime | None = None,
    *,
    timestamp_format: str = DEFAULT_OPERATOR_TIMESTAMP_FORMAT,
) -> str:
    resolved_now = now or datetime.now()
    return resolved_now.strftime(timestamp_format)


def timestamped_line(
    line: str,
    *,
    now: datetime | None = None,
    timestamp_format: str = DEFAULT_OPERATOR_TIMESTAMP_FORMAT,
) -> str:
    timestamp = operator_timestamp(now, timestamp_format=timestamp_format)
    return f"{timestamp} | {line}"


def operator_log(
    console: Console,
    line: str,
    *,
    style: str | None = None,
    now: datetime | None = None,
    timestamp_format: str = DEFAULT_OPERATOR_TIMESTAMP_FORMAT,
) -> None:
    console.print(
        timestamped_line(line, now=now, timestamp_format=timestamp_format),
        style=style,
    )
