"""Typed contract shared by the HumanEval eval experiments.

Each experiment (`DirectExperiment`, `EncDecExperiment`) implements this
protocol. The shared command orchestrators in `humaneval_dbos_flow` take a
single `ExperimentBackend` instead of a pile of injected callbacks, and the
per-experiment `@DBOS.step` bodies call backend methods directly. This keeps
type-checking intact and puts each experiment's behavior in one place, while
the experiment-specific SQL stays explicit on the concrete class.

`JobT` (the per-experiment prediction job) and the generation-result type are
opaque to the shared layer, so they are typed `Any` here; scoring always
yields `HumanEvalScoreResult`.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from dr_dspy.dbos_runtime import (
    EnqueueWorkflowsResult,
    EvalDbosConfig,
    QueueSelection,
)
from dr_dspy.eval_logging import PredictionLogContext
from dr_dspy.eval_repair import RepairApplyResult, RepairPlan
from dr_dspy.experiment_dimensions import Dimension
from dr_dspy.scoring import HumanEvalScoreResult
from dr_dspy.worker_monitor import WorkerMonitorConfig


class ExperimentBackend(Protocol):
    # --- spec / identity ---
    @property
    def prediction_table(self) -> str: ...
    @property
    def dimensions(self) -> tuple[Dimension, ...]: ...

    # --- submit / schema ---
    def create_schema(self, database_url: str) -> None: ...
    def upsert_experiment(
        self,
        database_url: str,
        *,
        experiment_name: str,
        seed: int,
        sample_count: int,
        metadata: Mapping[str, Any],
    ) -> None: ...
    def insert_prediction_jobs(
        self, database_url: str, jobs: Sequence[Any]
    ) -> int: ...
    def configure_runtime(
        self,
        config: EvalDbosConfig,
        experiment_name: str,
        *,
        consume_queues: bool = True,
    ) -> None: ...
    def enqueue_generation_jobs(
        self,
        database_url: str,
        jobs: Sequence[Any],
        *,
        score_timeout: float,
        retry_token: str | None = None,
    ) -> EnqueueWorkflowsResult: ...

    # --- generation step ops ---
    def mark_generation_started(
        self, database_url: str, prediction_id: str
    ) -> None: ...
    def fetch_prediction_job(
        self, database_url: str, prediction_id: str
    ) -> Any: ...
    def generate_code_for_job(self, job: Any) -> Any: ...
    def record_generation_success(
        self, database_url: str, result: Any
    ) -> None: ...
    def record_generation_error(
        self, database_url: str, prediction_id: str, error: str
    ) -> None: ...
    def generation_success_log_extra(
        self, result: Any
    ) -> Mapping[str, Any]: ...

    # --- logging ---
    def prediction_context_from_job(
        self, job: Any
    ) -> PredictionLogContext: ...
    def fetch_prediction_log_context(
        self, database_url: str, prediction_id: str
    ) -> PredictionLogContext: ...
    def emit_prediction_log_event(
        self,
        event: str,
        context: PredictionLogContext,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None: ...

    # --- scoring step ops ---
    def mark_scoring_started(
        self, database_url: str, prediction_id: str
    ) -> None: ...
    def mark_scoring_queued(
        self, database_url: str, prediction_ids: Sequence[str]
    ) -> int: ...
    def score_prediction(
        self, database_url: str, prediction_id: str, timeout: float
    ) -> HumanEvalScoreResult: ...
    def record_score_success(
        self, database_url: str, result: HumanEvalScoreResult
    ) -> None: ...
    def record_score_error(
        self, database_url: str, prediction_id: str, error: str
    ) -> None: ...

    # --- enqueue ---
    def enqueue_score(
        self,
        database_url: str,
        prediction_id: str,
        *,
        experiment_name: str,
        timeout: float,
    ) -> None: ...
    def enqueue_score_jobs(
        self,
        database_url: str,
        prediction_ids: Sequence[str],
        *,
        experiment_name: str,
        timeout: float,
        retry_token: str | None = None,
    ) -> None: ...

    # --- repair ---
    def fetch_scoreable_prediction_ids(
        self, database_url: str, *, experiment_name: str, limit: int
    ) -> list[str]: ...
    def build_repair_plan(
        self,
        database_url: str,
        *,
        dbos_system_database_url: str,
        experiment_name: str,
        generation_limit: int,
        scoring_limit: int,
    ) -> RepairPlan: ...
    def apply_repair(
        self,
        config: EvalDbosConfig,
        *,
        experiment_name: str,
        generation_limit: int,
        scoring_limit: int,
        score_timeout: float,
        repair_token: str | None = None,
    ) -> RepairApplyResult: ...

    # --- status / analysis ---
    def fetch_status_counts(
        self, database_url: str, *, experiment_name: str | None
    ) -> list[dict[str, Any]]: ...
    def status_counts_table(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        experiment_name: str | None,
    ) -> object: ...
    def fetch_analysis_records(
        self, database_url: str, *, experiment_name: str
    ) -> Sequence[Any]: ...
    def summarize_analysis_records(
        self, records: Sequence[Any]
    ) -> Sequence[Any]: ...
    def analysis_table(
        self, *, experiment_name: str, summaries: Sequence[Any]
    ) -> object: ...
    def analysis_markdown(
        self, *, experiment_name: str, summaries: Sequence[Any]
    ) -> str: ...
    def write_analysis_csv(
        self, summaries: Sequence[Any], csv_path: Path
    ) -> None: ...

    # --- worker ---
    def configure_pooled_worker_runtime(
        self,
        config: EvalDbosConfig,
        *,
        experiment_name: str,
        queue: QueueSelection,
        raw_db_pool_max_size: str,
    ) -> Any: ...
    def queue_names_for_selection(
        self, selection: QueueSelection, *, experiment_name: str
    ) -> tuple[str, ...]: ...
    def resolve_worker_log_path(
        self,
        *,
        experiment_name: str,
        queue: QueueSelection,
        log_file: Path | None,
    ) -> Path: ...
    def configure_worker_file_logging(self, log_file: Path) -> object: ...
    def start_worker_monitor(
        self,
        monitor_config: WorkerMonitorConfig,
        stop_event: threading.Event,
    ) -> threading.Thread: ...
