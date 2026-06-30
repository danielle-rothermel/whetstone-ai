"""Legacy v0 typed contract for shared HumanEval command orchestration.

This protocol belongs to the old DBOS experiment surfaces. Keep it available
for v0 data migration and validation, not as the new graph platform boundary.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol

from dr_dspy.harness.dbos import EvalDbosConfig, QueueSelection
from dr_dspy.harness.repair import RepairPlan
from dr_dspy.harness.workers.monitor import WorkerMonitorConfig


class ExperimentBackend(Protocol):
    # --- spec / identity ---
    @property
    def prediction_table(self) -> str: ...

    def create_schema(self, database_url: str) -> None: ...
    def configure_runtime(
        self,
        config: EvalDbosConfig,
        experiment_name: str,
        *,
        queue: QueueSelection | None = None,
        consume_queues: bool = True,
    ) -> None: ...

    # --- repair ---
    def build_repair_plan(
        self,
        database_url: str,
        *,
        dbos_system_database_url: str,
        experiment_name: str,
    ) -> RepairPlan: ...
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
        halt_event: threading.Event,
    ) -> threading.Thread: ...
