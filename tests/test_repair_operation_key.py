from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

from dr_dspy import batch_operation, dbos_runtime
from dr_dspy import humaneval_direct_dbos as direct
from dr_dspy import humaneval_encdec_dbos as encdec


class _ExperimentConfig(BaseModel):
    script_kind: str
    default_subprocess_timeout: float


def _config() -> dbos_runtime.EvalDbosConfig:
    return dbos_runtime.EvalDbosConfig(
        database_url="postgresql://db",
        dbos_system_database_url="postgresql://sys",
        generation_concurrency=7,
        scoring_concurrency=8,
    )


def _completed_progress(
    *,
    operation_kind: batch_operation.BatchOperationKind,
    operation_key: str,
) -> batch_operation.BatchOperationProgress:
    return batch_operation.BatchOperationProgress(
        operation_kind=operation_kind,
        operation_key=operation_key,
        experiment_name="repair-exp",
        script_kind="script",
        workflow_id=f"{operation_kind.value}:{operation_key}:1",
        attempt=1,
        status=batch_operation.BatchOperationStatus.COMPLETED,
        total_items=5,
        next_offset=0,
        metadata={},
        processed_count=0,
        inserted_count=0,
        enqueued_count=0,
        existing_workflow_count=0,
        marked_count=0,
        batch_count=0,
        counters={},
        last_error=None,
        log_file="/tmp/repair.log",
    )


def _patch_repair_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    config_builder_name: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    prepared_operation_keys: list[str] = []
    operation_specs: list[dict[str, Any]] = []

    monkeypatch.setattr(
        module,
        "experiment_config",
        lambda: _ExperimentConfig(
            script_kind="script", default_subprocess_timeout=3.0
        ),
    )
    monkeypatch.setattr(module, "load_optional_env_file", lambda _path: None)
    monkeypatch.setattr(
        module, config_builder_name, lambda **_kwargs: _config()
    )
    monkeypatch.setattr(module, "_create_eval_schema", lambda _url: None)
    monkeypatch.setattr(
        module,
        "_resolve_operation_log_path",
        lambda **_kwargs: Path("/tmp/repair.log"),
    )
    monkeypatch.setattr(
        module, "_configure_operation_file_logging", lambda _path: None
    )
    monkeypatch.setattr(
        module, "_configure_dbos_runtime", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module, "_emit_operation_log", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module.shared_batch,
        "ensure_operation_workflow",
        lambda **_kwargs: True,
    )

    def prepare_operation(
        _database_url: str,
        *,
        operation_kind: batch_operation.BatchOperationKind,
        operation_key: str,
        spec: dict[str, Any],
        **_kwargs: Any,
    ) -> batch_operation.BatchOperationProgress:
        prepared_operation_keys.append(operation_key)
        operation_specs.append(spec)
        return _completed_progress(
            operation_kind=operation_kind,
            operation_key=operation_key,
        )

    def tail_operation_progress(
        *,
        operation_kind: batch_operation.BatchOperationKind,
        operation_key: str,
        **_kwargs: Any,
    ) -> batch_operation.BatchOperationProgress:
        return _completed_progress(
            operation_kind=operation_kind,
            operation_key=operation_key,
        )

    monkeypatch.setattr(
        module.shared_batch, "prepare_operation", prepare_operation
    )
    monkeypatch.setattr(
        module.shared_batch, "tail_operation_progress", tail_operation_progress
    )
    return prepared_operation_keys, operation_specs


def _call_repair_command(
    command: Callable[..., None],
    *,
    operation_key: str | None,
) -> None:
    command(
        experiment_name="repair-exp",
        apply=True,
        score_timeout=1.5,
        database_url=None,
        dbos_system_database_url=None,
        generation_concurrency=7,
        scoring_concurrency=8,
        batch_size=11,
        operation_key=operation_key,
        env_file=None,
    )


@pytest.mark.parametrize(
    ("module", "command", "config_builder_name"),
    (
        (direct, direct.repair_command, "common_config"),
        (encdec, encdec.repair, "_build_eval_dbos_config"),
    ),
)
def test_repair_apply_uses_fresh_operation_key_by_default(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    command: Callable[..., None],
    config_builder_name: str,
) -> None:
    prepared_operation_keys, operation_specs = _patch_repair_dependencies(
        monkeypatch, module, config_builder_name=config_builder_name
    )
    monkeypatch.setattr(
        module.shared_batch, "new_operation_key", lambda: "fresh-key"
    )

    def stable_operation_key(_spec: dict[str, Any]) -> str:
        raise AssertionError("repair --apply default must not hash the spec")

    monkeypatch.setattr(
        module.shared_batch, "operation_key", stable_operation_key
    )

    _call_repair_command(command, operation_key=None)

    assert prepared_operation_keys == ["fresh-key"]
    assert operation_specs[0] == {
        "experiment_name": "repair-exp",
        "score_timeout": 1.5,
        "batch_size": 11,
        "database_url": "postgresql://db",
        "dbos_system_database_url": "postgresql://sys",
        "generation_concurrency": 7,
        "scoring_concurrency": 8,
    }


@pytest.mark.parametrize(
    ("module", "command", "config_builder_name"),
    (
        (direct, direct.repair_command, "common_config"),
        (encdec, encdec.repair, "_build_eval_dbos_config"),
    ),
)
def test_repair_apply_preserves_explicit_operation_key(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    command: Callable[..., None],
    config_builder_name: str,
) -> None:
    prepared_operation_keys, _operation_specs = _patch_repair_dependencies(
        monkeypatch, module, config_builder_name=config_builder_name
    )

    def new_operation_key() -> str:
        raise AssertionError("explicit repair operation key should be reused")

    monkeypatch.setattr(
        module.shared_batch, "new_operation_key", new_operation_key
    )

    _call_repair_command(command, operation_key="resume-key")

    assert prepared_operation_keys == ["resume-key"]
