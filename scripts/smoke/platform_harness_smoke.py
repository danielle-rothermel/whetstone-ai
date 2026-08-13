#!/usr/bin/env -S uv run python
"""Minimal smoke checks for platform harness Tier 1 paths."""

from __future__ import annotations

import importlib
import tempfile

from dr_store.content_addressing import parse_object_reference

from whetstone.core.blocking_store import open_blocking_sqlite_store
from whetstone.coordination.runtime_bootstrap import prepare_copro_run, register_runtime
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.platform.contracts import (
    STAGE_OPTIM_STEP,
    STAGE_RUN_COMPLETION,
    OptimWorkInput,
    persist_work_input,
)
from whetstone.platform.step_executor import (
    execute_optim_step_sync,
    execute_run_completion_sync,
)


def _bootstrap_import_smoke() -> None:
    importlib.import_module("whetstone.coordination.runtime_bootstrap")
    importlib.import_module("whetstone.coordination.run_controller_registry")


def _inline_platform_driver_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store_path = f"{tmp}/platform-smoke.sqlite"
        with open_blocking_sqlite_store(store_path) as store:
            runtime = register_runtime(store=store)
            control = runtime.adapter_registry.resolve("copro").control  # type: ignore[attr-defined]
            run_id = "platform-smoke-run"
            launch = prepare_copro_run(
                runtime,
                run_id=run_id,
                control=control,
                terminal_top_k=1,
            )
            runtime.controller.bind_launch(launch)
            work_input = OptimWorkInput(
                run_id=run_id,
                controller_identity_hash=runtime.controller.runtime_hash,
                control_identity_hash=control.identity_hash(),
            )
            current_ref = persist_work_input(runtime.store, work_input)
            stage_index = 0
            while True:
                completion = execute_optim_step_sync(
                    runtime,
                    input_reference=current_ref,
                )
                if not completion.successors:
                    break
                successor = completion.successors[0]
                if successor.stage_key.value == STAGE_RUN_COMPLETION:
                    terminal_ref = execute_run_completion_sync(
                        runtime,
                        input_reference=successor.input_reference,
                    )
                    parsed = parse_object_reference(terminal_ref)
                    assert parsed.schema == OPTIM_RESULT_SCHEMA
                    result = OptimResult.model_validate(
                        runtime.store.get(parsed)
                    )
                    assert result.run.record.run_id == run_id
                    assert result.proposals
                    return
                assert successor.stage_key.value == STAGE_OPTIM_STEP
                assert successor.stage_index == stage_index + 1
                current_ref = successor.input_reference
                stage_index = successor.stage_index
            raise RuntimeError("platform smoke did not reach run completion")


def main() -> None:
    _bootstrap_import_smoke()
    print("bootstrap import smoke ok")
    _inline_platform_driver_smoke()
    print("inline platform driver smoke ok")


if __name__ == "__main__":
    main()
