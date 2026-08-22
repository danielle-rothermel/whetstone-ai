"""Tier-2 CLI: whetstone-optim run / status / result against seeded store + Postgres."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from dr_store.sync import open_sqlite

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.platform.contracts import OptimPlatformRunResult
from whetstone.testing.runtime import (
    build_toy_copro_control,
    prepare_toy_copro_run,
    register_toy_runtime,
)

pytestmark = pytest.mark.integration


def _json_payload(output: str) -> dict:
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise AssertionError(f"CLI output had no JSON object:\n{output}")
    return json.loads(output[start : end + 1])


def _invoke(*args: str, database_url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "whetstone.platform.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "WHETSTONE_DATABASE_URL": database_url},
    )


def test_whetstone_optim_cli_run_status_result(
    pg_engine,
    clean_pg: str,
    tmp_path,
) -> None:
    store_path = tmp_path / "cli-store.sqlite"
    run_id = "cli-seeded-run"
    run_key = "cli-platform-run"
    campaign_key = "cli-campaign"
    with open_sqlite(str(store_path)) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
        runtime = register_toy_runtime(
            store=store,
            engine=engine,
            copro_control=control,
        )
        prepare_toy_copro_run(
            runtime,
            run_id=run_id,
            control=control,
            terminal_top_k=1,
        )
        runtime.close()

    shared = [
        "--store-path",
        str(store_path),
        "--database-url",
        clean_pg,
        "--run-key",
        run_key,
    ]
    submitted = _invoke(
        "run",
        "--run-id",
        run_id,
        "--campaign-key",
        campaign_key,
        "--adapter",
        "copro",
        "--proposer",
        "fake",
        "--application-version",
        "cli-test-v1",
        "--executor-id",
        "cli-test-exec",
        *shared,
        database_url=clean_pg,
    )
    assert submitted.returncode == 0, submitted.stdout + submitted.stderr
    receipt = _json_payload(submitted.stdout)
    assert receipt["run_key"] == run_key
    assert receipt["registered_member_count"] == 1
    assert receipt["membership_digest"]

    status = _invoke("status", *shared, database_url=clean_pg)
    assert status.returncode == 0, status.stdout + status.stderr
    status_payload = _json_payload(status.stdout)
    assert status_payload["run_key"] == run_key
    assert status_payload["released"] is True
    assert status_payload["members"][0]["run_id"] == run_id

    result = _invoke("result", *shared, database_url=clean_pg)
    assert result.returncode == 0, result.stdout + result.stderr
    run_result = OptimPlatformRunResult.model_validate(_json_payload(result.stdout))
    assert run_result.platform_run_key == run_key
    assert run_result.member_results[0].run_id == run_id
