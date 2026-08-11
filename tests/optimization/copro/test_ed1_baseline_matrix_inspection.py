from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest
from dr_store import MemoryBackend, ObjectStore

from tests.envs.support import execution_policy, synthetic_ed1_tasks
from tests.provider import support as provider_support
from whetstone.envs.ed1_runtime import Ed1RuntimeProbe
from whetstone.envs.ed1_scoring import CodeScore, CodeScoringInput
from whetstone.envs.encdec_rollout import build_encoder_provider_call_config
from whetstone.evaluation.code.power import PowerConfig
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.execution.prompt_cache import (
    PromptResultCache,
    execute_call,
    prompt_cache_key,
)
from whetstone.optimization.copro.ed1_baseline_preview import (
    Ed1BaselinePreviewTranscript,
    run_ed1_baseline_preview,
)
from whetstone.optimization.copro.ed1_scoring_preview import (
    Ed1ScoringRuntimeSummary,
)
from whetstone.optimization.copro.ed1_task_model import (
    Ed1TaskModelConfig,
    Ed1TaskModelKind,
)

_SCRIPT = (
    Path(__file__).parents[3]
    / "scripts"
    / "experiments"
    / "inspect_baseline_behavior_matrix.py"
)


def _load_script() -> ModuleType:
    name = "test_inspect_baseline_behavior_matrix"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _score(
    inputs: Sequence[CodeScoringInput],
    *,
    max_wall_seconds: float | None = None,
) -> tuple[CodeScore, ...]:
    del max_wall_seconds
    return tuple(
        CodeScore(
            passed=True,
            infrastructure_unknown=False,
            outcome="passed",
        )
        for _ in inputs
    )


def _runtime() -> Ed1ScoringRuntimeSummary:
    return Ed1ScoringRuntimeSummary(
        evaluation_python="/isolated/python",
        dr_code_version="0.1.5",
        runtime_identity_hash="a" * 64,
        probe=Ed1RuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable="/isolated/python",
            python_version="3.13.0",
        ),
    )


def _task_model() -> Ed1TaskModelConfig:
    return Ed1TaskModelConfig(
        kind=Ed1TaskModelKind.DUMMY,
        provider_call_config=build_encoder_provider_call_config(
            "test/task-model"
        ),
        execution_policy=execution_policy(),
    )


def _transcript() -> Ed1BaselinePreviewTranscript:
    tasks = synthetic_ed1_tasks(1)
    return run_ed1_baseline_preview(
        store=ObjectStore(MemoryBackend()),
        tasks=tasks,
        task_ids=("Synthetic/0",),
        pool_ceiling=1,
        task_model=_task_model(),
        batch_scorer=_score,
        runtime=_runtime(),
        budget_ratio=0.5,
        repeats=2,
        power_config=PowerConfig(trials=1, repeat_cap=2),
        bootstrap_resamples=2,
    )


def _write_run(tmp_path: Path) -> tuple[Path, Ed1BaselinePreviewTranscript]:
    output = tmp_path / "matrix"
    treatment = output / "treatment-1"
    treatment.mkdir(parents=True)
    transcript = _transcript()
    (treatment / "result.json").write_text(
        transcript.model_dump_json(), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "mode": "smoke",
        "task_ids": ["Synthetic/0"],
        "repeats": 2,
        "concurrency": 1,
        "treatments": [
            {
                "treatment_id": "treatment-1",
                "directory": "treatment-1",
                "budget_ratio": 0.5,
                "task_model": transcript.task_model.model_dump(mode="json"),
                "planned_rows": 4,
                "planned_provider_calls": 8,
            }
        ],
    }
    (output / "run-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    events = (
        {"schema_version": 1, "state": "run_started"},
        {
            "schema_version": 1,
            "state": "treatment_completed",
            "treatment_id": "treatment-1",
        },
        {"schema_version": 1, "state": "run_completed"},
    )
    (output / "process-log.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    _write_provider_cache(treatment, transcript)
    PartialLog(treatment / "partial-log").append(
        PartialCallRecord(
            phase="internal",
            instance_id="Synthetic/0",
            unit="baseline",
            repeat_id=0,
            request_identity="b" * 64,
            redrive_pending=False,
            score=1.0,
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            output_text="ok",
            raw_response="raw",
        )
    )
    record_dir = output / "execution-records" / "run-1"
    record_dir.mkdir(parents=True)
    (record_dir / "record.json").write_text(
        json.dumps(
            {
                "state": "finalized",
                "result": {
                    "execution_id": {"job_id": "job-1"},
                    "measurements": {"duration_ns": 12},
                    "outcome": {"kind": "exited", "exit_code": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    connection = sqlite3.connect(output / "execution-cache.sqlite3")
    try:
        connection.execute("CREATE TABLE scores (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO scores DEFAULT VALUES")
        connection.commit()
    finally:
        connection.close()
    return output, transcript


def _write_provider_cache(
    treatment: Path, transcript: Ed1BaselinePreviewTranscript
) -> None:
    request = provider_support.build_request(content="exact encoder prompt")
    policy = transcript.task_model.execution_policy
    candidate_id = transcript.baseline.evidence.candidate.record.candidate_id
    logical_call_id = f"{candidate_id}:Synthetic/0#0:enc"
    cache = PromptResultCache(treatment / "prompt-cache")
    key = prompt_cache_key(request, policy, 0, 0)

    def transport(provider_request):
        return provider_support.build_evidence(
            request=provider_request,
            policy=policy.transport_policy,
            outcome=provider_support.response_outcome(text="encoded"),
        )

    executed = execute_call(
        request=request,
        policy=policy,
        transport=transport,
        logical_call_id=logical_call_id,
        repeat_index=0,
        drive_ordinal=0,
        cache=cache,
        phase="internal",
        unit=candidate_id,
        clock=provider_support.FakeClock(),
        sleep=provider_support.SleepRecorder(),
    )
    assert executed.result.succeeded
    assert cache.get_result(key) is not None


def test_inspection_validates_and_exports_complete_matrix_evidence(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output, transcript = _write_run(tmp_path)

    report = module.inspect_matrix(output)

    assert report["manifest"] == {
        "mode": "smoke",
        "treatment_count": 1,
        "completed_result_count": 1,
        "run_completed": True,
        "integrity": "validated",
    }
    assert len(report["rows"]) == 4
    assert all(row["encoder_prompt"] for row in report["rows"])
    assert all(row["encoder_generation"] for row in report["rows"])
    assert all(row["decoder_prompt"] for row in report["rows"])
    assert all(row["decoder_generation"] for row in report["rows"])
    assert all(isinstance(row["over_budget"], bool) for row in report["rows"])
    assert report["treatments"][0]["budget"]["compliance_available"] == 4
    treatment = report["treatments"][0]
    assert treatment["paired_delta_ci"] == asdict(transcript.paired_delta_ci)
    assert treatment["power"]["recommendation"] == (
        asdict(transcript.power.recommendation)
    )
    assert len(report["paired_deltas"]) == 1
    assert report["paired_deltas"][0]["direction"] == ("HUMAN_BEST-BASELINE")
    assert len(report["provider_calls"]) == 1
    provider = report["provider_calls"][0]
    assert provider["returned_model"] == "test-model"
    assert provider["returned_model_availability"] == "available"
    assert provider["raw_response_availability"] == "available"
    assert provider["upstream_provider_availability"] == "unavailable"
    assert provider["telemetry"]["latency_s"] == 0.5
    linked = next(
        row
        for row in report["rows"]
        if row["arm"] == "BASELINE" and row["repeat"] == 0
    )
    assert linked["encoder_provider_cache_key"] == provider["cache_key"]
    assert len(report["partials"]) == 1
    assert report["execution_inventory"]["record_count"] == 1
    assert report["execution_inventory"]["outcomes"] == {"exited": 1}
    assert report["execution_inventory"]["databases"][0]["tables"] == {
        "scores": 1
    }
    inspection = output / "inspection"
    assert {path.name for path in inspection.iterdir()} == set(
        module.REPORT_FILES
    )
    assert module.main([str(output)]) == 0
    assert module.main(["power", str(output)]) == 0


def test_inspection_rejects_result_count_that_disagrees_with_status_log(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output, _ = _write_run(tmp_path)
    (output / "process-log.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "treatment_failed",
                "treatment_id": "treatment-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(module.InspectionError, match="completed results"):
        module.inspect_matrix(output)


def test_inspection_rejects_tampered_stored_per_task_reward(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output, transcript = _write_run(tmp_path)
    raw = transcript.model_dump(mode="json")
    raw["baseline"]["evidence"]["per_task_values"] = [0.0]
    (output / "treatment-1" / "result.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )

    with pytest.raises(module.InspectionError, match="stored per_task"):
        module.inspect_matrix(output)
