from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from dr_providers import ProviderKind, ReasoningEffort
from pydantic import ValidationError

from whetstone.envs.ed1_preview import Ed1ScoringRuntimeSummary
from whetstone.envs.ed1_runtime import Ed1RuntimeProbe
from whetstone.envs.task_pools import (
    select_lowest_historical_pass_rate_for_env,
)
from whetstone.evaluation.schema import RowAccounting
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitRole,
    load_task_split_manifest,
)
from whetstone.optimization.copro import ed1_baseline_matrix as matrix
from whetstone.optimization.copro.ed1_baseline_matrix import (
    EXCLUDED_TASK_IDS,
    FULL_BUDGET_RATIOS,
    Ed1BaselineMatrixPlan,
    Ed1BaselineTreatmentStatus,
    TreatmentState,
    baseline_provider_routes,
    build_matrix_plan,
    map_openai_credential,
    parse_args,
    raise_open_file_limit,
)

_EXPECTED_TASK_IDS = (
    "HumanEval/32",
    "HumanEval/163",
    "HumanEval/160",
    "HumanEval/124",
    "HumanEval/132",
    "HumanEval/151",
    "HumanEval/86",
    "HumanEval/120",
    "HumanEval/76",
    "HumanEval/55",
)


def _runtime(evaluation_python: Path) -> Ed1ScoringRuntimeSummary:
    return Ed1ScoringRuntimeSummary(
        evaluation_python=str(evaluation_python),
        dr_code_version="0.1.5",
        runtime_identity_hash="a" * 64,
        probe=Ed1RuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable=str(evaluation_python),
            python_version="3.13.0",
        ),
    )


def _selection(
    task_ids: tuple[str, ...] = _EXPECTED_TASK_IDS,
) -> TaskRoleSelection:
    return TaskRoleSelection(
        manifest_content_hash="b" * 64,
        pool_key="ed1",
        role=TaskSplitRole.TRAIN,
        task_ids=task_ids,
        source_role_count=46,
        eligible_pool_count=41,
        excluded_task_ids=EXCLUDED_TASK_IDS,
        historical_pass_rates=tuple(0.5 for _ in task_ids),
    )


def _plan(
    tmp_path: Path,
    *,
    mode: Literal["full", "smoke"] = "full",
) -> Ed1BaselineMatrixPlan:
    evaluation_python = tmp_path / "python"
    snapshot = tmp_path / "snapshot.json"
    manifest = tmp_path / "tasks.json"
    evaluation_python.write_bytes(b"python")
    snapshot.write_bytes(b"snapshot")
    manifest.write_bytes(b"manifest")
    task_selection = (
        _selection()
        if mode == "full"
        else _selection((_EXPECTED_TASK_IDS[0],))
    )
    return build_matrix_plan(
        mode=mode,
        evaluation_python=evaluation_python,
        snapshot_path=snapshot,
        task_manifest_path=manifest,
        output_dir=tmp_path / "output",
        task_selection=task_selection,
        concurrency=100,
        runtime=_runtime(evaluation_python),
    )


def test_provider_routes_are_the_exact_four_treatments() -> None:
    routes = baseline_provider_routes()

    assert tuple((route.lane, route.model) for route in routes) == (
        ("openai", "gpt-5.4-nano"),
        ("openrouter", "deepseek/deepseek-v4-flash"),
        ("openrouter", "qwen/qwen3-coder-flash"),
        ("openrouter", "google/gemini-3.1-flash-lite"),
    )
    assert tuple(
        route.call_config.definition.route.provider for route in routes
    ) == (
        ProviderKind.OPENAI,
        ProviderKind.OPENROUTER,
        ProviderKind.OPENROUTER,
        ProviderKind.OPENROUTER,
    )
    assert all(
        route.call_config.controls.temperature is None for route in routes
    )
    assert all(
        route.call_config.controls.token_limit == 4096 for route in routes
    )
    assert tuple(route.call_config.controls.reasoning for route in routes) == (
        ReasoningEffort.NONE,
        ReasoningEffort.NONE,
        None,
        ReasoningEffort.NONE,
    )
    assert all(route.execution_policy.max_attempts == 1 for route in routes)


def test_frozen_manifest_selection_matches_declared_screen() -> None:
    manifest = load_task_split_manifest(
        Path(__file__).parents[3]
        / "src/whetstone/optimization/copro/humaneval_copro_challenge_v1.json"
    )

    selection = select_lowest_historical_pass_rate_for_env(
        manifest,
        env="ed1",
        role=TaskSplitRole.TRAIN,
        count=10,
        excluded_task_ids=EXCLUDED_TASK_IDS,
    )

    assert selection.task_ids == _EXPECTED_TASK_IDS
    assert selection.excluded_task_ids == EXCLUDED_TASK_IDS
    assert selection.manifest_content_hash == (
        "fb0db70a652f070869080c13b60e067829eb2db36d86c36a625b90226602d8d2"
    )


def test_full_plan_has_all_model_budget_treatments_and_exact_counts(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    assert plan.budget_ratios == FULL_BUDGET_RATIOS
    assert plan.repeats == 3
    assert plan.concurrency == 100
    assert len(plan.treatments) == 24
    assert len({item.directory for item in plan.treatments}) == 24
    assert all(item.planned_rows == 60 for item in plan.treatments)
    assert all(item.planned_provider_calls == 120 for item in plan.treatments)
    assert sum(item.planned_rows for item in plan.treatments) == 1_440
    assert (
        sum(item.planned_provider_calls for item in plan.treatments) == 2_880
    )
    assert all(
        item.task_model.provider_call_config.definition.route.model
        == item.model
        for item in plan.treatments
    )


def test_smoke_plan_keeps_all_routes_but_one_unbudgeted_row_pair(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, mode="smoke")

    assert plan.mode == "smoke"
    assert plan.task_ids == ("HumanEval/32",)
    assert plan.budget_ratios == (None,)
    assert plan.repeats == 1
    assert plan.pool_ceiling == 1
    assert len(plan.treatments) == 4
    assert all(item.planned_rows == 2 for item in plan.treatments)
    assert all(item.planned_provider_calls == 4 for item in plan.treatments)


def test_manifest_resume_requires_exact_plan_equality(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    path = output / "run-manifest.json"

    matrix._prepare_manifest(plan, path=path, resume=False)
    matrix._prepare_manifest(plan, path=path, resume=True)

    restored = Ed1BaselineMatrixPlan.model_validate_json(path.read_text())
    assert restored == plan
    mismatched = plan.model_copy(update={"concurrency": 99})
    with pytest.raises(RuntimeError, match="does not exactly match"):
        matrix._prepare_manifest(mismatched, path=path, resume=True)


def test_process_log_status_is_typed_and_flushed(tmp_path: Path) -> None:
    rows = RowAccounting(
        planned=30,
        present=29,
        missing=0,
        failed=1,
        invalid=0,
    )
    status = Ed1BaselineTreatmentStatus(
        timestamp="2026-08-08T12:00:00+00:00",
        elapsed_seconds=1.25,
        state=TreatmentState.TREATMENT_COMPLETED,
        treatment_id="treatment-1",
        baseline_rows=rows,
        comparison_rows=rows,
    )
    path = tmp_path / "process-log.jsonl"

    matrix._append_status(path, status)

    assert (
        Ed1BaselineTreatmentStatus.model_validate_json(path.read_text())
        == status
    )
    with pytest.raises(ValidationError, match="both row accounts"):
        Ed1BaselineTreatmentStatus(
            timestamp="2026-08-08T12:00:00+00:00",
            elapsed_seconds=1.25,
            state=TreatmentState.TREATMENT_COMPLETED,
            treatment_id="treatment-1",
        )


def test_open_file_limit_uses_the_concurrency_requirement(monkeypatch) -> None:
    updated: list[tuple[int, int]] = []
    monkeypatch.setattr(
        matrix.resource, "getrlimit", lambda _kind: (1024, 10_000)
    )
    monkeypatch.setattr(
        matrix.resource,
        "setrlimit",
        lambda _kind, limits: updated.append(limits),
    )

    assert raise_open_file_limit(100) == 6_400
    assert updated == [(6_400, 10_000)]


def test_openai_credential_mapping_never_overwrites_runtime_name() -> None:
    environment = {"MARIMO_OPENAI_API_KEY": "mise-secret"}
    map_openai_credential(environment)
    assert environment == {
        "MARIMO_OPENAI_API_KEY": "mise-secret",
        "OPENAI_API_KEY": "mise-secret",
    }

    environment["OPENAI_API_KEY"] = "runtime-secret"
    map_openai_credential(environment)
    assert environment["OPENAI_API_KEY"] == "runtime-secret"


def test_cli_requires_paths_and_exposes_resume_smoke_and_concurrency() -> None:
    args = parse_args(
        [
            "--evaluation-python",
            "/runtime/python",
            "--snapshot-path",
            "/data/snapshot.json",
            "--output-dir",
            "/runs/smoke-1",
            "--resume",
            "--smoke",
            "--concurrency",
            "7",
        ]
    )

    assert args.evaluation_python == Path("/runtime/python")
    assert args.snapshot_path == Path("/data/snapshot.json")
    assert args.output_dir == Path("/runs/smoke-1")
    assert args.resume is True
    assert args.smoke is True
    assert args.concurrency == 7
