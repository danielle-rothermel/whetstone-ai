from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

import pytest
from dr_providers import ProviderKind, ReasoningEffort

from whetstone.envs.code_comp import behavior_matrix as matrix
from whetstone.envs.code_comp.behavior_matrix import (
    EXCLUDED_TASK_IDS,
    FULL_BUDGET_RATIOS,
    BehaviorMatrixPlan,
    build_matrix_plan,
    map_openai_credential,
)
from whetstone.envs.code_comp.registry import CodeCompMode
from whetstone.envs.code_comp.runtime import (
    CodeCompRuntimeProbe,
    EncDecScoringRuntimeSummary,
)
from whetstone.envs.task_pools import (
    select_lowest_historical_pass_rate_for_env,
)
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitRole,
    load_task_split_manifest,
)
from whetstone.optimization.validation.matrix import prepare_manifest

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
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO_ROOT / (
    "src/whetstone/experiment/task_selection/humaneval_copro_challenge_v1.json"
)
_ROUTES_MODULE_PATH = (
    _REPO_ROOT / "scripts/experiments/code_comp_matrix_routes.py"
)


def _baseline_provider_routes():
    spec = importlib.util.spec_from_file_location(
        "code_comp_matrix_routes",
        _ROUTES_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.baseline_provider_routes()


def _runtime(evaluation_python: Path) -> EncDecScoringRuntimeSummary:
    return EncDecScoringRuntimeSummary(
        evaluation_python=str(evaluation_python),
        dr_code_version="0.1.5",
        runtime_hash="a" * 64,
        probe=CodeCompRuntimeProbe(
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
        pool_key="encdec",
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
) -> BehaviorMatrixPlan:
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
        provider_routes=_baseline_provider_routes(),
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
    routes = _baseline_provider_routes()

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
    manifest = load_task_split_manifest(_MANIFEST)

    selection = select_lowest_historical_pass_rate_for_env(
        manifest,
        env="code_comp",
        mode=CodeCompMode.ENCDEC,
        role=TaskSplitRole.TRAIN,
        count=10,
        excluded_task_ids=EXCLUDED_TASK_IDS,
    )

    assert selection.task_ids == _EXPECTED_TASK_IDS
    assert selection.excluded_task_ids == EXCLUDED_TASK_IDS
    assert selection.manifest_content_hash == (
        "c3f3919ff8163c0331feefaed822e37c8b7c1b7b88af9fdc969a4a15856d49f2"
    )


def test_full_plan_has_all_model_budget_treatments_and_exact_counts(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    assert plan.budget_ratios == FULL_BUDGET_RATIOS
    assert plan.num_samples == 3
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
    assert plan.num_samples == 1
    assert plan.pool_ceiling == 1
    assert len(plan.treatments) == 4
    assert all(item.planned_rows == 2 for item in plan.treatments)
    assert all(item.planned_provider_calls == 4 for item in plan.treatments)


def test_manifest_resume_requires_exact_plan_equality(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    path = output / "run-manifest.json"

    prepare_manifest(
        plan,
        path=path,
        resume=False,
        plan_type=BehaviorMatrixPlan,
    )
    prepare_manifest(
        plan,
        path=path,
        resume=True,
        plan_type=BehaviorMatrixPlan,
    )

    restored = BehaviorMatrixPlan.model_validate_json(path.read_text())
    assert restored == plan
    mismatched = plan.model_copy(update={"concurrency": 99})
    with pytest.raises(RuntimeError, match="does not exactly match"):
        prepare_manifest(
            mismatched,
            path=path,
            resume=True,
            plan_type=BehaviorMatrixPlan,
        )


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


def test_run_code_comp_baseline_behavior_matrix_delegates_to_generic_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_python = tmp_path / "python"
    snapshot = tmp_path / "snapshot.json"
    manifest = tmp_path / "tasks.json"
    output = tmp_path / "output"
    for path in (evaluation_python, snapshot, manifest):
        path.write_bytes(b"x")
    monkeypatch.setenv("DR_CODE_DISPOSABLE_WORKER", "1")

    plan = _plan(tmp_path, mode="smoke")
    captured: dict[str, dict[str, object]] = {}

    class _FakeRuntime:
        runtime_hash = "a" * 64
        runtime_document = object()
        executor = object()
        probe = _runtime(evaluation_python).probe

    class _FakeScorer:
        def __enter__(self) -> _FakeScorer:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_run_behavior_matrix(**kwargs: object) -> BehaviorMatrixPlan:
        captured["kwargs"] = kwargs
        return plan

    monkeypatch.setattr(
        matrix,
        "_select_tasks",
        lambda **_kwargs: ((), _selection(("HumanEval/32",)), object()),
    )
    monkeypatch.setattr(
        matrix,
        "build_code_comp_scoring_runtime",
        lambda **_kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(matrix, "build_matrix_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        matrix,
        "run_behavior_matrix",
        fake_run_behavior_matrix,
    )
    monkeypatch.setattr(matrix, "raise_open_file_limit", lambda _c: 4096)
    monkeypatch.setattr(matrix, "map_openai_credential", lambda _e: None)
    monkeypatch.setattr(
        matrix,
        "CheckpointedCodeBatchScorer",
        lambda *_args, **_kwargs: _FakeScorer(),
    )

    result = matrix.run_code_comp_baseline_behavior_matrix(
        provider_routes=_baseline_provider_routes(),
        evaluation_python=evaluation_python,
        snapshot_path=snapshot,
        output_dir=output,
        task_manifest_path=manifest,
        smoke=True,
        concurrency=1,
    )

    assert result == plan
    assert captured["kwargs"]["plan"] == plan
    assert captured["kwargs"]["output_dir"] == output


def test_cli_requires_paths_and_exposes_resume_smoke_and_concurrency() -> None:
    script = _REPO_ROOT / "scripts/experiments/run_baseline_behavior_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "test_run_baseline_behavior_matrix_cli",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
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
