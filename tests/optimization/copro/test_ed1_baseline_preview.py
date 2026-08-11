from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from dr_store import MemoryBackend, ObjectStore

from tests.envs.support import execution_policy, synthetic_ed1_tasks
from whetstone.envs.ed1_runtime import Ed1RuntimeProbe
from whetstone.envs.ed1_scoring import CodeScore, CodeScoringInput
from whetstone.envs.encdec_rollout import build_encoder_provider_call_config
from whetstone.evaluation.analysis.power import PowerConfig
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitRole,
)
from whetstone.optimization.copro.ed1_baseline_preview import (
    Ed1BaselinePreviewTranscript,
    run_ed1_baseline_preview,
    run_ed1_baseline_sweep,
)
from whetstone.optimization.copro.ed1_scoring_preview import (
    Ed1ScoringRuntimeSummary,
)
from whetstone.optimization.copro.ed1_task_model import (
    Ed1TaskModelConfig,
    Ed1TaskModelKind,
)


def _score(
    inputs: Sequence[CodeScoringInput],
    *,
    max_wall_seconds: float | None = None,
) -> tuple[CodeScore, ...]:
    del max_wall_seconds
    return tuple(
        CodeScore(
            passed="return None" not in item.raw_submission,
            infrastructure_unknown=False,
            outcome=(
                "passed"
                if "return None" not in item.raw_submission
                else "tests_failed"
            ),
        )
        for item in inputs
    )


def _runtime() -> Ed1ScoringRuntimeSummary:
    return Ed1ScoringRuntimeSummary(
        evaluation_python="/copied/python",
        dr_code_version="0.1.5",
        runtime_identity_hash="a" * 64,
        probe=Ed1RuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable="/copied/python",
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


def test_baseline_preview_uses_one_binding_and_estimates_tiny_data() -> None:
    tasks = synthetic_ed1_tasks(3)
    task_ids = ("Synthetic/2", "Synthetic/0")

    transcript = run_ed1_baseline_preview(
        store=ObjectStore(MemoryBackend()),
        tasks=tasks,
        task_ids=task_ids,
        pool_ceiling=3,
        task_model=_task_model(),
        batch_scorer=_score,
        runtime=_runtime(),
        repeats=2,
        power_config=PowerConfig(repeat_cap=3, trials=100, seed=17),
        bootstrap_resamples=200,
        bootstrap_seed=19,
    )

    assert transcript.task_ids == task_ids
    assert transcript.task_model.kind is Ed1TaskModelKind.DUMMY
    assert transcript.baseline.evidence.evaluation_binding == (
        transcript.evaluation_binding
    )
    assert transcript.ceiling.evidence.evaluation_binding == (
        transcript.evaluation_binding
    )
    assert transcript.baseline.evidence.task_identities == task_ids
    assert transcript.ceiling.evidence.task_identities == task_ids
    assert transcript.baseline.evidence.repeat_count == 2
    assert transcript.ceiling.evidence.repeat_count == 2
    assert len(transcript.baseline.component_traces.rows) == 4
    assert len(transcript.ceiling.component_traces.rows) == 4
    assert transcript.paired_delta_ci.resamples == 200
    assert transcript.power.pool_ceiling == 3
    assert (
        Ed1BaselinePreviewTranscript.model_validate_json(
            transcript.model_dump_json()
        )
        == transcript
    )


def test_baseline_preview_labels_progress_with_budget_mode() -> None:
    tasks = synthetic_ed1_tasks(1)
    messages: list[str] = []

    run_ed1_baseline_preview(
        store=ObjectStore(MemoryBackend()),
        tasks=tasks,
        task_ids=("Synthetic/0",),
        pool_ceiling=1,
        task_model=_task_model(),
        batch_scorer=_score,
        runtime=_runtime(),
        budget_ratio=0.5,
        power_config=PowerConfig(trials=1),
        bootstrap_resamples=1,
        log=messages.append,
    )

    assert (
        messages[0] == "budget ratio 0.5: starting scoring-runtime preflight"
    )
    assert messages[1].startswith(
        "budget ratio 0.5: scoring-runtime preflight completed"
    )
    assert messages[2:] == [
        "budget ratio 0.5: Starting hand-engineered baseline evaluation "
        "(1 rows)",
        "budget ratio 0.5: Completed hand-engineered baseline evaluation "
        "(present=1/1, missing=0, failed=0, invalid=0)",
        (
            "budget ratio 0.5: Starting hand-engineered comparison "
            "anchor evaluation (1 rows)"
        ),
        (
            "budget ratio 0.5: Completed hand-engineered comparison "
            "anchor evaluation "
            "(present=1/1, missing=0, failed=0, invalid=0)"
        ),
    ]


def test_baseline_preview_threads_partial_log_and_prompt_cache(
    tmp_path: Path,
) -> None:
    partial_log = PartialLog(tmp_path / "baseline-partials.jsonl")

    run_ed1_baseline_preview(
        store=ObjectStore(MemoryBackend()),
        tasks=synthetic_ed1_tasks(1),
        task_ids=("Synthetic/0",),
        pool_ceiling=1,
        task_model=_task_model(),
        batch_scorer=_score,
        runtime=_runtime(),
        partial_log=partial_log,
        prompt_cache=PromptResultCache(tmp_path / "prompt-cache"),
        power_config=PowerConfig(trials=1),
        bootstrap_resamples=1,
    )

    assert len(partial_log.load()) == 2


def test_baseline_preview_rejects_unknown_selection_before_scoring() -> None:
    with pytest.raises(ValueError, match="task IDs are unknown"):
        run_ed1_baseline_preview(
            store=ObjectStore(MemoryBackend()),
            tasks=synthetic_ed1_tasks(1),
            task_ids=("Synthetic/missing",),
            pool_ceiling=1,
            task_model=_task_model(),
            batch_scorer=_score,
            runtime=_runtime(),
            power_config=PowerConfig(trials=1),
            bootstrap_resamples=1,
        )


def test_baseline_sweep_preserves_manifest_role_across_budget_modes() -> None:
    tasks = synthetic_ed1_tasks(2)
    task_ids = ("Synthetic/1", "Synthetic/0")
    selection = TaskRoleSelection(
        manifest_content_hash="b" * 64,
        pool_key="ed1",
        role=TaskSplitRole.TRAIN,
        task_ids=task_ids,
    )

    transcript = run_ed1_baseline_sweep(
        store=ObjectStore(MemoryBackend()),
        tasks=tasks,
        task_ids=task_ids,
        task_selection=selection,
        preflight_task=tasks[0],
        pool_ceiling=2,
        task_model=_task_model(),
        batch_scorer=_score,
        runtime=_runtime(),
        budget_ratios=(None, 0.5),
        concurrency=3,
        power_config=PowerConfig(repeat_cap=2, trials=10, seed=17),
        bootstrap_resamples=10,
        bootstrap_seed=19,
    )

    assert transcript.task_selection == selection
    assert all(item.concurrency == 3 for item in transcript.previews)
    assert all(
        item.preflight.task_id == "Synthetic/0" for item in transcript.previews
    )
    assert transcript.budget_ratios == (None, 0.5)
    assert tuple(item.budget_ratio for item in transcript.previews) == (
        None,
        0.5,
    )
    assert all(
        item.task_selection == selection for item in transcript.previews
    )
    assert all(item.task_ids == task_ids for item in transcript.previews)
    assert (
        transcript.previews[0].baseline.evidence.graph_hash
        != transcript.previews[1].baseline.evidence.graph_hash
    )
