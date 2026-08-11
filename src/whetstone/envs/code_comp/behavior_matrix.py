from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from dr_providers import ProviderCallConfig
from dr_store import ObjectStore, SqliteBackend
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from whetstone.envs.code_comp.dataset import CodeCompTaskInstance, load_tasks
from whetstone.envs.code_comp.modes.encdec import (
    EncDecTaskModelConfig,
    EncDecTaskModelKind,
    encdec_runtime_from_metadata,
    encdec_task_model_from_metadata,
)
from whetstone.envs.code_comp.preview import (
    run_code_comp_anchor_baseline_preview,
)
from whetstone.envs.code_comp.procedure import build_encdec_procedure_config
from whetstone.envs.code_comp.registry import CodeCompMode
from whetstone.envs.code_comp.runtime import (
    EncDecScoringRuntimeSummary,
    build_code_comp_scoring_runtime,
)
from whetstone.envs.code_comp.scoring import (
    CODE_COMP_SCORING_PREFLIGHT_TASK_ID,
    CheckpointedCodeBatchScorer,
)
from whetstone.envs.task_pools import (
    select_lowest_historical_pass_rate_for_env,
)
from whetstone.evaluation.preview.anchor import BaselinePreviewTranscript
from whetstone.evaluation.schema import RowAccounting
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitRole,
    load_task_split_manifest,
)
from whetstone.optimization.validation.matrix import (
    BehaviorMatrixHooks,
    MatrixTreatmentBase,
    atomic_write_model,
    map_openai_credential,
    raise_open_file_limit,
    run_behavior_matrix,
)
from whetstone.provider.policy import ProviderExecutionPolicy


class MatrixProviderRoute(Protocol):
    """Minimal provider-route surface the matrix plan builder reads."""

    lane: str
    model: str
    call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy


MATRIX_SCHEMA_VERSION = 1
DEFAULT_CONCURRENCY = 100
TOKEN_LIMIT = 4096
FULL_BUDGET_RATIOS: tuple[float | None, ...] = (
    None,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
)
EXCLUDED_TASK_IDS = (
    "HumanEval/39",
    "HumanEval/113",
    "HumanEval/116",
    "HumanEval/149",
    "HumanEval/162",
)
DEFAULT_TASK_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "experiment"
    / "task_selection"
    / "humaneval_copro_challenge_v1.json"
)


class CodeCompBehaviorMatrixTreatmentPlan(MatrixTreatmentBase):
    """One exact model-by-budget treatment and its durable location."""

    lane: StrictStr
    model: StrictStr
    provider_call_config_hash: StrictStr
    execution_policy_hash: StrictStr
    budget_ratio: StrictFloat | None
    task_model: EncDecTaskModelConfig
    planned_rows: StrictInt
    planned_provider_calls: StrictInt

    @model_validator(mode="after")
    def _validate_treatment(self) -> CodeCompBehaviorMatrixTreatmentPlan:
        if not self.treatment_id or not self.directory:
            raise ValueError("treatment ID and directory must be non-empty")
        if self.planned_rows < 1 or self.planned_provider_calls < 1:
            raise ValueError("treatment work counts must be positive")
        if self.task_model.model != self.model:
            raise ValueError("treatment model disagrees with task-model route")
        if (
            self.provider_call_config_hash
            != self.task_model.provider_call_config.identity_hash
        ):
            raise ValueError(
                "treatment provider config hash disagrees with "
                "task-model route"
            )
        if (
            self.execution_policy_hash
            != self.task_model.execution_policy.identity_hash
        ):
            raise ValueError(
                "treatment execution policy hash disagrees with "
                "task-model route"
            )
        return self


class BehaviorMatrixPlan(BaseModel):
    """Exact, restart-comparable plan persisted before provider calls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = MATRIX_SCHEMA_VERSION
    mode: Literal["full", "smoke"]
    evaluation_python: StrictStr
    evaluation_python_sha256: StrictStr
    snapshot_path: StrictStr
    snapshot_sha256: StrictStr
    task_manifest_path: StrictStr
    output_dir: StrictStr
    task_selection: TaskRoleSelection
    task_ids: tuple[StrictStr, ...]
    excluded_task_ids: tuple[StrictStr, ...]
    budget_ratios: tuple[StrictFloat | None, ...]
    repeats: StrictInt
    concurrency: StrictInt
    pool_ceiling: StrictInt
    procedure_config_hash: StrictStr
    runtime: EncDecScoringRuntimeSummary
    treatments: tuple[CodeCompBehaviorMatrixTreatmentPlan, ...]

    @model_validator(mode="after")
    def _validate_plan(self) -> BehaviorMatrixPlan:
        if not self.task_ids or self.task_selection.task_ids != self.task_ids:
            raise ValueError("matrix task selection must match its task IDs")
        if self.repeats < 1 or self.concurrency < 1:
            raise ValueError("matrix repeats and concurrency must be positive")
        if len({item.treatment_id for item in self.treatments}) != len(
            self.treatments
        ):
            raise ValueError("matrix treatment IDs must be unique")
        if len({item.directory for item in self.treatments}) != len(
            self.treatments
        ):
            raise ValueError("matrix treatment directories must be unique")
        expected = len(self.budget_ratios) * 4
        if len(self.treatments) != expected:
            raise ValueError(
                "matrix requires four routes per budget "
                f"({expected} treatments)"
            )
        return self


@dataclass(frozen=True, slots=True)
class _CodeCompMatrixShared:
    tasks: tuple[CodeCompTaskInstance, ...]
    preflight_task: CodeCompTaskInstance
    scorer: CheckpointedCodeBatchScorer


def _task_model(route: MatrixProviderRoute) -> EncDecTaskModelConfig:
    return EncDecTaskModelConfig(
        kind=EncDecTaskModelKind.PROVIDER,
        provider_call_config=route.call_config,
        execution_policy=route.execution_policy,
    )


def _safe_fragment(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-" for character in value
    )


def _budget_label(value: float | None) -> str:
    return "none" if value is None else f"{value:g}".replace(".", "p")


def build_matrix_plan(
    *,
    provider_routes: tuple[MatrixProviderRoute, ...],
    mode: Literal["full", "smoke"],
    evaluation_python: Path,
    snapshot_path: Path,
    task_manifest_path: Path,
    output_dir: Path,
    task_selection: TaskRoleSelection,
    concurrency: int,
    runtime: EncDecScoringRuntimeSummary,
) -> BehaviorMatrixPlan:
    """Build the exact immutable plan for a full matrix or four-model smoke."""

    task_ids = task_selection.task_ids
    budget_ratios = (None,) if mode == "smoke" else FULL_BUDGET_RATIOS
    repeats = 1 if mode == "smoke" else 3
    rows = len(task_ids) * repeats * 2
    treatments: list[CodeCompBehaviorMatrixTreatmentPlan] = []
    ordinal = 0
    for route in provider_routes:
        for budget_ratio in budget_ratios:
            ordinal += 1
            treatment_id = (
                f"t{ordinal:02d}-{route.lane}-{_safe_fragment(route.model)}-"
                f"budget-{_budget_label(budget_ratio)}"
            )
            treatments.append(
                CodeCompBehaviorMatrixTreatmentPlan(
                    treatment_id=treatment_id,
                    directory=treatment_id,
                    lane=route.lane,
                    model=route.model,
                    provider_call_config_hash=route.call_config.identity_hash,
                    execution_policy_hash=(
                        route.execution_policy.identity_hash
                    ),
                    budget_ratio=budget_ratio,
                    task_model=_task_model(route),
                    planned_rows=rows,
                    planned_provider_calls=rows * 2,
                )
            )
    pool_ceiling = (
        len(task_ids)
        if mode == "smoke"
        else task_selection.eligible_pool_count or len(task_ids)
    )
    return BehaviorMatrixPlan(
        mode=mode,
        evaluation_python=str(evaluation_python),
        evaluation_python_sha256=_sha256_file(evaluation_python),
        snapshot_path=str(snapshot_path),
        snapshot_sha256=_sha256_file(snapshot_path),
        task_manifest_path=str(task_manifest_path),
        output_dir=str(output_dir),
        task_selection=task_selection,
        task_ids=task_ids,
        excluded_task_ids=EXCLUDED_TASK_IDS,
        budget_ratios=budget_ratios,
        repeats=repeats,
        concurrency=concurrency,
        pool_ceiling=pool_ceiling,
        procedure_config_hash=(
            build_encdec_procedure_config().config_identity_hash
        ),
        runtime=runtime,
        treatments=tuple(treatments),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tasks_by_id(
    pool: tuple[CodeCompTaskInstance, ...], task_ids: tuple[str, ...]
) -> tuple[CodeCompTaskInstance, ...]:
    by_id = {task.humaneval_task.task_id: task for task in pool}
    missing = tuple(task_id for task_id in task_ids if task_id not in by_id)
    if missing:
        raise RuntimeError(f"snapshot is missing selected tasks: {missing}")
    return tuple(by_id[task_id] for task_id in task_ids)


def _select_tasks(
    *, manifest_path: Path, snapshot_path: Path, smoke: bool
) -> tuple[
    tuple[CodeCompTaskInstance, ...], TaskRoleSelection, CodeCompTaskInstance
]:
    manifest = load_task_split_manifest(manifest_path)
    selected = select_lowest_historical_pass_rate_for_env(
        manifest,
        env="code_comp",
        mode=CodeCompMode.ENCDEC,
        role=TaskSplitRole.TRAIN,
        count=10,
        excluded_task_ids=EXCLUDED_TASK_IDS,
    )
    if smoke:
        selected = TaskRoleSelection.model_validate(
            {
                **selected.model_dump(mode="python"),
                "task_ids": selected.task_ids[:1],
                "historical_pass_rates": selected.historical_pass_rates[:1],
            }
        )
    pool = load_tasks(snapshot_path=snapshot_path)
    return (
        _tasks_by_id(pool, selected.task_ids),
        selected,
        _tasks_by_id(pool, (CODE_COMP_SCORING_PREFLIGHT_TASK_ID,))[0],
    )


def _validate_result(
    transcript: BaselinePreviewTranscript,
    *,
    plan: BehaviorMatrixPlan,
    treatment: CodeCompBehaviorMatrixTreatmentPlan,
) -> None:
    baseline_binding = transcript.baseline.evidence.evaluation_binding
    comparison_binding = transcript.ceiling.evidence.evaluation_binding
    expected = {
        "task IDs": transcript.task_ids == plan.task_ids,
        "task selection": transcript.task_selection == plan.task_selection,
        "pool ceiling": transcript.pool_ceiling == plan.pool_ceiling,
        "budget ratio": transcript.budget_ratio == treatment.budget_ratio,
        "concurrency": transcript.concurrency == plan.concurrency,
        "task model": (
            encdec_task_model_from_metadata(transcript.metadata)
            == treatment.task_model
        ),
        "runtime": encdec_runtime_from_metadata(transcript.metadata)
        == plan.runtime,
        "baseline procedure": (
            baseline_binding.eval_config.record.evaluation_procedure_config_hash
            == plan.procedure_config_hash
        ),
        "comparison procedure": (
            comparison_binding.eval_config.record.evaluation_procedure_config_hash
            == plan.procedure_config_hash
        ),
        "baseline repeats": (
            transcript.baseline.evidence.repeat_count == plan.repeats
        ),
        "comparison repeats": (
            transcript.ceiling.evidence.repeat_count == plan.repeats
        ),
        "baseline rows": (
            transcript.baseline.evidence.row_accounting.planned
            == treatment.planned_rows // 2
        ),
        "comparison rows": (
            transcript.ceiling.evidence.row_accounting.planned
            == treatment.planned_rows // 2
        ),
    }
    mismatches = tuple(
        name for name, matches in expected.items() if not matches
    )
    if mismatches:
        raise ValueError(
            "treatment result disagrees with run manifest: "
            + ", ".join(mismatches)
        )


def _load_valid_result(
    path: Path,
    *,
    plan: BehaviorMatrixPlan,
    treatment: CodeCompBehaviorMatrixTreatmentPlan,
) -> BaselinePreviewTranscript | None:
    if not path.exists():
        return None
    try:
        transcript = BaselinePreviewTranscript.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        _validate_result(transcript, plan=plan, treatment=treatment)
    except (OSError, ValidationError, ValueError):
        return None
    return transcript


def _row_summary(transcript: BaselinePreviewTranscript) -> str:
    def arm(label: str, rows: RowAccounting) -> str:
        return (
            f"{label} present={rows.present}/{rows.planned}, "
            f"missing={rows.missing}, failed={rows.failed}, "
            f"invalid={rows.invalid}"
        )

    return "; ".join(
        (
            arm("baseline", transcript.baseline.evidence.row_accounting),
            arm("comparison", transcript.ceiling.evidence.row_accounting),
        )
    )


def _status_row_accounts(
    transcript: BaselinePreviewTranscript,
) -> tuple[RowAccounting, RowAccounting]:
    return (
        transcript.baseline.evidence.row_accounting,
        transcript.ceiling.evidence.row_accounting,
    )


def _build_hooks(
    *,
    shared: _CodeCompMatrixShared,
) -> BehaviorMatrixHooks[
    BehaviorMatrixPlan,
    CodeCompBehaviorMatrixTreatmentPlan,
    BaselinePreviewTranscript,
    _CodeCompMatrixShared,
]:
    @contextmanager
    def shared_context(
        _output_dir: Path, _plan: BehaviorMatrixPlan
    ) -> Iterator[_CodeCompMatrixShared]:
        yield shared

    def execute_treatment(
        treatment: CodeCompBehaviorMatrixTreatmentPlan,
        plan: BehaviorMatrixPlan,
        matrix_shared: _CodeCompMatrixShared,
        log: Callable[[str], None],
    ) -> BaselinePreviewTranscript:
        treatment_dir = Path(plan.output_dir) / treatment.directory
        return run_code_comp_anchor_baseline_preview(
            store=ObjectStore(
                SqliteBackend(treatment_dir / "objects.sqlite3")
            ),
            tasks=matrix_shared.tasks,
            task_ids=plan.task_ids,
            task_selection=plan.task_selection,
            preflight_task=matrix_shared.preflight_task,
            pool_ceiling=plan.pool_ceiling,
            task_model=treatment.task_model,
            batch_scorer=matrix_shared.scorer,
            runtime=plan.runtime,
            budget_ratio=treatment.budget_ratio,
            concurrency=plan.concurrency,
            repeats=plan.repeats,
            partial_log=PartialLog(treatment_dir / "partial-log"),
            prompt_cache=PromptResultCache(treatment_dir / "prompt-cache"),
            log=log,
        )

    return BehaviorMatrixHooks(
        shared_context=shared_context,
        execute_treatment=execute_treatment,
        validate_result=lambda result, plan, treatment: _validate_result(
            result,
            plan=plan,
            treatment=treatment,
        ),
        load_valid_result=lambda path, plan, treatment: _load_valid_result(
            path,
            plan=plan,
            treatment=treatment,
        ),
        persist_result=lambda path, result: atomic_write_model(path, result),
        row_summary=_row_summary,
        status_row_accounts=_status_row_accounts,
    )


def run_code_comp_baseline_behavior_matrix(
    *,
    provider_routes: tuple[MatrixProviderRoute, ...],
    evaluation_python: Path,
    snapshot_path: Path,
    output_dir: Path,
    task_manifest_path: Path = DEFAULT_TASK_MANIFEST,
    resume: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    smoke: bool = False,
) -> BehaviorMatrixPlan:
    """Run or exactly resume the fixed ED1 baseline behavior matrix."""

    launched_at = perf_counter()
    evaluation_python = evaluation_python.expanduser().absolute()
    snapshot_path = snapshot_path.expanduser().absolute()
    task_manifest_path = task_manifest_path.expanduser().absolute()
    output_dir = output_dir.expanduser().absolute()
    for label, path in (
        ("evaluation Python", evaluation_python),
        ("snapshot", snapshot_path),
        ("task manifest", task_manifest_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")
    if os.environ.get("DR_CODE_DISPOSABLE_WORKER") != "1":
        raise RuntimeError(
            "refusing to execute generated code: set "
            "DR_CODE_DISPOSABLE_WORKER=1 in a disposable worker environment"
        )
    map_openai_credential(os.environ)
    required_files = raise_open_file_limit(concurrency)

    def progress(message: str) -> None:
        elapsed = perf_counter() - launched_at
        print(f"[{elapsed:8.1f}s] {message}", flush=True)

    progress(
        f"open-file soft limit prepared (required={required_files}); "
        f"output={output_dir}"
    )
    tasks, selection, preflight_task = _select_tasks(
        manifest_path=task_manifest_path,
        snapshot_path=snapshot_path,
        smoke=smoke,
    )
    runtime = build_code_comp_scoring_runtime(
        runtime_executable=evaluation_python,
        record_root=output_dir / "code-execution-records",
    )
    plan = build_matrix_plan(
        provider_routes=provider_routes,
        mode="smoke" if smoke else "full",
        evaluation_python=evaluation_python,
        snapshot_path=snapshot_path,
        task_manifest_path=task_manifest_path,
        output_dir=output_dir,
        task_selection=selection,
        concurrency=concurrency,
        runtime=EncDecScoringRuntimeSummary(
            evaluation_python=runtime.probe.python_executable,
            dr_code_version=version("dr-code"),
            runtime_identity_hash=runtime.runtime_identity_hash,
            probe=runtime.probe,
        ),
    )
    with CheckpointedCodeBatchScorer(
        output_dir / "code-scoring-cache.sqlite3",
        runtime_identity=runtime.runtime_identity,
        executor=runtime.executor,
    ) as scorer:
        matrix_shared = _CodeCompMatrixShared(
            tasks=tasks,
            preflight_task=preflight_task,
            scorer=scorer,
        )
        return run_behavior_matrix(
            plan=plan,
            output_dir=output_dir,
            resume=resume,
            hooks=_build_hooks(shared=matrix_shared),
            progress=progress,
        )


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_TASK_MANIFEST",
    "EXCLUDED_TASK_IDS",
    "FULL_BUDGET_RATIOS",
    "BehaviorMatrixPlan",
    "CodeCompBehaviorMatrixTreatmentPlan",
    "build_matrix_plan",
    "run_code_comp_baseline_behavior_matrix",
]
