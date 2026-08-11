from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import resource
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import UNIQUE, StrEnum, verify
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from dr_providers import ReasoningEffort
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

from whetstone.envs.ed1 import (
    Ed1Instance,
    build_ed1_procedure_config,
    load_ed1_tasks,
)
from whetstone.envs.ed1_preview import (
    ED1_SCORING_PREFLIGHT_TASK_ID,
    Ed1ScoringRuntimeSummary,
)
from whetstone.envs.ed1_runtime import build_ed1_scoring_runtime
from whetstone.envs.ed1_scoring import CheckpointedCodeBatchScorer
from whetstone.envs.task_pools import (
    select_lowest_historical_pass_rate_for_env,
)
from whetstone.evaluation.schema import RowAccounting
from whetstone.execution._file_lock import (
    fsync_parent_directory,
    open_private_regular_file,
)
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.task_selection import (
    TaskRoleSelection,
    TaskSplitRole,
    load_task_split_manifest,
)
from whetstone.optimization.copro.ed1_baseline_preview import (
    Ed1BaselinePreviewTranscript,
    run_ed1_baseline_preview,
)
from whetstone.optimization.copro.ed1_task_model import (
    Ed1TaskModelConfig,
    Ed1TaskModelKind,
)
from whetstone.runner.routes import (
    ProviderRoute,
    canonical_task_route,
    openai_direct_route,
)

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
DEFAULT_TASK_MANIFEST = Path(__file__).with_name(
    "humaneval_copro_challenge_v1.json"
)


class Ed1BaselineMatrixTreatmentPlan(BaseModel):
    """One exact model-by-budget treatment and its durable location."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment_id: StrictStr
    directory: StrictStr
    lane: StrictStr
    model: StrictStr
    provider_call_config_hash: StrictStr
    execution_policy_hash: StrictStr
    budget_ratio: StrictFloat | None
    task_model: Ed1TaskModelConfig
    planned_rows: StrictInt
    planned_provider_calls: StrictInt

    @model_validator(mode="after")
    def _validate_treatment(self) -> Ed1BaselineMatrixTreatmentPlan:
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


class Ed1BaselineMatrixPlan(BaseModel):
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
    runtime: Ed1ScoringRuntimeSummary
    treatments: tuple[Ed1BaselineMatrixTreatmentPlan, ...]

    @model_validator(mode="after")
    def _validate_plan(self) -> Ed1BaselineMatrixPlan:
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


@verify(UNIQUE)
class TreatmentState(StrEnum):
    """Durably logged matrix lifecycle states."""

    RUN_STARTED = "run_started"
    TREATMENT_STARTED = "treatment_started"
    TREATMENT_SKIPPED = "treatment_skipped"
    TREATMENT_COMPLETED = "treatment_completed"
    TREATMENT_FAILED = "treatment_failed"
    RUN_COMPLETED = "run_completed"


class Ed1BaselineTreatmentStatus(BaseModel):
    """One flushed process-log event with optional terminal row accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = MATRIX_SCHEMA_VERSION
    timestamp: StrictStr
    elapsed_seconds: float
    state: TreatmentState
    treatment_id: StrictStr | None = None
    baseline_rows: RowAccounting | None = None
    comparison_rows: RowAccounting | None = None
    failure_type: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> Ed1BaselineTreatmentStatus:
        treatment_state = self.state not in {
            TreatmentState.RUN_STARTED,
            TreatmentState.RUN_COMPLETED,
        }
        if treatment_state != (self.treatment_id is not None):
            raise ValueError(
                "treatment lifecycle states require a treatment ID"
            )
        terminal_success = self.state in {
            TreatmentState.TREATMENT_SKIPPED,
            TreatmentState.TREATMENT_COMPLETED,
        }
        has_rows = (
            self.baseline_rows is not None and self.comparison_rows is not None
        )
        if terminal_success != has_rows:
            raise ValueError(
                "successful terminal states require both row accounts"
            )
        if (self.state is TreatmentState.TREATMENT_FAILED) != (
            self.failure_type is not None
        ):
            raise ValueError("only failed treatments carry a failure type")
        return self


def _provider_route(
    *, lane: str, model: str, reasoning: ReasoningEffort | None
) -> ProviderRoute:
    route = openai_direct_route if lane == "openai" else canonical_task_route
    return route(
        model=model,
        temperature=None,
        reasoning=reasoning,
        token_limit=TOKEN_LIMIT,
        max_attempts=1,
    )


def baseline_provider_routes() -> tuple[ProviderRoute, ...]:
    """Return the four ordered, fixed behavior-matrix provider routes."""

    return (
        _provider_route(
            lane="openai",
            model="gpt-5.4-nano",
            reasoning=ReasoningEffort.NONE,
        ),
        _provider_route(
            lane="openrouter",
            model="deepseek/deepseek-v4-flash",
            reasoning=ReasoningEffort.NONE,
        ),
        _provider_route(
            lane="openrouter",
            model="qwen/qwen3-coder-flash",
            reasoning=None,
        ),
        _provider_route(
            lane="openrouter",
            model="google/gemini-3.1-flash-lite",
            reasoning=ReasoningEffort.NONE,
        ),
    )


def _task_model(route: ProviderRoute) -> Ed1TaskModelConfig:
    return Ed1TaskModelConfig(
        kind=Ed1TaskModelKind.PROVIDER,
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
    mode: Literal["full", "smoke"],
    evaluation_python: Path,
    snapshot_path: Path,
    task_manifest_path: Path,
    output_dir: Path,
    task_selection: TaskRoleSelection,
    concurrency: int,
    runtime: Ed1ScoringRuntimeSummary,
) -> Ed1BaselineMatrixPlan:
    """Build the exact immutable plan for a full matrix or four-model smoke."""

    task_ids = task_selection.task_ids
    budget_ratios = (None,) if mode == "smoke" else FULL_BUDGET_RATIOS
    repeats = 1 if mode == "smoke" else 3
    rows = len(task_ids) * repeats * 2
    treatments: list[Ed1BaselineMatrixTreatmentPlan] = []
    ordinal = 0
    for route in baseline_provider_routes():
        for budget_ratio in budget_ratios:
            ordinal += 1
            treatment_id = (
                f"t{ordinal:02d}-{route.lane}-{_safe_fragment(route.model)}-"
                f"budget-{_budget_label(budget_ratio)}"
            )
            treatments.append(
                Ed1BaselineMatrixTreatmentPlan(
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
    return Ed1BaselineMatrixPlan(
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
            build_ed1_procedure_config().config_identity_hash
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


def _atomic_write_model(path: Path, model: BaseModel) -> None:
    body = model.model_dump_json(indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = open_private_regular_file(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            descriptor = None
            destination.write(body)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        fsync_parent_directory(path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _prepare_manifest(
    plan: Ed1BaselineMatrixPlan,
    *,
    path: Path,
    resume: bool,
) -> None:
    if resume:
        try:
            existing = Ed1BaselineMatrixPlan.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise RuntimeError(
                "--resume requires a valid existing run-manifest.json"
            ) from exc
        if existing != plan:
            raise RuntimeError(
                "refusing unsafe resume: current plan does not exactly match "
                "run-manifest.json"
            )
        return
    if path.exists():
        raise RuntimeError(
            "output already contains run-manifest.json; use --resume with the "
            "exact same plan or choose a new output directory"
        )
    _atomic_write_model(path, plan)


def _append_status(path: Path, status: Ed1BaselineTreatmentStatus) -> None:
    body = status.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as destination:
        destination.write(body)
        destination.flush()
        os.fsync(destination.fileno())


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    descriptor = open_private_regular_file(
        path,
        os.O_RDWR | os.O_CREAT,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another behavior-matrix process holds {path}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def raise_open_file_limit(concurrency: int) -> int:
    """Raise RLIMIT_NOFILE to the matrix fanout requirement and return it."""

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    required = max(4096, concurrency * 64)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= required:
        return required
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError(
            f"open-file hard limit {hard} is below required limit {required}"
        )
    resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    return required


def map_openai_credential(
    environment: dict[str, str] | os._Environ[str],
) -> None:
    """Expose the mise OpenAI credential only when runtime naming is absent."""

    if (
        "OPENAI_API_KEY" not in environment
        and "MARIMO_OPENAI_API_KEY" in environment
    ):
        environment["OPENAI_API_KEY"] = environment["MARIMO_OPENAI_API_KEY"]


def _tasks_by_id(
    pool: tuple[Ed1Instance, ...], task_ids: tuple[str, ...]
) -> tuple[Ed1Instance, ...]:
    by_id = {task.humaneval_task.task_id: task for task in pool}
    missing = tuple(task_id for task_id in task_ids if task_id not in by_id)
    if missing:
        raise RuntimeError(f"snapshot is missing selected tasks: {missing}")
    return tuple(by_id[task_id] for task_id in task_ids)


def _select_tasks(
    *, manifest_path: Path, snapshot_path: Path, smoke: bool
) -> tuple[tuple[Ed1Instance, ...], TaskRoleSelection, Ed1Instance]:
    manifest = load_task_split_manifest(manifest_path)
    selected = select_lowest_historical_pass_rate_for_env(
        manifest,
        env="ed1",
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
    pool = load_ed1_tasks(snapshot_path=snapshot_path)
    return (
        _tasks_by_id(pool, selected.task_ids),
        selected,
        _tasks_by_id(pool, (ED1_SCORING_PREFLIGHT_TASK_ID,))[0],
    )


def _validate_result(
    transcript: Ed1BaselinePreviewTranscript,
    *,
    plan: Ed1BaselineMatrixPlan,
    treatment: Ed1BaselineMatrixTreatmentPlan,
) -> None:
    baseline_binding = transcript.baseline.evidence.evaluation_binding
    comparison_binding = transcript.ceiling.evidence.evaluation_binding
    expected = {
        "task IDs": transcript.task_ids == plan.task_ids,
        "task selection": transcript.task_selection == plan.task_selection,
        "pool ceiling": transcript.pool_ceiling == plan.pool_ceiling,
        "budget ratio": transcript.budget_ratio == treatment.budget_ratio,
        "concurrency": transcript.concurrency == plan.concurrency,
        "task model": transcript.task_model == treatment.task_model,
        "runtime": transcript.runtime == plan.runtime,
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
    plan: Ed1BaselineMatrixPlan,
    treatment: Ed1BaselineMatrixTreatmentPlan,
) -> Ed1BaselinePreviewTranscript | None:
    if not path.exists():
        return None
    try:
        transcript = Ed1BaselinePreviewTranscript.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        _validate_result(transcript, plan=plan, treatment=treatment)
    except (OSError, ValidationError, ValueError):
        return None
    return transcript


def _row_summary(transcript: Ed1BaselinePreviewTranscript) -> str:
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


def run_baseline_behavior_matrix(
    *,
    evaluation_python: Path,
    snapshot_path: Path,
    output_dir: Path,
    task_manifest_path: Path = DEFAULT_TASK_MANIFEST,
    resume: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    smoke: bool = False,
) -> Ed1BaselineMatrixPlan:
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
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(message: str) -> None:
        elapsed = perf_counter() - launched_at
        print(f"[{elapsed:8.1f}s] {message}", flush=True)

    progress(
        f"open-file soft limit prepared (required={required_files}); "
        f"output={output_dir}"
    )
    with _run_lock(output_dir / ".run.lock"):
        manifest_path = output_dir / "run-manifest.json"
        if resume and not manifest_path.is_file():
            raise RuntimeError(
                "--resume requires an existing run-manifest.json"
            )
        if not resume:
            unexpected = tuple(
                path.name
                for path in output_dir.iterdir()
                if path.name != ".run.lock"
            )
            if unexpected:
                raise RuntimeError(
                    "refusing to start in a non-empty output directory: "
                    + ", ".join(sorted(unexpected))
                )
        tasks, selection, preflight_task = _select_tasks(
            manifest_path=task_manifest_path,
            snapshot_path=snapshot_path,
            smoke=smoke,
        )
        runtime = build_ed1_scoring_runtime(
            runtime_executable=evaluation_python,
            record_root=output_dir / "code-execution-records",
        )
        runtime_summary = Ed1ScoringRuntimeSummary(
            evaluation_python=runtime.probe.python_executable,
            dr_code_version=version("dr-code"),
            runtime_identity_hash=runtime.runtime_identity_hash,
            probe=runtime.probe,
        )
        plan = build_matrix_plan(
            mode="smoke" if smoke else "full",
            evaluation_python=evaluation_python,
            snapshot_path=snapshot_path,
            task_manifest_path=task_manifest_path,
            output_dir=output_dir,
            task_selection=selection,
            concurrency=concurrency,
            runtime=runtime_summary,
        )
        _prepare_manifest(plan, path=manifest_path, resume=resume)
        process_log = output_dir / "process-log.jsonl"

        def status(
            state: TreatmentState,
            *,
            treatment_id: str | None = None,
            transcript: Ed1BaselinePreviewTranscript | None = None,
            failure_type: str | None = None,
        ) -> None:
            _append_status(
                process_log,
                Ed1BaselineTreatmentStatus(
                    timestamp=datetime.now(UTC).isoformat(),
                    elapsed_seconds=perf_counter() - launched_at,
                    state=state,
                    treatment_id=treatment_id,
                    baseline_rows=(
                        None
                        if transcript is None
                        else transcript.baseline.evidence.row_accounting
                    ),
                    comparison_rows=(
                        None
                        if transcript is None
                        else transcript.ceiling.evidence.row_accounting
                    ),
                    failure_type=failure_type,
                ),
            )

        status(TreatmentState.RUN_STARTED)
        progress(
            f"manifest ready; treatments={len(plan.treatments)}, "
            f"tasks={len(plan.task_ids)}, repeats={plan.repeats}"
        )
        with CheckpointedCodeBatchScorer(
            output_dir / "code-scoring-cache.sqlite3",
            runtime_identity=runtime.runtime_identity,
            executor=runtime.executor,
        ) as scorer:
            for index, treatment in enumerate(plan.treatments, start=1):
                treatment_dir = output_dir / treatment.directory
                treatment_dir.mkdir(parents=True, exist_ok=True)
                result_path = treatment_dir / "result.json"
                existing = (
                    _load_valid_result(
                        result_path,
                        plan=plan,
                        treatment=treatment,
                    )
                    if resume
                    else None
                )
                if existing is not None:
                    status(
                        TreatmentState.TREATMENT_SKIPPED,
                        treatment_id=treatment.treatment_id,
                        transcript=existing,
                    )
                    progress(
                        f"[{index}/{len(plan.treatments)}] skipped valid "
                        f"{treatment.treatment_id}: {_row_summary(existing)}"
                    )
                    continue
                status(
                    TreatmentState.TREATMENT_STARTED,
                    treatment_id=treatment.treatment_id,
                )
                progress(
                    f"[{index}/{len(plan.treatments)}] starting "
                    f"{treatment.treatment_id} "
                    f"(rows={treatment.planned_rows}, "
                    f"provider_calls={treatment.planned_provider_calls})"
                )

                def treatment_progress(
                    message: str,
                    treatment_id: str = treatment.treatment_id,
                ) -> None:
                    progress(f"{treatment_id}: {message}")

                try:
                    transcript = run_ed1_baseline_preview(
                        store=ObjectStore(
                            SqliteBackend(treatment_dir / "objects.sqlite3")
                        ),
                        tasks=tasks,
                        task_ids=plan.task_ids,
                        task_selection=plan.task_selection,
                        preflight_task=preflight_task,
                        pool_ceiling=plan.pool_ceiling,
                        task_model=treatment.task_model,
                        batch_scorer=scorer,
                        runtime=plan.runtime,
                        budget_ratio=treatment.budget_ratio,
                        concurrency=plan.concurrency,
                        repeats=plan.repeats,
                        partial_log=PartialLog(treatment_dir / "partial-log"),
                        prompt_cache=PromptResultCache(
                            treatment_dir / "prompt-cache"
                        ),
                        log=treatment_progress,
                    )
                    _validate_result(
                        transcript,
                        plan=plan,
                        treatment=treatment,
                    )
                    _atomic_write_model(result_path, transcript)
                except BaseException as exc:
                    status(
                        TreatmentState.TREATMENT_FAILED,
                        treatment_id=treatment.treatment_id,
                        failure_type=type(exc).__name__,
                    )
                    progress(
                        f"[{index}/{len(plan.treatments)}] failed "
                        f"{treatment.treatment_id} ({type(exc).__name__})"
                    )
                    raise
                status(
                    TreatmentState.TREATMENT_COMPLETED,
                    treatment_id=treatment.treatment_id,
                    transcript=transcript,
                )
                progress(
                    f"[{index}/{len(plan.treatments)}] completed "
                    f"{treatment.treatment_id}: {_row_summary(transcript)}"
                )
        status(TreatmentState.RUN_COMPLETED)
        progress("baseline behavior matrix completed")
        return plan


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed ED1 baseline behavior matrix."
    )
    parser.add_argument("--evaluation-python", required=True, type=Path)
    parser.add_argument("--snapshot-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run all four routes on one task, one repeat, unbudgeted",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_baseline_behavior_matrix(
        evaluation_python=args.evaluation_python,
        snapshot_path=args.snapshot_path,
        output_dir=args.output_dir,
        resume=args.resume,
        concurrency=args.concurrency,
        smoke=args.smoke,
    )


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_TASK_MANIFEST",
    "EXCLUDED_TASK_IDS",
    "FULL_BUDGET_RATIOS",
    "Ed1BaselineMatrixPlan",
    "Ed1BaselineMatrixTreatmentPlan",
    "Ed1BaselineTreatmentStatus",
    "TreatmentState",
    "baseline_provider_routes",
    "build_matrix_plan",
    "main",
    "map_openai_credential",
    "parse_args",
    "raise_open_file_limit",
    "run_baseline_behavior_matrix",
]
