from __future__ import annotations

import fcntl
import os
import resource
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from time import perf_counter
from typing import Literal, TypeVar, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    ValidationError,
    model_validator,
)

from whetstone.eval.schema import RowAccounting
from dr_store.localfs import (
    fsync_parent_directory,
    open_private_regular_file,
)

MATRIX_SCHEMA_VERSION = 1


class MatrixTreatmentBase(BaseModel):
    """Shared treatment identity fields required by the matrix runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment_id: StrictStr
    directory: StrictStr


TPlan = TypeVar("TPlan", bound=BaseModel)
TResult = TypeVar("TResult")
TShared = TypeVar("TShared")
TTreatment = TypeVar("TTreatment", bound=MatrixTreatmentBase)


@verify(UNIQUE)
class MatrixTreatmentState(StrEnum):
    """Durably logged matrix lifecycle states."""

    RUN_STARTED = "run_started"
    TREATMENT_STARTED = "treatment_started"
    TREATMENT_SKIPPED = "treatment_skipped"
    TREATMENT_COMPLETED = "treatment_completed"
    TREATMENT_FAILED = "treatment_failed"
    RUN_COMPLETED = "run_completed"


class MatrixTreatmentStatus(BaseModel):
    """One flushed process-log event with optional terminal row accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = MATRIX_SCHEMA_VERSION
    timestamp: StrictStr
    elapsed_seconds: float
    state: MatrixTreatmentState
    treatment_id: StrictStr | None = None
    baseline_rows: RowAccounting | None = None
    comparison_rows: RowAccounting | None = None
    failure_type: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> MatrixTreatmentStatus:
        treatment_state = self.state not in {
            MatrixTreatmentState.RUN_STARTED,
            MatrixTreatmentState.RUN_COMPLETED,
        }
        if treatment_state != (self.treatment_id is not None):
            raise ValueError(
                "treatment lifecycle states require a treatment ID"
            )
        terminal_success = self.state in {
            MatrixTreatmentState.TREATMENT_SKIPPED,
            MatrixTreatmentState.TREATMENT_COMPLETED,
        }
        has_rows = (
            self.baseline_rows is not None and self.comparison_rows is not None
        )
        if terminal_success != has_rows:
            raise ValueError(
                "successful terminal states require both row accounts"
            )
        if (self.state is MatrixTreatmentState.TREATMENT_FAILED) != (
            self.failure_type is not None
        ):
            raise ValueError("only failed treatments carry a failure type")
        return self


@dataclass(frozen=True, slots=True)
class BehaviorMatrixHooks[TPlan, TTreatment, TResult, TShared]:
    shared_context: Callable[[Path, TPlan], AbstractContextManager[TShared]]
    execute_treatment: Callable[
        [TTreatment, TPlan, TShared, Callable[[str], None]], TResult
    ]
    validate_result: Callable[[TResult, TPlan, TTreatment], None]
    load_valid_result: Callable[[Path, TPlan, TTreatment], TResult | None]
    persist_result: Callable[[Path, TResult], None]
    row_summary: Callable[[TResult], str]
    status_row_accounts: Callable[
        [TResult], tuple[RowAccounting, RowAccounting]
    ]


def atomic_write_model(path: Path, model: BaseModel) -> None:
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


def prepare_manifest[TPlan: BaseModel](
    plan: TPlan,
    *,
    path: Path,
    resume: bool,
    plan_type: type[TPlan],
) -> None:
    if resume:
        try:
            existing = plan_type.model_validate_json(
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
    atomic_write_model(path, plan)


def append_status(path: Path, status: MatrixTreatmentStatus) -> None:
    body = status.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as destination:
        destination.write(body)
        destination.flush()
        os.fsync(destination.fileno())


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
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
    if (
        "OPENAI_API_KEY" not in environment
        and "MARIMO_OPENAI_API_KEY" in environment
    ):
        environment["OPENAI_API_KEY"] = environment["MARIMO_OPENAI_API_KEY"]


def _treatments(plan: BaseModel) -> object:
    raw = getattr(plan, "treatments", None)
    if raw is None:
        raise ValueError("behavior matrix plan must expose treatments")
    return raw


def run_behavior_matrix[
    TPlan: BaseModel,
    TTreatment,
    TResult,
    TShared,
](
    *,
    plan: TPlan,
    output_dir: Path,
    resume: bool,
    hooks: BehaviorMatrixHooks[TPlan, TTreatment, TResult, TShared],
    progress: Callable[[str], None] | None = None,
) -> TPlan:
    output_dir = output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    launched_at = perf_counter()

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    with run_lock(output_dir / ".run.lock"):
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
        prepare_manifest(
            plan,
            path=manifest_path,
            resume=resume,
            plan_type=type(plan),
        )
        process_log = output_dir / "process-log.jsonl"
        treatments = cast(tuple[MatrixTreatmentBase, ...], _treatments(plan))

        def status(
            state: MatrixTreatmentState,
            *,
            treatment_id: str | None = None,
            result: TResult | None = None,
            failure_type: str | None = None,
        ) -> None:
            baseline_rows: RowAccounting | None = None
            comparison_rows: RowAccounting | None = None
            if result is not None:
                baseline_rows, comparison_rows = hooks.status_row_accounts(
                    result
                )
            append_status(
                process_log,
                MatrixTreatmentStatus(
                    timestamp=datetime.now(UTC).isoformat(),
                    elapsed_seconds=perf_counter() - launched_at,
                    state=state,
                    treatment_id=treatment_id,
                    baseline_rows=baseline_rows,
                    comparison_rows=comparison_rows,
                    failure_type=failure_type,
                ),
            )

        status(MatrixTreatmentState.RUN_STARTED)
        emit(
            f"manifest ready; treatments={len(treatments)}, "
            f"output={output_dir}"
        )
        with hooks.shared_context(output_dir, plan) as shared:
            for index, treatment in enumerate(treatments, start=1):
                typed_treatment = cast(TTreatment, treatment)
                treatment_dir = output_dir / treatment.directory
                treatment_dir.mkdir(parents=True, exist_ok=True)
                result_path = treatment_dir / "result.json"
                existing = (
                    hooks.load_valid_result(
                        result_path,
                        plan,
                        typed_treatment,
                    )
                    if resume
                    else None
                )
                if existing is not None:
                    status(
                        MatrixTreatmentState.TREATMENT_SKIPPED,
                        treatment_id=treatment.treatment_id,
                        result=existing,
                    )
                    emit(
                        f"[{index}/{len(treatments)}] skipped valid "
                        f"{treatment.treatment_id}: "
                        f"{hooks.row_summary(existing)}"
                    )
                    continue
                status(
                    MatrixTreatmentState.TREATMENT_STARTED,
                    treatment_id=treatment.treatment_id,
                )
                emit(
                    f"[{index}/{len(treatments)}] starting "
                    f"{treatment.treatment_id}"
                )

                def treatment_progress(
                    message: str,
                    treatment_id: str = treatment.treatment_id,
                ) -> None:
                    emit(f"{treatment_id}: {message}")

                try:
                    result = hooks.execute_treatment(
                        typed_treatment,
                        plan,
                        shared,
                        treatment_progress,
                    )
                    hooks.validate_result(result, plan, typed_treatment)
                    hooks.persist_result(result_path, result)
                except BaseException as exc:
                    status(
                        MatrixTreatmentState.TREATMENT_FAILED,
                        treatment_id=treatment.treatment_id,
                        failure_type=type(exc).__name__,
                    )
                    emit(
                        f"[{index}/{len(treatments)}] failed "
                        f"{treatment.treatment_id} ({type(exc).__name__})"
                    )
                    raise
                status(
                    MatrixTreatmentState.TREATMENT_COMPLETED,
                    treatment_id=treatment.treatment_id,
                    result=result,
                )
                emit(
                    f"[{index}/{len(treatments)}] completed "
                    f"{treatment.treatment_id}: {hooks.row_summary(result)}"
                )
        status(MatrixTreatmentState.RUN_COMPLETED)
        emit("behavior matrix completed")
        return plan
