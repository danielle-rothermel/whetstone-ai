from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from whetstone.evaluation.schema import RowAccounting
from whetstone.optimization.validation import matrix as matrix_module
from whetstone.optimization.validation.matrix import (
    BehaviorMatrixHooks,
    MatrixTreatmentBase,
    MatrixTreatmentState,
    MatrixTreatmentStatus,
    append_status,
    prepare_manifest,
    run_behavior_matrix,
    run_lock,
)


class _FakeTreatment(MatrixTreatmentBase):
    pass


class _FakePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    treatments: tuple[_FakeTreatment, ...]


class _FakeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment_id: str
    value: int


def _rows(value: int) -> tuple[RowAccounting, RowAccounting]:
    rows = RowAccounting(
        planned=value,
        present=value,
        missing=0,
        failed=0,
        invalid=0,
    )
    return rows, rows


def _hooks(
    *,
    executed: list[str] | None = None,
) -> BehaviorMatrixHooks[_FakePlan, _FakeTreatment, _FakeResult, None]:
    executed = executed if executed is not None else []

    @contextmanager
    def shared_context(_output_dir: Path, _plan: _FakePlan):
        yield None

    def execute_treatment(
        treatment: _FakeTreatment,
        _plan: _FakePlan,
        _shared: None,
        _log: object,
    ) -> _FakeResult:
        executed.append(treatment.treatment_id)
        return _FakeResult(treatment_id=treatment.treatment_id, value=1)

    def persist_result(path: Path, result: _FakeResult) -> None:
        path.write_text(result.model_dump_json(), encoding="utf-8")

    def load_valid_result(
        path: Path,
        _plan: _FakePlan,
        treatment: _FakeTreatment,
    ) -> _FakeResult | None:
        if not path.exists():
            return None
        try:
            result = _FakeResult.model_validate_json(path.read_text())
        except ValidationError:
            return None
        if result.treatment_id != treatment.treatment_id:
            return None
        return result

    return BehaviorMatrixHooks(
        shared_context=shared_context,
        execute_treatment=execute_treatment,
        validate_result=lambda _result, _plan, _treatment: None,
        load_valid_result=load_valid_result,
        persist_result=persist_result,
        row_summary=lambda result: f"value={result.value}",
        status_row_accounts=lambda result: _rows(result.value),
    )


def test_manifest_resume_requires_exact_plan_equality(tmp_path: Path) -> None:
    plan = _FakePlan(
        treatments=(_FakeTreatment(treatment_id="t01", directory="t01"),)
    )
    output = tmp_path / "output"
    output.mkdir()
    path = output / "run-manifest.json"

    prepare_manifest(plan, path=path, resume=False, plan_type=_FakePlan)
    prepare_manifest(plan, path=path, resume=True, plan_type=_FakePlan)

    restored = _FakePlan.model_validate_json(path.read_text())
    assert restored == plan
    mismatched = plan.model_copy(
        update={
            "treatments": (
                _FakeTreatment(treatment_id="t02", directory="t02"),
            )
        }
    )
    with pytest.raises(RuntimeError, match="does not exactly match"):
        prepare_manifest(
            mismatched,
            path=path,
            resume=True,
            plan_type=_FakePlan,
        )


def test_process_log_status_is_typed_and_flushed(tmp_path: Path) -> None:
    rows = RowAccounting(
        planned=30,
        present=29,
        missing=0,
        failed=1,
        invalid=0,
    )
    status = MatrixTreatmentStatus(
        timestamp="2026-08-08T12:00:00+00:00",
        elapsed_seconds=1.25,
        state=MatrixTreatmentState.TREATMENT_COMPLETED,
        treatment_id="treatment-1",
        baseline_rows=rows,
        comparison_rows=rows,
    )
    path = tmp_path / "process-log.jsonl"

    append_status(path, status)

    assert (
        MatrixTreatmentStatus.model_validate_json(path.read_text()) == status
    )
    with pytest.raises(ValidationError, match="both row accounts"):
        MatrixTreatmentStatus(
            timestamp="2026-08-08T12:00:00+00:00",
            elapsed_seconds=1.25,
            state=MatrixTreatmentState.TREATMENT_COMPLETED,
            treatment_id="treatment-1",
        )


def test_run_behavior_matrix_skips_valid_results_on_resume(
    tmp_path: Path,
) -> None:
    plan = _FakePlan(
        treatments=(
            _FakeTreatment(treatment_id="t01", directory="t01"),
            _FakeTreatment(treatment_id="t02", directory="t02"),
        )
    )
    output = tmp_path / "output"
    executed: list[str] = []
    hooks = _hooks(executed=executed)

    run_behavior_matrix(
        plan=plan,
        output_dir=output,
        resume=False,
        hooks=hooks,
    )
    assert executed == ["t01", "t02"]

    executed.clear()
    run_behavior_matrix(
        plan=plan,
        output_dir=output,
        resume=True,
        hooks=hooks,
    )
    assert executed == []


def test_run_behavior_matrix_writes_process_log_sequence(
    tmp_path: Path,
) -> None:
    plan = _FakePlan(
        treatments=(_FakeTreatment(treatment_id="t01", directory="t01"),)
    )
    output = tmp_path / "output"

    run_behavior_matrix(
        plan=plan,
        output_dir=output,
        resume=False,
        hooks=_hooks(),
    )

    states = [
        json.loads(line)["state"]
        for line in (output / "process-log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert states == [
        "run_started",
        "treatment_started",
        "treatment_completed",
        "run_completed",
    ]


def test_run_lock_prevents_concurrent_execution(tmp_path: Path) -> None:
    lock_path = tmp_path / ".run.lock"
    with run_lock(lock_path):
        with pytest.raises(RuntimeError, match="another behavior-matrix"):
            with run_lock(lock_path):
                pass


def test_open_file_limit_uses_the_concurrency_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updated: list[tuple[int, int]] = []
    monkeypatch.setattr(
        matrix_module.resource, "getrlimit", lambda _kind: (1024, 10_000)
    )
    monkeypatch.setattr(
        matrix_module.resource,
        "setrlimit",
        lambda _kind, limits: updated.append(limits),
    )

    assert matrix_module.raise_open_file_limit(100) == 6_400
    assert updated == [(6_400, 10_000)]
