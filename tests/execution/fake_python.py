"""Cross-platform Python execution for tests of dr-code's executor seam."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from uuid import uuid4

from dr_exec import (
    AttemptId,
    BudgetAxis,
    BudgetExceededOutcome,
    CompletedExecution,
    ExecutionAttribution,
    ExecutionId,
    ExecutionJob,
    ExecutionMeasurements,
    ExecutionOutcome,
    ExecutionResult,
    ExitedOutcome,
    FailureOwner,
    FakeExecutor,
    FakeRecordReceipt,
    FiniteDurationLimit,
    PayloadOutputs,
    RetainedPayloadStream,
    SignaledOutcome,
    UntrustedPythonTarget,
)

_LOCAL_BOOTSTRAP = (
    "\n"
    "import json as _stub_json\n"
    "import sys as _stub_sys\n"
    "dr_exec_main(\n"
    "    _stub_json.loads(_stub_sys.stdin.buffer.read().decode('utf-8')),\n"
    "    lambda _document: None,\n"
    ")\n"
)


def _stream(data: bytes) -> RetainedPayloadStream:
    return RetainedPayloadStream(
        head=data,
        tail=b"",
        produced_bytes=len(data),
        dropped_bytes=0,
    )


def _completion(
    job: ExecutionJob,
    *,
    outcome: ExecutionOutcome,
    stdout: str = "",
    stderr: str = "",
) -> CompletedExecution:
    execution_id = ExecutionId(
        job_id=job.job_id,
        attempt_id=AttemptId(uuid4()),
    )
    now = datetime.now(UTC)
    return CompletedExecution(
        result=ExecutionResult(
            execution_id=execution_id,
            outcome=outcome,
            attribution=ExecutionAttribution(
                owner=(
                    FailureOwner.NONE
                    if isinstance(outcome, ExitedOutcome)
                    and outcome.exit_code == 0
                    else FailureOwner.PAYLOAD
                )
            ),
            protocol_outputs=(),
            payload_outputs=PayloadOutputs(
                stdout=_stream(stdout.encode()),
                stderr=_stream(stderr.encode()),
            ),
            measurements=ExecutionMeasurements(
                started_at=now,
                finished_at=now,
                duration_ns=0,
                teardown_duration_ns=0,
                input_bytes=0,
                protocol_bytes_received=0,
            ),
        ),
        record_receipt=FakeRecordReceipt(execution_id=execution_id),
    )


def local_python_executor() -> FakeExecutor:
    """Run each declared Python driver locally behind dr-exec's fake seam."""

    def respond(job: ExecutionJob, cancellation: object) -> CompletedExecution:
        del cancellation
        target = job.target
        assert isinstance(target, UntrustedPythonTarget)
        wall_time = job.budgets.wall_time
        timeout = (
            wall_time.max_ns / 1e9
            if isinstance(wall_time, FiniteDurationLimit)
            else None
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    target.driver_source + _LOCAL_BOOTSTRAP,
                ],
                input=json.dumps(target.request.to_json_dict()),
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _completion(
                job,
                outcome=BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME),
            )
        outcome: ExecutionOutcome = (
            ExitedOutcome(exit_code=completed.returncode)
            if completed.returncode >= 0
            else SignaledOutcome(signal_number=-completed.returncode)
        )
        return _completion(
            job,
            outcome=outcome,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return FakeExecutor(responder=respond)


__all__ = ["local_python_executor"]
