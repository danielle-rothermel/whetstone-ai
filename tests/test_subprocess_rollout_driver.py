"""Contract tests for the dr-exec worker-pool rollout driver.

Two things are pinned here. First, the subprocess driver and the default
in-process driver must produce the same evidence and attribution for the
same rollout, so that switching drivers is not an experiment change.
Second, the driver's deadline vocabulary: a row that outruns its own wall
budget, and a batch that outruns the operation deadline, must surface as
this driver's row statuses rather than as silent absence.
"""

from __future__ import annotations

import gc
import os
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from whetstone.eval.drivers.eval_result import InternalEvalResult
from whetstone.eval.drivers.graph_row_request import RowDispatchStatus
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.drivers.subprocess_graph_rollout import (
    RowWorkerError,
    SubprocessGraphRolloutEvalDriver,
)
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.traces import ExecutedRowState
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.env import Experiment
from whetstone.provider.policy import (
    ProviderExecutionPolicy,
    default_transport_policy,
)
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.row_worker import GATED_ROW_RELEASE_ENV
from whetstone.testing.fakes.transport import fake_llm_transport_factory
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    ToyTask,
    build_toy_experiment,
    toy_template_render_contract,
)

GATED_ENTRYPOINT = "whetstone.testing.fakes.row_worker:gated_run_row"

#: dr-exec names the scheduler's own worker threads with this prefix.
#: Refusing to start one is how this test forces a scheduler break.
SCHEDULER_WORKER_THREAD_PREFIX = "dr-exec-pool-worker"

#: Per-row and per-batch budgets short enough to fire promptly on a gated
#: row. The tests assert on the resulting status, never on elapsed time.
SHORT_ROW_BUDGET_SECONDS = 1.0
SHORT_BATCH_BUDGET_SECONDS = 1.0

#: Long enough that neither budget can fire during an ordinary rollout.
GENEROUS_BUDGET_SECONDS = 3_600.0


def _execution_policy() -> ProviderExecutionPolicy:
    return ProviderExecutionPolicy(
        transport_policy=default_transport_policy(
            api_key_env="WHETSTONE_TOY_API_KEY"
        )
    )


def _eval_request(experiment: Experiment, request_id: str) -> EvalRequest:
    return EvalRequest(
        request_id=request_id,
        candidate=experiment.initial_candidate,
        metadata=metadata_with_purpose("test"),
    )


def _in_process_driver() -> GraphRolloutEvalDriver:
    return GraphRolloutEvalDriver(
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
        transport_factory=fake_llm_transport_factory,
    )


def _subprocess_driver(
    *,
    row_job_entrypoint: str = "whetstone.eval.drivers.graph_worker:run_row",
    unit_deadline_seconds: float = GENEROUS_BUDGET_SECONDS,
) -> Iterator[SubprocessGraphRolloutEvalDriver]:
    driver = SubprocessGraphRolloutEvalDriver(
        row_job_entrypoint=row_job_entrypoint,
        unit_deadline_seconds=unit_deadline_seconds,
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
        transport_factory=fake_llm_transport_factory,
    )
    with driver:
        yield driver


@pytest.fixture
def subprocess_driver() -> Iterator[SubprocessGraphRolloutEvalDriver]:
    yield from _subprocess_driver()


def _run(
    driver: GraphRolloutEvalDriver,
    *,
    experiment: Experiment,
    request_id: str,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
    concurrency: int = 2,
    max_wall_seconds: float | None = None,
) -> InternalEvalResult:
    sampling = experiment.eval_configs.internal
    return driver.run(
        experiment=experiment,
        sampling=sampling,
        request=_eval_request(experiment, request_id),
        eval_config_hash=sampling.eval_config.config_hash,
        execution_policy=_execution_policy(),
        concurrency=concurrency,
        max_wall_seconds=max_wall_seconds,
        partial_log=partial_log,
        prompt_cache=prompt_cache,
    )


def _comparable_row(row: RolloutRowOutput) -> RolloutRowOutput:
    """Strip nothing: both drivers must agree on every row field."""
    return replace(row)


def test_both_drivers_produce_identical_rows_and_identities(
    tmp_path: Path,
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    experiment = build_toy_experiment(num_seeds=2)

    in_process_log = PartialLog(tmp_path / "in-process-partials")
    in_process_cache = PromptResultCache(root=tmp_path / "in-process-cache")
    subprocess_log = PartialLog(tmp_path / "subprocess-partials")
    subprocess_cache = PromptResultCache(root=tmp_path / "subprocess-cache")

    in_process = _run(
        _in_process_driver(),
        experiment=experiment,
        request_id="align:in-process",
        partial_log=in_process_log,
        prompt_cache=in_process_cache,
    )
    subprocess = _run(
        subprocess_driver,
        experiment=experiment,
        request_id="align:subprocess",
        partial_log=subprocess_log,
        prompt_cache=subprocess_cache,
    )

    assert [_comparable_row(row) for row in subprocess.outputs] == [
        _comparable_row(row) for row in in_process.outputs
    ]
    assert subprocess.request_identities == in_process.request_identities
    assert subprocess.request_identities
    assert subprocess.aggregate == in_process.aggregate
    assert subprocess.per_task_scores == in_process.per_task_scores
    assert subprocess.per_task_counts == in_process.per_task_counts
    assert subprocess.deadline_reached is in_process.deadline_reached


def test_both_drivers_agree_on_the_rows_an_elapsed_deadline_produces(
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    """Switching drivers must not change the deadline's shape of evidence.

    The failure codes are deliberately not compared here. The in-process
    driver has no notion of dispatch, so it calls every deadline-stopped row
    ``deadline``; the subprocess driver can tell a row that never reached a
    worker from one that ran and was killed, and says so. Giving the
    in-process driver the same distinction is a separate change.
    """
    experiment = build_toy_experiment()

    in_process = _run(
        _in_process_driver(),
        experiment=experiment,
        request_id="align-deadline:in-process",
        concurrency=1,
        max_wall_seconds=0.0,
    )
    subprocess_result = _run(
        subprocess_driver,
        experiment=experiment,
        request_id="align-deadline:subprocess",
        concurrency=1,
        max_wall_seconds=0.0,
    )

    assert in_process.deadline_reached is subprocess_result.deadline_reached
    assert [row.row_state for row in subprocess_result.outputs] == [
        row.row_state for row in in_process.outputs
    ]
    assert subprocess_result.request_identities == (
        in_process.request_identities
    )


def test_both_drivers_write_the_same_partial_log_entries(
    tmp_path: Path,
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    experiment = build_toy_experiment(num_seeds=2)
    in_process_log = PartialLog(tmp_path / "in-process-partials")
    subprocess_log = PartialLog(tmp_path / "subprocess-partials")

    _run(
        _in_process_driver(),
        experiment=experiment,
        request_id="partials:in-process",
        partial_log=in_process_log,
    )
    _run(
        subprocess_driver,
        experiment=experiment,
        request_id="partials:subprocess",
        partial_log=subprocess_log,
    )

    in_process_keys = sorted(
        record.key() for record in in_process_log.load()
    )
    subprocess_keys = sorted(
        record.key() for record in subprocess_log.load()
    )
    assert subprocess_keys == in_process_keys
    assert subprocess_keys


def test_both_drivers_populate_the_same_prompt_cache(
    tmp_path: Path,
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    experiment = build_toy_experiment()
    shared_cache = PromptResultCache(root=tmp_path / "shared-cache")

    _run(
        _in_process_driver(),
        experiment=experiment,
        request_id="cache:in-process",
        prompt_cache=shared_cache,
    )
    warm = _run(
        subprocess_driver,
        experiment=experiment,
        request_id="cache:subprocess",
        prompt_cache=shared_cache,
    )

    assert all(
        row.row_state is ExecutedRowState.SUCCESS for row in warm.outputs
    )


def _gated_experiment(release_path: Path, *, task_count: int) -> Experiment:
    tasks = tuple(
        ToyTask(
            task_id=f"gated-{index}",
            prompt_inputs={
                "prompt": f"hello {index}",
                GATED_ROW_RELEASE_ENV: str(release_path),
            },
            gold=str(index),
        )
        for index in range(task_count)
    )
    return build_toy_experiment(internal_tasks=tasks)


def test_row_exceeding_its_wall_budget_reports_the_per_row_deadline(
    tmp_path: Path,
) -> None:
    never_released = tmp_path / "never-released"
    experiment = _gated_experiment(never_released, task_count=1)

    driver_context = _subprocess_driver(
        row_job_entrypoint=GATED_ENTRYPOINT,
        unit_deadline_seconds=SHORT_ROW_BUDGET_SECONDS,
    )
    driver = next(driver_context)
    try:
        result = _run(
            driver,
            experiment=experiment,
            request_id="deadline:per-row",
            concurrency=1,
        )
    finally:
        driver_context.close()

    assert not never_released.exists()
    assert [row.failure_code for row in result.outputs] == [
        RowDispatchStatus.UNIT_TIMEOUT.value
    ]
    assert all(
        row.row_state is ExecutedRowState.MISSING for row in result.outputs
    )
    assert result.deadline_reached is False


def test_batch_expiry_separates_running_rows_from_never_started_rows(
    tmp_path: Path,
) -> None:
    """A row killed at the deadline ran; only unstarted rows are undispatched.

    One worker takes the first row and blocks on it, so when the batch wall
    fires exactly one row is inside a worker and the other two never left the
    queue. Persisting all three as "not dispatched" would claim no provider
    work was attempted for a row that in fact ran.
    """
    never_released = tmp_path / "never-released"
    experiment = _gated_experiment(never_released, task_count=3)

    driver_context = _subprocess_driver(
        row_job_entrypoint=GATED_ENTRYPOINT,
        unit_deadline_seconds=GENEROUS_BUDGET_SECONDS,
    )
    driver = next(driver_context)
    try:
        result = _run(
            driver,
            experiment=experiment,
            request_id="deadline:batch",
            concurrency=1,
            max_wall_seconds=SHORT_BATCH_BUDGET_SECONDS,
        )
    finally:
        driver_context.close()

    assert not never_released.exists()
    assert result.deadline_reached is True
    codes = [row.failure_code for row in result.outputs]
    assert codes.count(RowDispatchStatus.OPERATION_DEADLINE.value) == 1
    assert codes.count(RowDispatchStatus.NOT_DISPATCHED.value) == 2
    assert all(
        row.row_state is ExecutedRowState.MISSING for row in result.outputs
    )


def test_every_running_row_is_reported_as_the_operation_deadline(
    tmp_path: Path,
) -> None:
    """With a worker per row, no row is undispatched when the wall fires."""
    never_released = tmp_path / "never-released"
    experiment = _gated_experiment(never_released, task_count=2)

    driver_context = _subprocess_driver(
        row_job_entrypoint=GATED_ENTRYPOINT,
        unit_deadline_seconds=GENEROUS_BUDGET_SECONDS,
    )
    driver = next(driver_context)
    try:
        result = _run(
            driver,
            experiment=experiment,
            request_id="deadline:batch-all-running",
            concurrency=2,
            max_wall_seconds=SHORT_BATCH_BUDGET_SECONDS,
        )
    finally:
        driver_context.close()

    assert not never_released.exists()
    assert result.deadline_reached is True
    assert {row.failure_code for row in result.outputs} == {
        RowDispatchStatus.OPERATION_DEADLINE.value
    }


def test_already_elapsed_deadline_dispatches_no_rows(
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    """A zero wall means the operation is already over before any row runs."""
    experiment = build_toy_experiment()

    result = _run(
        subprocess_driver,
        experiment=experiment,
        request_id="deadline:already-elapsed",
        concurrency=1,
        max_wall_seconds=0.0,
    )

    assert result.deadline_reached is True
    assert {row.failure_code for row in result.outputs} == {
        RowDispatchStatus.NOT_DISPATCHED.value
    }
    assert not result.request_identities


def test_in_process_driver_also_dispatches_nothing_at_a_zero_deadline() -> None:
    """The drivers agree that a zero wall yields no completed rows."""
    experiment = build_toy_experiment()

    result = _run(
        _in_process_driver(),
        experiment=experiment,
        request_id="deadline:already-elapsed-in-process",
        concurrency=1,
        max_wall_seconds=0.0,
    )

    assert result.deadline_reached is True
    assert all(
        row.row_state is ExecutedRowState.MISSING for row in result.outputs
    )
    assert not result.request_identities


@pytest.mark.parametrize(
    "max_wall_seconds", [pytest.param(1e-12, id="sub-nanosecond")]
)
def test_an_unrepresentable_deadline_expires_instead_of_raising(
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
    max_wall_seconds: float,
) -> None:
    """A deadline too small for dr-exec is already elapsed, not an error."""
    experiment = build_toy_experiment()

    result = _run(
        subprocess_driver,
        experiment=experiment,
        request_id="deadline:unrepresentable",
        concurrency=1,
        max_wall_seconds=max_wall_seconds,
    )

    assert result.deadline_reached is True
    assert {row.failure_code for row in result.outputs} == {
        RowDispatchStatus.NOT_DISPATCHED.value
    }


@pytest.mark.parametrize(
    "unit_deadline_seconds",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_an_unusable_row_budget_is_rejected_at_construction(
    unit_deadline_seconds: float,
) -> None:
    """The caller learns at its own call site, not mid-rollout."""
    with pytest.raises(ValueError, match="unit_deadline_seconds"):
        SubprocessGraphRolloutEvalDriver(
            unit_deadline_seconds=unit_deadline_seconds,
            eval_runner=FakeEvalProcedureRunner(),
            mutation_field=TOY_MUTATION_FIELD,
            render_contract=toy_template_render_contract(),
            transport_factory=fake_llm_transport_factory,
        )


def test_pool_width_is_fixed_by_the_first_run(
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    """A pool serves one width for its lifetime, so widening must be loud."""
    experiment = build_toy_experiment()
    _run(
        subprocess_driver,
        experiment=experiment,
        request_id="width:first",
        concurrency=1,
    )

    with pytest.raises(RowWorkerError, match="cannot widen"):
        _run(
            subprocess_driver,
            experiment=experiment,
            request_id="width:wider",
            concurrency=4,
        )


def test_a_narrower_later_run_is_honoured(
    tmp_path: Path,
) -> None:
    """Narrowing schedules fewer rows at once instead of being ignored.

    Two gated rows at concurrency 1 mean only one can be inside a worker when
    the batch wall fires, even though the pool is two workers wide. If the
    narrower request were ignored both would run and both would report the
    operation deadline.
    """
    never_released = tmp_path / "never-released"
    experiment = _gated_experiment(never_released, task_count=2)

    driver_context = _subprocess_driver(
        row_job_entrypoint=GATED_ENTRYPOINT,
        unit_deadline_seconds=GENEROUS_BUDGET_SECONDS,
    )
    driver = next(driver_context)
    try:
        _run(
            driver,
            experiment=build_toy_experiment(),
            request_id="narrow:wide-first",
            concurrency=2,
        )
        result = _run(
            driver,
            experiment=experiment,
            request_id="narrow:narrow-second",
            concurrency=1,
            max_wall_seconds=SHORT_BATCH_BUDGET_SECONDS,
        )
    finally:
        driver_context.close()

    codes = [row.failure_code for row in result.outputs]
    assert codes.count(RowDispatchStatus.OPERATION_DEADLINE.value) == 1
    assert codes.count(RowDispatchStatus.NOT_DISPATCHED.value) == 1


def test_a_closed_driver_refuses_to_run_rather_than_respawning() -> None:
    """Closing is terminal, so it cannot silently reset the pool's width."""
    experiment = build_toy_experiment()
    before = _child_pids()
    driver = SubprocessGraphRolloutEvalDriver(
        unit_deadline_seconds=GENEROUS_BUDGET_SECONDS,
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
        transport_factory=fake_llm_transport_factory,
    )
    _run(driver, experiment=experiment, request_id="closed:run", concurrency=1)
    driver.close()

    with pytest.raises(RowWorkerError, match="closed"):
        _run(
            driver,
            experiment=experiment,
            request_id="closed:rerun",
            concurrency=4,
        )

    assert _child_pids() - before == frozenset()


def _child_pids() -> frozenset[str]:
    listing = subprocess.run(
        ["pgrep", "-P", str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
    )
    return frozenset(listing.stdout.split())


def test_closing_the_driver_stops_its_workers() -> None:
    """Workers are owned, not leaked: closing releases exactly them."""
    experiment = build_toy_experiment()
    before = _child_pids()
    driver_context = _subprocess_driver()
    driver = next(driver_context)
    _run(driver, experiment=experiment, request_id="close:run", concurrency=2)
    assert _child_pids() - before

    driver.close()

    assert _child_pids() - before == frozenset()


def test_dropping_the_driver_stops_its_workers() -> None:
    """A caller that never closes still releases workers on collection."""
    experiment = build_toy_experiment()
    before = _child_pids()
    driver = SubprocessGraphRolloutEvalDriver(
        unit_deadline_seconds=GENEROUS_BUDGET_SECONDS,
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
        transport_factory=fake_llm_transport_factory,
    )
    _run(driver, experiment=experiment, request_id="drop:run", concurrency=2)
    assert _child_pids() - before

    del driver
    gc.collect()

    assert _child_pids() - before == frozenset()


def test_a_broken_scheduler_surfaces_as_this_driver_s_error(
    monkeypatch: pytest.MonkeyPatch,
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    """A caller catches RowWorkerError without importing dr-exec."""
    experiment = build_toy_experiment()
    real_start = threading.Thread.start

    def refuse_pool_worker(self: threading.Thread) -> None:
        if self.name.startswith(SCHEDULER_WORKER_THREAD_PREFIX):
            raise RuntimeError("cannot start a scheduling thread")
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", refuse_pool_worker)

    with pytest.raises(RowWorkerError, match="scheduler broke"):
        _run(
            subprocess_driver,
            experiment=experiment,
            request_id="broken:scheduler",
            concurrency=1,
        )


def test_row_dispatch_status_values_are_pinned() -> None:
    """These strings are persisted as row failure codes."""
    assert {
        status.name: status.value for status in RowDispatchStatus
    } == {
        "COMPLETED": "completed",
        "UNIT_TIMEOUT": "unit-timeout",
        "OPERATION_DEADLINE": "deadline",
        "NOT_DISPATCHED": "not-dispatched",
    }
