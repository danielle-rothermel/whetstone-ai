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
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from dr_exec.declarations.models import FiniteDurationLimit

from whetstone.eval.drivers.eval_result import InternalEvalResult
from whetstone.eval.drivers.graph_row_request import RowDispatchStatus
from whetstone.eval.drivers import graph_rollout as graph_rollout_module
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.drivers.row_common import (
    MAX_REPRESENTABLE_WALL_SECONDS,
    RolloutRowOutput,
    validated_phase_wall_seconds,
)
from whetstone.eval.drivers.subprocess_graph_rollout import (
    RowWorkerError,
    SubprocessGraphRolloutEvalDriver,
)
from whetstone.eval.protocol import EvalRequest, EvalTaskView, eval_is_success
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.eval.traces import ExecutedRowState
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.candidate import TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.provider.policy import (
    ProviderExecutionPolicy,
    default_transport_policy,
)
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.row_worker import (
    GATED_ROW_RELEASE_ENV,
    RAISING_ROW_MARKER,
    RAISING_ROW_MESSAGE,
)
from whetstone.testing.fakes.transport import fake_llm_transport_factory
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    ToyTask,
    build_toy_experiment,
    toy_template_render_contract,
)

GATED_ENTRYPOINT = "whetstone.testing.fakes.row_worker:gated_run_row"
RAISING_ENTRYPOINT = (
    "whetstone.testing.fakes.row_worker:selectively_raising_run_row"
)

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

    Both drivers share one dispatch vocabulary: a row the deadline stopped
    before it was submitted never attempted provider work and is
    ``not-dispatched``, and a row that was running when the deadline fired is
    the operation ``deadline``. An already-elapsed deadline submits nothing,
    so both must say ``not-dispatched`` for every row.
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
    assert [row.failure_code for row in subprocess_result.outputs] == [
        row.failure_code for row in in_process.outputs
    ]
    assert {row.failure_code for row in in_process.outputs} == {
        RowDispatchStatus.NOT_DISPATCHED.value
    }
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
    assert {row.failure_code for row in result.outputs} == {
        RowDispatchStatus.NOT_DISPATCHED.value
    }
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


def _raising_experiment(*, task_count: int, raising_index: int) -> Experiment:
    tasks = tuple(
        ToyTask(
            task_id=f"raising-{index}",
            prompt_inputs=(
                {"prompt": f"hello {index}", RAISING_ROW_MARKER: "1"}
                if index == raising_index
                else {"prompt": f"hello {index}"}
            ),
            gold=str(index),
        )
        for index in range(task_count)
    )
    return build_toy_experiment(internal_tasks=tasks)


def test_a_raising_row_is_attributed_to_the_payload(
    tmp_path: Path,
) -> None:
    """One bad row surfaces as this driver's error, blamed on the payload.

    The worker pool discards the row's exception, so the driver reports the
    failing row's coordinates and dr-exec's payload attribution rather than
    the original type or message. That upstream limitation is pinned here so
    a dr-exec fix that starts carrying the exception is noticed.
    """
    experiment = _raising_experiment(task_count=3, raising_index=1)
    partial_log = PartialLog(tmp_path / "raising-partials")

    driver_context = _subprocess_driver(
        row_job_entrypoint=RAISING_ENTRYPOINT,
    )
    driver = next(driver_context)
    try:
        with pytest.raises(RowWorkerError) as failure:
            _run(
                driver,
                experiment=experiment,
                request_id="raising:payload",
                concurrency=3,
                partial_log=partial_log,
            )
    finally:
        driver_context.close()

    message = str(failure.value)
    assert "owner=payload" in message
    assert "(1, 0)" in message
    # The upstream limitation: dr-exec keeps the exception to itself, so the
    # driver cannot name it. Reproduce a raising row in-process to see it.
    assert RAISING_ROW_MESSAGE not in message

    completed_rows = {record.key() for record in partial_log.load()}
    assert len(completed_rows) == 2


@dataclass(frozen=True, slots=True)
class _RaisingRenderContract:
    """Render every row except the marked one, which raises instead.

    Rendering happens before the rollout graph runs, so this raises out of
    the row the way the worker's entry point raises out of its row — rather
    than being captured as a graph node error.
    """

    delegate: TemplateRenderContract

    def validate_template(self, template: object) -> None:
        self.delegate.validate_template(template)

    def render(
        self, template: str, prompt_inputs: Mapping[str, str]
    ) -> str:
        if RAISING_ROW_MARKER in prompt_inputs:
            raise RuntimeError(RAISING_ROW_MESSAGE)
        return self.delegate.render(template, prompt_inputs)


def test_a_raising_row_is_reproducible_in_process() -> None:
    """The in-process driver still names the row's own exception.

    This is the documented workaround for the worker pool discarding it: the
    same failing row, run in process, carries its type and message.
    """
    experiment = _raising_experiment(task_count=3, raising_index=1)
    driver = GraphRolloutEvalDriver(
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=_RaisingRenderContract(
            toy_template_render_contract()
        ),  # type: ignore[arg-type]
        transport_factory=fake_llm_transport_factory,
    )

    with pytest.raises(RuntimeError, match=RAISING_ROW_MESSAGE):
        _run(
            driver,
            experiment=experiment,
            request_id="raising:in-process",
            concurrency=3,
        )


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


def _subprocess_engine(store: object) -> RuntimeEvalEngine:
    runtime = ReferenceEvalRuntimeConfig.model_validate(
        {"driver_mode": "subprocess"}
    )
    engine = runtime.build_engine(store)  # type: ignore[arg-type]
    assert isinstance(engine, RuntimeEvalEngine)
    return engine


def _evaluate(engine: RuntimeEvalEngine, request_id: str) -> None:
    evaluated = engine.evaluate(
        EvalRequest(
            request_id=request_id,
            candidate=engine.experiment.initial_candidate,
            metadata=metadata_with_purpose("test"),
        )
    )
    assert eval_is_success(evaluated)


def test_closing_the_engine_stops_the_subprocess_driver_s_workers(
    sqlite_store: object,
) -> None:
    """The only production construction path can end its driver's lifetime.

    ``build_engine`` owns the driver it constructs, so a caller that never
    touches the driver directly can still release its worker processes.
    """
    before = _child_pids()
    engine = _subprocess_engine(sqlite_store)
    _evaluate(engine, "engine-close:run")
    assert _child_pids() - before

    engine.close()

    assert _child_pids() - before == frozenset()
    # Closing twice must not fail or reach a driver that is already closed.
    engine.close()


def test_the_engine_context_manager_stops_its_workers(
    sqlite_store: object,
) -> None:
    """Leaving the ``with`` block releases the driver's workers."""
    before = _child_pids()
    with _subprocess_engine(sqlite_store) as engine:
        _evaluate(engine, "engine-context:run")
        assert _child_pids() - before

    assert _child_pids() - before == frozenset()


def test_a_derived_engine_does_not_close_the_root_engine_s_driver(
    sqlite_store: object,
) -> None:
    """Derived engines borrow the shared driver; only the root owns it."""
    before = _child_pids()
    engine = _subprocess_engine(sqlite_store)
    try:
        _evaluate(engine, "engine-derived:run")
        workers = _child_pids() - before
        assert workers

        task_id = engine.sampling.tasks[0].task_id
        engine.for_task_ids((task_id,)).close()
        engine.for_task_seed(task_id, 0).close()

        assert _child_pids() - before == workers
    finally:
        engine.close()

    assert _child_pids() - before == frozenset()


def test_closing_an_in_process_engine_is_a_no_op(
    sqlite_store: object,
) -> None:
    """A driver with nothing to release needs no close of its own."""
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store  # type: ignore[arg-type]
    )
    assert isinstance(engine, RuntimeEvalEngine)
    with engine:
        _evaluate(engine, "engine-in-process:run")
    engine.close()


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


def test_a_row_finished_before_the_deadline_keeps_its_real_result(
    tmp_path: Path,
) -> None:
    """A finished-but-uncollected row is evidence, not a deadline miss.

    ``Future.cancel()`` reports False both for a running future and for one
    that already finished, so a driver that reads False as "was executing"
    overwrites a completed row with a deadline miss and drops its result.

    The race is forced on state, not timing: the in-process driver's own
    clock is replaced by a gate that reports "time remains" until the row's
    future is observed done, then reports "expired" forever. That is exactly
    the window in which the row has a result nobody has collected yet.
    """
    experiment = build_toy_experiment(
        internal_tasks=(
            ToyTask(task_id="finished-0", prompt_inputs={"prompt": "hi"}, gold="0"),
        )
    )
    driver = _in_process_driver()

    submitted: list[object] = []
    real_submit = ThreadPoolExecutor.submit

    def _recording_submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        future = real_submit(self, fn, *args, **kwargs)
        submitted.append(future)
        return future

    def _gated_remaining(_deadline: float | None) -> float | None:
        """Report the wall as elapsed only once the row has truly finished."""
        if submitted and all(future.done() for future in submitted):
            return 0.0
        return GENEROUS_BUDGET_SECONDS

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ThreadPoolExecutor, "submit", _recording_submit)
        patch.setattr(
            graph_rollout_module,
            "remaining_phase_wall_seconds",
            _gated_remaining,
        )
        result = _run(
            driver,
            experiment=experiment,
            request_id="deadline:done-but-uncollected",
            concurrency=1,
            max_wall_seconds=GENEROUS_BUDGET_SECONDS,
        )

    assert submitted, "the row must have been submitted for the race to exist"
    row = result.outputs[0]
    assert row.failure_code == ""
    assert row.row_state is not ExecutedRowState.MISSING
    assert row.output_text is not None
    assert result.request_identities


@pytest.mark.parametrize(
    "max_wall_seconds",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_both_drivers_reject_an_invalid_batch_wall(
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
    max_wall_seconds: float,
) -> None:
    """A wall naming no interval is a caller error, not an expired batch.

    Silently expiring would turn a configuration mistake into a full set of
    rows persisted as deadline misses — evidence that reads as "the
    operation ran out of time" when in fact nothing was ever asked to run.
    """
    experiment = build_toy_experiment()

    for driver in (subprocess_driver, _in_process_driver()):
        with pytest.raises(ValueError, match="max_wall_seconds"):
            _run(
                driver,
                experiment=experiment,
                request_id="deadline:invalid",
                concurrency=1,
                max_wall_seconds=max_wall_seconds,
            )


def test_both_drivers_treat_an_infinite_batch_wall_as_unbounded(
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    """Positive infinity names no deadline, exactly as None does."""
    experiment = build_toy_experiment()

    subprocess_result = _run(
        subprocess_driver,
        experiment=experiment,
        request_id="deadline:infinite-subprocess",
        concurrency=2,
        max_wall_seconds=float("inf"),
    )
    in_process_result = _run(
        _in_process_driver(),
        experiment=experiment,
        request_id="deadline:infinite-in-process",
        concurrency=2,
        max_wall_seconds=float("inf"),
    )

    for result in (subprocess_result, in_process_result):
        assert result.deadline_reached is False
        assert {row.failure_code for row in result.outputs} == {""}
        assert all(
            row.row_state is not ExecutedRowState.MISSING
            for row in result.outputs
        )


def test_both_drivers_treat_an_overlarge_batch_wall_as_unbounded(
    subprocess_driver: SubprocessGraphRolloutEvalDriver,
) -> None:
    """A wall too long for dr-exec is generous, not already elapsed.

    dr-exec expresses a finite wall as a nanosecond count and refuses one
    that overflows. Reading that refusal as "cannot express it, so expire
    now" would turn the most generous deadline a caller can name into the
    harshest possible outcome: every row persisted as never dispatched. The
    shared validator instead reads a wall beyond dr-exec's range the same
    way it reads infinity — as no deadline at all.
    """
    experiment = build_toy_experiment()
    overlarge_wall_seconds = 1e18

    assert overlarge_wall_seconds > MAX_REPRESENTABLE_WALL_SECONDS
    with pytest.raises(ValueError):
        FiniteDurationLimit.from_seconds(overlarge_wall_seconds)

    subprocess_result = _run(
        subprocess_driver,
        experiment=experiment,
        request_id="deadline:overlarge-subprocess",
        concurrency=2,
        max_wall_seconds=overlarge_wall_seconds,
    )
    in_process_result = _run(
        _in_process_driver(),
        experiment=experiment,
        request_id="deadline:overlarge-in-process",
        concurrency=2,
        max_wall_seconds=overlarge_wall_seconds,
    )

    assert validated_phase_wall_seconds(overlarge_wall_seconds) is None
    for result in (subprocess_result, in_process_result):
        assert result.deadline_reached is False
        assert {row.failure_code for row in result.outputs} == {""}
        assert all(
            row.row_state is not ExecutedRowState.MISSING
            for row in result.outputs
        )
        assert result.request_identities
