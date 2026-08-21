"""Contract tests for the dr-exec worker-pool rollout driver.

Two things are pinned here. First, the subprocess driver and the default
in-process driver must produce the same evidence and attribution for the
same rollout, so that switching drivers is not an experiment change.
Second, the driver's deadline vocabulary: a row that outruns its own wall
budget, and a batch that outruns the operation deadline, must surface as
this driver's row statuses rather than as silent absence.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from whetstone.eval.drivers.eval_result import InternalEvalResult
from whetstone.eval.drivers.graph_row_request import RowDispatchStatus
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.drivers.subprocess_graph_rollout import (
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


def test_batch_expiry_reports_remaining_rows_as_not_dispatched(
    tmp_path: Path,
) -> None:
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
    assert {row.failure_code for row in result.outputs} == {
        RowDispatchStatus.NOT_DISPATCHED.value
    }
    assert all(
        row.row_state is ExecutedRowState.MISSING for row in result.outputs
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
