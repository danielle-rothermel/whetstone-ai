from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from dspy.utils.dummies import dotdict  # type: ignore[attr-defined]
from rich.console import Console


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.rowcount = 0

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: str, params: object = None) -> None:
        self.statements.append(statement)

    def executemany(self, statement: str, params: object) -> None:
        self.statements.append(statement)
        self.rowcount = len(list(params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


class FakeCompletions:
    def __init__(self, *, reject_temperatures: set[float] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reject_temperatures = reject_temperatures or set()

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        temperature = kwargs.get("temperature")
        if temperature in self.reject_temperatures:
            raise ValueError(f"temperature rejected: {temperature}")
        return dotdict(
            model=kwargs["model"],
            choices=[
                dotdict(
                    message=dotdict(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=dotdict(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cost=0.01,
            ),
        )


class FakeClient:
    def __init__(self, *, reject_temperatures: set[float] | None = None) -> None:
        self.chat = dotdict(
            completions=FakeCompletions(
                reject_temperatures=reject_temperatures
            )
        )


def test_build_eval_dbos_config_uses_database_url_for_dbos_default(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DBOS_SYSTEM_DATABASE_URL", raising=False)

    config = eval_dbos_harness.build_eval_dbos_config(
        database_url="postgresql:///app",
        dbos_system_database_url=None,
        generation_concurrency=7,
        scoring_concurrency=3,
    )

    assert config.database_url == "postgresql:///app"
    assert config.dbos_system_database_url == "postgresql:///app"
    assert config.generation_concurrency == 7
    assert config.scoring_concurrency == 3
    assert eval_dbos_harness.build_dbos_config(config) == {
        "name": eval_dbos_harness.DBOS_APP_NAME,
        "system_database_url": "postgresql:///app",
    }


def test_build_eval_dbos_config_requires_database_url(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="DATABASE_URL"):
        eval_dbos_harness.build_eval_dbos_config(
            database_url=None,
            dbos_system_database_url=None,
            generation_concurrency=7,
            scoring_concurrency=3,
        )


def test_create_eval_schema_executes_expected_statements(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()

    def fake_connect(database_url: str) -> FakeConnection:
        assert database_url == "postgresql:///unit"
        return fake_conn

    monkeypatch.setattr(eval_dbos_harness.psycopg, "connect", fake_connect)

    eval_dbos_harness.create_eval_schema("postgresql:///unit")

    statements = fake_conn.cursor_instance.statements
    assert statements[0] == eval_dbos_harness.EXPERIMENTS_TABLE_SQL
    assert statements[1] == eval_dbos_harness.PREDICTIONS_TABLE_SQL
    assert statements[2:] == list(eval_dbos_harness.PREDICTION_INDEX_SQL)


def test_register_and_listen_to_eval_queues(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    registered: list[dict[str, Any]] = []
    listened: list[list[str]] = []

    def fake_register_queue(name: str, **kwargs: Any) -> None:
        registered.append({"name": name, **kwargs})

    def fake_listen_queues(queues: list[str]) -> None:
        listened.append(queues)

    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "register_queue", fake_register_queue
    )
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "listen_queues", fake_listen_queues
    )

    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )
    queue_names = eval_dbos_harness.eval_queue_names("exp")
    eval_dbos_harness.register_eval_queues(config, experiment_name="exp")
    eval_dbos_harness.listen_to_selected_queue(
        eval_dbos_harness.QueueSelection.BOTH, experiment_name="exp"
    )

    assert registered == [
        {
            "name": queue_names.generation,
            "worker_concurrency": 11,
        },
        {
            "name": queue_names.scoring,
            "worker_concurrency": 5,
        },
    ]
    assert listened == [[queue_names.generation, queue_names.scoring]]


def test_worker_queue_selection_and_log_path(eval_dbos_harness) -> None:
    queue_names = eval_dbos_harness.eval_queue_names("exp")
    assert eval_dbos_harness.queue_names_for_selection(
        eval_dbos_harness.QueueSelection.GENERATION, experiment_name="exp"
    ) == (queue_names.generation,)
    assert eval_dbos_harness.queue_names_for_selection(
        eval_dbos_harness.QueueSelection.SCORING, experiment_name="exp"
    ) == (queue_names.scoring,)
    assert eval_dbos_harness.queue_names_for_selection(
        eval_dbos_harness.QueueSelection.BOTH, experiment_name="exp"
    ) == (queue_names.generation, queue_names.scoring)

    experiment_name = "local mock/dbos smoke"
    path = eval_dbos_harness.default_worker_log_path(
        experiment_name=experiment_name,
        queue=eval_dbos_harness.QueueSelection.GENERATION,
        now=datetime(2026, 1, 2, 3, 4, 5),
        pid=123,
    )

    assert path == (
        eval_dbos_harness.DEFAULT_WORKER_LOG_ROOT
        / eval_dbos_harness.hashed_experiment_log_name(experiment_name)
        / "20260102-030405-generation-pid123.log"
    )


def test_worker_monitor_phase_counts_filter_experiment(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    class FetchCursor:
        def __init__(self) -> None:
            self.statement: str | None = None
            self.params: object = None

        def __enter__(self) -> FetchCursor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: str, params: object = None) -> None:
            self.statement = statement
            self.params = params

        def fetchall(self) -> list[tuple[str, int]]:
            return [("pending", 3)]

    class FetchConnection:
        def __init__(self) -> None:
            self.cursor_instance = FetchCursor()

        def __enter__(self) -> FetchConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def cursor(self) -> FetchCursor:
            return self.cursor_instance

    fake_conn = FetchConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )

    counts = eval_dbos_harness.fetch_prediction_phase_counts(
        "postgresql:///unit",
        status_column="scoring_status",
        experiment_name="exp-1",
    )

    assert counts == {"pending": 3}
    assert fake_conn.cursor_instance.statement is not None
    assert "WHERE experiment_name = %s" in fake_conn.cursor_instance.statement
    assert fake_conn.cursor_instance.params == ("exp-1",)


def test_worker_monitor_line_has_aligned_metrics(
    eval_dbos_harness,
) -> None:
    snapshot = eval_dbos_harness.WorkerQueueSnapshot(
        dbos_status_counts={"ENQUEUED": 3, "SUCCESS": 2},
        generation_status_counts={"generated": 2, "started": 3},
    )
    line = eval_dbos_harness.worker_monitor_line(
        snapshot,
        was_active=None,
        initial_success_total=2,
        initial_failure_total=0,
        force_summary=False,
    )
    assert line is not None

    assert line.startswith("Queue Active | active=   3 | enqueued=   3")
    assert "completed=   0" in line
    assert "gen pend=   0 start=   3 done=   2 err=   0" in line
    assert "score pend=   -" in line


def test_worker_monitor_line_can_force_empty_summary(
    eval_dbos_harness,
) -> None:
    snapshot = eval_dbos_harness.WorkerQueueSnapshot(
        dbos_status_counts={"SUCCESS": 7},
        scoring_status_counts={"pending": 2, "scored": 5},
    )

    line = eval_dbos_harness.worker_monitor_line(
        snapshot,
        was_active=False,
        initial_success_total=2,
        initial_failure_total=0,
        force_summary=True,
    )

    assert line is not None
    assert line.startswith("Queue Empty  | active=   0")
    assert "completed=   5" in line
    assert "score pend=   2 queue=   0 start=   0 done=   5 err=   0" in line


def test_timestamped_operator_lines(eval_dbos_harness) -> None:
    line = eval_dbos_harness.timestamped_line(
        "Queue Empty  | active=   0",
        now=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert line == "03:04:05 | Queue Empty  | active=   0"


def test_enqueue_scores_line_is_fixed_width(eval_dbos_harness) -> None:
    line = eval_dbos_harness.enqueue_scores_line(
        experiment_name="local-mock-dbos-smoke",
        selected_count=10,
        limit=1000,
        timeout=15.0,
    )

    assert line.startswith(
        "Enqueue Scores | selected=   10 | limit= 1000 | "
        "timeout=  15.0s |"
    )
    assert line.endswith("experiment=local-mock-dbos-smoke")
    assert eval_dbos_harness.enqueue_scores_style(0) == "yellow"
    assert eval_dbos_harness.enqueue_scores_style(10) == "green"


def test_worker_detail_log_writes_prediction_context(
    eval_dbos_harness, tmp_path
) -> None:
    log_file = tmp_path / "worker.log"
    logger = eval_dbos_harness.configure_worker_file_logging(log_file)
    context = eval_dbos_harness.PredictionLogContext(
        prediction_id="pred-1",
        experiment_name="exp",
        task_id="HumanEval/1",
        sample_index=0,
        model="model/a",
        temperature=0.0,
        repetition_seed=0,
    )

    eval_dbos_harness.emit_prediction_log_event(
        "generation_started",
        context,
        extra={"use_mock_lm": True},
    )
    for handler in logger.handlers:
        handler.flush()

    text = log_file.read_text()
    assert '"event":"generation_started"' in text
    assert '"prediction_id":"pred-1"' in text
    assert '"task_id":"HumanEval/1"' in text
    assert '"use_mock_lm":true' in text


def test_configure_dbos_runtime_launches_before_registering_queues(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_dbos(*_args: object, **_kwargs: object) -> None:
        calls.append("init")

    def fake_listen_queues(_queues: list[str]) -> None:
        calls.append("listen")

    def fake_launch() -> None:
        calls.append("launch")

    def fake_register_queue(_name: str, **_kwargs: object) -> None:
        calls.append("register")

    monkeypatch.setattr(eval_dbos_harness, "DBOS", fake_dbos)
    fake_dbos.listen_queues = fake_listen_queues  # type: ignore[attr-defined]
    fake_dbos.launch = fake_launch  # type: ignore[attr-defined]
    fake_dbos.register_queue = fake_register_queue  # type: ignore[attr-defined]

    config = eval_dbos_harness.EvalDbosConfig(
        database_url="postgresql:///app",
        dbos_system_database_url="postgresql:///dbos",
        generation_concurrency=11,
        scoring_concurrency=5,
    )

    eval_dbos_harness.configure_dbos_runtime(
        config, experiment_name="exp", consume_queues=False
    )

    assert calls == ["init", "listen", "launch", "register", "register"]


def test_build_humaneval_samples_from_rows_is_seeded(eval_dbos_harness) -> None:
    rows = [
        {
            "task_id": f"task/{index}",
            "prompt": f"def f_{index}(): pass",
            "test": "def check(candidate): pass",
            "entry_point": f"f_{index}",
        }
        for index in range(6)
    ]

    first = eval_dbos_harness.build_humaneval_samples_from_rows(
        rows, seed=3, sample_count=4
    )
    second = eval_dbos_harness.build_humaneval_samples_from_rows(
        rows, seed=3, sample_count=4
    )
    different = eval_dbos_harness.build_humaneval_samples_from_rows(
        rows, seed=4, sample_count=4
    )

    assert [sample.task_id for sample in first] == [
        sample.task_id for sample in second
    ]
    assert [sample.task_id for sample in first] != [
        sample.task_id for sample in different
    ]
    assert [sample.sample_index for sample in first] == [0, 1, 2, 3]


def test_build_prediction_jobs_uses_stable_ids(eval_dbos_harness) -> None:
    samples = [
        eval_dbos_harness.HumanEvalSample(
            task_id="task/add",
            sample_index=0,
            prompt="def add(a, b): pass",
            test="def check(candidate): pass",
            entry_point="add",
        )
    ]
    models = [
        eval_dbos_harness.ModelConfig(
            model="model/a", reasoning={"enabled": False}
        ),
        eval_dbos_harness.ModelConfig(
            model="model/b", reasoning={"effort": "low"}
        ),
    ]

    first = eval_dbos_harness.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub-1",
        samples=samples,
        model_configs=models,
        temperatures=[0.0, 0.2],
        repetitions=2,
    )
    second = eval_dbos_harness.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub-2",
        samples=samples,
        model_configs=models,
        temperatures=[0.0, 0.2],
        repetitions=2,
    )

    assert len(first) == 8
    assert {job.prediction_id for job in first} == {
        job.prediction_id for job in second
    }
    assert first[0].submission_id == "sub-1"
    assert second[0].submission_id == "sub-2"


def test_insert_prediction_jobs_does_not_overwrite_existing_rows(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()

    def fake_connect(database_url: str) -> FakeConnection:
        assert database_url == "postgresql:///unit"
        return fake_conn

    monkeypatch.setattr(eval_dbos_harness.psycopg, "connect", fake_connect)

    jobs = eval_dbos_harness.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub",
        samples=[
            eval_dbos_harness.HumanEvalSample(
                task_id="task/add",
                sample_index=0,
                prompt="def add(a, b): pass",
                test="def check(candidate): pass",
                entry_point="add",
            )
        ],
        model_configs=[
            eval_dbos_harness.ModelConfig(model="model/a", reasoning={})
        ],
        temperatures=[0.0],
        repetitions=1,
    )

    eval_dbos_harness.insert_prediction_jobs("postgresql:///unit", jobs)

    statement = fake_conn.cursor_instance.statements[0]
    assert "ON CONFLICT (prediction_id) DO NOTHING" in statement


def test_enqueue_generation_jobs_uses_stable_workflow_ids(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    workflow_ids: list[str] = []
    enqueued: list[dict[str, Any]] = []

    class FakeSetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

        def __enter__(self) -> None:
            workflow_ids.append(self.workflow_id)

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_enqueue_workflow(
        queue_name: str,
        workflow: object,
        database_url: str,
        prediction_id: str,
        experiment_name: str,
        use_mock_lm: bool,
        score_timeout: float,
    ) -> None:
        enqueued.append(
            {
                "queue_name": queue_name,
                "workflow": workflow,
                "database_url": database_url,
                "prediction_id": prediction_id,
                "experiment_name": experiment_name,
                "use_mock_lm": use_mock_lm,
                "score_timeout": score_timeout,
            }
        )

    monkeypatch.setattr(eval_dbos_harness, "SetWorkflowID", FakeSetWorkflowID)
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "enqueue_workflow", fake_enqueue_workflow
    )
    job = eval_dbos_harness.PredictionJob(
        prediction_id="abc",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="model/a",
        temperature=0.0,
        repetition_seed=0,
        prompt="def add(a, b): pass",
        test="def check(candidate): pass",
        entry_point="add",
    )

    eval_dbos_harness.enqueue_generation_jobs(
        "postgresql:///unit",
        [job],
        use_mock_lm=True,
        score_timeout=7.0,
    )

    assert workflow_ids == ["generate:abc"]
    assert enqueued == [
        {
            "queue_name": eval_dbos_harness.eval_queue_names(
                "exp"
            ).generation,
            "workflow": eval_dbos_harness.generate_prediction_workflow,
            "database_url": "postgresql:///unit",
            "prediction_id": "abc",
            "experiment_name": "exp",
            "use_mock_lm": True,
            "score_timeout": 7.0,
        }
    ]


def test_generation_workflow_enqueues_scoring_after_success(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    result = eval_dbos_harness.GenerationResult(
        prediction_id="abc",
        code="def add(a, b): return a + b",
    )

    def fake_generate_prediction_step(
        database_url: str, prediction_id: str, use_mock_lm: bool
    ) -> object:
        calls.append(
            (
                "generate",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                    "use_mock_lm": use_mock_lm,
                },
            )
        )
        return result

    def fake_record_generation_success_step(
        database_url: str, generation_result: object
    ) -> None:
        calls.append(
            (
                "record_generation_success",
                {
                    "database_url": database_url,
                    "result": generation_result,
                },
            )
        )

    def fake_enqueue_score_job(
        database_url: str,
        prediction_id: str,
        *,
        experiment_name: str,
        timeout: float,
    ) -> None:
        calls.append(
            (
                "enqueue_score",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                    "experiment_name": experiment_name,
                    "timeout": timeout,
                },
            )
        )

    def fake_mark_scoring_queued_step(
        database_url: str, prediction_id: str
    ) -> None:
        calls.append(
            (
                "mark_scoring_queued",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                },
            )
        )

    def fail_record_generation_error_step(
        _database_url: str, _prediction_id: str, _error: str
    ) -> None:
        raise AssertionError("generation should not be marked failed")

    monkeypatch.setattr(
        eval_dbos_harness,
        "generate_prediction_step",
        fake_generate_prediction_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "record_generation_success_step",
        fake_record_generation_success_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "enqueue_score_job",
        fake_enqueue_score_job,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "mark_scoring_queued_step",
        fake_mark_scoring_queued_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "record_generation_error_step",
        fail_record_generation_error_step,
    )

    status = eval_dbos_harness.generate_prediction_workflow.__wrapped__(
        "postgresql:///unit",
        "abc",
        "exp",
        True,
        9.0,
    )

    assert status == "generated"
    assert calls == [
        (
            "generate",
            {
                "database_url": "postgresql:///unit",
                "prediction_id": "abc",
                "use_mock_lm": True,
            },
        ),
        (
            "record_generation_success",
            {"database_url": "postgresql:///unit", "result": result},
        ),
        (
            "enqueue_score",
            {
                "database_url": "postgresql:///unit",
                "prediction_id": "abc",
                "experiment_name": "exp",
                "timeout": 9.0,
            },
        ),
        (
            "mark_scoring_queued",
            {"database_url": "postgresql:///unit", "prediction_id": "abc"},
        ),
    ]


def test_generation_workflow_does_not_enqueue_scoring_after_failure(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_generate_prediction_step(
        _database_url: str, _prediction_id: str, _use_mock_lm: bool
    ) -> object:
        raise RuntimeError("generation failed")

    def fake_record_generation_error_step(
        database_url: str, prediction_id: str, error: str
    ) -> None:
        calls.append(
            (
                "record_generation_error",
                {
                    "database_url": database_url,
                    "prediction_id": prediction_id,
                    "error": error,
                },
            )
        )

    def fail_enqueue_score_job(
        _database_url: str,
        _prediction_id: str,
        *,
        experiment_name: str,
        timeout: float,
    ) -> None:
        raise AssertionError("scoring should not be enqueued")

    def fail_mark_scoring_queued_step(
        _database_url: str, _prediction_id: str
    ) -> None:
        raise AssertionError("scoring should not be marked queued")

    monkeypatch.setattr(
        eval_dbos_harness,
        "generate_prediction_step",
        fake_generate_prediction_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "record_generation_error_step",
        fake_record_generation_error_step,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "enqueue_score_job",
        fail_enqueue_score_job,
    )
    monkeypatch.setattr(
        eval_dbos_harness,
        "mark_scoring_queued_step",
        fail_mark_scoring_queued_step,
    )

    status = eval_dbos_harness.generate_prediction_workflow.__wrapped__(
        "postgresql:///unit",
        "abc",
        "exp",
        True,
        9.0,
    )

    assert status == "generation_error"
    assert calls == [
        (
            "record_generation_error",
            {
                "database_url": "postgresql:///unit",
                "prediction_id": "abc",
                "error": "RuntimeError('generation failed')",
            },
        )
    ]


def test_build_generation_lm_passes_temperature_and_reasoning(
    eval_dbos_harness,
) -> None:
    job = eval_dbos_harness.PredictionJob(
        prediction_id="abc",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="openai/gpt-5-nano",
        temperature=0.2,
        repetition_seed=0,
        prompt="def add(a, b): pass",
        test="def check(candidate): pass",
        entry_point="add",
        reasoning={"effort": "minimal", "exclude": False},
    )
    client = FakeClient()
    buffer = eval_dbos_harness.LmEventBuffer()

    lm = eval_dbos_harness.build_generation_lm(
        job, use_mock_lm=False, event_buffer=buffer, client=client
    )
    lm.forward(messages=[{"role": "user", "content": "hello"}])

    assert client.chat.completions.calls == [
        {
            "model": "openai/gpt-5-nano",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": (
                eval_dbos_harness.DEFAULT_MAX_COMPLETION_TOKENS
            ),
            "temperature": 0.2,
            "extra_body": {
                "reasoning": {"effort": "minimal", "exclude": False}
            },
        }
    ]


def test_generate_code_for_job_uses_mock_lm(eval_dbos_harness) -> None:
    job = eval_dbos_harness.PredictionJob(
        prediction_id="abc",
        experiment_name="exp",
        submission_id="sub",
        task_id="task/add",
        sample_index=0,
        model="callable/mock",
        temperature=0.0,
        repetition_seed=0,
        prompt="def add(a, b):\n    pass\n",
        test="def check(candidate): pass",
        entry_point="add",
    )

    result = eval_dbos_harness.generate_code_for_job(
        job, use_mock_lm=True
    )

    assert result.prediction_id == "abc"
    assert "def add" in result.code


def test_score_generated_code_records_pass_and_failure(
    eval_dbos_harness,
) -> None:
    passing = eval_dbos_harness.ScoringTarget(
        prediction_id="pass",
        task_id="task/add",
        code="def add(a, b):\n    return a + b\n",
        test=(
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
        ),
        entry_point="add",
    )
    failing = eval_dbos_harness.ScoringTarget(
        prediction_id="fail",
        task_id="task/add",
        code="def add(a, b):\n    return a - b\n",
        test=(
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
        ),
        entry_point="add",
    )

    pass_result = eval_dbos_harness.score_generated_code(
        passing, timeout=5.0
    )
    fail_result = eval_dbos_harness.score_generated_code(
        failing, timeout=5.0
    )

    assert pass_result.score == 1.0
    assert pass_result.error is None
    assert fail_result.score == 0.0
    assert "AssertionError" in fail_result.error


def test_enqueue_score_jobs_uses_stable_workflow_ids(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    workflow_ids: list[str] = []
    enqueued: list[dict[str, Any]] = []

    class FakeSetWorkflowID:
        def __init__(self, workflow_id: str) -> None:
            self.workflow_id = workflow_id

        def __enter__(self) -> None:
            workflow_ids.append(self.workflow_id)

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_enqueue_workflow(
        queue_name: str,
        workflow: object,
        database_url: str,
        prediction_id: str,
        timeout: float,
    ) -> None:
        enqueued.append(
            {
                "queue_name": queue_name,
                "workflow": workflow,
                "database_url": database_url,
                "prediction_id": prediction_id,
                "timeout": timeout,
            }
        )

    monkeypatch.setattr(eval_dbos_harness, "SetWorkflowID", FakeSetWorkflowID)
    monkeypatch.setattr(
        eval_dbos_harness.DBOS, "enqueue_workflow", fake_enqueue_workflow
    )

    eval_dbos_harness.enqueue_score_jobs(
        "postgresql:///unit",
        ["abc"],
        experiment_name="exp",
        timeout=7.0,
    )

    assert workflow_ids == ["score:abc"]
    assert enqueued == [
        {
            "queue_name": eval_dbos_harness.eval_queue_names("exp").scoring,
            "workflow": eval_dbos_harness.score_prediction_workflow,
            "database_url": "postgresql:///unit",
            "prediction_id": "abc",
            "timeout": 7.0,
        }
    ]


def test_mark_scoring_queued_only_updates_waiting_predictions(
    eval_dbos_harness,
    monkeypatch,
) -> None:
    fake_conn = FakeConnection()
    monkeypatch.setattr(
        eval_dbos_harness.psycopg,
        "connect",
        lambda _database_url: fake_conn,
    )

    eval_dbos_harness.mark_scoring_queued("postgresql:///unit", ["abc"])

    statement = fake_conn.cursor_instance.statements[0]
    assert "scoring_status = 'queued'" in statement
    assert "AND scoring_status IN ('pending', 'score_error')" in statement


def test_summarize_analysis_records_includes_variance(
    eval_dbos_harness,
) -> None:
    records = [
        eval_dbos_harness.AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/1",
            repetition_seed=0,
            score=1.0,
            provider_cost=0.01,
        ),
        eval_dbos_harness.AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/1",
            repetition_seed=1,
            score=0.0,
            provider_cost=0.03,
        ),
        eval_dbos_harness.AnalysisRecord(
            model="model/a",
            temperature=0.0,
            task_id="task/2",
            repetition_seed=0,
            score=1.0,
            provider_cost=0.02,
        ),
    ]

    summaries = eval_dbos_harness.summarize_analysis_records(records)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.model == "model/a"
    assert summary.sample_count == 2
    assert summary.scored_count == 3
    assert summary.total_price == pytest.approx(0.06)
    assert summary.avg_price_per_sample == pytest.approx(0.02)
    assert summary.avg_performance == pytest.approx(2 / 3)
    assert summary.price_variance == pytest.approx(0.0001)
    assert summary.performance_variance == pytest.approx(1 / 3)
    assert summary.avg_repetition_variance == pytest.approx(0.5)


def test_analysis_markdown_and_csv(eval_dbos_harness, tmp_path) -> None:
    summaries = [
        eval_dbos_harness.AnalysisSummary(
            model="model/a",
            temperature=0.0,
            sample_count=2,
            scored_count=3,
            total_price=0.06,
            avg_price_per_sample=0.02,
            price_variance=0.0001,
            avg_performance=2 / 3,
            performance_variance=1 / 3,
            avg_repetition_variance=0.5,
        )
    ]

    markdown = eval_dbos_harness.analysis_markdown(
        experiment_name="exp", summaries=summaries
    )
    csv_path = tmp_path / "analysis.csv"
    eval_dbos_harness.write_analysis_csv(summaries, csv_path=csv_path)

    assert "# Eval Analysis: exp" in markdown
    assert "model/a" in markdown
    assert "Avg Price/1k Samples" in markdown
    assert "0.06" in markdown
    assert "20" in markdown
    assert "0.666667" in markdown
    assert "| Total |  | 2 | 3 | 0.06 |  |  |  |  |  |" in markdown
    csv_text = csv_path.read_text()
    assert "model,temperature,sample_count" in csv_text
    assert "model/a" in csv_text


def test_cost_formatting_uses_fixed_decimal(eval_dbos_harness) -> None:
    assert eval_dbos_harness.format_cost(5.0862e-05) == "0.000050862"
    assert eval_dbos_harness.format_cost(0.0007023) == "0.0007023"
    assert eval_dbos_harness.format_cost(47.12) == "47.12"
    assert eval_dbos_harness.format_cost(0.0) == "0"
    assert eval_dbos_harness.format_cost(None) == ""
    assert eval_dbos_harness.format_cost_column(
        [5.0862e-05, 0.0007023, 0.00011185, 5.492e-05, 4.712e-05]
    ) == [
        "0.000050862",
        "0.000702300",
        "0.000111850",
        "0.000054920",
        "0.000047120",
    ]
    assert eval_dbos_harness.format_cost_column(
        [0.050862, 0.7023, 0.11185, 0.05492, 0.04712]
    ) == [
        "0.050862",
        "0.702300",
        "0.111850",
        "0.054920",
        "0.047120",
    ]
    assert eval_dbos_harness.format_cost_column([0.0, 0.1]) == [
        "0.0",
        "0.1",
    ]
    assert eval_dbos_harness.price_per_thousand_samples(
        5.0862e-05
    ) == pytest.approx(0.050862)


def test_status_and_analysis_tables_render(eval_dbos_harness) -> None:
    status_table = eval_dbos_harness.status_counts_table(
        [
            {
                "experiment_name": "exp",
                "model": "model/a",
                "temperature": 0.0,
                "generation_status": "generated",
                "scoring_status": "scored",
                "count": 2,
            }
        ],
        experiment_name="exp",
    )
    analysis_table = eval_dbos_harness.analysis_table(
        experiment_name="exp",
        summaries=[
            eval_dbos_harness.AnalysisSummary(
                model="model/a",
                temperature=0.0,
                sample_count=2,
                scored_count=2,
                total_price=None,
                avg_price_per_sample=None,
                price_variance=None,
                avg_performance=1.0,
                performance_variance=None,
                avg_repetition_variance=None,
            ),
            eval_dbos_harness.AnalysisSummary(
                model="model/b",
                temperature=0.0,
                sample_count=2,
                scored_count=2,
                total_price=5.0862e-05,
                avg_price_per_sample=5.0862e-05,
                price_variance=None,
                avg_performance=1.0,
                performance_variance=None,
                avg_repetition_variance=None,
            ),
            eval_dbos_harness.AnalysisSummary(
                model="model/c",
                temperature=0.0,
                sample_count=2,
                scored_count=2,
                total_price=0.0007023,
                avg_price_per_sample=0.0007023,
                price_variance=None,
                avg_performance=1.0,
                performance_variance=None,
                avg_repetition_variance=None,
            ),
        ],
    )
    console = Console(record=True, width=120)
    console.print(status_table)
    console.print(analysis_table)
    text = console.export_text()
    performance_table, cost_table, variance_table = analysis_table.renderables

    assert "Eval Status: exp" in text
    assert "Generation" in text
    assert "Scoring" in text
    assert status_table.row_styles == list(
        eval_dbos_harness.TABLE_ROW_STYLES
    )
    assert "Eval Analysis: exp" in text
    assert "Avg $/1k Samples" in text
    assert performance_table.row_styles == list(
        eval_dbos_harness.TABLE_ROW_STYLES
    )
    assert cost_table.row_styles == list(eval_dbos_harness.TABLE_ROW_STYLES)
    assert variance_table.row_styles == list(
        eval_dbos_harness.TABLE_ROW_STYLES
    )
    assert (
        performance_table.rows[-1].style
        == eval_dbos_harness.TABLE_TOTAL_ROW_STYLE
    )
    assert (
        cost_table.rows[-1].style
        == eval_dbos_harness.TABLE_TOTAL_ROW_STYLE
    )
    assert "0.000050862" in text
    assert "0.000702300" in text
    assert "0.050862" in text
    assert "0.702300" in text
    assert "Total" in text
    assert "0.000753162" in text
    assert "Variance" in text


def test_run_temperature_probe_records_accept_and_reject(
    eval_dbos_harness,
) -> None:
    client = FakeClient(reject_temperatures={1.0})

    results = eval_dbos_harness.run_temperature_probe(
        model="openai/gpt-5.4-nano",
        reasoning={"enabled": False},
        temperatures=[0.0, 1.0],
        client=client,
    )

    assert [result.accepted for result in results] == [True, False]
    assert results[0].response_metadata["usage"]["cost"] == 0.01
    assert "temperature rejected" in results[1].error
    assert [
        call["temperature"] for call in client.chat.completions.calls
    ] == [0.0, 1.0]
    assert all(
        call["extra_body"]["reasoning"] == {"enabled": False}
        for call in client.chat.completions.calls
    )
