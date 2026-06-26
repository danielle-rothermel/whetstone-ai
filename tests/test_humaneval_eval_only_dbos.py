from __future__ import annotations

from typing import Any

import pytest
from dspy.utils.dummies import dotdict  # type: ignore[attr-defined]


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
    eval_dbos_harness.register_eval_queues(config)
    eval_dbos_harness.listen_to_selected_queue(
        eval_dbos_harness.QueueSelection.BOTH
    )

    assert registered == [
        {
            "name": eval_dbos_harness.GENERATION_QUEUE_NAME,
            "worker_concurrency": 11,
        },
        {
            "name": eval_dbos_harness.SCORING_QUEUE_NAME,
            "worker_concurrency": 5,
        },
    ]
    assert listened == [
        [
            eval_dbos_harness.GENERATION_QUEUE_NAME,
            eval_dbos_harness.SCORING_QUEUE_NAME,
        ]
    ]


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

    eval_dbos_harness.configure_dbos_runtime(config, consume_queues=False)

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
        use_mock_lm: bool,
    ) -> None:
        enqueued.append(
            {
                "queue_name": queue_name,
                "workflow": workflow,
                "database_url": database_url,
                "prediction_id": prediction_id,
                "use_mock_lm": use_mock_lm,
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
        "postgresql:///unit", [job], use_mock_lm=True
    )

    assert workflow_ids == ["generate:abc"]
    assert enqueued == [
        {
            "queue_name": eval_dbos_harness.GENERATION_QUEUE_NAME,
            "workflow": eval_dbos_harness.generate_prediction_workflow,
            "database_url": "postgresql:///unit",
            "prediction_id": "abc",
            "use_mock_lm": True,
        }
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
        "postgresql:///unit", ["abc"], timeout=7.0
    )

    assert workflow_ids == ["score:abc"]
    assert enqueued == [
        {
            "queue_name": eval_dbos_harness.SCORING_QUEUE_NAME,
            "workflow": eval_dbos_harness.score_prediction_workflow,
            "database_url": "postgresql:///unit",
            "prediction_id": "abc",
            "timeout": 7.0,
        }
    ]


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
    assert "0.666667" in markdown
    csv_text = csv_path.read_text()
    assert "model,temperature,sample_count" in csv_text
    assert "model/a" in csv_text


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
