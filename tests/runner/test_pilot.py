"""Full checklist-B pilot against scripted responses (no live calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.registry import TokenEstimate, env_spec
from whetstone.runner.budget import CreditsSnapshot
from whetstone.runner.pilot import PILOT_REPEATS, run_pilot

from .support import (
    SPLIT,
    TASK_MODEL,
    FailingTransport,
    FakeTransport,
    _split_fits,
    correct_reply,
    no_improvement_reply,
    runner_execution_policy,
    tiny_experiment,
)


def _pool_n(env: str) -> int:
    n = 1
    while not _split_fits(env_spec(env), n):
        n += 1
    return n


def test_pilot_full_run_clean_pass() -> None:
    env = "c11"
    exp = tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        spec_estimate_tokens=100,
    )
    # Both probes score a clean 1.0; ceiling >= naive direction holds.
    assert report.naive.mean_score == pytest.approx(1.0)
    assert report.ceiling.mean_score == pytest.approx(1.0)
    assert report.direction_ok
    # 2 instances x 2 probes x 3 repeats = 12 per-call spot-records.
    assert len(report.calls) == 2 * 2 * len(PILOT_REPEATS)
    # Temp-0 agreement: all three repeats agree per instance.
    assert report.naive.agreement_rate == pytest.approx(1.0)


def test_pilot_records_extraction_spot_records() -> None:
    env = "c11"
    exp = tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
    )
    first = report.calls[0]
    # Each spot-record carries the raw response, the extracted answer, and 0/1.
    assert first.raw_response
    assert first.extracted == first.raw_response
    assert first.score in (0.0, 1.0)


def test_pilot_token_counts_vs_spec() -> None:
    env = "c11"
    exp = tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        spec_estimate_tokens=50,
    )
    # The fake transport carries no usage, so token_mean_total is None here;
    # the field is present and the ratio degrades to None rather than crashing.
    assert "token_mean_total" in report.as_dict()
    assert report.token_vs_spec is None or report.token_vs_spec >= 0


def test_pilot_spend_recorded_when_openrouter() -> None:
    env = "c11"
    exp = tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        spend_before=CreditsSnapshot(total_credits=710.0, total_usage=616.0),
        spend_after=CreditsSnapshot(total_credits=710.0, total_usage=616.5),
    )
    assert report.spend_usd == pytest.approx(0.5)


def test_pilot_writes_env_json(tmp_path: Path) -> None:
    env = "c11"
    exp = tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FakeTransport(reply=no_improvement_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
    )
    path = report.write(tmp_path)
    # --root is used exactly as given: pilots at <root>/pilots/, no extra
    # 'validation' segment (live round-1 path-duplication fix).
    assert path == tmp_path / "pilots" / "c11.json"
    assert path.exists()
    import json

    data = json.loads(path.read_text())
    assert data["env"] == "c11"
    assert "naive" in data and "ceiling" in data


def test_pilot_all_calls_failed_zero_success_rate() -> None:
    # Live round-1: every call fails pre-flight (missing_base_url). The pilot
    # must report success_rate 0.0 and a failure summary keyed by the code, not
    # a silent naive=None/ceiling=None result.
    env = "c11"
    tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FailingTransport(code="missing_base_url"),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
    )
    assert report.success_rate == 0.0
    assert report.success_count == 0
    assert report.failed_count == report.call_count == 12
    assert report.failure_summary() == {"missing_base_url": 12}
    assert report.naive.mean_score is None
    assert report.ceiling.mean_score is None
    assert report.as_dict()["failure_summary"] == {"missing_base_url": 12}


def test_pilot_uses_committed_per_env_token_estimates() -> None:
    # --spec-estimate-tokens defaults to None; the pilot falls back to the
    # env's per-probe estimates so the token-sanity check runs without a flag,
    # recorded separately per probe. c11 is LIVE-MEASURED (naive 1735 /
    # ceiling 1831) after the round-3 measured-mean update.
    env = "c11"
    exp = tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        spec_estimate_tokens=None,
    )
    expected = TokenEstimate(
        naive=1735, ceiling=1831, estimate_source="live-measured"
    )
    assert env_spec(env).token_estimate == expected
    assert report.naive.spec_estimate_tokens == 1735
    assert report.ceiling.spec_estimate_tokens == 1831
    assert report.naive.estimate_source == "live-measured"


def test_pilot_resume_skips_recorded_calls(tmp_path: Path) -> None:
    # A pilot with a partial log that already records some calls must NOT
    # re-drive them: a counting transport proves the recorded (instance, probe,
    # repeat) observations are restored from disk, not re-called.
    from whetstone.execution.partials import PartialCallRecord, PartialLog

    env = "c11"
    exp = tiny_experiment(env)
    partial_log = PartialLog(path=tmp_path / "c11.partial.jsonl")

    # Pre-seed the partial log with the naive probe's first instance/repeat.
    first_instance = exp.eval_configs.internal.instances[0]
    partial_log.append(
        PartialCallRecord(
            phase="pilot", instance_id=str(first_instance.id), unit="naive",
            repeat_id=0, score=1.0, total_tokens=100,
        )
    )

    class _Counter(FakeTransport):
        physical: int = 0

        def __call__(self, request):  # type: ignore[override]
            _Counter.physical += 1
            return super().__call__(request)

    _Counter.physical = 0
    transport = _Counter(reply=correct_reply(exp))
    report = run_pilot(
        env=env, lane="openrouter", model=TASK_MODEL, transport=transport,
        execution_policy=runner_execution_policy(), instance_count=2,
        pool_n_per_stratum=_pool_n(env), split_sizes=SPLIT,
        partial_log=partial_log,
    )
    # 2 instances x 2 probes x 3 repeats = 12 planned; 1 was pre-recorded, so
    # exactly 11 physical calls are made on this run.
    assert _Counter.physical == 11
    # The report still covers all 12 spot-records (restored + driven).
    assert len(report.calls) == 12


def test_pilot_records_spend_via_credits_fetcher() -> None:
    # Round-3 fix: the pilot self-reports spend when a credits fetcher is
    # injected (the gap the cell path already closed). before/after snapshots.
    env = "c11"
    exp = tiny_experiment(env)
    snapshots = [
        CreditsSnapshot(total_credits=710.0, total_usage=616.0),
        CreditsSnapshot(total_credits=710.0, total_usage=616.4),
    ]

    def _fetch() -> CreditsSnapshot:
        return snapshots.pop(0)

    report = run_pilot(
        env=env, lane="openrouter", model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(), instance_count=2,
        pool_n_per_stratum=_pool_n(env), split_sizes=SPLIT,
        credits_fetcher=_fetch,
    )
    assert report.spend_usd == pytest.approx(0.4)


def test_pilot_cli_flag_overrides_committed_estimate() -> None:
    # An explicit --spec-estimate-tokens overrides BOTH probes' committed
    # estimates.
    env = "c11"
    exp = tiny_experiment(env)
    report = run_pilot(
        env=env,
        lane="openrouter",
        model=TASK_MODEL,
        transport=FakeTransport(reply=correct_reply(exp)),
        execution_policy=runner_execution_policy(),
        instance_count=2,
        pool_n_per_stratum=_pool_n(env),
        split_sizes=SPLIT,
        spec_estimate_tokens=999,
    )
    assert report.naive.spec_estimate_tokens == 999
    assert report.ceiling.spec_estimate_tokens == 999
