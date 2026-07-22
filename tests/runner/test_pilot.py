"""Full checklist-B pilot against scripted responses (no live calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.registry import env_spec
from whetstone.runner.budget import CreditsSnapshot
from whetstone.runner.pilot import PILOT_REPEATS, run_pilot

from .support import (
    SPLIT,
    TASK_MODEL,
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
    assert path == tmp_path / "validation" / "pilots" / "c11.json"
    assert path.exists()
    import json

    data = json.loads(path.read_text())
    assert data["env"] == "c11"
    assert "naive" in data and "ceiling" in data
