"""Bootstrap CI tests (pure, deterministic, no LLM calls)."""

from __future__ import annotations

import pytest

from whetstone.runner.statistics import bootstrap_delta_ci, mean


def test_point_delta_is_mean_difference() -> None:
    naive = (0.0, 0.0, 1.0, 0.0)
    best = (1.0, 1.0, 1.0, 0.0)
    ci = bootstrap_delta_ci(naive, best, seed=0)
    assert ci.delta == pytest.approx(mean(best) - mean(naive))


def test_ci_brackets_the_delta() -> None:
    naive = tuple([0.0] * 10)
    best = tuple([1.0] * 10)
    ci = bootstrap_delta_ci(naive, best, seed=1)
    # Every task improves by 1.0, so the whole interval sits at 1.0.
    assert ci.low == pytest.approx(1.0)
    assert ci.high == pytest.approx(1.0)
    assert ci.delta == pytest.approx(1.0)


def test_no_improvement_delta_zero() -> None:
    scores = (1.0, 0.0, 1.0, 0.0, 1.0)
    ci = bootstrap_delta_ci(scores, scores, seed=2)
    assert ci.delta == pytest.approx(0.0)
    assert ci.low == pytest.approx(0.0)
    assert ci.high == pytest.approx(0.0)


def test_deterministic_given_seed() -> None:
    naive = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    best = (1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
    a = bootstrap_delta_ci(naive, best, seed=7)
    b = bootstrap_delta_ci(naive, best, seed=7)
    assert a.as_dict() == b.as_dict()


def test_single_task_returns_point_interval() -> None:
    ci = bootstrap_delta_ci((0.0,), (1.0,), seed=0)
    assert ci.low == ci.high == ci.delta == pytest.approx(1.0)


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        bootstrap_delta_ci((0.0, 1.0), (1.0,), seed=0)


def test_empty_rejected() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        bootstrap_delta_ci((), (), seed=0)
