from __future__ import annotations

from dr_dspy.humaneval_dbos_flow import stable_prediction_id_from_dimensions


def _identity(**dimensions: object) -> str:
    return stable_prediction_id_from_dimensions(
        experiment_name="exp",
        task_id="task/0",
        dimensions=dimensions,
        repetition_seed=0,
        digest_length=32,
    )


def test_deterministic() -> None:
    first = _identity(model="m", temperature=0.0)
    second = _identity(model="m", temperature=0.0)
    assert first == second
    assert len(first) == 32


def test_dimension_order_does_not_matter() -> None:
    assert _identity(model="m", temperature=0.0) == _identity(
        temperature=0.0, model="m"
    )


def test_changes_when_any_dimension_changes() -> None:
    base = _identity(model="m", temperature=0.0, budget_ratio=None)
    assert base != _identity(model="m2", temperature=0.0, budget_ratio=None)
    assert base != _identity(model="m", temperature=0.5, budget_ratio=None)
    assert base != _identity(model="m", temperature=0.0, budget_ratio=1.0)


def test_none_budget_distinct_from_zero() -> None:
    assert _identity(budget_ratio=None) != _identity(budget_ratio=0.0)


def test_repetition_seed_changes_identity() -> None:
    a = stable_prediction_id_from_dimensions(
        experiment_name="exp",
        task_id="t",
        dimensions={"model": "m"},
        repetition_seed=0,
        digest_length=32,
    )
    b = stable_prediction_id_from_dimensions(
        experiment_name="exp",
        task_id="t",
        dimensions={"model": "m"},
        repetition_seed=1,
        digest_length=32,
    )
    assert a != b
