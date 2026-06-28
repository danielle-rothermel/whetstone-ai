from __future__ import annotations

from dr_dspy import job_ordering


def test_stable_shuffle_is_deterministic_and_reorders_items() -> None:
    items = ["a", "b", "c", "d", "e", "f"]

    first = job_ordering.stable_shuffle(items, seed="seed", key=str)
    second = job_ordering.stable_shuffle(items, seed="seed", key=str)

    assert first == second
    assert sorted(first) == sorted(items)
    assert first != items


def test_stable_shuffle_seed_changes_order() -> None:
    items = ["a", "b", "c", "d", "e", "f"]

    assert job_ordering.stable_shuffle(
        items, seed="seed-1", key=str
    ) != job_ordering.stable_shuffle(items, seed="seed-2", key=str)
