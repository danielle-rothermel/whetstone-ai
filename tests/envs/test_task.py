from __future__ import annotations

from whetstone_envs.core import make_instance

from whetstone.envs.task import Task

_ENV_NAME = "example"


def _instance():
    return make_instance(
        id="example-D1-1",
        seed=42,
        strata=("D1",),
        prompt_inputs={"question": "Sally is a brimpus.", "query": "q?"},
        gold="True",
    )


def test_env_task_wraps_instance_fields() -> None:
    task = Task.from_instance(_ENV_NAME, _instance())
    assert task.env_name == _ENV_NAME
    assert task.task_id == "example-D1-1"
    assert task.seed == 42
    assert task.strata == ("D1",)
    assert task.prompt_inputs_dict() == {
        "question": "Sally is a brimpus.",
        "query": "q?",
    }
    assert task.gold == "True"


def test_external_input_fields_are_task_namespaced() -> None:
    task = Task.from_instance(_ENV_NAME, _instance())
    assert set(task.external_input_fields()) == {"task.query", "task.question"}


def test_identity_is_stable_and_full_hash() -> None:
    a = Task.from_instance(_ENV_NAME, _instance())
    b = Task.from_instance(_ENV_NAME, _instance())
    identity = a.task_hash()
    assert identity == b.task_hash()
    assert len(identity) == 64
    assert all(c in "0123456789abcdef" for c in identity)


def test_identity_changes_with_gold() -> None:
    base = _instance()
    other = make_instance(
        id=base.id,
        seed=base.seed,
        strata=base.strata,
        prompt_inputs=dict(base.prompt_inputs),
        gold="False",
    )
    assert (
        Task.from_instance(_ENV_NAME, base).task_hash()
        != Task.from_instance(_ENV_NAME, other).task_hash()
    )


def test_identity_changes_with_prompt_input() -> None:
    base = _instance()
    other = make_instance(
        id=base.id,
        seed=base.seed,
        strata=base.strata,
        prompt_inputs={"question": "Rex is a wumpus.", "query": "q?"},
        gold=base.gold,
    )
    assert (
        Task.from_instance(_ENV_NAME, base).task_hash()
        != Task.from_instance(_ENV_NAME, other).task_hash()
    )


def test_identity_changes_across_env_name() -> None:
    inst = _instance()
    assert (
        Task.from_instance("example", inst).task_hash()
        != Task.from_instance("other", inst).task_hash()
    )


def test_gold_is_not_in_prompt_inputs() -> None:
    task = Task.from_instance(_ENV_NAME, _instance())
    assert task.gold not in task.prompt_inputs_dict().values()
    assert len(task.task_hash()) == 64
    assert task.task_content_hash
