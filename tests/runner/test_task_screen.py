from __future__ import annotations

import json

import pytest

from whetstone.envs import input_transform
from whetstone.runner import task_screen


def test_screen_uses_pr6_transform_functions() -> None:
    assert task_screen.rename_identifier is input_transform.rename_identifier
    assert task_screen.split_prompt is input_transform.split_prompt


def test_screen_key_lock_refuses_a_second_writer(tmp_path) -> None:
    path = tmp_path / "summary.json"

    with task_screen.screen_key_lock(path):
        with pytest.raises(task_screen.ScreenKeyLocked):
            with task_screen.screen_key_lock(path):
                pass


def test_summary_replacement_is_complete_and_restart_safe(tmp_path) -> None:
    path = tmp_path / "summary.json"
    task_screen.write_screen_summary_atomic(path, ({"round": 1},))
    task_screen.write_screen_summary_atomic(
        path,
        ({"round": 2}, {"complete": True}),
    )

    assert json.loads(path.read_text()) == [
        {"round": 2},
        {"complete": True},
    ]
    assert not list(tmp_path.glob("*.tmp"))
