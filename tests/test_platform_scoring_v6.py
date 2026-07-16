from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from whetstone.platform.targets import (
    SCORING_QUEUE_NAME,
    ScoringTargetSpec,
    scoring_target,
    target_registry,
)
from whetstone.records import DatasetSnapshotIdentityPayload


def _target(generation_run_id: str) -> ScoringTargetSpec:
    return ScoringTargetSpec(
        prediction_id="prediction",
        generation_run_id=generation_run_id,
        scoring_profile_id="humaneval",
        scoring_profile_version="1",
        parser_profile_id="python",
        parser_version="1",
        dataset_name="evalplus/humanevalplus",
        dataset_split="test",
        dataset_snapshot=DatasetSnapshotIdentityPayload.model_validate(
            {
                "sha256": "a" * 64,
                "header": {
                    "schema_version": 1,
                    "dataset_id": "evalplus/humanevalplus",
                    "hf_revision": "frozen",
                    "overrides_digest": "b" * 64,
                },
            }
        ),
    )


def test_scoring_target_is_managed_top_level_priority_queue() -> None:
    target = scoring_target()

    assert target.queue_name == SCORING_QUEUE_NAME
    assert target.topology.value == "top_level_only"
    assert target_registry().resolve(target.ref) == target


def test_scoring_workflow_arguments_carry_every_recipe_axis() -> None:
    target = _target("run-1")
    registered_item = SimpleNamespace(spec=target.spec, item_id="item-id")

    arguments = scoring_target().args_for(cast("Any", registered_item), 2)

    assert arguments[:8] == (
        "run-1",
        2,
        "humaneval",
        "1",
        "python",
        "1",
        "evalplus/humanevalplus",
        "test",
    )
    assert arguments[8] == target.dataset_snapshot.model_dump(mode="json")
    assert arguments[9]
    assert arguments[10] == "item-id"
