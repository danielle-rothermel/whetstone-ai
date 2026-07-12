from __future__ import annotations

from whetstone.platform.submission import (
    ScoringTargetSpec,
    prepare_scoring_manifest,
)
from whetstone.platform.targets import (
    SCORING_QUEUE_NAME,
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


def test_scoring_manifest_freezes_ordered_selection_axes() -> None:
    targets = (_target("run-1"), _target("run-2"))

    manifest, source, digest = prepare_scoring_manifest(
        operation_key="whetstone:scoring:exp:operation",
        experiment_name="exp",
        targets=targets,
    )

    assert source.targets == targets
    assert manifest.item_count == 2
    assert digest
    assert manifest.workflow_role == "scoring"
    assert manifest.manifest_digest
