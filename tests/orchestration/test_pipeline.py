"""The concrete versioned Orchestration Pipeline registration.

Proves the pipeline is a linear native Pipeline Definition with one
rollout-execution Stage, registers cleanly in dr-platform, maps the Whetstone
namespaces onto native platform keys, and carries the Rollout Work Request only
as an opaque typed reference the platform never parses.
"""

from __future__ import annotations

from dr_platform.staging import (
    PipelineKey,
    PipelineRegistry,
    StageKey,
    WorkKey,
)
from dr_platform.staging.handoff import is_pipeline_wrapped
from dr_platform.staging.submission import WorkInput

from whetstone.orchestration import (
    ORCHESTRATION_PIPELINE_KEY,
    ORCHESTRATION_PIPELINE_VERSION,
    ROLLOUT_EXECUTION_STAGE_KEY,
    encode_work_request_ref,
    orchestration_pipeline,
    orchestration_pipeline_identity,
    quota_labels_for,
    rollout_work_input,
    work_key_for_execution_key,
)

from .support import execution_key, quota, work_request


def _stage_body(_input_ref: str) -> str:
    return "objref://x/?content_hash=" + "a" * 64


def test_pipeline_is_linear_with_one_rollout_execution_stage() -> None:
    pipeline = orchestration_pipeline(_stage_body, wrap=False)
    assert pipeline.key == PipelineKey(ORCHESTRATION_PIPELINE_KEY)
    assert pipeline.version == ORCHESTRATION_PIPELINE_VERSION
    assert len(pipeline.stages) == 1
    assert pipeline.stages[0].key == StageKey(ROLLOUT_EXECUTION_STAGE_KEY)


def test_wrapped_pipeline_registers_and_is_dbos_wrapped() -> None:
    pipeline = orchestration_pipeline(_stage_body)
    assert is_pipeline_wrapped(pipeline)
    registry = PipelineRegistry()
    registry.register(pipeline)
    resolved = registry.get(
        key=PipelineKey(ORCHESTRATION_PIPELINE_KEY),
        version=ORCHESTRATION_PIPELINE_VERSION,
    )
    assert resolved.identity == orchestration_pipeline_identity()


def test_identity_is_a_pipeline_key_version_pair() -> None:
    key, version = orchestration_pipeline_identity()
    assert isinstance(key, PipelineKey)
    assert version == ORCHESTRATION_PIPELINE_VERSION


def test_work_item_maps_execution_key_to_native_work_key() -> None:
    key = execution_key()
    work_input = rollout_work_input(
        execution_key=key,
        input_ref=encode_work_request_ref(work_request(key=key)),
        quotas=(quota(),),
    )
    assert isinstance(work_input, WorkInput)
    # Rollout Work Item -> one Rollout Execution Key (one-to-one via WorkKey).
    assert work_input.work_key == work_key_for_execution_key(key)
    assert isinstance(work_input.work_key, WorkKey)


def test_work_item_carries_opaque_reference_and_quota_labels() -> None:
    key = execution_key()
    request = work_request(key=key)
    input_ref = encode_work_request_ref(request)
    route = quota()
    work_input = rollout_work_input(
        execution_key=key,
        input_ref=input_ref,
        quotas=(route,),
    )
    # The input_ref is the opaque typed reference verbatim (never parsed).
    assert work_input.input_ref == input_ref
    # The labels are exactly the collision-free quota labels.
    assert dict(work_input.labels) == quota_labels_for([route])


def test_distinct_execution_keys_derive_distinct_work_keys() -> None:
    a = work_key_for_execution_key(execution_key(task_identity="task-a"))
    b = work_key_for_execution_key(execution_key(task_identity="task-b"))
    assert a != b


def test_multi_route_work_item_carries_every_label() -> None:
    key = execution_key()
    encoder = quota(model="encoder")
    decoder = quota(model="decoder")
    work_input = rollout_work_input(
        execution_key=key,
        input_ref=encode_work_request_ref(work_request(key=key)),
        quotas=(encoder, decoder),
    )
    assert len(work_input.labels) == 2
