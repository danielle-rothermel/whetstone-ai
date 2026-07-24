"""The concrete versioned Whetstone Orchestration Pipeline.

An Orchestration Pipeline is a concrete versioned *native linear Pipeline
Definition* registered in dr-platform and used to execute Whetstone work — not
a Rollout Definition and not a Graph (``design/vocab_and_defs.html`` ·
*Orchestration Pipeline*). This module declares that pipeline and the
Whetstone→platform key mapping.

The pipeline is **linear with one rollout-execution Stage to start**
(:data:`ROLLOUT_EXECUTION_STAGE_KEY`). dr-platform's generic
``PipelineDefinition`` / ``PipelineRegistry`` own version identity and
registration; this module supplies the concrete stage chain and the
domain-neutral ``args_for`` boundary.

Key mapping (Whetstone namespace → native platform key):

* **Evaluation Campaign → Platform Campaign** — the Whetstone Evaluation
  Campaign namespace is one native ``CampaignKey``.
* **Submission Batch → Submission Run** — one immutable Submission Batch is
  one native ``RunKey`` (a Platform Submission Run).
* **Rollout Work Item → one Rollout Execution Key** — one Work Item maps
  one-to-one to one Rollout Execution Key, carried as the native ``WorkKey``.

The Rollout Work Request (execution key, Graph Config + Evaluation Context
refs, task inputs, repeat data, expected schema identities; **no**
Materialization Record ref) rides opaquely: it is persisted through dr-store
and its typed Object Reference is encoded as a scheme-tagged string in
``WorkInput.input_ref``. The stage returns the terminal Rollout Result's typed
reference as an equally-opaque ``output_reference``. dr-platform validates both
only as non-empty strings and never parses them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dr_platform.staging import (
    PipelineDefinition,
    PipelineKey,
    StageDefinition,
    StageKey,
    WorkKey,
)
from dr_platform.staging.handoff import wrap_pipeline_workflows
from dr_platform.staging.submission import WorkInput

from whetstone.orchestration.labels import quota_labels_for
from whetstone.result import encode_rollout_execution_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dr_platform.staging.admission import AdmissionPayload
    from dr_providers import ProviderQuotaIdentity

    from whetstone.graph.rollout import RolloutExecutionKey

__all__ = [
    "ORCHESTRATION_PIPELINE_KEY",
    "ORCHESTRATION_PIPELINE_VERSION",
    "ROLLOUT_EXECUTION_STAGE_KEY",
    "ROLLOUT_EXECUTION_STAGE_QUEUE",
    "orchestration_pipeline",
    "orchestration_pipeline_identity",
    "rollout_work_input",
    "work_key_for_execution_key",
]

#: The Orchestration Pipeline key. One declared linear pipeline across all
#: its versions.
ORCHESTRATION_PIPELINE_KEY = "whetstone.rollout"

#: The current concrete pipeline version. Registering an immutable version that
#: differs replaces the superseded revision through the platform registry.
ORCHESTRATION_PIPELINE_VERSION = 1

#: The single rollout-execution Stage key (linear pipeline, one stage for now).
ROLLOUT_EXECUTION_STAGE_KEY = "rollout-execution"

#: The DBOS queue the rollout-execution stage is admitted onto.
ROLLOUT_EXECUTION_STAGE_QUEUE = "whetstone-rollout-execution"


def orchestration_pipeline_identity() -> tuple[PipelineKey, int]:
    """The ``(PipelineKey, version)`` identity of the pipeline."""
    return (
        PipelineKey(ORCHESTRATION_PIPELINE_KEY),
        ORCHESTRATION_PIPELINE_VERSION,
    )


def _rollout_execution_args(payload: AdmissionPayload) -> tuple[object, ...]:
    """Produce the stage workflow's positional arguments.

    Domain-neutral platform boundary: the only argument the stage body needs is
    the opaque ``input_ref`` string (the encoded Rollout Work Request Object
    Reference). The platform never interprets it; Whetstone's stage body
    resolves it back to the immutable Work Request.
    """
    return (payload.input_ref,)


def orchestration_pipeline(
    stage_callable: Callable[[str], str],
    *,
    wrap: bool = True,
) -> PipelineDefinition:
    """Build the concrete versioned Orchestration Pipeline.

    Args:
        stage_callable: the plain rollout-execution stage body — takes the
            opaque input-ref, returns the opaque output-ref. Typically
            ``ExecutorContext.stage_callable()``.
        wrap: when True (default) the stages are wrapped with dr-platform's
            DBOS handoff workflows so admission enqueues durable stage
            workflows and the completion transaction commits SUCCEEDED/FAILED.
            Set False to inspect the raw declaration.

    Returns:
        the registrable ``PipelineDefinition`` (wrapped unless ``wrap=False``).
    """
    key, version = orchestration_pipeline_identity()
    declared = PipelineDefinition(
        key=key,
        version=version,
        stages=(
            StageDefinition(
                key=StageKey(ROLLOUT_EXECUTION_STAGE_KEY),
                queue_name=ROLLOUT_EXECUTION_STAGE_QUEUE,
                workflow=stage_callable,
                args_for=_rollout_execution_args,
            ),
        ),
    )
    return wrap_pipeline_workflows(declared) if wrap else declared


def work_key_for_execution_key(key: RolloutExecutionKey) -> WorkKey:
    """The native ``WorkKey`` for one Rollout Execution Key.

    The Rollout Work Item maps one-to-one to one Rollout Execution Key, so its
    campaign-scoped work identity is derived deterministically from the key.
    The platform key alphabet is restricted, so the canonical key encoding is
    hashed to a stable, collision-free platform work key.
    """
    from dr_serialize import json_hash

    digest = json_hash(
        {"rollout_execution_key": encode_rollout_execution_key(key)},
        length=48,
    )
    return WorkKey(f"rxk-{digest}")


def rollout_work_input(
    *,
    execution_key: RolloutExecutionKey,
    input_ref: str,
    quotas: Iterable[ProviderQuotaIdentity],
) -> WorkInput:
    """Build one immutable ``WorkInput`` for a Rollout Work Item.

    The Work Item's native ``WorkKey`` is the Rollout Execution Key's derived
    key (one-to-one); ``input_ref`` is the opaque encoded Rollout Work Request
    reference; and the labels are the collision-free Provider Quota labels for
    **every** applicable route (multi-route work carries every label).
    """
    return WorkInput(
        work_key=work_key_for_execution_key(execution_key),
        input_ref=input_ref,
        labels=quota_labels_for(quotas),
    )
