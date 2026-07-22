"""Shared builders for Rollout Result and Result Store tests.

Constructs a real dr-graph ``GraphRunResult`` nested inside a real
:class:`~whetstone.result.RolloutResult`, wired through the real
Whetstone Rollout Execution Key identities, so the persistence and binding
proofs run against the released dependency contracts rather than stand-ins.
"""

from __future__ import annotations

from typing import Any

from dr_graph import GraphRunResult, GraphRunStatus, NodeOutcome, NodeOutput

from whetstone.graph.rollout import (
    EvaluationContext,
    EvaluationRole,
    RolloutExecutionKey,
    RolloutKey,
    rollout_execution_key,
)
from whetstone.result.rollout_result import (
    ExhaustedCausalFailure,
    PlatformStageAttemptEvidence,
    ProviderCallAttemptObservation,
    RolloutResult,
    ScoreFact,
)


def full_hash(char: str) -> str:
    """A syntactically valid full 64-char lowercase hex hash."""
    return char * 64


GRAPH_HASH = full_hash("a")
EVAL_CONFIG_HASH = full_hash("b")


def evaluation_context(
    *,
    campaign: str = "camp-1",
    role: EvaluationRole = EvaluationRole.INTERNAL,
    authority: str | None = None,
    provenance_ordinal: int | None = None,
    eval_config_hash: str = EVAL_CONFIG_HASH,
) -> EvaluationContext:
    return EvaluationContext(
        eval_config_hash=eval_config_hash,
        role=role,
        authority=authority,
        campaign=campaign,
        provenance_ordinal=provenance_ordinal,
    )


def rollout_key(
    *,
    task_identity: str = "task-1",
    repeat_id: str = "r0",
    graph_hash: str = GRAPH_HASH,
    eval_config_hash: str = EVAL_CONFIG_HASH,
) -> RolloutKey:
    return RolloutKey(
        graph_hash=graph_hash,
        eval_config_hash=eval_config_hash,
        task_identity=task_identity,
        repeat_id=repeat_id,
    )


def execution_key(
    *,
    task_identity: str = "task-1",
    repeat_id: str = "r0",
    context: EvaluationContext | None = None,
) -> RolloutExecutionKey:
    ctx = context or evaluation_context()
    return rollout_execution_key(
        rollout_key=rollout_key(
            task_identity=task_identity,
            repeat_id=repeat_id,
            eval_config_hash=ctx.eval_config_hash,
        ),
        context=ctx,
    )


def graph_run_result(
    *,
    graph_hash: str = GRAPH_HASH,
    terminal_output: dict[str, Any] | None = None,
    attempt_evidence_refs: tuple[str, ...] = (),
) -> GraphRunResult:
    output = terminal_output or {"score": 1}
    return GraphRunResult(
        graph_hash=graph_hash,
        external_inputs={"prompt": "task.prompt"},
        status=GraphRunStatus.SUCCESS,
        outcomes={
            "evaluate": NodeOutcome.success(
                node_id="evaluate",
                output=NodeOutput(values=output),
            ),
        },
        execution_order=("generate", "evaluate"),
        terminal_node_id="evaluate",
        terminal_output=output,
        attempt_evidence_refs=attempt_evidence_refs,
    )


def provider_attempt(
    *,
    evidence_ref: str = "attempt-1",
    logical_call_id: str = "call-1",
    attempt_number: int = 1,
) -> ProviderCallAttemptObservation:
    return ProviderCallAttemptObservation(
        evidence_ref=evidence_ref,
        logical_call_id=logical_call_id,
        attempt_number=attempt_number,
        semantic_classification="accepted",
        provider_invocation_evidence={"raw": {"body": "..."}},
    )


def success_rollout_result(
    *,
    key: RolloutExecutionKey | None = None,
    score: float = 1.0,
    with_attempt: bool = True,
    provenance_ordinal: int | None = None,
) -> RolloutResult:
    """A complete semantic-success Rollout Result with facts + scores."""
    exec_key = key or execution_key()
    attempts: tuple[ProviderCallAttemptObservation, ...] = ()
    attempt_refs: tuple[str, ...] = ()
    if with_attempt:
        attempts = (provider_attempt(),)
        attempt_refs = ("attempt-1",)
    return RolloutResult(
        rollout_execution_key=exec_key,
        graph_config_ref="graphcfg://a",
        graph_hash=exec_key.rollout_key.graph_hash,
        eval_config_ref="evalcfg://b",
        eval_config_hash=exec_key.rollout_key.eval_config_hash,
        evaluation_context_id=exec_key.evaluation_context_id,
        input_identities={"task_identity": exec_key.rollout_key.task_identity},
        graph_run_result=graph_run_result(
            graph_hash=exec_key.rollout_key.graph_hash,
            attempt_evidence_refs=attempt_refs,
        ),
        metric_facts=(
            {
                "name": "compressed_description_length",
                "value": 42,
                "unit": "bytes",
            },
        ),
        scores=(ScoreFact(name="reward", value=score),),
        provider_call_attempts=attempts,
        stage_attempt_evidence=PlatformStageAttemptEvidence(
            platform_stage_attempt_id="psa-1",
            dbos_workflow_id="wf-1",
            durability_replay_count=0,
        ),
        provenance_ordinal=provenance_ordinal,
    )


def failure_rollout_result(
    *,
    key: RolloutExecutionKey | None = None,
) -> RolloutResult:
    """A complete exhausted-causal-failure Rollout Result (no facts)."""
    exec_key = key or execution_key()
    return RolloutResult(
        rollout_execution_key=exec_key,
        graph_config_ref="graphcfg://a",
        graph_hash=exec_key.rollout_key.graph_hash,
        eval_config_ref="evalcfg://b",
        eval_config_hash=exec_key.rollout_key.eval_config_hash,
        evaluation_context_id=exec_key.evaluation_context_id,
        graph_run_result=graph_run_result(
            graph_hash=exec_key.rollout_key.graph_hash,
        ),
        exhausted_failure=ExhaustedCausalFailure(
            failure_class="rate_limited",
            failure_exception_type="whetstone.RateLimitedFailureError",
            underlying_exception_type="httpx.HTTPStatusError",
            message="exhausted bounded retries",
        ),
    )
