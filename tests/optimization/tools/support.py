from __future__ import annotations

from dr_store import (
    MemoryBackend,
    ObjectStore,
    PutOutcome,
    SqliteBackend,
)

from tests.optimization.support import eval_config
from whetstone.core.effects.authority import (
    EffectAuthority,
)
from whetstone.core.identity import TypedRef, typed_ref_for_record
from whetstone.core.roles import EvaluationRole
from whetstone.experiment.reward import (
    Reward,
    RewardInputCitation,
    RewardPolicy,
)
from whetstone.optimization.tools.contracts import (
    RUN_CAPACITY_SUBJECT_SCHEMA,
    STEP_CAPACITY_SUBJECT_SCHEMA,
    ToolCall,
    ToolCapacity,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    ToolResult,
    tool_call_reference,
    tool_config_reference,
    tool_definition_reference,
)
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


def tool_config(
    *,
    capacity: int = 2,
    namespace: str = "tool-ns",
    scope: ToolCapacityScope = ToolCapacityScope.RUN,
    reward_policy_hash: str = FULL_B,
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("model_route", "template"),
        output_fields=("generation_refs", "accepted_ordinal"),
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="evaluate_candidate",
        eval_config=eval_config(FULL_A),
        reward_policy_hash=reward_policy_hash,
        capacity=ToolCapacity(
            max_accepted_calls=capacity,
            scope=scope,
        ),
        store_namespace_key=namespace,
    )


def tool_call(
    config: ToolConfig,
    call_id: str,
    *,
    template: str | None = None,
    scope_id: str = "run-1",
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_config=tool_config_reference(config),
        capacity_binding=capacity_binding(config.capacity.scope, scope_id),
        args={
            "model_route": "route",
            "template": template if template is not None else call_id,
        },
    )


def capacity_binding(
    scope: ToolCapacityScope,
    subject: str = "run-1",
) -> ToolCapacityBinding:
    if scope is ToolCapacityScope.GLOBAL:
        return ToolCapacityBinding(scope=scope)
    schema = (
        RUN_CAPACITY_SUBJECT_SCHEMA
        if scope is ToolCapacityScope.RUN
        else STEP_CAPACITY_SUBJECT_SCHEMA
    )
    return ToolCapacityBinding(
        scope=scope,
        subject_ref=typed_ref_for_record(schema, {"subject": subject}),
    )


def sqlite_store(
    database,
    *,
    effect_authority: EffectAuthority | None = None,
) -> ToolCallStore:
    return ToolCallStore(
        ObjectStore(SqliteBackend(database)),
        ToolAdmissionAuthority.sqlite(database),
        effect_authority or EffectAuthority.memory(),
    )


def successful_result(call: ToolCall, ordinal: int) -> ToolResult:
    return ToolResult(
        call=tool_call_reference(call),
        output={
            "generation_refs": [],
            "accepted_ordinal": ordinal,
        },
        provenance_ordinal=ordinal,
    )


def reward_record(policy: RewardPolicy) -> Reward:
    return Reward(
        reward_name="reward",
        value=1.0,
        reward_policy=policy,
        evidence_role=EvaluationRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="score",
                value=1.0,
                contributed=1.0,
            ),
        ),
        evidence_refs=(reward_evidence_ref(),),
    )


def reward_evidence_ref() -> TypedRef:
    return typed_ref_for_record(
        "whetstone.test.reward_evidence",
        {"evidence": "score"},
    )


class CollisionBackend(MemoryBackend):
    def put_object(
        self,
        *,
        schema: str,
        content_hash: str,
        canonical: str,
    ) -> PutOutcome:
        del content_hash, canonical
        return PutOutcome(
            inserted=False,
            stored_schema=schema,
            stored_canonical="{}",
        )
