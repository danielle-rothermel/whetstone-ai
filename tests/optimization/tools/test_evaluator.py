from __future__ import annotations

import pytest
from dr_store import (
    ObjectStore,
    SqliteBackend,
)

from tests.evaluation.support import (
    _engine,
)
from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.identity import (
    TypedRef,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.engine import (
    EvaluationEngine,
)
from whetstone.evaluation.schema import (
    EvaluationEvidence,
)
from whetstone.optimization.tools.contracts import (
    RefusalClass,
    ToolCall,
    ToolCapacity,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    tool_capacity_binding,
    tool_config_reference,
    tool_definition_reference,
)
from whetstone.optimization.tools.evaluator import EngineToolEvaluator
from whetstone.optimization.tools.execution import (
    EvaluatingToolExecutor,
    ToolValidationError,
)
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)


def _tool_config(
    engine: EvaluationEngine,
    *,
    store_namespace_key: str,
    output_fields: tuple[str, ...] = (
        "evaluation_evidence_ref",
        "output_artifact_ref",
    ),
    input_fields: tuple[str, ...] = ("base_ref", "model_route", "template"),
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=input_fields,
        output_fields=output_fields,
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="evaluate_candidate",
        eval_config=engine.sampling.eval_config,
        reward_policy_hash=engine.experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(
            max_accepted_calls=1,
            scope=ToolCapacityScope.GLOBAL,
        ),
        store_namespace_key=store_namespace_key,
    )


def _tool_call(
    engine: EvaluationEngine,
    config: ToolConfig,
    *,
    call_id: str,
    model_route: str = "openai/test",
    task_ids: list[str] | None = None,
) -> ToolCall:
    base = engine.experiment.initial_candidate
    args: dict[str, object] = {
        "base_ref": base.base_ref.model_dump(mode="json"),
        "model_route": model_route,
        "template": base.payload["user_prompt_template"],
    }
    if task_ids is not None:
        args["task_ids"] = task_ids
    return ToolCall(
        call_id=call_id,
        tool_config=tool_config_reference(config),
        capacity_binding=tool_capacity_binding(ToolCapacityScope.GLOBAL),
        args=args,
    )


@pytest.mark.process_integration
def test_tool_projection_uses_same_engine_evidence(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "tool.sqlite"))
    engine = _engine(tmp_path, store=store)
    config = _tool_config(engine, store_namespace_key="tool-projection")
    call = _tool_call(engine, config, call_id="tool-call")

    projected = EngineToolEvaluator(engine).evaluate(call, config)

    assert projected.eval_config_hash == engine.eval_config_ref.identity_hash
    assert len(projected.rollout_refs) == 1
    assert projected.output["evaluation_evidence_ref"] == (
        projected.rollout_refs[0].model_dump(mode="json")
    )
    artifact = TypedRef.model_validate(projected.output["output_artifact_ref"])
    assert store.get(artifact.reference)


def _subset_tool_config(
    engine: EvaluationEngine, *, store_namespace_key: str
) -> ToolConfig:
    return _tool_config(
        engine,
        store_namespace_key=store_namespace_key,
        input_fields=("base_ref", "model_route", "template", "task_ids"),
    )


def test_tool_projection_rejects_malformed_task_subsets(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "tool-malformed.sqlite"))
    engine = _engine(tmp_path, store=store)
    config = _subset_tool_config(engine, store_namespace_key="tool-malformed")

    mismatched = _tool_call(
        engine,
        config,
        call_id="wrong-task",
        task_ids=["not-the-bound-task"],
    )
    with pytest.raises(ToolValidationError, match="unknown task IDs"):
        EngineToolEvaluator(engine).evaluate(mismatched, config)
    duplicate = _tool_call(
        engine,
        config,
        call_id="duplicate-task",
        task_ids=[
            engine.sampling.task_set.task_identities[0],
            engine.sampling.task_set.task_identities[0],
        ],
    )
    with pytest.raises(ToolValidationError, match="must be unique"):
        EngineToolEvaluator(engine).evaluate(duplicate, config)
    empty = _tool_call(engine, config, call_id="empty-task", task_ids=[])
    with pytest.raises(ToolValidationError, match="at least one task"):
        EngineToolEvaluator(engine).evaluate(empty, config)


@pytest.mark.process_integration
def test_tool_projection_accepts_a_validated_task_subset(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "tool-subset.sqlite"))
    engine = _engine(tmp_path, store=store)
    config = _subset_tool_config(engine, store_namespace_key="tool-subset")
    bound_task = engine.sampling.task_set.task_identities[0]
    call = _tool_call(
        engine,
        config,
        call_id="subset-call",
        task_ids=[bound_task],
    )

    assert not isinstance(call.args["task_ids"], list)

    projected = EngineToolEvaluator(engine).evaluate(call, config)

    assert projected.eval_config_hash == engine.eval_config_ref.identity_hash
    assert len(projected.rollout_refs) == 1


@pytest.mark.process_integration
def test_engine_tool_evaluator_drives_a_call_through_the_executor(
    tmp_path,
) -> None:
    database = tmp_path / "tool-executor.sqlite"
    store = ObjectStore(SqliteBackend(database))
    engine = _engine(tmp_path, store=store)
    config = _tool_config(engine, store_namespace_key="tool-executor")
    effect_authority = EffectAuthority.memory()
    call_store = ToolCallStore(
        ObjectStore(SqliteBackend(database)),
        ToolAdmissionAuthority.sqlite(database),
        effect_authority,
    )
    handle = EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        engine.experiment.reward_policy,
        effect_authority,
        owner_id="tool-executor-owner",
        replay_policy=ReplayPolicy.IDEMPOTENT,
    ).runtime_handle(
        config,
        call_store,
        tool_capacity_binding(ToolCapacityScope.GLOBAL),
    )

    result = handle(_tool_call(engine, config, call_id="executed-call"))

    assert result.refusal is None
    assert result.terminal_failure is None
    assert result.output is not None
    assert result.reward is not None
    assert len(result.evaluation_evidence_refs) == 1
    evidence = EvaluationEvidence.model_validate(
        store.get(result.evaluation_evidence_refs[0].reference)
    )
    assert evidence.evaluation_binding.eval_config == engine.eval_config_ref
    assert evidence.evaluation_binding.role is EvaluationRole.INTERNAL


def test_engine_tool_evaluator_refuses_a_foreign_route_before_admission(
    tmp_path,
) -> None:
    database = tmp_path / "tool-refusal.sqlite"
    store = ObjectStore(SqliteBackend(database))
    engine = _engine(tmp_path, store=store)
    config = _tool_config(engine, store_namespace_key="tool-refusal")
    effect_authority = EffectAuthority.memory()
    call_store = ToolCallStore(
        ObjectStore(SqliteBackend(database)),
        ToolAdmissionAuthority.sqlite(database),
        effect_authority,
    )
    capacity_binding = tool_capacity_binding(ToolCapacityScope.GLOBAL)
    handle = EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        engine.experiment.reward_policy,
        effect_authority,
        owner_id="tool-refusal-owner",
        replay_policy=ReplayPolicy.IDEMPOTENT,
    ).runtime_handle(config, call_store, capacity_binding)

    result = handle(
        _tool_call(
            engine,
            config,
            call_id="foreign-route-call",
            model_route="openai/other",
        )
    )

    assert result.refusal is not None
    assert result.refusal.refusal_class is RefusalClass.VALIDATION
    assert call_store.accepted_count(config, capacity_binding) == 0
