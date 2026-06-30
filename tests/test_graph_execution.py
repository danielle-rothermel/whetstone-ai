from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from dr_dspy.eval_failures import PermanentFailureError
from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphRunResult,
    GraphRunStatus,
    GraphSpec,
    InputResolutionError,
    NodeConfig,
    NodeError,
    NodeExecutionError,
    NodeOutcome,
    NodeOutcomeStatus,
    NodeOutput,
    NodeSpec,
    TerminalError,
    execute_graph,
    graph_digest,
    validate_task_bindings,
)


def _node(
    node_id: str,
    *,
    bindings: dict[str, str] | None = None,
    output_field: str = "output",
) -> NodeSpec:
    input_bindings = {
        name: BindingRef.model_validate(ref)
        for name, ref in (bindings or {}).items()
    }
    fields = [
        FieldSpec(name=name, role=FieldRole.INPUT)
        for name in input_bindings
    ]
    fields.append(FieldSpec(name=output_field, role=FieldRole.OUTPUT))
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            fields=tuple(fields),
            input_bindings=input_bindings,
            output_field=output_field,
        ),
    )


def _output(value: Any, *, field: str = "output") -> NodeOutput:
    return NodeOutput(values={field: value})


def _graph(
    *nodes: NodeSpec,
    terminal_node_id: str,
) -> GraphSpec:
    return GraphSpec(
        nodes=nodes,
        terminal_node_id=terminal_node_id,
    )


def test_direct_one_node_graph_success() -> None:
    graph = _graph(
        _node("direct", bindings={"prompt": "task.prompt"}),
        terminal_node_id="direct",
    )

    result = execute_graph(
        graph=graph,
        inputs={"prompt": "write add"},
        run_node=lambda node, inputs: _output(f"code for {inputs['prompt']}"),
    )

    assert result.status is GraphRunStatus.SUCCESS
    assert result.execution_order == ("direct",)
    assert result.terminal_output == "code for write add"
    assert result.outcomes["direct"].status is NodeOutcomeStatus.SUCCESS


def test_two_node_graph_binds_upstream_output_into_downstream_input() -> None:
    encoder = _node(
        "encoder",
        bindings={"prompt": "task.prompt"},
        output_field="description",
    )
    decoder = _node(
        "decoder",
        bindings={"description": "encoder.description"},
        output_field="code",
    )
    graph = _graph(
        decoder,
        encoder,
        terminal_node_id="decoder",
    )
    seen_inputs: dict[str, Mapping[str, Any]] = {}

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        seen_inputs[node.id] = dict(inputs)
        if node.id == "encoder":
            return _output("plain description", field="description")
        return _output(
            f"def f(): return {inputs['description']!r}",
            field="code",
        )

    result = execute_graph(
        graph=graph,
        inputs={"prompt": "write f"},
        run_node=run_node,
    )

    assert result.status is GraphRunStatus.SUCCESS
    assert result.execution_order == ("encoder", "decoder")
    assert seen_inputs["decoder"] == {"description": "plain description"}
    assert result.terminal_output == "def f(): return 'plain description'"


def test_topological_order_is_deterministic_for_independent_nodes() -> None:
    graph = _graph(
        _node("zeta"),
        _node("alpha"),
        _node("middle"),
        terminal_node_id="middle",
    )

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: _output(node.id),
    )

    assert result.execution_order == ("alpha", "middle", "zeta")
    assert result.terminal_output == "middle"


def test_duplicate_node_id_validation() -> None:
    with pytest.raises(ValueError, match="duplicate node ids"):
        GraphSpec(
            nodes=(_node("same"), _node("same")),
            terminal_node_id="same",
        )


def test_unknown_dependency_validation() -> None:
    with pytest.raises(ValueError, match="points at unknown node"):
        GraphSpec(
            nodes=(_node("decoder", bindings={"description": "encoder"}),),
            terminal_node_id="decoder",
        )


def test_cycle_detection() -> None:
    with pytest.raises(ValueError, match="graph has a cycle"):
        GraphSpec(
            nodes=(
                _node("a", bindings={"value": "b"}),
                _node("b", bindings={"value": "a"}),
            ),
            terminal_node_id="a",
        )


def test_empty_graph_validation() -> None:
    with pytest.raises(ValueError, match="graph must have at least one node"):
        GraphSpec(nodes=(), terminal_node_id="missing")


def test_missing_terminal_node_validation() -> None:
    with pytest.raises(
        ValueError,
        match="terminal_node_id 'missing' not in graph",
    ):
        GraphSpec(nodes=(_node("direct"),), terminal_node_id="missing")


def test_node_config_requires_declared_fields() -> None:
    with pytest.raises(
        ValueError,
        match="node config must declare at least one field",
    ):
        NodeConfig(fields=(), output_field="output")


def test_node_config_rejects_duplicate_field_names() -> None:
    with pytest.raises(ValueError, match="duplicate field names"):
        NodeConfig(
            fields=(
                FieldSpec(name="output", role=FieldRole.OUTPUT),
                FieldSpec(name="output", role=FieldRole.OUTPUT),
            ),
            output_field="output",
        )


def test_node_config_rejects_unknown_output_field() -> None:
    with pytest.raises(
        ValueError,
        match="output_field 'missing' is not an output field",
    ):
        NodeConfig(
            fields=(FieldSpec(name="output", role=FieldRole.OUTPUT),),
            output_field="missing",
        )


def test_node_config_rejects_binding_to_undeclared_input_field() -> None:
    with pytest.raises(
        ValueError,
        match="input binding 'prompt' is not an input field",
    ):
        NodeConfig(
            fields=(FieldSpec(name="output", role=FieldRole.OUTPUT),),
            input_bindings={
                "prompt": BindingRef.model_validate("task.prompt")
            },
            output_field="output",
        )


def test_unknown_task_binding_field_validation() -> None:
    graph = _graph(
        _node("direct", bindings={"prompt": "task.promt"}),
        terminal_node_id="direct",
    )
    with pytest.raises(
        ValueError,
        match="task binding field\\(s\\) 'promt' not in allowed task fields",
    ):
        validate_task_bindings(graph, allowed_task_fields=("prompt",))


def test_graph_with_task_bindings_does_not_require_task_fields() -> None:
    graph = _graph(
        _node("direct", bindings={"prompt": "task.prompt"}),
        terminal_node_id="direct",
    )
    assert graph.model_dump(mode="json")["nodes"]
    validate_task_bindings(graph, allowed_task_fields=("prompt",))


def test_missing_task_input_becomes_error_outcome() -> None:
    graph = _graph(
        _node("direct", bindings={"prompt": "task.prompt"}),
        terminal_node_id="direct",
    )

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: _output("unreachable"),
    )

    outcome = result.outcomes["direct"]
    assert result.status is GraphRunStatus.ERROR
    assert outcome.status is NodeOutcomeStatus.ERROR
    assert outcome.error is not None
    assert outcome.error.error_type == (
        f"{InputResolutionError.__module__}."
        f"{InputResolutionError.__qualname__}"
    )
    assert result.terminal_error is not None
    assert result.terminal_error.error == outcome.error


def test_missing_returned_output_field_becomes_node_execution_error() -> None:
    graph = _graph(
        _node("direct", output_field="code"),
        terminal_node_id="direct",
    )

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: _output("wrong", field="text"),
    )

    outcome = result.outcomes["direct"]
    assert result.status is GraphRunStatus.ERROR
    assert outcome.status is NodeOutcomeStatus.ERROR
    assert outcome.error is not None
    assert outcome.error.error_type == (
        f"{NodeExecutionError.__module__}."
        f"{NodeExecutionError.__qualname__}"
    )


def test_missing_named_upstream_output_field_validation() -> None:
    with pytest.raises(ValueError, match="points at unknown field 'other'"):
        GraphSpec(
            nodes=(
                _node("encoder", output_field="description"),
                _node("decoder", bindings={"description": "encoder.other"}),
            ),
            terminal_node_id="decoder",
        )


def test_binding_ref_rejects_empty_node_field() -> None:
    with pytest.raises(ValueError, match="non-empty field"):
        BindingRef.model_validate("encoder.")


def test_graph_digest_rejects_invalid_length() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")
    with pytest.raises(ValueError, match="digest length must be"):
        graph_digest(graph, length=0)
    with pytest.raises(ValueError, match="digest length must be"):
        graph_digest(graph, length=-1)
    with pytest.raises(ValueError, match="digest length must be"):
        graph_digest(graph, length=65)


def test_node_exception_captures_persistable_error() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")
    error = PermanentFailureError(
        "provider rejected request",
        metadata={"provider": "test"},
    )

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        raise error

    result = execute_graph(graph=graph, inputs={}, run_node=run_node)

    outcome = result.outcomes["direct"]
    dumped = result.model_dump(mode="json")
    assert result.status is GraphRunStatus.ERROR
    assert outcome.status is NodeOutcomeStatus.ERROR
    assert outcome.error is not None
    assert outcome.error.failure_class == "permanent"
    assert outcome.error.metadata == {"provider": "test"}
    assert "exception" not in dumped["outcomes"]["direct"]


def test_node_error_preserves_wrapped_step_failure_diagnostics() -> None:
    class StepFailure(Exception):
        def __init__(self) -> None:
            super().__init__("provider failed")
            self.error_type = "builtins.RuntimeError"
            self.failure_class = "permanent"
            self.metadata = {"provider": "test"}

    graph = _graph(_node("direct"), terminal_node_id="direct")

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        raise StepFailure()

    result = execute_graph(graph=graph, inputs={}, run_node=run_node)

    outcome = result.outcomes["direct"]
    assert outcome.error is not None
    assert outcome.error.error_type == "builtins.RuntimeError"
    assert outcome.error.failure_class == "permanent"
    assert outcome.error.metadata == {"provider": "test"}
    assert result.terminal_error is not None
    assert result.terminal_error.error == outcome.error


def test_node_error_preserves_underlying_exception_type() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")
    error = PermanentFailureError(
        "classified failure",
        underlying=ValueError("bad payload"),
        metadata={"stage": "parse"},
    )

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        raise error

    result = execute_graph(graph=graph, inputs={}, run_node=run_node)

    outcome = result.outcomes["direct"]
    assert outcome.error is not None
    assert outcome.error.metadata == {
        "stage": "parse",
        "underlying_exception_type": "builtins.ValueError",
    }


def test_node_error_preserves_chained_underlying_exception_type() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")
    error = PermanentFailureError(
        "outer",
        underlying=PermanentFailureError(
            "middle",
            underlying=ValueError("inner"),
        ),
        metadata={"stage": "parse"},
    )

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        raise error

    result = execute_graph(graph=graph, inputs={}, run_node=run_node)

    outcome = result.outcomes["direct"]
    assert outcome.error is not None
    assert outcome.error.metadata["underlying_exception_type"] == (
        "builtins.ValueError"
    )


def test_independent_nodes_continue_after_unrelated_failure() -> None:
    graph = _graph(
        _node("terminal"),
        _node("bad"),
        terminal_node_id="terminal",
    )

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        if node.id == "bad":
            raise RuntimeError("boom")
        return _output("ok")

    result = execute_graph(graph=graph, inputs={}, run_node=run_node)

    assert result.status is GraphRunStatus.PARTIAL
    assert result.terminal_output == "ok"
    assert result.outcomes["bad"].status is NodeOutcomeStatus.ERROR
    assert result.outcomes["terminal"].status is NodeOutcomeStatus.SUCCESS


def test_downstream_nodes_are_blocked_when_dependency_errors() -> None:
    graph = _graph(
        _node("encoder", bindings={"prompt": "task.prompt"}),
        _node("decoder", bindings={"description": "encoder"}),
        terminal_node_id="decoder",
    )

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        if node.id == "encoder":
            raise RuntimeError("encoder failed")
        return _output("unreachable")

    result = execute_graph(
        graph=graph,
        inputs={"prompt": "write f"},
        run_node=run_node,
    )

    assert result.status is GraphRunStatus.BLOCKED
    assert result.outcomes["encoder"].status is NodeOutcomeStatus.ERROR
    assert result.outcomes["decoder"].status is NodeOutcomeStatus.BLOCKED
    assert result.outcomes["decoder"].blocked_by == ("encoder",)
    assert result.terminal_error is not None
    assert result.terminal_error.status is NodeOutcomeStatus.BLOCKED
    assert result.terminal_error.blocked_by == ("encoder",)


def test_blocked_nodes_do_not_invoke_run_node() -> None:
    graph = _graph(
        _node("encoder", bindings={"prompt": "task.prompt"}),
        _node("decoder", bindings={"description": "encoder"}),
        terminal_node_id="decoder",
    )
    invoked: list[str] = []

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        invoked.append(node.id)
        if node.id == "encoder":
            raise RuntimeError("encoder failed")
        return _output("unreachable")

    result = execute_graph(
        graph=graph,
        inputs={"prompt": "write f"},
        run_node=run_node,
    )

    assert invoked == ["encoder"]
    assert result.status is GraphRunStatus.BLOCKED
    assert result.outcomes["decoder"].status is NodeOutcomeStatus.BLOCKED


def test_blocked_node_lists_all_failed_dependencies() -> None:
    graph = _graph(
        _node("terminal", bindings={"left": "a", "right": "b"}),
        _node("b"),
        _node("a"),
        terminal_node_id="terminal",
    )

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        if node.id in {"a", "b"}:
            raise RuntimeError(f"{node.id} errored")
        return _output("unreachable")

    result = execute_graph(graph=graph, inputs={}, run_node=run_node)

    assert result.status is GraphRunStatus.BLOCKED
    assert result.outcomes["terminal"].status is NodeOutcomeStatus.BLOCKED
    assert result.outcomes["terminal"].blocked_by == ("a", "b")


def test_default_node_ref_uses_upstream_configured_output_field() -> None:
    graph = _graph(
        _node("encoder", output_field="description"),
        _node("decoder", bindings={"description": "encoder"}),
        terminal_node_id="decoder",
    )

    def run_node(node: NodeSpec, inputs: Mapping[str, Any]) -> NodeOutput:
        if node.id == "encoder":
            return _output("summary", field="description")
        return _output(inputs["description"])

    result = execute_graph(graph=graph, inputs={}, run_node=run_node)

    assert result.terminal_output == "summary"


def test_result_json_dump_is_persistable_shape() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: _output("ok", field="output"),
    )

    assert result.model_dump(mode="json") == {
        "status": "success",
        "outcomes": {
            "direct": {
                "node_id": "direct",
                "status": "success",
                "output": {"values": {"output": "ok"}, "metadata": {}},
                "error": None,
                "blocked_by": [],
            }
        },
        "execution_order": ["direct"],
        "terminal_node_id": "direct",
        "terminal_output": "ok",
        "terminal_error": None,
    }


def test_graph_digest_is_stable_for_equivalent_graph_specs() -> None:
    graph = _graph(
        _node("direct", bindings={"prompt": "task.prompt"}),
        terminal_node_id="direct",
    )
    same_graph = GraphSpec.model_validate(graph.model_dump(mode="json"))

    assert graph_digest(graph) == graph_digest(same_graph)


def test_graph_digest_does_not_depend_on_allowed_task_fields() -> None:
    graph = _graph(
        _node("direct", bindings={"prompt": "task.prompt"}),
        terminal_node_id="direct",
    )
    digest = graph_digest(graph)
    validate_task_bindings(graph, allowed_task_fields=("prompt",))
    validate_task_bindings(
        graph,
        allowed_task_fields=("prompt", "task_id", "entry_point"),
    )
    assert graph_digest(graph) == digest


def test_graph_digest_changes_with_node_declaration_order() -> None:
    first = _graph(_node("a"), _node("b"), terminal_node_id="a")
    second = _graph(_node("b"), _node("a"), terminal_node_id="a")

    assert first.topological_order() == tuple(
        sorted(first.nodes, key=lambda n: n.id)
    )
    assert graph_digest(first) != graph_digest(second)


def test_run_node_dict_return_is_coerced_to_node_output() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: {
            "values": {"output": "ok"},
            "metadata": {},
        },
    )

    assert result.status is GraphRunStatus.SUCCESS
    assert result.terminal_output == "ok"


def test_invalid_run_node_return_shape_becomes_error_outcome() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: {"values": "not a dict"},
    )

    outcome = result.outcomes["direct"]
    assert result.status is GraphRunStatus.ERROR
    assert outcome.status is NodeOutcomeStatus.ERROR
    assert outcome.error is not None
    assert "ValidationError" in outcome.error.error_type


def test_error_outcome_json_dump_is_persistable_shape() -> None:
    graph = _graph(_node("direct"), terminal_node_id="direct")

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )

    outcome = result.outcomes["direct"]
    assert outcome.error is not None
    assert result.model_dump(mode="json") == {
        "status": "error",
        "outcomes": {
            "direct": {
                "node_id": "direct",
                "status": "error",
                "output": None,
                "error": {
                    "error_type": (
                        f"{RuntimeError.__module__}."
                        f"{RuntimeError.__qualname__}"
                    ),
                    "message": "boom",
                    "failure_class": None,
                    "metadata": {},
                },
                "blocked_by": [],
            }
        },
        "execution_order": ["direct"],
        "terminal_node_id": "direct",
        "terminal_output": None,
        "terminal_error": {
            "node_id": "direct",
            "status": "error",
            "error": {
                "error_type": (
                    f"{RuntimeError.__module__}."
                    f"{RuntimeError.__qualname__}"
                ),
                "message": "boom",
                "failure_class": None,
                "metadata": {},
            },
            "blocked_by": [],
        },
    }


def test_blocked_outcome_json_dump_is_persistable_shape() -> None:
    graph = _graph(
        _node("encoder", bindings={"prompt": "task.prompt"}),
        _node("decoder", bindings={"description": "encoder"}),
        terminal_node_id="decoder",
    )

    result = execute_graph(
        graph=graph,
        inputs={"prompt": "write f"},
        run_node=lambda node, inputs: (_ for _ in ()).throw(
            RuntimeError("encoder failed")
        ),
    )

    assert result.model_dump(mode="json") == {
        "status": "blocked",
        "outcomes": {
            "encoder": {
                "node_id": "encoder",
                "status": "error",
                "output": None,
                "error": {
                    "error_type": (
                        f"{RuntimeError.__module__}."
                        f"{RuntimeError.__qualname__}"
                    ),
                    "message": "encoder failed",
                    "failure_class": None,
                    "metadata": {},
                },
                "blocked_by": [],
            },
            "decoder": {
                "node_id": "decoder",
                "status": "blocked",
                "output": None,
                "error": None,
                "blocked_by": ["encoder"],
            },
        },
        "execution_order": ["encoder", "decoder"],
        "terminal_node_id": "decoder",
        "terminal_output": None,
        "terminal_error": {
            "node_id": "decoder",
            "status": "blocked",
            "error": None,
            "blocked_by": ["encoder"],
        },
    }


def test_partial_outcome_json_dump_is_persistable_shape() -> None:
    graph = _graph(
        _node("terminal"),
        _node("bad"),
        terminal_node_id="terminal",
    )

    result = execute_graph(
        graph=graph,
        inputs={},
        run_node=lambda node, inputs: (
            _output("ok")
            if node.id == "terminal"
            else (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )

    assert result.model_dump(mode="json") == {
        "status": "partial",
        "outcomes": {
            "bad": {
                "node_id": "bad",
                "status": "error",
                "output": None,
                "error": {
                    "error_type": (
                        f"{RuntimeError.__module__}."
                        f"{RuntimeError.__qualname__}"
                    ),
                    "message": "boom",
                    "failure_class": None,
                    "metadata": {},
                },
                "blocked_by": [],
            },
            "terminal": {
                "node_id": "terminal",
                "status": "success",
                "output": {"values": {"output": "ok"}, "metadata": {}},
                "error": None,
                "blocked_by": [],
            },
        },
        "execution_order": ["bad", "terminal"],
        "terminal_node_id": "terminal",
        "terminal_output": "ok",
        "terminal_error": None,
    }


def test_binding_ref_round_trips_through_graph_spec_json_dump() -> None:
    graph = _graph(
        _node(
            "encoder",
            bindings={"prompt": "task.prompt"},
            output_field="description",
        ),
        _node(
            "decoder",
            bindings={"description": "encoder.description"},
            output_field="code",
        ),
        terminal_node_id="decoder",
    )

    payload = graph.model_dump(mode="json")
    assert payload["nodes"][0]["config"]["input_bindings"]["prompt"] == (
        "task.prompt"
    )
    assert payload["nodes"][1]["config"]["input_bindings"]["description"] == (
        "encoder.description"
    )

    round_tripped = GraphSpec.model_validate(payload)
    assert round_tripped == graph


def test_node_id_rejects_ref_grammar_tokens() -> None:
    with pytest.raises(ValueError, match=r"cannot contain '\.'"):
        _node("a.b")
    with pytest.raises(ValueError, match=r"'task' is reserved"):
        _node("task")


def test_binding_ref_rejects_reserved_node_id() -> None:
    with pytest.raises(ValueError, match=r"'task' is reserved"):
        BindingRef.model_validate("task")


def test_terminal_error_rejects_success_status() -> None:
    with pytest.raises(ValueError, match="must be error or blocked"):
        TerminalError(
            node_id="direct",
            status=NodeOutcomeStatus.SUCCESS,
        )


def test_graph_run_result_rejects_mismatched_outcome_keys() -> None:
    outcome = NodeOutcome.success(
        node_id="direct",
        output=_output("ok"),
    )
    with pytest.raises(ValueError, match="does not match node_id"):
        GraphRunResult(
            status=GraphRunStatus.SUCCESS,
            outcomes={"other": outcome},
            execution_order=("direct",),
            terminal_node_id="direct",
            terminal_output="ok",
        )


def test_graph_run_result_rejects_conflicting_terminal_fields() -> None:
    with pytest.raises(
        ValueError,
        match="both terminal_output and terminal_error",
    ):
        GraphRunResult(
            status=GraphRunStatus.ERROR,
            outcomes={},
            execution_order=(),
            terminal_node_id="direct",
            terminal_output="ok",
            terminal_error=TerminalError(
                node_id="direct",
                status=NodeOutcomeStatus.ERROR,
                error=NodeError(error_type="test", message="failed"),
            ),
        )
