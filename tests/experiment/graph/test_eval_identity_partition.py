from __future__ import annotations

import pytest
from dr_code.eval import EvalConfig
from dr_graph import GraphConfig, GraphDefinition, graph_hash

from tests.experiment.graph.support import (
    EVALUATION_PROCEDURE_CONFIG_SCHEMA,
    PROVIDER_CALL_CONFIG_SCHEMA,
    build_graph_config,
    eval_config,
    fake_hash,
    procedure_config,
)
from whetstone.experiment.graph.eval_identity import (
    EvalIdentityMismatchError,
    EvalNodeError,
    eval_config_hash,
    sole_eval_node_procedure_hash,
    validate_eval_identity_partition,
)
from whetstone.experiment.graph.nodes import (
    EVALUATION_PROCEDURE_CONFIG_VARIABLE,
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)


def _graph_for(ec: EvalConfig) -> GraphConfig:
    return build_graph_config(
        provider_call_config_hash=fake_hash("a"),
        evaluation_procedure_config_hash=(ec.evaluation_procedure_config_hash),
    )


def _graph_with_eval_nodes(count: int, *, procedure_hash: str) -> GraphConfig:
    llm = llm_call_node_definition("generate", prompt_source="task.prompt")
    eval_nodes = []
    upstream_node_id = "generate"
    for index in range(count):
        node_id = f"evaluate-{index}"
        eval_nodes.append(
            eval_node_definition(
                node_id,
                upstream_sources={"candidate": upstream_node_id},
            )
        )
        upstream_node_id = node_id
    definition = GraphDefinition(
        nodes=(llm, *eval_nodes),
        terminal_node_id=upstream_node_id,
    )
    assignments = {
        "generate": llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=fake_hash("a"),
        ),
        **{
            node.node_id: eval_variable_assignment(
                evaluation_procedure_config_schema=(
                    EVALUATION_PROCEDURE_CONFIG_SCHEMA
                ),
                evaluation_procedure_config_hash=procedure_hash,
            )
            for node in eval_nodes
        },
    }
    return definition.materialize(assignments)


def test_matching_procedure_identity_validates() -> None:
    ec = eval_config()
    graph = _graph_for(ec)
    validate_eval_identity_partition(graph, ec)
    assert sole_eval_node_procedure_hash(graph) == (
        ec.evaluation_procedure_config_hash
    )


def test_mismatched_procedure_identity_rejected() -> None:
    ec = eval_config()
    graph = build_graph_config(
        provider_call_config_hash=fake_hash("a"),
        evaluation_procedure_config_hash=fake_hash("f"),
    )
    with pytest.raises(EvalIdentityMismatchError):
        validate_eval_identity_partition(graph, ec)


@pytest.mark.parametrize("eval_node_count", [0, 2])
def test_eval_identity_partition_requires_exactly_one_eval_node(
    eval_node_count: int,
) -> None:
    ec = eval_config()
    graph = _graph_with_eval_nodes(
        eval_node_count,
        procedure_hash=ec.evaluation_procedure_config_hash,
    )

    with pytest.raises(
        EvalNodeError,
        match=rf"expected exactly one .* found {eval_node_count}",
    ):
        validate_eval_identity_partition(graph, ec)


def test_eval_identity_partition_propagates_missing_procedure_reference() -> (
    None
):
    ec = eval_config()
    payload = _graph_for(ec).model_dump(mode="json")
    payload["nodes"][-1]["variables"].pop(EVALUATION_PROCEDURE_CONFIG_VARIABLE)
    graph = GraphConfig.model_validate(payload)

    with pytest.raises(KeyError, match=EVALUATION_PROCEDURE_CONFIG_VARIABLE):
        validate_eval_identity_partition(graph, ec)


def test_eval_identity_partition_rejects_malformed_procedure_reference() -> (
    None
):
    ec = eval_config()
    payload = _graph_for(ec).model_dump(mode="json")
    payload["nodes"][-1]["variables"][EVALUATION_PROCEDURE_CONFIG_VARIABLE] = (
        None
    )
    graph = GraphConfig.model_validate(payload)

    with pytest.raises(ValueError, match="reference is malformed"):
        validate_eval_identity_partition(graph, ec)


def test_procedure_change_alters_both_hashes() -> None:
    base_proc = procedure_config(zero_denominator="not_applicable")
    changed_proc = procedure_config(zero_denominator="error")
    base_ec = eval_config(procedure=base_proc)
    changed_ec = eval_config(procedure=changed_proc)

    base_graph = _graph_for(base_ec)
    changed_graph = _graph_for(changed_ec)

    assert graph_hash(base_graph) != graph_hash(changed_graph)
    assert eval_config_hash(base_ec) != eval_config_hash(changed_ec)


def test_sampling_or_aggregation_change_alters_only_eval_config_hash() -> None:
    proc = procedure_config()
    base_ec = eval_config(procedure=proc, reduction="mean")
    changed_ec = eval_config(procedure=proc, reduction="sum")

    base_graph = _graph_for(base_ec)
    changed_graph = _graph_for(changed_ec)

    assert (
        base_ec.evaluation_procedure_config_hash
        == changed_ec.evaluation_procedure_config_hash
    )
    assert graph_hash(base_graph) == graph_hash(changed_graph)
    assert eval_config_hash(base_ec) != eval_config_hash(changed_ec)


def test_eval_config_hash_is_dr_code_composite_identity() -> None:
    ec = eval_config()
    assert eval_config_hash(ec) == ec.config_identity_hash
    assert len(eval_config_hash(ec)) == 64
