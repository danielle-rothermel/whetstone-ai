"""Rollout Definition role: dr-graph Graph Definition used directly, with no
parallel type/schema/artifact/identity (deliverable 2)."""

from __future__ import annotations

from dr_graph import (
    GraphConfig,
    GraphDefinition,
    graph_config_identity_document,
    graph_hash,
)

from tests.graph.support import (
    EVALUATION_PROCEDURE_CONFIG_SCHEMA,
    PROVIDER_CALL_CONFIG_SCHEMA,
    eval_config,
    fake_hash,
    llm_eval_graph_definition,
)
from whetstone.graph.nodes import (
    eval_variable_assignment,
    llm_call_variable_assignment,
)


def _config(provider_hash: str, procedure_hash: str) -> GraphConfig:
    definition = llm_eval_graph_definition()
    return definition.materialize(
        {
            "generate": llm_call_variable_assignment(
                provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
                provider_call_config_hash=provider_hash,
            ),
            "evaluate": eval_variable_assignment(
                evaluation_procedure_config_schema=(
                    EVALUATION_PROCEDURE_CONFIG_SCHEMA
                ),
                evaluation_procedure_config_hash=procedure_hash,
            ),
        }
    )


def test_rollout_definition_role_uses_native_graph_definition() -> None:
    # The Rollout Definition role is played by a native dr-graph
    # GraphDefinition; whetstone introduces no parallel Definition type.
    definition = llm_eval_graph_definition()
    assert isinstance(definition, GraphDefinition)


def test_one_definition_materializes_multiple_distinct_configs() -> None:
    proc = eval_config().evaluation_procedure_config_hash
    config_a = _config(fake_hash("a"), proc)
    config_b = _config(fake_hash("b"), proc)
    assert isinstance(config_a, GraphConfig)
    assert graph_hash(config_a) != graph_hash(config_b)


def test_rollout_variant_identity_is_the_native_graph_hash() -> None:
    proc = eval_config().evaluation_procedure_config_hash
    config = _config(fake_hash("a"), proc)
    # The variant identity is exactly the native dr-graph graph_hash over the
    # native Graph Config Identity Document; whetstone builds no second one.
    document = graph_config_identity_document(config)
    assert document.schema == "dr_graph.graph_config"
    assert len(graph_hash(config)) == 64


def test_whetstone_defines_no_parallel_graph_config_identity() -> None:
    # Whetstone owns no generic Graph Config identity payload/document/schema.
    # The GraphConfig and graph_hash used across whetstone are dr-graph's.
    import whetstone.graph.materialization as materialization

    assert materialization.GraphConfig.__module__.startswith("dr_graph")
    assert materialization.graph_hash.__module__.startswith("dr_graph")
