"""Shared builders for whetstone graph-contract tests.

Constructs real dr-code Eval Configs and real dr-graph Graph Configs so the
identity-partition and execution-planning proofs run against the released
dependency contracts, not stand-ins.
"""

from __future__ import annotations

from dr_code.eval.lifecycle import (
    AggregationDefinition,
    EvalConfig,
    EvalDefinition,
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    PreprocessingStepBinding,
    SamplingConfig,
    SamplingDefinition,
)
from dr_graph import GraphConfig, GraphDefinition

from whetstone.graph.nodes import (
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)

PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"
EVALUATION_PROCEDURE_CONFIG_SCHEMA = "dr_code.evaluation_procedure.config"


def sampling_config() -> SamplingConfig:
    definition = SamplingDefinition(definition_id="samp", version="1")
    return definition.materialize(
        {"task_set_hash": "ts1", "repeat_plan_hash": "rp1"}
    )


def procedure_config(
    *, zero_denominator: str = "not_applicable"
) -> EvaluationProcedureConfig:
    preprocessing = PreprocessingDefinition(
        definition_id="pre",
        version="1",
        steps=(
            PreprocessingStepBinding(
                instance_name="sf", step="select_first"
            ),
        ),
    ).materialize()
    metric = MetricExtractionDefinition(
        definition_id="met",
        version="1",
        questions=(
            MetricQuestionBinding(metric="code_leakage", on="output"),
        ),
    ).materialize()
    return EvaluationProcedureDefinition(
        definition_id="proc", version="1"
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric,
        assignment={"zero_denominator": zero_denominator},
    )


def eval_config(
    *,
    procedure: EvaluationProcedureConfig | None = None,
    reduction: str = "mean",
) -> EvalConfig:
    """Build a composite Eval Config; override the Procedure or Aggregation
    to probe the identity partition."""
    procedure = procedure or procedure_config()
    aggregation = AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize({"reduction": reduction})
    return EvalDefinition(definition_id="ev", version="1").materialize(
        sampling=sampling_config(),
        evaluation_procedure=procedure,
        aggregation=aggregation,
    )


def llm_eval_graph_definition() -> GraphDefinition:
    """A minimal LLM Call -> Eval Graph Definition with one terminal Node."""
    llm = llm_call_node_definition(
        "generate",
        prompt_source="task.prompt",
    )
    ev = eval_node_definition(
        "evaluate",
        upstream_sources={"candidate": "generate"},
    )
    return GraphDefinition(nodes=(llm, ev), terminal_node_id="evaluate")


def build_graph_config(
    *,
    provider_call_config_hash: str,
    evaluation_procedure_config_hash: str,
) -> GraphConfig:
    """Materialize a Graph Config with the given static config references."""
    definition = llm_eval_graph_definition()
    assignments = {
        "generate": llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=provider_call_config_hash,
        ),
        "evaluate": eval_variable_assignment(
            evaluation_procedure_config_schema=(
                EVALUATION_PROCEDURE_CONFIG_SCHEMA
            ),
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        ),
    }
    return definition.materialize(assignments)


def fake_hash(char: str) -> str:
    return char * 64
