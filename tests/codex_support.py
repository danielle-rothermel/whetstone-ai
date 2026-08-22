"""Shared builders for the Codex-direct optimizer tests.

Every Codex test needs the same three things: an engine, the one Tool
Config a Codex run grants, and a run whose terminal contract permits a
seed-retaining terminal step. Building them once here keeps each test
about the behavior it names.
"""

from __future__ import annotations

from typing import Any

from whetstone.coordination.runtime_bootstrap import (
    CODEX_MCP_ENDPOINT_KEY,
    codex_tool_config,
)
from whetstone.core.identity import ImmutableJsonObject
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY
from whetstone.optim.codex.control import CodexControl
from whetstone.optim.codex.step_contract import (
    CODEX_TOOL_CALLS_BUDGET_LABEL,
    codex_step_output_contract,
)
from whetstone.optim.contracts import (
    BudgetState,
    OptimRun,
    OptimStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    optimization_run_reference,
)
from whetstone.optim.tools.contracts import (
    ToolCapacityScope,
    ToolConfig,
    tool_capacity_binding,
    tool_config_reference,
)
from whetstone.testing.runtime import build_toy_codex_control
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

TOY_CODEX_RUN_ID = "codex-toy-run"


def toy_codex_control(*, engine, max_tool_calls: int = 3) -> CodexControl:
    return build_toy_codex_control(
        engine=engine,
        max_tool_calls=max_tool_calls,
    )


def toy_codex_run(
    *,
    control: CodexControl,
    engine,
    experiment=None,
    run_id: str = TOY_CODEX_RUN_ID,
) -> tuple[OptimRun, ToolConfig, Candidate]:
    """Build the run, its Tool Config, and the seed candidate together.

    The Tool Config has to be attached to the run before the run is
    referenced: ``OptimStepRequest`` rejects a ``TOOL_USING`` run with no
    Tool Configs, and the config's identity is part of the run's.
    """
    resolved = experiment or build_toy_experiment(num_seeds=1)
    candidate = resolved.initial_candidate
    config = codex_tool_config(
        control=control,
        engine=engine,
        reward_policy_hash=resolved.reward_policy.identity_hash(),
        store_namespace_key=f"{CODEX_MCP_ENDPOINT_KEY}:{run_id}",
    )
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=CODEX_ADAPTER_KEY,
        mode=StepMode.TOOL_USING,
        tool_configs=(tool_config_reference(config),),
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(candidate),
        mutation_field=TOY_MUTATION_FIELD,
        # A TOOL_USING run carries no Reward Policy: the Tool Config pins
        # the policy hash, and the executor holds the policy itself.
    )
    return run, config, candidate


def toy_codex_step_request(
    *,
    control: CodexControl,
    run: OptimRun,
    candidate: Candidate,
    tool_calls: int | None = None,
) -> OptimStepRequest:
    run_ref = optimization_run_reference(run)
    return OptimStepRequest(
        run=run_ref,
        step_id=f"{run.run_id}:codex:0",
        kind=StepKind.TOOL,
        kind_label="codex_direct",
        step_index=0,
        candidates=(candidate,),
        hyperparameters=ImmutableJsonObject(
            control.step_hyperparameters(iteration=0)
        ),
        budget=BudgetState(
            remaining=ImmutableJsonObject(
                {
                    CODEX_TOOL_CALLS_BUDGET_LABEL: (
                        control.max_tool_calls
                        if tool_calls is None
                        else tool_calls
                    )
                }
            )
        ),
        step_output_contract=codex_step_output_contract(run_ref),
    )


def toy_capacity_binding(run: OptimRun):
    return tool_capacity_binding(
        ToolCapacityScope.RUN,
        optimization_run_reference(run).record_ref,
    )


def toy_tool_args(
    *,
    candidate: Candidate,
    engine,
    template: str,
) -> dict[str, Any]:
    base_ref = candidate_reference(candidate).record_ref
    args: dict[str, Any] = {
        "base_ref": {
            "schema_name": base_ref.schema_name,
            "content_hash": base_ref.content_hash,
        },
        "model_route": engine.expected_model_route(),
        "template": template,
    }
    return args


def transcript_json(steps: list[dict[str, Any]]) -> str:
    """Serialize a fake-Codex transcript for inline environment passing."""
    import json

    return json.dumps(steps)
