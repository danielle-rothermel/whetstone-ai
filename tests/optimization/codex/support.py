from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio
from dr_store import ObjectStore, SqliteBackend
from mcp.client import Client

from tests.envs.support import (
    code_comp_direct_experiment,
    execution_policy,
    process_row_job_factory,
)
from tests.optimization.support import (
    memory_tool_call_store,
    optimizer_config_ref,
    python_format_contract,
)
from whetstone.core.effects.authority import EffectAuthority, ReplayPolicy
from whetstone.core.identity import TypedRef
from whetstone.envs.code_comp import CodeCompMode
from whetstone.envs.code_comp.config import default_code_comp_config
from whetstone.envs.code_comp.constants import MUTATION_FIELD
from whetstone.envs.code_comp.experiment import CodeCompExperiment
from whetstone.envs.code_comp.runtime_config import (
    CodeCompEvaluationRuntimeConfig,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optimization.codex.adapter import (
    CodexOutputArtifact,
    CodexRunResult,
    OpaqueStepError,
)
from whetstone.optimization.codex.mcp_bridge import EvaluateCandidateServer
from whetstone.optimization.contracts import (
    BudgetState,
    OptimizationRun,
    OptimizationRunRef,
    OptimizationStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    optimization_run_reference,
)
from whetstone.optimization.tools.contracts import (
    RuntimeToolHandle,
    ToolCapacity,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    tool_capacity_binding,
    tool_config_reference,
    tool_definition_reference,
)
from whetstone.optimization.tools.evaluator import EngineToolEvaluator
from whetstone.optimization.tools.execution import EvaluatingToolExecutor
from whetstone.optimization.tools.facade import ToolCallStore

ROW_JOB_ENTRYPOINT = "tests.envs.process_workers:drive_d1_success"
MODEL_ROUTE = "openai/test"


def experiment() -> EnvExperiment:
    return code_comp_direct_experiment(
        model=MODEL_ROUTE,
        num_samples=1,
        task_count=3,
        internal_n=1,
        official_n=1,
    )


def engine(store: ObjectStore, experiment: EnvExperiment) -> EvaluationEngine:
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=process_row_job_factory(ROW_JOB_ENTRYPOINT),
    )


def tool_config(
    engine: EvaluationEngine,
    experiment: EnvExperiment,
    namespace: str,
    *,
    tool_name: str = "evaluate_candidate",
    supports_task_ids: bool = False,
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name=tool_name,
        input_fields=(
            "base_ref",
            "model_route",
            "template",
            *(("task_ids",) if supports_task_ids else ()),
        ),
        output_fields=(
            "evaluation_evidence_ref",
            "output_artifact_ref",
            "per_task_values",
            "per_task_counts",
            "row_accounting",
        ),
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="mcp://whetstone/evaluate_candidate",
        eval_config=engine.eval_config_ref.record,
        reward_policy_hash=experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(
            max_accepted_calls=4,
            scope=ToolCapacityScope.RUN,
        ),
        store_namespace_key=namespace,
        idempotent_replay=False,
    )


def proposals(base: Candidate) -> tuple[Candidate, Candidate]:
    base_record_ref = candidate_reference(base).record_ref
    body = base.payload[MUTATION_FIELD]
    assert isinstance(body, str)
    return (
        Candidate(
            candidate_id="codex-a",
            base_ref=base_record_ref,
            payload={MUTATION_FIELD: body + " Respond with complete code."},
        ),
        Candidate(
            candidate_id="codex-b",
            base_ref=base_record_ref,
            payload={MUTATION_FIELD: body + " Output only valid Python."},
        ),
    )


def optimization_run(
    config: ToolConfig, contract: OutputContract, run_id: str
) -> OptimizationRunRef:
    return optimization_run_reference(
        OptimizationRun(
            run_id=run_id,
            optimizer_config=optimizer_config_ref("codex"),
            adapter_key="codex",
            mode=StepMode.TOOL_USING,
            terminal_output_contract=contract,
            template_render_contract=python_format_contract(
                available_fields=("input_code",)
            ),
            tool_configs=(tool_config_reference(config),),
        )
    )


def request(
    base: Candidate,
    config: ToolConfig,
    *,
    distinct: bool = False,
    proposal_count: int = 2,
    run_id: str = "codex-run",
    hyperparameters: dict[str, object] | None = None,
) -> OptimizationStepRequest:
    contract = OutputContract(
        returned_proposal_count=proposal_count,
        require_distinct_bases=distinct,
    )
    return OptimizationStepRequest(
        run=optimization_run(config, contract, run_id),
        step_id="codex-opaque",
        kind=StepKind.TOOL,
        step_index=0,
        candidates=(base,),
        step_output_contract=contract,
        budget=BudgetState(remaining={"tool_calls": 4}),
        hyperparameters=hyperparameters or {},
    )


def binding(request: OptimizationStepRequest) -> ToolCapacityBinding:
    return tool_capacity_binding(ToolCapacityScope.RUN, request.run.record_ref)


def executor(
    engine: EvaluationEngine,
    experiment: EnvExperiment,
    authority: EffectAuthority,
) -> EvaluatingToolExecutor:
    return EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        experiment.reward_policy,
        authority,
        owner_id="codex-test-owner",
        replay_policy=ReplayPolicy.NO_REDRIVE,
    )


@dataclass(frozen=True, slots=True)
class ScriptedAgentCall:
    call_id: str
    base_ref: TypedRef
    model_route: str
    template: str
    task_ids: tuple[str, ...] | None = None


class FakeCodexRunner:
    def __init__(
        self,
        *,
        server: EvaluateCandidateServer | None = None,
        scripted_calls: Sequence[ScriptedAgentCall],
        final_proposals: Sequence[Candidate],
        artifact_run_id: str | None = None,
    ) -> None:
        self._server = server
        self._calls = tuple(scripted_calls)
        self._proposals = tuple(final_proposals)
        self._artifact_run_id = artifact_run_id
        self.observed_payloads: list[dict[str, object]] = []

    def _boundary(self, handle: RuntimeToolHandle) -> EvaluateCandidateServer:
        if self._server is not None:
            return self._server
        return EvaluateCandidateServer(handle=handle)

    async def _evaluate_calls(
        self,
        server: EvaluateCandidateServer,
        handle: RuntimeToolHandle,
    ) -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
            if [tool.name for tool in tools.tools] != [
                str(handle.config.tool_name)
            ]:
                raise OpaqueStepError("external MCP process omitted the tool")
            for call in self._calls:
                arguments: dict[str, object] = {
                    "call_id": call.call_id,
                    "base_ref": call.base_ref.model_dump(mode="json"),
                    "model_route": call.model_route,
                    "template": call.template,
                }
                if call.task_ids is not None:
                    arguments["task_ids"] = list(call.task_ids)
                result = await client.call_tool(
                    str(handle.config.tool_name), arguments
                )
                payload = result.structured_content
                if not isinstance(payload, dict):
                    raise OpaqueStepError(
                        "MCP tool returned no structured result"
                    )
                self.observed_payloads.append(payload)

    def run(
        self, request: OptimizationStepRequest, handle: RuntimeToolHandle
    ) -> CodexRunResult:
        anyio.run(
            self._evaluate_calls,
            self._boundary(handle),
            handle,
        )
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=self._artifact_run_id or request.run_id,
                proposals=self._proposals,
                conversation_evidence={
                    "process": "fake",
                    "jsonrpc_call_count": len(self._calls),
                },
                control_cost={"agent_tokens": 0},
            )
        )


def fake_runner(
    base: Candidate,
    *,
    call_id: str = "agent-call-1",
    final_proposals: Sequence[Candidate] | None = None,
    scripted_calls: Sequence[ScriptedAgentCall] | None = None,
    artifact_run_id: str | None = None,
) -> FakeCodexRunner:
    template = base.payload[MUTATION_FIELD]
    assert isinstance(template, str)
    return FakeCodexRunner(
        scripted_calls=(
            (
                ScriptedAgentCall(
                    call_id=call_id,
                    base_ref=base.base_ref,
                    model_route=MODEL_ROUTE,
                    template=template,
                ),
            )
            if scripted_calls is None
            else scripted_calls
        ),
        final_proposals=(
            proposals(base) if final_proposals is None else final_proposals
        ),
        artifact_run_id=artifact_run_id,
    )


@dataclass(frozen=True, slots=True)
class CodexStack:
    database: Path
    store: ObjectStore
    config: ToolConfig
    tool_store: ToolCallStore
    executor: EvaluatingToolExecutor
    runner: FakeCodexRunner
    base: Candidate


def stack(
    tmp_path: Path,
    experiment: EnvExperiment,
    *,
    namespace: str = "codex-durable",
    tool_name: str = "evaluate_candidate",
    supports_task_ids: bool = False,
) -> CodexStack:
    database = tmp_path / "codex.sqlite"
    store = ObjectStore(SqliteBackend(database))
    evaluation_engine = engine(store, experiment)
    config = tool_config(
        evaluation_engine,
        experiment,
        namespace,
        tool_name=tool_name,
        supports_task_ids=supports_task_ids,
    )
    authority = EffectAuthority.memory()
    call_store = memory_tool_call_store(store, authority)
    evaluation_executor = executor(evaluation_engine, experiment, authority)
    base = experiment.initial_candidate
    return CodexStack(
        database=database,
        store=store,
        config=config,
        tool_store=call_store,
        executor=evaluation_executor,
        runner=fake_runner(base),
        base=base,
    )


def runtime_config(
    engine: EvaluationEngine,
    *,
    partial_log_path: str | None = None,
    prompt_cache_path: str | None = None,
) -> CodeCompEvaluationRuntimeConfig:
    experiment = engine.experiment
    if isinstance(experiment, CodeCompExperiment):
        config = experiment.config
    else:
        config = default_code_comp_config(
            CodeCompMode.DIRECT,
            pool={"tasks": ()},
            split={"internal_n": 1, "official_n": 1},
            sampling={"num_samples": 1},
        )
    return CodeCompEvaluationRuntimeConfig(
        experiment_config=config,
        expected_eval_config_hash=engine.eval_config_ref.config_hash,
        execution_policy=execution_policy(),
        row_job_entrypoint=ROW_JOB_ENTRYPOINT,
        partial_log_path=partial_log_path,
        prompt_cache_path=prompt_cache_path,
    )
