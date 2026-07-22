"""Shared builders for the tool-using optimizer tests (GEPA + Codex).

Everything here builds real released-contract objects (dr-store ObjectStore,
the
real Tool Config/Reward Policy identities, the real Tool Call Store, the real
harness) plus the shipped deterministic doubles (``StubToolEvaluator``,
``FakeProposerTransport``, ``FakeCodexRunner``). No network, no real CLI.
"""

from __future__ import annotations

from dr_store import MemoryBackend, ObjectStore

from whetstone.optimization import (
    Candidate,
    EvaluateCandidateServer,
    EvaluatingToolExecutor,
    MissingDataPolicy,
    OptimizationStepRequest,
    OutputContract,
    ProposerConfig,
    RewardPolicy,
    RewardTerm,
    StubToolEvaluator,
    ToolCapacity,
    ToolConfig,
    ToolDefinition,
)
from whetstone.optimization.gepa import (
    TOOL_EVALUATE_MINIBATCH,
    TOOL_EVALUATE_SUBSET,
)
from whetstone.optimization.schema import StepKind, StepMode

FULL_A = "a" * 64
FULL_B = "b" * 64


def make_store() -> ObjectStore:
    return ObjectStore(MemoryBackend())


def reward_policy() -> RewardPolicy:
    """The pass-up / compression-down Reward Policy the Tool Configs cite."""
    return RewardPolicy(
        policy_name="pass_up_compression_down/v1",
        reward_name="reward",
        terms=(
            RewardTerm(name="pass_rate", weight=1.0, maximize=True),
            RewardTerm(name="compression", weight=1.0, maximize=False),
        ),
        missing_data=MissingDataPolicy.FAIL,
    )


def _tool_config(
    *,
    tool_name: str,
    namespace: str,
    capacity: int,
    reward_hash: str,
    eval_hash: str = FULL_B,
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name=tool_name,
        input_fields=("model_route", "template"),
        output_fields=("rollout_refs", "objective_values", "reward"),
    )
    return ToolConfig(
        tool_name=tool_name,
        tool_definition_ref=f"tooldef://{tool_name}",
        tool_definition_identity_hash=definition.identity_hash(),
        endpoint=f"mcp://bridge/{tool_name}",
        eval_config_ref="evalcfg://internal",
        eval_config_identity_hash=eval_hash,
        reward_policy_ref=reward_hash,
        capacity=ToolCapacity(max_accepted_calls=capacity),
        store_namespace=namespace,
    )


def evaluate_candidate_config(
    *, capacity: int = 20, namespace: str = "codex-ns"
) -> ToolConfig:
    return _tool_config(
        tool_name="evaluate_candidate",
        namespace=namespace,
        capacity=capacity,
        reward_hash=reward_policy().identity_hash(),
    )


def gepa_tool_configs(
    *, minibatch_capacity: int = 64, subset_capacity: int = 64
) -> tuple[ToolConfig, ToolConfig]:
    reward_hash = reward_policy().identity_hash()
    minibatch = _tool_config(
        tool_name=TOOL_EVALUATE_MINIBATCH,
        namespace="gepa-mini",
        capacity=minibatch_capacity,
        reward_hash=reward_hash,
    )
    subset = _tool_config(
        tool_name=TOOL_EVALUATE_SUBSET,
        namespace="gepa-subset",
        capacity=subset_capacity,
        reward_hash=reward_hash,
    )
    return minibatch, subset


def evaluating_executor(*, rollout_count: int = 20) -> EvaluatingToolExecutor:
    return EvaluatingToolExecutor(
        StubToolEvaluator(rollout_count=rollout_count), reward_policy()
    )


def mcp_server(tool_store, *, capacity: int = 20) -> EvaluateCandidateServer:
    config = evaluate_candidate_config(capacity=capacity)
    return EvaluateCandidateServer(
        tool_config=config,
        store=tool_store,
        executor=evaluating_executor(),
    )


def candidate(cid: str, base: str, template: str = "t") -> Candidate:
    return Candidate(
        candidate_id=cid,
        base_ref=base,
        payload={"user_prompt_template": template},
    )


def proposer_config() -> ProposerConfig:
    return ProposerConfig(
        provider_call_config_ref="pcc://reflection",
        provider_call_config_hash=FULL_A,
        temperature=1.0,
    )


def gepa_request(
    *,
    run_id: str = "run-gepa",
    step_index: int = 0,
    configs: tuple[ToolConfig, ToolConfig],
    candidates: tuple[Candidate, ...],
    pools: dict | None = None,
    prior_step_result_ref=None,
    returned_proposal_count: int = 2,
    max_reflection_lm_calls: int = 8,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s{step_index}",
        optimizer_config_hash=FULL_A,
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        kind_label="gepa_step",
        step_index=step_index,
        candidates=candidates,
        pools=pools or {},
        hyperparameters={"max_reflection_lm_calls": max_reflection_lm_calls},
        output_contract=OutputContract(
            returned_proposal_count=returned_proposal_count
        ),
        tool_configs=configs,
        prior_step_result_ref=prior_step_result_ref,
    )


def codex_request(
    *,
    run_id: str = "run-codex",
    config: ToolConfig,
    candidates: tuple[Candidate, ...],
    returned_proposal_count: int = 4,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s0",
        optimizer_config_hash=FULL_A,
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        kind_label="codex_opaque_step",
        step_index=0,
        candidates=candidates,
        output_contract=OutputContract(
            returned_proposal_count=returned_proposal_count
        ),
        tool_configs=(config,),
    )
