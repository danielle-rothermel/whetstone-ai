from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from whetstone.coordination.eval_service import EvalEngineService
from whetstone.coordination.harness_run_controller import (
    HarnessRunController,
    OptimRunLaunch,
)
from whetstone.coordination.run_controller_registry import register_run_controller
from whetstone.coordination.step_request_builder import StepRequestBuilder
from dr_store.sync import (
    BlockingObjectStore,
    persistent_sqlite,
)
from whetstone.core.effects.authority import EffectAuthority, ReplayPolicy
from whetstone.core.identity import compute_identity_hash
from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import Candidate
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.contracts import OptimRun, OutputContract, StepMode
from whetstone.optim.harness import OptimHarness
from whetstone.optim.proposal.proposer import (
    ProposalExecutorDurabilityContract,
    ProposerConfig,
    _durable_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
from whetstone.testing.fakes.proposer import DummyProposerTransport
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from whetstone.optim.copro.control import CoproControl

RUNTIME_BOOTSTRAP_SCHEMA = "whetstone.runtime_bootstrap"
RUNTIME_BOOTSTRAP_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RegisteredRuntime:
    store: BlockingObjectStore
    harness: OptimHarness
    controller: HarnessRunController
    eval_service: EvalEngineService
    adapter_registry: MappingAdapterRegistry
    ledger_engine: Engine | None = None


def _inline_proposal_executor(*, policy_identity_hash: str):
    def execute(
        *,
        config,
        request,
        transport,
        count: int,
    ):
        return transport.draft(config, request, count)

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=policy_identity_hash,
        ),
        execute=execute,
    )


def build_toy_copro_control(
    *,
    breadth: int = 2,
    depth: int = 1,
    engine: object | None = None,
) -> CoproControl:
    from whetstone.optim.copro.control import CoproInjectedDefaults, configure_copro
    from whetstone.sandbox.copro_step import toy_copro_proposal_contract

    experiment = build_toy_experiment(num_seeds=1)
    if engine is None:
        raise ValueError("engine is required to bind COPRO evaluation authority")
    execution_policy = ProviderExecutionPolicy(
        transport_policy=default_transport_policy(
            api_key_env="WHETSTONE_TOY_API_KEY",
        )
    )
    prompt_adapter = PlainPromptAdapter()
    defaults = CoproInjectedDefaults(
        prompt_model=ProposerConfig(
            provider_call_config=engine.provider_execution_policy_ref,
            temperature=None,
        ),
        proposal_contract=toy_copro_proposal_contract(
            task_context="Reply briefly.",
        ),
        eval_config_ref=engine.eval_config_ref,
        eval_role=EvalRole.INTERNAL,
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
        expected_reward_policy_hash=experiment.reward_policy.identity_hash(),
        provider_execution_policy_hash=execution_policy.identity_hash,
        prompt_adapter=prompt_adapter,
    )
    return configure_copro(
        breadth=breadth,
        depth=depth,
        track_stats=False,
        defaults=defaults,
    )


def register_runtime(
    *,
    store: BlockingObjectStore | None = None,
    sqlite_path: str | None = None,
    copro_control: CoproControl | None = None,
    ledger_engine: Engine | None = None,
) -> RegisteredRuntime:
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
    from whetstone.optim.tools.facade import ToolAdmissionAuthority, ToolCallStore

    if store is None:
        if sqlite_path is None:
            sqlite_path = f"/tmp/whetstone-runtime-{uuid4().hex}.sqlite"
        store = persistent_sqlite(sqlite_path)
    effect_authority = EffectAuthority.memory()
    runtime_config = ReferenceEvalRuntimeConfig()
    engine = runtime_config.build_engine(store)
    eval_service = EvalEngineService(store=store, engine=engine)
    control = copro_control or build_toy_copro_control(engine=engine)
    prompt_adapter = PlainPromptAdapter()
    execution_policy = runtime_config.execution_policy
    proposal_policy_hash = compute_identity_hash(
        schema="whetstone.testing.inline_proposal_executor",
        schema_version=1,
        payload={"mode": "inline"},
    )
    transport = DummyProposerTransport(
        scripted_bodies=(
            "Reply briefly to: {prompt} with a concise greeting.",
            "Answer {prompt} in one short friendly sentence.",
        ),
        execution_policy_hash=execution_policy.identity_hash,
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(prompt_adapter),
        proposal_mode="seed_proposal",
        request_ordinal=0,
    )
    copro_adapter = CoproAdapter(
        control=control,
        transport=transport,
        proposal_executor=_inline_proposal_executor(
            policy_identity_hash=proposal_policy_hash,
        ),
    )
    adapter_registry = MappingAdapterRegistry(
        {
            COPRO_ADAPTER_KEY: copro_adapter,
        }
    )
    tool_store = ToolCallStore(
        store,
        ToolAdmissionAuthority.memory(),
        effect_authority,
    )
    owner_id = uuid4().hex
    harness = OptimHarness(
        store=store,
        adapter_registry=adapter_registry,
        tool_store=tool_store,
        effect_authority=effect_authority,
        owner_id=owner_id,
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
        lease_duration=timedelta(minutes=5),
        evaluation_service=eval_service,
    )
    runtime_hash = compute_identity_hash(
        schema=RUNTIME_BOOTSTRAP_SCHEMA,
        schema_version=RUNTIME_BOOTSTRAP_SCHEMA_VERSION,
        payload={
            "owner_id": owner_id,
            "adapter_keys": [COPRO_ADAPTER_KEY],
        },
    )
    controller = HarnessRunController(
        store=store,
        harness=harness,
        runtime_hash=runtime_hash,
        step_builder=StepRequestBuilder(store=store),
    )
    register_run_controller(controller)
    return RegisteredRuntime(
        store=store,
        harness=harness,
        controller=controller,
        eval_service=eval_service,
        adapter_registry=adapter_registry,
        ledger_engine=ledger_engine,
    )


def prepare_copro_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: CoproControl,
    initial_candidate: Candidate | None = None,
    terminal_top_k: int = 1,
) -> OptimRunLaunch:
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY

    experiment = build_toy_experiment(num_seeds=1)
    candidate = initial_candidate or experiment.initial_candidate
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=COPRO_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(
            returned_proposal_count=terminal_top_k,
        ),
        template_render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )
    launch = OptimRunLaunch(
        run=run,
        initial_candidate=candidate,
        control=control,
    )
    runtime.controller.bind_launch(launch)
    return launch


def copro_run_request(
    launch: OptimRunLaunch,
    *,
    controller_identity_hash: str,
) -> "RunRequest":
    from whetstone.coordination.run_workflow import RunRequest

    if launch.control is None:
        raise ValueError("COPRO launch requires control")
    return RunRequest(
        controller_identity_hash=controller_identity_hash,
        run_id=launch.run.run_id,
        control_identity_hash=launch.control.identity_hash(),
    )


__all__ = [
    "RegisteredRuntime",
    "build_toy_copro_control",
    "copro_run_request",
    "prepare_copro_run",
    "register_runtime",
]
