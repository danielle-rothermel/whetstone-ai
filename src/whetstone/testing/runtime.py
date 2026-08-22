from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

from dr_store.sync import BlockingObjectStore, persistent_sqlite

from whetstone.coordination.harness_run_controller import OptimRunLaunch
from whetstone.coordination.runtime_bootstrap import (
    RegisteredRuntime,
    build_runtime,
    prepare_copro_run,
)
from whetstone.core.leasing import EffectLeaseAuthority
from whetstone.core.identity import compute_identity_hash
from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import Candidate
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.proposal.proposer import (
    ProposerConfig,
    build_inline_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.testing.fakes.proposer import DummyProposerTransport
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.candidate import TemplateRenderContract
    from whetstone.experiment.env import Experiment
    from whetstone.optim.adapters import OptimizerAdapter
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.proposal.proposer import ProposerTransport


def build_toy_copro_control(
    *,
    breadth: int = 2,
    depth: int = 1,
    engine: object | None = None,
) -> CoproControl:
    from whetstone.optim.copro.control import CoproInjectedDefaults, configure_copro
    from whetstone.sandbox.copro_step import toy_copro_proposal_contract

    if engine is None:
        raise ValueError("engine is required to bind COPRO evaluation authority")
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
        expected_reward_policy_hash=engine.reward_policy_identity_hash(),
        provider_execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter=prompt_adapter,
    )
    return configure_copro(
        breadth=breadth,
        depth=depth,
        track_stats=False,
        defaults=defaults,
    )


def register_toy_runtime(
    *,
    store: BlockingObjectStore | None = None,
    sqlite_path: str | None = None,
    copro_control: CoproControl | None = None,
    ledger_engine: Engine | None = None,
    engine: EvalEngine | None = None,
    extra_adapters: Mapping[str, OptimizerAdapter] | None = None,
    proposal_bodies: tuple[str, ...] | None = None,
    proposer_transport: ProposerTransport | None = None,
    platform: bool = False,
) -> RegisteredRuntime:
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter

    if engine is not None and store is None:
        raise ValueError(
            "register_toy_runtime(engine=...) requires store= the engine "
            "was built against"
        )
    if engine is not None and copro_control is None:
        raise ValueError(
            "register_toy_runtime(engine=...) requires copro_control= "
            "built against that engine"
        )
    if store is None:
        if sqlite_path is None:
            sqlite_path = f"/tmp/whetstone-runtime-{uuid4().hex}.sqlite"
        store = persistent_sqlite(sqlite_path)
    effect_authority = EffectLeaseAuthority.memory()
    runtime_config = ReferenceEvalRuntimeConfig()
    resolved_engine = engine or runtime_config.build_engine(store)
    control = copro_control or build_toy_copro_control(engine=resolved_engine)
    if (
        control.expected_reward_policy_hash
        != resolved_engine.reward_policy_identity_hash()
    ):
        raise ValueError(
            "register_toy_runtime copro_control reward policy must match "
            "the engine reward_policy_identity_hash"
        )
    if (
        control.provider_execution_policy_hash
        != resolved_engine.execution_policy_identity_hash()
    ):
        raise ValueError(
            "register_toy_runtime copro_control provider execution policy "
            "must match the engine execution_policy_identity_hash"
        )
    if control.eval_config_ref != resolved_engine.eval_config_ref:
        raise ValueError(
            "register_toy_runtime copro_control eval_config_ref must match "
            "the engine eval_config_ref"
        )
    prompt_adapter = PlainPromptAdapter()
    execution_policy = runtime_config.execution_policy
    engine_policy_hash = getattr(
        resolved_engine, "execution_policy_identity_hash", None
    )
    dummy_policy_hash = (
        engine_policy_hash()
        if callable(engine_policy_hash)
        else execution_policy.identity_hash
    )
    proposal_policy_hash = compute_identity_hash(
        schema="whetstone.testing.inline_proposal_executor",
        schema_version=1,
        payload={"mode": "inline"},
    )
    transport = proposer_transport or DummyProposerTransport(
        scripted_bodies=proposal_bodies
        or (
            "Reply briefly to: {prompt} with a concise greeting.",
            "Answer {prompt} in one short friendly sentence.",
        ),
        execution_policy_hash=dummy_policy_hash,
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(prompt_adapter),
        proposal_mode="seed_proposal",
        request_ordinal=0,
    )
    copro_adapter = CoproAdapter(
        control=control,
        transport=transport,
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=proposal_policy_hash,
        ),
    )
    adapters = {COPRO_ADAPTER_KEY: copro_adapter}
    if extra_adapters:
        adapters.update(extra_adapters)
    return build_runtime(
        store=store,
        engine=resolved_engine,
        adapter_registry=MappingAdapterRegistry(adapters),
        effect_authority=effect_authority,
        ledger_engine=ledger_engine,
        platform=platform,
    )


def prepare_toy_copro_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: CoproControl,
    initial_candidate: Candidate | None = None,
    terminal_top_k: int = 1,
    experiment: Experiment | None = None,
    render_contract: TemplateRenderContract | None = None,
    mutation_field: str | None = None,
) -> OptimRunLaunch:
    return prepare_copro_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment or build_toy_experiment(num_seeds=1),
        render_contract=render_contract or toy_template_render_contract(),
        mutation_field=mutation_field or TOY_MUTATION_FIELD,
        initial_candidate=initial_candidate,
        terminal_top_k=terminal_top_k,
    )


__all__ = [
    "build_toy_copro_control",
    "prepare_toy_copro_run",
    "register_toy_runtime",
]
