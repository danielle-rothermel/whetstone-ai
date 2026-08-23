from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

from dr_store.sync import BlockingObjectStore, persistent_sqlite

from whetstone.coordination.harness_run_controller import OptimRunLaunch
from whetstone.coordination.runtime_bootstrap import (
    RegisteredRuntime,
    build_runtime,
    prepare_copro_run,
    prepare_miprov2_run,
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
    from whetstone.optim.codex.adapter import CodexRunner
    from whetstone.optim.codex.control import CodexControl
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.gepa.control import GepaControl
    from whetstone.optim.gepa.harness_adapter import GepaHarnessAdapter
    from whetstone.optim.miprov2.control import Miprov2Control
    from whetstone.optim.miprov2.runtime import Miprov2State
    from whetstone.optim.proposal.proposer import ProposerTransport

TOY_GEPA_COMPONENT = "generate"
TOY_GEPA_INLINE_POLICY_HASH = compute_identity_hash(
    schema="whetstone.testing.inline_proposal_executor",
    schema_version=1,
    payload={"mode": "inline"},
)
TOY_GEPA_REFLECTION_BODIES = (
    "Answer {prompt} in one short friendly sentence.",
)


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
    num_seeds: int = 1,
    render_contract: TemplateRenderContract | None = None,
    mutation_field: str | None = None,
) -> OptimRunLaunch:
    return prepare_copro_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment or build_toy_experiment(num_seeds=num_seeds),
        render_contract=render_contract or toy_template_render_contract(),
        mutation_field=mutation_field or TOY_MUTATION_FIELD,
        initial_candidate=initial_candidate,
        terminal_top_k=terminal_top_k,
    )



def build_miprov2_adapter(
    *,
    store: BlockingObjectStore,
    control: Miprov2Control,
    engine: EvalEngine,
    proposal_bodies: tuple[str, ...] | None = None,
    proposer_transport: ProposerTransport | None = None,
) -> OptimizerAdapter:
    """Build a MIPROv2 adapter bound to ``engine`` and ``control``.

    The adapter derives every per-effect Eval Config through the engine that
    will run it, and drafts instructions through the canonical inline
    proposal executor, so the routes the control names are the routes the
    run actually uses.
    """

    from whetstone.optim.miprov2.adapter import Miprov2Adapter
    from whetstone.optim.miprov2.engine_binding import (
        EngineEvalBindingResolver,
    )
    from whetstone.testing.toy.miprov2 import (
        toy_proposal_policy_identity_hash,
    )

    prompt_adapter = PlainPromptAdapter()
    transport = proposer_transport or DummyProposerTransport(
        scripted_bodies=proposal_bodies
        or (
            "Reply briefly to: {prompt} with a concise greeting.",
            "Answer {prompt} in one short friendly sentence.",
        ),
        execution_policy_hash=engine.execution_policy_identity_hash(),
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
        proposal_mode="miprov2_instruction",
        request_ordinal=0,
    )
    return Miprov2Adapter(
        store=store,
        proposer_config=control.prompt_model,
        transport=transport,
        eval_config_resolver=EngineEvalBindingResolver(engine=engine),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=toy_proposal_policy_identity_hash(),
        ),
    )


def prepare_toy_miprov2_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: Miprov2Control,
    engine: EvalEngine,
    initial_candidate: Candidate | None = None,
    experiment: Experiment | None = None,
    num_seeds: int = 1,
    render_contract: TemplateRenderContract | None = None,
    mutation_field: str | None = None,
    initial_state: Miprov2State | None = None,
) -> OptimRunLaunch:
    from whetstone.experiment.candidate import candidate_reference
    from whetstone.optim.contracts import (
        OptimRun,
        OutputContract,
        StepMode,
        optimization_run_reference,
    )
    from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY
    from whetstone.testing.toy.miprov2 import build_toy_miprov2_state

    resolved = experiment or build_toy_experiment(num_seeds=num_seeds)
    candidate = initial_candidate or control.base_candidate.record
    if (
        mutation_field is not None
        and mutation_field != control.mutation_field
    ):
        raise ValueError(
            "prepare_toy_miprov2_run mutation_field must match the control "
            "mutation_field"
        )
    if initial_state is None:
        try:
            adapter = runtime.adapter_registry.resolve(MIPROV2_ADAPTER_KEY)
        except KeyError as exc:
            raise ValueError(
                "prepare_toy_miprov2_run requires a MIPROv2 adapter; pass it "
                "via register_toy_runtime(..., extra_adapters={...})"
            ) from exc
        preview = OptimRun(
            run_id=run_id,
            optimizer_config=control.reference(),
            adapter_key=MIPROV2_ADAPTER_KEY,
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=(
                render_contract or control.template_render_contract
            ),
            initial_candidate_ref=candidate_reference(candidate),
            mutation_field=control.mutation_field,
            reward_policy=resolved.reward_policy,
        )
        initial_state = build_toy_miprov2_state(
            run=optimization_run_reference(preview),
            control=control,
            engine=engine,
            proposal_executor_policy_identity_hash=(
                adapter.proposal_executor_policy_identity_hash
            ),
            proposal_transport_durability_identity_hash=(
                adapter.proposal_transport_durability_identity_hash
            ),
        )
    return prepare_miprov2_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=resolved,
        initial_state=initial_state,
        initial_candidate=initial_candidate,
        render_contract=render_contract,
        mutation_field=mutation_field,
    )


def build_toy_gepa_control(
    *,
    engine: EvalEngine,
    max_metric_calls: int = 4,
    mutation_field: str = TOY_MUTATION_FIELD,
    reflection_minibatch_size: int = 1,
    use_merge: bool = False,
    trainset_task_hashes: tuple[str, ...] | None = None,
    valset_task_hashes: tuple[str, ...] | None = None,
) -> GepaControl:
    """A toy GEPA control over the engine's tasks.

    By default the whole sampling plan is the trainset and validation binds
    back to it. Pass both ``trainset_task_hashes`` and ``valset_task_hashes``
    to exercise a real split; together they must be the engine's task order.
    """
    from whetstone.optim.gepa.control import configure_gepa
    from whetstone.optim.gepa.factory import default_gepa_prompt_services

    if valset_task_hashes is not None and trainset_task_hashes is None:
        raise ValueError(
            "a toy GEPA valset requires an explicit trainset"
        )
    resolved_trainset = (
        engine.sampling.task_hashes
        if trainset_task_hashes is None
        else trainset_task_hashes
    )

    prompt_adapter = PlainPromptAdapter()
    services = default_gepa_prompt_services(
        component_names=(TOY_GEPA_COMPONENT,),
        mutation_field=mutation_field,
    )
    return configure_gepa(
        reflection_model=ProposerConfig(
            provider_call_config=engine.provider_execution_policy_ref,
        ),
        metric=engine.eval_config_ref,
        reward_policy_hash=engine.reward_policy_identity_hash(),
        evaluation_execution_policy_hash=engine.execution_policy_identity_hash(),
        proposal_execution_policy_hash=engine.execution_policy_identity_hash(),
        proposal_prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            prompt_adapter
        ),
        proposal_durability_policy_identity_hash=TOY_GEPA_INLINE_POLICY_HASH,
        task_model_identity_hash=engine.task_model_identity_hash(),
        prompt_format_identity_hash=services.descriptor.identity_hash(),
        prompt_binding_identity_hash=services.binding.identity_hash(),
        trainset_task_hashes=resolved_trainset,
        valset_task_hashes=valset_task_hashes,
        component_names=(TOY_GEPA_COMPONENT,),
        num_predictors=1,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        use_merge=use_merge,
    )


def build_toy_gepa_adapter(
    *,
    store: BlockingObjectStore,
    engine: EvalEngine,
    control: GepaControl,
    run_id: str,
    initial_candidate: Candidate,
    mutation_field: str = TOY_MUTATION_FIELD,
    evaluation_service=None,
    reflection_bodies: tuple[str, ...] = TOY_GEPA_REFLECTION_BODIES,
) -> GepaHarnessAdapter:
    from whetstone.optim.gepa.factory import (
        build_gepa_harness_adapter,
        default_gepa_prompt_services,
    )
    from whetstone.optim.proposal.proposer import FakeProposerTransport

    prompt_adapter = PlainPromptAdapter()
    services = default_gepa_prompt_services(
        component_names=control.component_names,
        mutation_field=mutation_field,
    )
    return build_gepa_harness_adapter(
        store=store,
        engine=engine,
        control=control,
        run_id=run_id,
        initial_candidate=initial_candidate,
        mutation_field=mutation_field,
        prompt_services=services,
        transport=FakeProposerTransport(
            {("gepa_reflection", 0): reflection_bodies},
            default=reflection_bodies,
            execution_policy_hash=engine.execution_policy_identity_hash(),
            prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                prompt_adapter
            ),
        ),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=TOY_GEPA_INLINE_POLICY_HASH,
        ),
        evaluation_service=evaluation_service,
    )


def prepare_toy_gepa_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: GepaControl,
    initial_candidate: Candidate | None = None,
    experiment: Experiment | None = None,
    num_seeds: int = 1,
    render_contract: TemplateRenderContract | None = None,
    mutation_field: str | None = None,
) -> OptimRunLaunch:
    from whetstone.coordination.runtime_bootstrap import prepare_gepa_run

    return prepare_gepa_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment or build_toy_experiment(num_seeds=num_seeds),
        render_contract=render_contract or toy_template_render_contract(),
        mutation_field=mutation_field or TOY_MUTATION_FIELD,
        initial_candidate=initial_candidate,
    )


TOY_CODEX_MODEL = "toy-codex-model"


def build_toy_codex_control(
    *,
    engine: EvalEngine,
    max_tool_calls: int = 3,
    mutation_field: str = TOY_MUTATION_FIELD,
    wall_seconds: float = 120.0,
    codex_binary: str = "codex",
) -> CodexControl:
    from whetstone.optim.codex.control import configure_codex

    return configure_codex(
        model=TOY_CODEX_MODEL,
        max_tool_calls=max_tool_calls,
        eval_config_ref=engine.eval_config_ref,
        reward_policy_hash=engine.reward_policy_identity_hash(),
        evaluation_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        task_model_identity_hash=engine.task_model_identity_hash(),
        internal_task_hashes=engine.sampling.task_hashes,
        wall_seconds=wall_seconds,
        codex_binary=codex_binary,
        mutation_field=mutation_field,
    )


def build_toy_codex_adapter(
    *,
    store: BlockingObjectStore,
    runner: CodexRunner,
) -> OptimizerAdapter:
    """Wrap ``runner`` in a Codex adapter awaiting its Tool Call Store.

    Bind the runtime's exact store with ``bind_tool_store`` once
    ``build_runtime`` has produced it.
    """
    from whetstone.optim.codex.adapter import CodexAdapter

    return CodexAdapter(runner, store=store)


def scripted_codex_preflight() -> None:
    """The preflight a scripted Codex stand-in satisfies.

    ``prepare_codex_run`` requires proof of a usable Codex session before
    it commits capacity or eval budget. A test driving the scripted fake
    CLI has no real session to prove and no spend to protect, so it names
    this explicitly. It lives in ``whetstone.testing`` and is not a
    default anywhere, so no production path can reach it: the CLI must
    pass the real :func:`codex_auth_preflight`.
    """


def prepare_toy_codex_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: CodexControl,
    initial_candidate: Candidate | None = None,
    experiment: Experiment | None = None,
    num_seeds: int = 1,
    render_contract: TemplateRenderContract | None = None,
    mutation_field: str | None = None,
    preflight: Callable[[], None] = scripted_codex_preflight,
) -> OptimRunLaunch:
    from whetstone.coordination.runtime_bootstrap import prepare_codex_run

    return prepare_codex_run(
        runtime,
        run_id=run_id,
        control=control,
        experiment=experiment or build_toy_experiment(num_seeds=num_seeds),
        render_contract=render_contract or toy_template_render_contract(),
        mutation_field=mutation_field or TOY_MUTATION_FIELD,
        initial_candidate=initial_candidate,
        preflight=preflight,
    )


__all__ = [
    "TOY_CODEX_MODEL",
    "TOY_GEPA_COMPONENT",
    "TOY_GEPA_INLINE_POLICY_HASH",
    "TOY_GEPA_REFLECTION_BODIES",
    "build_miprov2_adapter",
    "build_toy_codex_adapter",
    "build_toy_codex_control",
    "build_toy_copro_control",
    "build_toy_gepa_adapter",
    "build_toy_gepa_control",
    "prepare_toy_codex_run",
    "scripted_codex_preflight",
    "prepare_toy_copro_run",
    "prepare_toy_gepa_run",
    "prepare_toy_miprov2_run",
    "register_toy_runtime",
]
