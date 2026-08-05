"""Shared cell fixtures: a real engine driving real in-process evaluations.

These build the genuine article rather than a stub: ``build_env_experiment``
produces a real experiment, ``drive_internal_row`` executes the real row
adapter against a fake transport in-process, and the engine persists real
Evaluation Evidence. A cell test therefore exercises the same evidence path
production does, which is what makes assertions about scores, per-task vectors,
and viewer rows mean anything.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from dr_store import ObjectStore
from pydantic import BaseModel

from tests.envs.support import FakeTransport, execution_policy
from tests.optimization.support import (
    candidate,
    make_store,
    memory_tool_call_store,
    optimizer_config_ref,
    python_format_contract,
    registry,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.envs.internal_eval import (
    InternalRowRequest,
    InternalRowResult,
    drive_internal_row,
)
from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import env_spec
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.evaluation_role import EvaluationRole
from whetstone.execution.fanout import ProcessJob
from whetstone.optimization.adapters import (
    IdentityOptimizerAdapter,
    MappingAdapterRegistry,
)
from whetstone.optimization.effect_authority import (
    EffectAuthority,
    ReplayPolicy,
)
from whetstone.optimization.harness import OptimizationHarness
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.schema import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    BudgetState,
    Candidate,
    EvaluationBinding,
    OptimizationRun,
    OptimizationRunRef,
    OutputContract,
    StepKind,
    StepMode,
    optimization_run_reference,
)
from whetstone.runner.cell import CellConfig
from whetstone.runner.ledger import Ledger
from whetstone.runner.optimization_run import (
    HarnessRunController,
    OptimizationRunControl,
)

ENV_NAME = "c18"
#: Real c18 prompt placeholders, so preflight and rendering exercise the actual
#: template contract rather than a template the env would reject.
BASELINE_TEMPLATE = "{question}\n{query}\nAnswer:"
CEILING_TEMPLATE = "{question}\n{query}\nThink, then answer:"
TASK_MODEL = "openai/test"
PROPOSER_MODEL = "openai/test-proposer"


def in_process_row_job_factory(
    reply_for: Callable[[str], str] | None = None,
) -> Callable[[BaseModel], ProcessJob]:
    """Drive the real internal row adapter and return its payload directly.

    The row still travels the engine's exact ProcessJob seam, but the work
    happens here rather than in a spawned worker, so the test stays fast while
    the evidence it produces is real.
    """

    def build(request: BaseModel) -> ProcessJob:
        if not isinstance(request, InternalRowRequest):
            raise TypeError(f"unsupported row request {type(request)!r}")
        instance = request.instance.to_instance()
        env = env_spec(request.env_name)
        if (
            env_procedure_config(env).config_identity_hash
            != request.procedure_config_hash
        ):
            raise ValueError("row procedure identity is not canonical")
        answer = (
            reply_for(instance.gold)
            if reply_for is not None
            else instance.gold
        )
        outcome = drive_internal_row(
            env,
            candidate=request.candidate,
            instance=instance,
            provider_call_config=request.provider_call_config,
            execution_policy=request.execution_policy,
            transport=FakeTransport(lambda _prompt: answer),
            procedure_config_hash=request.procedure_config_hash,
            logical_call_id=request.logical_call_id,
            repeat_index=request.repeat_index,
            drive_ordinal=request.drive_ordinal,
            cache=None,
            cache_phase=request.cache_phase,
            cache_unit=request.cache_unit,
            render_guard=request.render_guard,
        )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=InternalRowResult(
                request_identity=request.request_identity,
                outcome=outcome,
            ).model_dump(mode="json"),
        )

    return build


def official_engine(
    store: ObjectStore,
    *,
    reply_for: Callable[[str], str] | None = None,
    repeats: int = 1,
) -> EvaluationEngine:
    """A real engine bound to the official split."""
    experiment = build_env_experiment(
        ENV_NAME,
        model=TASK_MODEL,
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=repeats,
    )
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        row_job_factory=in_process_row_job_factory(reply_for),
    )


def official_binding(engine: EvaluationEngine) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=engine.eval_config_ref,
        role=EvaluationRole.OFFICIAL,
        authority_principal="runner-test-authority",
        campaign="runner-cell-test",
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
    )


def identity_run(
    run_id: str, *, contract: OutputContract
) -> OptimizationRunRef:
    return optimization_run_reference(
        OptimizationRun(
            run_id=run_id,
            optimizer_config=optimizer_config_ref("identity"),
            adapter_key="identity",
            mode=StepMode.PURE,
            terminal_output_contract=contract,
            template_render_contract=python_format_contract(
                available_fields=("question", "query"),
            ),
        )
    )


def identity_controller(
    tmp_path: Path,
    *,
    store: ObjectStore,
    run_id: str,
    candidates: tuple[Candidate, ...] | None = None,
) -> HarnessRunController:
    """A controller over the identity adapter: one pure step, then COMPLETE."""
    records = candidates if candidates is not None else (candidate(),)
    contract = OutputContract(returned_proposal_count=len(records))
    run = identity_run(run_id, contract=contract)
    authority = EffectAuthority.memory()
    adapter_registry = MappingAdapterRegistry(
        {"identity": IdentityOptimizerAdapter()}
    )
    harness = OptimizationHarness(
        store=store,
        adapter_registry=adapter_registry,
        tool_store=memory_tool_call_store(store, authority),
        effect_authority=authority,
        owner_id="runner-cell-owner",
        adapter_replay_policy=ReplayPolicy.IDEMPOTENT,
        lease_duration=timedelta(seconds=1),
    )
    control = OptimizationRunControl(
        run=run,
        initial_candidates=records,
        initial_budget=BudgetState(remaining={"rollouts": 10}),
        step_kind=StepKind.IDENTITY,
        adapter_replay_policy=ReplayPolicy.IDEMPOTENT,
        owner_id="runner-cell-owner",
        step_output_contract=contract,
    )
    return HarnessRunController(
        control=control,
        harness=harness,
        adapter_registry=adapter_registry,
        store=store,
    )


def cell_config(
    tmp_path: Path,
    *,
    attempt: int = 0,
    ceiling: bool = True,
    store: ObjectStore | None = None,
    reply_for: Callable[[str], str] | None = None,
    **overrides,
) -> CellConfig:
    """One complete cell over real evidence, driven by the identity adapter."""
    exact_store = store if store is not None else make_store(tmp_path)
    engine = official_engine(exact_store, reply_for=reply_for)
    baseline = candidate("baseline", text=BASELINE_TEMPLATE)
    ceiling_candidate = (
        candidate("ceiling", text=CEILING_TEMPLATE) if ceiling else None
    )
    run_id = f"identity:{ENV_NAME}:a{attempt}"
    controller = identity_controller(
        tmp_path,
        store=exact_store,
        run_id=run_id,
        candidates=(baseline,),
    )

    def driver() -> TypedRef:
        return controller.drive(
            controller.control.run_request(
                controller_identity_hash=controller.runtime_identity_hash
            )
        )

    fields: dict[str, object] = {
        "env": ENV_NAME,
        "attempt": attempt,
        "canonical": False,
        "task_model": TASK_MODEL,
        "proposer_model": PROPOSER_MODEL,
        "lane": "test-lane",
        "baseline": baseline,
        "controller": controller,
        "driver": driver,
        "official_engine": engine,
        "official_evaluation_binding": official_binding(engine),
        "store": exact_store,
        "ledger": Ledger(tmp_path / "ledger"),
        "ceiling": ceiling_candidate,
    }
    fields.update(overrides)
    return CellConfig(**fields)  # ty: ignore[invalid-argument-type]


__all__ = [
    "ENV_NAME",
    "PROPOSER_MODEL",
    "TASK_MODEL",
    "cell_config",
    "identity_controller",
    "in_process_row_job_factory",
    "official_binding",
    "official_engine",
    "registry",
]
