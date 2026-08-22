from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from dr_store.sync import BlockingObjectStore

from whetstone.coordination.eval_service import EvalEngineService
from whetstone.coordination.harness_run_controller import (
    HarnessRunController,
    OptimRunLaunch,
    RunRequest,
)
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.core.identity import compute_identity_hash
from whetstone.core.roles import EvalRole
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.contracts import (
    OptimRun,
    OutputContract,
    StepMode,
    optimization_run_reference,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.tools.facade import ToolAdmissionAuthority, ToolCallStore

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from whetstone.optim.harness import ToolExecutor
    from whetstone.optim.tools.contracts import ToolConfig

    from whetstone.eval.protocol import EvalEngine
    from whetstone.experiment.candidate import TemplateRenderContract
    from whetstone.experiment.env import Experiment
    from whetstone.optim.codex.control import CodexControl
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.gepa.control import GepaControl
    from whetstone.optim.miprov2.control import Miprov2Control
    from whetstone.optim.miprov2.runtime import Miprov2State

#: Every Codex Tool Config endpoint key starts with ``mcp``; the adapter
#: refuses a handle whose endpoint is not external MCP.
CODEX_MCP_ENDPOINT_KEY = "mcp.whetstone.evaluate_candidate"

RUNTIME_BOOTSTRAP_SCHEMA = "whetstone.runtime_bootstrap"
RUNTIME_BOOTSTRAP_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RegisteredRuntime:
    store: BlockingObjectStore
    harness: OptimHarness
    controller: HarnessRunController
    eval_service: EvalEngineService
    adapter_registry: MappingAdapterRegistry
    engine: EvalEngine
    effect_authority: EffectLeaseAuthority
    #: The exact Tool Call Store the harness admits and ledgers through. A
    #: TOOL_USING adapter must read its durable entries from this instance,
    #: not a second store built over the same database.
    tool_store: ToolCallStore
    ledger_engine: Engine | None = None

    def close(self) -> None:
        """Release the eval engine and any closeable authority.

        The caller owns the store passed to :func:`build_runtime`; this does
        not close it.
        """
        closer = getattr(self.engine, "close", None)
        if callable(closer):
            closer()
        closer = getattr(self.effect_authority, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> RegisteredRuntime:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _derived_adapter_replay_policy(
    adapter_registry: MappingAdapterRegistry,
) -> ReplayPolicy:
    """Resolve the one replay policy every registered adapter accepts.

    ``OptimHarness`` takes a single policy but each adapter declares the
    one it requires, and ``AdapterReplayPolicyMismatchError`` fires on a
    disagreement. Codex is the first adapter that needs ``NO_REDRIVE``:
    its Step spawns a foreign agent, so a redrive would re-run paid work.
    Deriving from the registry keeps one construction path rather than a
    Codex-only branch.
    """
    required = {
        key: adapter.required_replay_policy
        for key, adapter in adapter_registry.adapters.items()
    }
    distinct = set(required.values())
    if not distinct:
        return ReplayPolicy.DURABLE_WORKFLOW
    if len(distinct) > 1:
        detail = ", ".join(
            f"{key}={policy.value}" for key, policy in sorted(required.items())
        )
        raise ValueError(
            "registered adapters disagree on the required replay policy; "
            "register them in separate runtimes: " + detail
        )
    return distinct.pop()


def build_runtime(
    *,
    store: BlockingObjectStore,
    engine: EvalEngine,
    adapter_registry: MappingAdapterRegistry,
    effect_authority: EffectLeaseAuthority,
    ledger_engine: Engine | None = None,
    platform: bool = False,
    owner_id: str | None = None,
    lease_duration: timedelta = timedelta(minutes=5),
    admission: ToolAdmissionAuthority | None = None,
    adapter_replay_policy: ReplayPolicy | None = None,
    tool_executor: ToolExecutor | None = None,
) -> RegisteredRuntime:
    """Assemble a runtime from explicit collaborators.

    Registry membership is the caller's: the controller identity hash
    includes ``adapter_keys``, so adding or removing an adapter changes
    ``controller.runtime_hash`` and therefore submit / work-input identity.

    ``admission`` defaults to an in-memory authority, which is right for a
    run whose capacity never outlives the process. A run whose per-run
    capacity gates paid work across processes -- Codex evaluates through an
    out-of-process MCP server -- must pass a durable one.

    ``adapter_replay_policy`` defaults to the policy the registered
    adapters require, and errors when they disagree.

    ``tool_executor`` is required only for a ``TOOL_USING`` run. Codex is
    the only such optimizer: the harness needs an executor so every Tool
    Call its agent made out of process is re-issued through the Step's
    Issued Tool Call ledger.
    """
    if platform and ledger_engine is None:
        raise ValueError("platform mode requires ledger_engine")
    eval_service = EvalEngineService(store=store, engine=engine)
    tool_store = ToolCallStore(
        store,
        admission or ToolAdmissionAuthority.memory(),
        effect_authority,
    )
    resolved_owner_id = owner_id or uuid4().hex
    harness = OptimHarness(
        store=store,
        adapter_registry=adapter_registry,
        tool_store=tool_store,
        effect_authority=effect_authority,
        owner_id=resolved_owner_id,
        adapter_replay_policy=(
            adapter_replay_policy
            if adapter_replay_policy is not None
            else _derived_adapter_replay_policy(adapter_registry)
        ),
        lease_duration=lease_duration,
        evaluation_service=eval_service,
        tool_executor=tool_executor,
    )
    # Adapter keys are part of controller identity: a deployment that adds
    # GEPA (or any other adapter) is a different runtime than COPRO-only.
    runtime_hash = compute_identity_hash(
        schema=RUNTIME_BOOTSTRAP_SCHEMA,
        schema_version=RUNTIME_BOOTSTRAP_SCHEMA_VERSION,
        payload={
            "owner_id": resolved_owner_id,
            "adapter_keys": sorted(adapter_registry.adapters),
        },
    )
    controller = HarnessRunController(
        store=store,
        harness=harness,
        runtime_hash=runtime_hash,
        step_builder=StepRequestBuilder(store=store),
    )
    return RegisteredRuntime(
        store=store,
        harness=harness,
        controller=controller,
        eval_service=eval_service,
        adapter_registry=adapter_registry,
        engine=engine,
        effect_authority=effect_authority,
        tool_store=tool_store,
        ledger_engine=ledger_engine,
    )


def prepare_copro_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: CoproControl,
    experiment: Experiment,
    render_contract: TemplateRenderContract,
    mutation_field: str,
    initial_candidate: Candidate | None = None,
    terminal_top_k: int = 1,
) -> OptimRunLaunch:
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY

    try:
        adapter = runtime.adapter_registry.resolve(COPRO_ADAPTER_KEY)
    except KeyError as exc:
        raise ValueError(
            "prepare_copro_run requires a COPRO adapter"
        ) from exc
    bound_control = getattr(adapter, "control", None)
    if (
        bound_control is not None
        and bound_control.reference() != control.reference()
    ):
        raise ValueError(
            "prepare_copro_run control must match the registered COPRO adapter"
        )
    if (
        experiment.reward_policy.identity_hash()
        != control.expected_reward_policy_hash
    ):
        raise ValueError(
            "prepare_copro_run experiment reward policy must match "
            "the control expected_reward_policy_hash"
        )
    candidate = initial_candidate or experiment.initial_candidate
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=COPRO_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(
            returned_proposal_count=terminal_top_k,
        ),
        template_render_contract=render_contract,
        initial_candidate_ref=candidate_reference(candidate),
        mutation_field=mutation_field,
        reward_policy=experiment.reward_policy,
    )
    launch = OptimRunLaunch(
        run=run,
        initial_candidate=candidate,
        control=control,
    )
    runtime.controller.bind_launch(launch)
    return launch


def prepare_gepa_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: GepaControl,
    experiment: Experiment,
    render_contract: TemplateRenderContract,
    mutation_field: str,
    initial_candidate: Candidate | None = None,
) -> OptimRunLaunch:
    from whetstone.optim.gepa.harness_adapter import (
        GEPA_ADAPTER_KEY,
        seed_components_from_candidate,
    )

    try:
        adapter = runtime.adapter_registry.resolve(GEPA_ADAPTER_KEY)
    except KeyError as exc:
        raise ValueError(
            "prepare_gepa_run requires a GEPA adapter; pass it in the "
            "adapter registry given to build_runtime"
        ) from exc
    if experiment.reward_policy.identity_hash() != control.reward_policy_hash:
        raise ValueError(
            "prepare_gepa_run experiment reward policy must match "
            "the control reward_policy_hash"
        )
    candidate = initial_candidate or experiment.initial_candidate
    bound_control = getattr(adapter, "control", None)
    if (
        bound_control is not None
        and bound_control.reference() != control.reference()
    ):
        raise ValueError(
            "prepare_gepa_run control must match the registered GEPA adapter"
        )
    bound_seed = getattr(adapter, "seed_candidate", None)
    if bound_seed is not None:
        component_names = (
            bound_control.component_names
            if bound_control is not None
            else tuple(bound_seed)
        )
        mapped = seed_components_from_candidate(
            candidate,
            component_names=component_names,
            mutation_field=mutation_field,
        )
        if mapped != dict(bound_seed):
            raise ValueError(
                "prepare_gepa_run initial candidate must match "
                "the adapter seed candidate"
            )
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=GEPA_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(
            returned_proposal_count=1,
        ),
        template_render_contract=render_contract,
        initial_candidate_ref=candidate_reference(candidate),
        mutation_field=mutation_field,
        reward_policy=experiment.reward_policy,
    )
    launch = OptimRunLaunch(
        run=run,
        initial_candidate=candidate,
        control=control,
    )
    runtime.controller.bind_launch(launch)
    return launch


def prepare_miprov2_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: Miprov2Control,
    experiment: Experiment,
    initial_state: Miprov2State,
    initial_candidate: Candidate | None = None,
    render_contract: TemplateRenderContract | None = None,
    mutation_field: str | None = None,
) -> OptimRunLaunch:
    """Bind one MIPROv2 run and its opening durable state.

    Unlike COPRO and GEPA, MIPROv2 cannot start from the control alone: its
    search reads a labeled trainset, rendered proposal examples, and a
    durable RNG checkpoint, all of which are part of the opening state. That
    state is carried on the launch's extra pools rather than rebuilt per
    Step, so a resumed run reads exactly what was bound, not a rebuild of it.
    """

    from whetstone.optim.miprov2.adapter import (
        MIPROV2_ADAPTER_KEY,
        MIPROV2_STATE_KEY,
    )

    try:
        runtime.adapter_registry.resolve(MIPROV2_ADAPTER_KEY)
    except KeyError as exc:
        raise ValueError(
            "prepare_miprov2_run requires a MIPROv2 adapter; pass it in the "
            "adapter registry given to build_runtime"
        ) from exc
    if experiment.reward_policy.identity_hash() != control.reward_policy_hash:
        raise ValueError(
            "prepare_miprov2_run experiment reward policy must match "
            "the control reward_policy_hash"
        )
    if control.eval_role is not EvalRole.INTERNAL:
        raise ValueError("MIPROv2 runs evaluate under the internal role")
    candidate = initial_candidate or control.base_candidate.record
    if candidate_reference(candidate) != control.base_candidate:
        raise ValueError(
            "prepare_miprov2_run initial candidate must be the control "
            "base candidate"
        )
    if (
        mutation_field is not None
        and mutation_field != control.mutation_field
    ):
        raise ValueError(
            "prepare_miprov2_run mutation_field must match the control "
            "mutation_field"
        )
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=MIPROV2_ADAPTER_KEY,
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(
            returned_proposal_count=1,
        ),
        template_render_contract=(
            render_contract or control.template_render_contract
        ),
        initial_candidate_ref=candidate_reference(candidate),
        mutation_field=control.mutation_field,
        reward_policy=experiment.reward_policy,
    )
    run_ref = optimization_run_reference(run)
    if initial_state.run != run_ref:
        raise ValueError(
            "prepare_miprov2_run initial state does not bind the exact run"
        )
    launch = OptimRunLaunch(
        run=run,
        initial_candidate=candidate,
        control=control,
        extra_pools={MIPROV2_STATE_KEY: initial_state.model_dump(mode="json")},
    )
    runtime.controller.bind_launch(launch)
    return launch


def codex_tool_config(
    *,
    control: CodexControl,
    engine: EvalEngine,
    reward_policy_hash: str,
    store_namespace_key: str,
) -> ToolConfig:
    """Build the one Tool Config a Codex run grants its agent (D12).

    The capacity is the control's ``max_tool_calls``, scoped to the run:
    the same number is simultaneously the Step's ``tool_calls`` budget, so
    the ledger's hard limit and the admission authority's cap cannot drift.
    """
    from whetstone.optim.codex.mcp_bridge import (
        CODEX_EVAL_INPUT_FIELDS,
        CODEX_EVAL_OUTPUT_FIELDS,
        CODEX_EVAL_TOOL_NAME,
    )
    from whetstone.optim.tools.contracts import (
        ToolCapacity,
        ToolCapacityScope,
        ToolConfig,
        ToolDefinition,
        tool_definition_reference,
    )

    definition = ToolDefinition(
        tool_name=CODEX_EVAL_TOOL_NAME,
        input_fields=CODEX_EVAL_INPUT_FIELDS,
        output_fields=CODEX_EVAL_OUTPUT_FIELDS,
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key=CODEX_MCP_ENDPOINT_KEY,
        eval_config=engine.eval_config_ref.record,
        reward_policy_hash=reward_policy_hash,
        capacity=ToolCapacity(
            max_accepted_calls=control.max_tool_calls,
            scope=ToolCapacityScope.RUN,
        ),
        store_namespace_key=store_namespace_key,
        candidate_template_field=control.mutation_field,
    )


def prepare_codex_run(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    control: CodexControl,
    experiment: Experiment,
    render_contract: TemplateRenderContract,
    mutation_field: str,
    initial_candidate: Candidate | None = None,
    preflight: Callable[[], None] | None = None,
) -> OptimRunLaunch:
    """Bind one Codex-direct run and the single Tool it may use.

    Codex is the only ``TOOL_USING`` optimizer, so this is the only
    ``prepare_*`` that constructs a Tool Config and attaches it to the run.
    ``preflight`` runs before the launch is bound, so a broken Codex
    session fails before any capacity or eval budget is committed; pass
    ``None`` only when the caller has already proven the session.
    """
    from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY
    from whetstone.optim.tools.contracts import tool_config_reference

    try:
        adapter = runtime.adapter_registry.resolve(CODEX_ADAPTER_KEY)
    except KeyError as exc:
        raise ValueError(
            "prepare_codex_run requires a Codex adapter; pass it in the "
            "adapter registry given to build_runtime"
        ) from exc
    bound_control = getattr(adapter, "control", None)
    if (
        bound_control is not None
        and bound_control.reference() != control.reference()
    ):
        raise ValueError(
            "prepare_codex_run control must match the registered Codex "
            "adapter"
        )
    reward_policy_hash = experiment.reward_policy.identity_hash()
    if reward_policy_hash != control.reward_policy_hash:
        raise ValueError(
            "prepare_codex_run experiment reward policy must match "
            "the control reward_policy_hash"
        )
    if control.eval_config_ref != runtime.engine.eval_config_ref:
        raise ValueError(
            "prepare_codex_run control eval_config_ref must match the "
            "runtime engine eval_config_ref"
        )
    if mutation_field != control.mutation_field:
        raise ValueError(
            "prepare_codex_run mutation_field must match the control "
            "mutation_field"
        )
    candidate = initial_candidate or experiment.initial_candidate
    tool_config = codex_tool_config(
        control=control,
        engine=runtime.engine,
        reward_policy_hash=reward_policy_hash,
        store_namespace_key=f"{CODEX_MCP_ENDPOINT_KEY}:{run_id}",
    )
    if preflight is not None:
        preflight()
    run = OptimRun(
        run_id=run_id,
        optimizer_config=control.reference(),
        adapter_key=CODEX_ADAPTER_KEY,
        mode=StepMode.TOOL_USING,
        tool_configs=(tool_config_reference(tool_config),),
        terminal_output_contract=OutputContract(
            returned_proposal_count=1,
        ),
        template_render_contract=render_contract,
        initial_candidate_ref=candidate_reference(candidate),
        mutation_field=mutation_field,
        # A TOOL_USING run carries no Reward Policy: the Tool Config pins
        # the policy hash, and the tool executor holds the policy itself.
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
) -> RunRequest:
    if launch.control is None:
        raise ValueError("COPRO launch requires control")
    return RunRequest(
        controller_identity_hash=controller_identity_hash,
        run_id=launch.run.run_id,
        control_identity_hash=launch.control.identity_hash(),
    )


__all__ = [
    "CODEX_MCP_ENDPOINT_KEY",
    "RegisteredRuntime",
    "build_runtime",
    "copro_run_request",
    "codex_tool_config",
    "prepare_codex_run",
    "prepare_copro_run",
    "prepare_gepa_run",
    "prepare_miprov2_run",
]
