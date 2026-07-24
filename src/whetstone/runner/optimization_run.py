"""Canonical algorithm controllers for one durable optimization run."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from dr_store import BindingConflictError, ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.code_eval.power import PowerResult
from whetstone.envs.sampling import (
    EnvSplitSampling,
    derive_split_sampling,
)
from whetstone.evaluation import EvaluationEvidence
from whetstone.evaluation.schema import EVALUATION_EVIDENCE_SCHEMA
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    COMPLETION,
    MIPROV2_ADAPTER_KEY,
    AdapterOutput,
    BudgetDelta,
    BudgetState,
    Candidate,
    CandidateRef,
    CoproAttempt,
    CoproConfig,
    CoproControl,
    CoproDriver,
    CoproFinalization,
    CoproState,
    EvalConfigRef,
    EvaluationService,
    Miprov2Driver,
    OptimizationHarness,
    OptimizationResult,
    OptimizationStepRequest,
    OptimizationStepResult,
    OutputContract,
    ProposerConfig,
    Reward,
    StepKind,
    StepMode,
    StepStatus,
    ToolConfig,
    TypedRef,
    candidate_reference,
    eval_config_reference,
    typed_ref_for_record,
)

OPTIMIZATION_TRACE_SCHEMA = "whetstone.runner.optimization_trace"
OPTIMIZATION_RUN_CONTROL_SCHEMA = "whetstone.runner.optimization_run_control"


class Optimizer(StrEnum):
    IDENTITY = "identity"
    COPRO = "copro"
    MIPROV2 = "miprov2"
    GEPA = "gepa"
    CODEX = "codex"


class OptimizationRunControlRecord(BaseModel):
    """Exact serialized identity bound before a run can be observed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    optimizer: Optimizer
    candidates: tuple[CandidateRef, ...]
    output_count: StrictInt
    budget: BudgetState
    hyperparameters: dict[str, Any]
    pools: dict[str, Any]
    eval_configs: dict[str, EvalConfigRef]
    tool_configs: tuple[ToolConfig, ...]
    proposer_config: ProposerConfig | None
    reflection_config: ProposerConfig | None
    copro_control: CoproControl | None
    copro_control_identity_hash: StrictStr | None

    @model_validator(mode="after")
    def _validate_copro_identity(self) -> OptimizationRunControlRecord:
        expected = (
            self.copro_control.identity_hash()
            if self.copro_control is not None
            else None
        )
        if self.copro_control_identity_hash != expected:
            raise ValueError(
                "copro_control_identity_hash does not match CoproControl"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class OptimizationRunControl:
    """Internal controller input with exact persisted configuration records."""

    run_id: str
    optimizer: Optimizer
    candidates: tuple[Candidate, ...]
    output_count: int = 1
    budget: BudgetState = field(default_factory=BudgetState)
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    pools: Mapping[str, Any] = field(default_factory=dict)
    eval_configs: Mapping[str, EvalConfigRef] = field(default_factory=dict)
    tool_configs: tuple[ToolConfig, ...] = ()
    proposer_config: ProposerConfig | None = None
    reflection_config: ProposerConfig | None = None
    copro_control: CoproControl | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.candidates:
            raise ValueError("optimization requires at least one candidate")
        if self.output_count < 1:
            raise ValueError("output_count must be positive")
        expected = {
            Optimizer.IDENTITY: set(),
            Optimizer.COPRO: {"internal"},
            Optimizer.MIPROV2: {"bootstrap", "minibatch", "full"},
            Optimizer.GEPA: set(),
            Optimizer.CODEX: set(),
        }[self.optimizer]
        if set(self.eval_configs) != expected:
            raise ValueError(
                f"{self.optimizer.value} requires exact Eval Configs "
                f"{sorted(expected)}"
            )
        if (
            self.optimizer is Optimizer.MIPROV2
            and len(
                {config.identity_hash for config in self.eval_configs.values()}
            )
            != 3
        ):
            raise ValueError(
                "MIPROv2 bootstrap, minibatch, and full Eval Configs "
                "must be distinct"
            )
        if self.optimizer is Optimizer.GEPA:
            names = {config.tool_name for config in self.tool_configs}
            if names != {"evaluate_minibatch", "evaluate_subset"}:
                raise ValueError(
                    "GEPA requires exact minibatch and subset Tool Configs"
                )
        elif self.optimizer is Optimizer.CODEX:
            if len(self.tool_configs) != 1:
                raise ValueError("Codex requires one exact Tool Config")
            if not self.tool_configs[0].endpoint.startswith("mcp"):
                raise ValueError("Codex Tool Config must use an MCP endpoint")
        elif self.tool_configs:
            raise ValueError("only GEPA and Codex controls carry Tool Configs")
        if self.optimizer in {Optimizer.COPRO, Optimizer.MIPROV2}:
            if self.proposer_config is None:
                raise ValueError(
                    f"{self.optimizer.value} requires one exact "
                    "Proposer Config"
                )
            if self.reflection_config is not None:
                raise ValueError(
                    "only GEPA controls carry a Reflection Config"
                )
        elif self.optimizer is Optimizer.GEPA:
            if self.reflection_config is None:
                raise ValueError("GEPA requires one exact Reflection Config")
            if self.proposer_config is not None:
                raise ValueError(
                    "GEPA controls carry Reflection Config, not Proposer "
                    "Config"
                )
        elif (
            self.proposer_config is not None
            or self.reflection_config is not None
        ):
            raise ValueError(
                "only COPRO/MIPROv2/GEPA controls carry proposer configuration"
            )
        if self.optimizer is Optimizer.COPRO:
            if len(self.candidates) != 1:
                raise ValueError(
                    "COPRO requires exactly one initial candidate"
                )
            if self.copro_control is None:
                raise ValueError("COPRO requires one exact CoproControl")
            if self.proposer_config != self.copro_control.prompt_model:
                raise ValueError(
                    "COPRO proposer_config must match CoproControl "
                    "prompt_model"
                )
            if self.eval_configs["internal"] != self.copro_control.metric:
                raise ValueError(
                    "COPRO internal Eval Config must match CoproControl metric"
                )
            if self.hyperparameters:
                raise ValueError(
                    "COPRO hyperparameters are owned by CoproControl"
                )
            valid_keys = self.pools.get("valid_template_keys")
            if not isinstance(valid_keys, list):
                raise ValueError("COPRO requires explicit valid_template_keys")
            if set(self.pools) != {"valid_template_keys"}:
                raise ValueError(
                    "COPRO pools contain noncanonical controller state"
                )
            if any(not isinstance(key, str) or not key for key in valid_keys):
                raise ValueError(
                    "COPRO valid_template_keys must be non-empty strings"
                )
            if len(valid_keys) != len(set(valid_keys)):
                raise ValueError("COPRO valid_template_keys must be unique")
        elif self.copro_control is not None:
            raise ValueError("only COPRO controls carry CoproControl")

    @property
    def record(self) -> OptimizationRunControlRecord:
        return OptimizationRunControlRecord(
            run_id=self.run_id,
            optimizer=self.optimizer,
            candidates=tuple(
                candidate_reference(candidate) for candidate in self.candidates
            ),
            output_count=self.output_count,
            budget=self.budget,
            hyperparameters=dict(self.hyperparameters),
            pools=dict(self.pools),
            eval_configs=dict(self.eval_configs),
            tool_configs=self.tool_configs,
            proposer_config=self.proposer_config,
            reflection_config=self.reflection_config,
            copro_control=self.copro_control,
            copro_control_identity_hash=(
                self.copro_control.identity_hash()
                if self.copro_control is not None
                else None
            ),
        )

    @property
    def config_hash(self) -> str:
        return typed_ref_for_record(
            OPTIMIZATION_RUN_CONTROL_SCHEMA,
            self.record.record_content(),
        ).content_hash


class TraceStep(BaseModel):
    """Human-readable projection of one canonical durable step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_ref: TypedRef
    result_ref: TypedRef
    proposal_refs: tuple[TypedRef, ...] = ()
    dispositions: tuple[str, ...] = ()
    evidence_refs: tuple[TypedRef, ...] = ()
    tool_evidence_count: int = 0
    state_ref: TypedRef | None = None
    history_ref: TypedRef | None = None


class CanonicalOptimizationTrace(BaseModel):
    """Reference projection; canonical records remain typed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    optimizer: Optimizer
    optimization_result_ref: TypedRef
    proposals: tuple[TypedRef, ...]
    steps: tuple[TraceStep, ...]


@dataclass(frozen=True, slots=True)
class OptimizationExecution:
    result: OptimizationResult
    result_ref: TypedRef
    trace: CanonicalOptimizationTrace
    trace_ref: TypedRef


@dataclass(frozen=True, slots=True)
class OptimizationRunServices:
    store: ObjectStore
    harness: OptimizationHarness
    evaluation_service: EvaluationService | None = None
    before_paid_step: Callable[[str], None] | None = None


class OptimizationControllerError(RuntimeError):
    """The controller cannot truthfully advance a canonical run."""


@dataclass(frozen=True, slots=True)
class PowerDerivation:
    sampling: EnvSplitSampling
    recommended_n_tasks: int
    recommended_repeats: int
    used_n_tasks: int
    used_repeats: int
    pool_ceiling: int
    note: str


def derive_power_sampling(
    base: EnvSplitSampling,
    power: PowerResult,
    *,
    minimum_n_tasks: int = 1,
) -> PowerDerivation:
    """Materialize a new exact sampling record from a power recommendation."""
    if minimum_n_tasks < 1:
        raise ValueError("minimum_n_tasks must be positive")
    pool_ceiling = min(power.pool_ceiling, len(base.instances))
    recommended = power.recommendation
    used_n = min(
        pool_ceiling,
        max(minimum_n_tasks, recommended.recommended_n_tasks),
    )
    used_repeats = recommended.recommended_repeats
    instances = base.instances[:used_n]
    identities = {
        id(instance): identity
        for instance, identity in zip(
            base.instances,
            base.task_set.task_identities,
            strict=True,
        )
    }
    namespace = base.task_set.manifest_id.removesuffix(f".{base.split_role}")
    sampling = derive_split_sampling(
        namespace=namespace,
        dataset_revision=base.task_set.dataset_revision,
        split_role=base.split_role,
        instances=instances,
        task_identity_of=lambda instance: identities[id(instance)],
        procedure=base.procedure_config,
        aggregation=base.aggregation_config,
        repeats=used_repeats,
    )
    notes: list[str] = []
    if used_n != recommended.recommended_n_tasks:
        notes.append(
            f"tasks clamped/floored from "
            f"{recommended.recommended_n_tasks} to {used_n}"
        )
    if recommended.pool_limited:
        notes.append("pool-limited recommendation")
    return PowerDerivation(
        sampling=sampling,
        recommended_n_tasks=recommended.recommended_n_tasks,
        recommended_repeats=recommended.recommended_repeats,
        used_n_tasks=used_n,
        used_repeats=used_repeats,
        pool_ceiling=pool_ceiling,
        note="; ".join(notes),
    )


def derive_powered_control(
    control: OptimizationRunControl,
    *,
    samplings: Mapping[str, EnvSplitSampling],
    tool_configs: tuple[ToolConfig, ...] = (),
) -> OptimizationRunControl:
    """Replace every algorithm scope with its newly derived exact config."""
    expected = {
        Optimizer.IDENTITY: set(),
        Optimizer.COPRO: {"internal"},
        Optimizer.MIPROV2: {"bootstrap", "minibatch", "full"},
        Optimizer.GEPA: {"evaluate_minibatch", "evaluate_subset"},
        Optimizer.CODEX: {"evaluate_candidate"},
    }[control.optimizer]
    if set(samplings) != expected:
        raise ValueError(
            f"{control.optimizer.value} power derivation requires scopes "
            f"{sorted(expected)}"
        )
    if control.optimizer in {Optimizer.GEPA, Optimizer.CODEX}:
        by_name = {config.tool_name: config for config in tool_configs}
        if set(by_name) != expected:
            raise ValueError(
                "power-derived tool configs must cover every exact scope"
            )
        for name, sampling in samplings.items():
            if (
                by_name[name].eval_config_identity_hash
                != sampling.eval_config.config_identity_hash
            ):
                raise ValueError(
                    f"Tool Config {name!r} does not bind its derived sampling"
                )
        return replace(
            control,
            tool_configs=tool_configs,
        )
    derived_eval_configs = {
        name: eval_config_reference(sampling.eval_config)
        for name, sampling in samplings.items()
    }
    if control.optimizer is Optimizer.COPRO:
        assert control.copro_control is not None
        return replace(
            control,
            eval_configs=derived_eval_configs,
            copro_control=control.copro_control.model_copy(
                update={"metric": derived_eval_configs["internal"]}
            ),
        )
    return replace(control, eval_configs=derived_eval_configs)


def _load(
    store: ObjectStore, reference: TypedRef, model: type[BaseModel]
) -> Any:
    return model.model_validate(store.get(reference.reference))


def _typed_put(
    store: ObjectStore, schema: str, content: dict[str, Any]
) -> TypedRef:
    reference, _ = store.put(schema, content)
    return TypedRef(
        schema_name=reference.schema,
        content_hash=reference.content_hash,
    )


def bind_optimization_control(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
) -> TypedRef:
    """Validate and bind the exact control before any terminal reuse."""
    _validate_adapter_configuration(control, services)
    reference = _typed_put(
        services.store,
        OPTIMIZATION_RUN_CONTROL_SCHEMA,
        control.record.record_content(),
    )
    key = f"{OPTIMIZATION_RUN_CONTROL_SCHEMA}:{control.run_id}"
    try:
        services.store.bind(key, reference.reference)
    except BindingConflictError as conflict:
        raise OptimizationControllerError(
            f"optimization run {control.run_id!r} is already bound to "
            f"control {conflict.existing.content_hash}; refusing "
            f"{reference.content_hash}"
        ) from conflict
    if reference.content_hash != control.config_hash:
        raise OptimizationControllerError(
            "persisted Optimization Run Control failed content validation"
        )
    return reference


def _validate_adapter_configuration(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
) -> None:
    adapter = services.harness.resolve_adapter(control.optimizer.value)
    if control.optimizer in {Optimizer.COPRO, Optimizer.MIPROV2}:
        actual = getattr(adapter, "proposer_config", None)
        if actual != control.proposer_config:
            raise OptimizationControllerError(
                f"{control.optimizer.value} adapter Proposer Config does not "
                "match its serialized run control"
            )
        if control.optimizer is Optimizer.COPRO:
            assert control.copro_control is not None
            if (
                getattr(adapter, "provider_execution_policy_hash", None)
                != control.copro_control.provider_execution_policy_hash
            ):
                raise OptimizationControllerError(
                    "COPRO adapter Provider Execution Policy does not match "
                    "its serialized CoproControl"
                )
            if (
                getattr(adapter, "prompt_adapter_identity_hash", None)
                != control.copro_control.prompt_adapter_identity_hash
            ):
                raise OptimizationControllerError(
                    "COPRO adapter Prompt Adapter does not match its "
                    "serialized CoproControl"
                )
    elif control.optimizer is Optimizer.GEPA:
        if (
            getattr(adapter, "reflection_config", None)
            != control.reflection_config
        ):
            raise OptimizationControllerError(
                "GEPA adapter Reflection Config does not match its serialized "
                "run control"
            )


def _prior_results(
    services: OptimizationRunServices,
    control: OptimizationRunControl,
    control_ref: TypedRef,
) -> list[tuple[OptimizationStepResult, TypedRef]]:
    results: list[tuple[OptimizationStepResult, TypedRef]] = []
    index = 0
    while reference := services.harness.resolve_step_result(
        control.run_id, index
    ):
        result = _load(services.store, reference, OptimizationStepResult)
        request = _load(
            services.store, result.request_ref, OptimizationStepRequest
        )
        if result.run_id != control.run_id or result.step_index != index:
            raise OptimizationControllerError(
                "bound prior Step Result does not match its run and index"
            )
        if (
            request.run_id != control.run_id
            or request.step_index != index
            or request.optimizer_config_hash != control_ref.content_hash
        ):
            raise OptimizationControllerError(
                "prior Step Request does not match the bound Optimization Run "
                "Control"
            )
        results.append((result, reference))
        index += 1
    return results


def _request(
    control: OptimizationRunControl,
    *,
    index: int,
    adapter_key: str,
    mode: StepMode,
    kind: StepKind,
    candidates: tuple[Candidate, ...],
    budget: BudgetState,
    prior: tuple[OptimizationStepResult, TypedRef] | None,
    output_count: int,
    kind_label: str | None = None,
    pools: Mapping[str, Any] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    tool_configs: tuple[ToolConfig, ...] = (),
) -> OptimizationStepRequest:
    prior_result, prior_ref = prior if prior is not None else (None, None)
    return OptimizationStepRequest(
        run_id=control.run_id,
        step_id=f"{control.run_id}:{index}:{kind_label or adapter_key}",
        optimizer_config_hash=control.config_hash,
        adapter_key=adapter_key,
        mode=mode,
        kind=kind,
        kind_label=kind_label,
        step_index=index,
        prior_step_result_ref=prior_ref,
        prior_state_ref=(
            prior_result.state_ref if prior_result is not None else None
        ),
        prior_history_ref=(
            prior_result.history_ref if prior_result is not None else None
        ),
        candidates=candidates,
        pools=dict(pools or {}),
        hyperparameters=dict(hyperparameters or {}),
        budget=budget,
        output_contract=OutputContract(returned_proposal_count=output_count),
        tool_configs=tool_configs,
    )


def _run_step(
    services: OptimizationRunServices,
    request: OptimizationStepRequest,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> OptimizationStepResult:
    if (
        request.mode is not StepMode.PURE
        and services.harness.resolve_step_result(
            request.run_id, request.step_index
        )
        is None
        and services.before_paid_step is not None
    ):
        services.before_paid_step(
            f"optimization:{request.step_index}:"
            f"{request.kind_label or request.adapter_key}"
        )
    result, reference = services.harness.run_step(request)
    prior.append((result, reference))
    return result


def _terminalize(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> OptimizationExecution:
    result, result_ref = services.harness.terminalize(
        run_id=control.run_id,
        step_result_refs=tuple(reference for _, reference in prior),
    )
    steps: list[TraceStep] = []
    for step, step_ref in prior:
        evidence: list[TypedRef] = []
        dispositions: list[str] = []
        for resolution in step.resolved_intents:
            dispositions.append(resolution.outcome.value)
            evidence.extend(resolution.evaluation_evidence_refs)
            if resolution.reward_ref is not None:
                evidence.append(resolution.reward_ref)
        evidence.extend(item.tool_result_ref for item in step.tool_evidence)
        steps.append(
            TraceStep(
                request_ref=step.request_ref,
                result_ref=step_ref,
                proposal_refs=tuple(
                    proposal.record_ref
                    for proposal in step.proposed_candidates
                ),
                dispositions=tuple(dispositions),
                evidence_refs=tuple(evidence),
                tool_evidence_count=len(step.tool_evidence),
                state_ref=step.state_ref,
                history_ref=step.history_ref,
            )
        )
    trace = CanonicalOptimizationTrace(
        run_id=control.run_id,
        optimizer=control.optimizer,
        optimization_result_ref=result_ref,
        proposals=tuple(
            proposal.candidate.record_ref for proposal in result.proposals
        ),
        steps=tuple(steps),
    )
    trace_ref = _typed_put(
        services.store,
        OPTIMIZATION_TRACE_SCHEMA,
        trace.model_dump(mode="json"),
    )
    return OptimizationExecution(result, result_ref, trace, trace_ref)


def _budget(
    control: OptimizationRunControl,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> BudgetState:
    return prior[-1][0].budget if prior else control.budget


def _identity(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> OptimizationExecution:
    if not prior:
        _run_step(
            services,
            _request(
                control,
                index=0,
                adapter_key=Optimizer.IDENTITY.value,
                mode=StepMode.PURE,
                kind=StepKind.IDENTITY,
                candidates=control.candidates,
                budget=control.budget,
                prior=None,
                output_count=len(control.candidates),
            ),
            prior,
        )
    return _terminalize(control, services, prior)


def _copro_round_attempts(
    control: OptimizationRunControl,
    result: OptimizationStepResult,
    store: ObjectStore,
) -> tuple[CoproAttempt, ...]:
    """Load and validate one exact measured COPRO occurrence batch."""

    copro = control.copro_control
    assert copro is not None
    round_index = result.step_index
    if len(result.resolved_intents) != copro.breadth:
        raise OptimizationControllerError(
            "COPRO round did not resolve exactly breadth occurrences"
        )
    resolved_candidates = tuple(
        resolution.intent.candidate for resolution in result.resolved_intents
    )
    if (
        resolved_candidates != result.proposed_candidates
        or resolved_candidates != result.accepted_candidates
    ):
        raise OptimizationControllerError(
            "COPRO resolved occurrences must exactly match the ordered "
            "proposed and accepted candidate batch"
        )
    proposed_hashes = {
        candidate.identity_hash for candidate in result.proposed_candidates
    }
    attempts: list[CoproAttempt] = []
    expected_purpose = (
        "seed_proposal" if round_index == 0 else "history_proposal"
    )
    for offset, resolution in enumerate(result.resolved_intents):
        occurrence_ordinal = round_index * copro.breadth + offset
        expected_intent_id = (
            f"{control.run_id}:{round_index}:{occurrence_ordinal}:"
            f"{resolution.intent.candidate.identity_hash}"
        )
        if resolution.intent.intent_id != expected_intent_id:
            raise OptimizationControllerError(
                "COPRO resolution order conflicts with its occurrence intent"
            )
        if resolution.intent.candidate.identity_hash not in proposed_hashes:
            raise OptimizationControllerError(
                "COPRO resolution candidate is absent from proposed batch"
            )
        if resolution.intent.purpose != expected_purpose:
            raise OptimizationControllerError(
                "COPRO resolution purpose does not match its round"
            )
        if len(resolution.evaluation_evidence_refs) != 1:
            raise OptimizationControllerError(
                "COPRO measured resolution requires one Evaluation Evidence"
            )
        evidence_ref = resolution.evaluation_evidence_refs[0]
        if evidence_ref.schema_name != EVALUATION_EVIDENCE_SCHEMA:
            raise OptimizationControllerError(
                "COPRO resolution cites noncanonical Evaluation Evidence"
            )
        evidence = _load(store, evidence_ref, EvaluationEvidence)
        if (
            typed_ref_for_record(
                EVALUATION_EVIDENCE_SCHEMA,
                evidence.record_content(),
            )
            != evidence_ref
        ):
            raise OptimizationControllerError(
                "COPRO Evaluation Evidence content does not match its ref"
            )
        if (
            evidence.candidate != resolution.intent.candidate
            or evidence.eval_config != copro.metric
            or evidence.evaluation_role is not EvaluationRole.INTERNAL
            or evidence.evaluation_context_id != resolution.intent.intent_id
            or evidence.purpose != resolution.intent.purpose
            or evidence.reward_ref != resolution.reward_ref
        ):
            raise OptimizationControllerError(
                "COPRO Evaluation Evidence conflicts with its resolution"
            )
        if resolution.reward_ref is None:
            raise OptimizationControllerError(
                "COPRO measured resolution carries no Reward"
            )
        reward = _load(store, resolution.reward_ref, Reward)
        if (
            reward.evidence_ref_content_hash
            != evidence.aggregate_ref.content_hash
        ):
            raise OptimizationControllerError(
                "COPRO Reward does not cite the evidence aggregate"
            )
        if not any(
            citation.name == evidence.aggregate_name
            and citation.value == evidence.aggregate_value
            for citation in reward.input_citations
        ):
            raise OptimizationControllerError(
                "COPRO Reward does not cite its Evaluation Evidence aggregate"
            )
        try:
            attempt = CoproAttempt.from_resolution(
                occurrence_ordinal=occurrence_ordinal,
                round_index=round_index,
                resolution=resolution,
                reward=reward,
                expected_run_id=control.run_id,
                expected_eval_config=copro.metric,
                expected_reward_policy_hash=copro.reward_policy_hash,
            )
        except ValueError as exc:
            raise OptimizationControllerError(
                "COPRO measured occurrence failed provenance validation"
            ) from exc
        attempts.append(attempt)
    return tuple(attempts)


def _validate_copro_round_request(
    control: OptimizationRunControl,
    result: OptimizationStepResult,
    request: OptimizationStepRequest,
    *,
    round_index: int,
    proposal_mode: str,
    attempts: tuple[CoproAttempt, ...],
    prior: tuple[OptimizationStepResult, TypedRef] | None,
) -> None:
    copro = control.copro_control
    assert copro is not None
    expected_pools = {
        **dict(control.pools),
        "attempt_history": [
            attempt.model_dump(mode="json") for attempt in attempts
        ],
    }
    prior_result, prior_ref = prior if prior is not None else (None, None)
    if (
        result.step_id != f"{control.run_id}:{round_index}:{proposal_mode}"
        or request.step_id != f"{control.run_id}:{round_index}:{proposal_mode}"
        or request.adapter_key != Optimizer.COPRO.value
        or request.mode is not StepMode.PROPOSAL_ONLY
        or request.kind is not StepKind.PROPOSAL
        or request.kind_label != proposal_mode
        or request.step_index != round_index
        or request.candidates != control.candidates
        or request.output_contract
        != OutputContract(returned_proposal_count=copro.breadth)
        or request.hyperparameters
        != copro.step_hyperparameters(iteration=round_index)
        or request.pools != expected_pools
        or request.budget
        != (
            prior_result.budget if prior_result is not None else control.budget
        )
        or request.prior_step_result_ref != prior_ref
        or request.prior_state_ref
        != (prior_result.state_ref if prior_result is not None else None)
        or request.prior_history_ref
        != (prior_result.history_ref if prior_result is not None else None)
        or request.tool_configs
    ):
        raise OptimizationControllerError(
            "COPRO prior Step Request does not match reconstructed round plan"
        )


def _copro_finalization_pools(
    state: CoproState,
    finalization: CoproFinalization,
) -> dict[str, Any]:
    return {
        "attempt_history": [
            attempt.model_dump(mode="json") for attempt in state.attempts
        ],
        "copro_finalization": finalization.model_dump(
            mode="json",
            exclude_none=True,
        ),
    }


def _copro(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> OptimizationExecution:
    copro = control.copro_control
    assert copro is not None
    driver = CoproDriver(
        CoproConfig(
            breadth=copro.breadth,
            depth=copro.depth,
            init_temperature=copro.init_temperature,
            track_stats=copro.track_stats,
        )
    )
    state = driver.initial_state(control.candidates[0])

    proposal_results = prior[: min(len(prior), copro.depth)]
    for round_index, (result, _) in enumerate(proposal_results):
        if result.step_index != round_index:
            raise OptimizationControllerError(
                "COPRO prior round index is not contiguous"
            )
        plan = driver.advance(state)
        request = _load(
            services.store, result.request_ref, OptimizationStepRequest
        )
        _validate_copro_round_request(
            control,
            result,
            request,
            round_index=round_index,
            proposal_mode=plan.proposal_mode,
            attempts=state.attempts,
            prior=prior[round_index - 1] if round_index else None,
        )
        if result.status is StepStatus.FAILED:
            if round_index != len(prior) - 1:
                raise OptimizationControllerError(
                    "COPRO has steps after a failed proposal round"
                )
            return _terminalize(control, services, prior)
        if result.status is not StepStatus.CONTINUE:
            raise OptimizationControllerError(
                "successful COPRO proposal rounds must continue"
            )
        state = driver.fold_round(
            state,
            _copro_round_attempts(control, result, services.store),
        )

    if len(prior) > copro.depth:
        if len(prior) != copro.depth + 1:
            raise OptimizationControllerError(
                "COPRO has unexpected steps after final projection"
            )
        final_step = prior[-1][0]
        projection_prior_result, projection_prior_ref = prior[copro.depth - 1]
        if (
            final_step.step_index != copro.depth
            or final_step.step_id
            != f"{control.run_id}:{copro.depth}:copro_finalize"
            or final_step.status is not StepStatus.COMPLETE
            or final_step.resolved_intents
            or final_step.tool_evidence
            or final_step.state_ref is not None
            or final_step.history_ref is not None
            or final_step.budget_delta != BudgetDelta()
            or final_step.budget != projection_prior_result.budget
        ):
            raise OptimizationControllerError(
                "COPRO terminal projection is not one pure completed step"
            )
        final = driver.finalize(state)
        final_request = _load(
            services.store,
            final_step.request_ref,
            OptimizationStepRequest,
        )
        expected_candidates = tuple(
            attempt.candidate for attempt in final.ranked_attempts
        )
        expected_records = tuple(
            attempt.candidate.record for attempt in final.ranked_attempts
        )
        if (
            final_step.accepted_candidates != expected_candidates
            or final_step.proposed_candidates != expected_candidates
            or final_request.adapter_key != Optimizer.IDENTITY.value
            or final_request.mode is not StepMode.PURE
            or final_request.kind is not StepKind.IDENTITY
            or final_request.kind_label != "copro_finalize"
            or final_request.step_id
            != f"{control.run_id}:{copro.depth}:copro_finalize"
            or final_request.candidates != expected_records
            or final_request.output_contract
            != OutputContract(returned_proposal_count=len(expected_records))
            or final_request.hyperparameters
            or final_request.budget != projection_prior_result.budget
            or final_request.prior_step_result_ref != projection_prior_ref
            or final_request.prior_state_ref
            != projection_prior_result.state_ref
            or final_request.prior_history_ref
            != projection_prior_result.history_ref
            or final_request.tool_configs
        ):
            raise OptimizationControllerError(
                "COPRO terminal projection is not the full unique ranking"
            )
        expected_pools = _copro_finalization_pools(state, final)
        if final_request.pools != expected_pools:
            raise OptimizationControllerError(
                "COPRO terminal pools do not match reconstructed finalization"
            )
        return _terminalize(control, services, prior)

    while state.completed_rounds < copro.depth:
        plan = driver.advance(state)
        index = state.completed_rounds
        result = _run_step(
            services,
            _request(
                control,
                index=index,
                adapter_key=Optimizer.COPRO.value,
                mode=StepMode.PROPOSAL_ONLY,
                kind=StepKind.PROPOSAL,
                kind_label=plan.proposal_mode,
                candidates=control.candidates,
                budget=_budget(control, prior),
                prior=prior[-1] if prior else None,
                output_count=copro.breadth,
                pools={
                    **dict(control.pools),
                    "attempt_history": [
                        attempt.model_dump(mode="json")
                        for attempt in state.attempts
                    ],
                },
                hyperparameters=copro.step_hyperparameters(iteration=index),
            ),
            prior,
        )
        if result.status is StepStatus.FAILED:
            return _terminalize(control, services, prior)
        if result.status is not StepStatus.CONTINUE:
            raise OptimizationControllerError(
                "successful COPRO proposal rounds must continue"
            )
        state = driver.fold_round(
            state,
            _copro_round_attempts(control, result, services.store),
        )

    final = driver.finalize(state)
    winners = tuple(
        attempt.candidate.record for attempt in final.ranked_attempts
    )
    if not winners:
        raise OptimizationControllerError(
            "COPRO produced no measured terminal candidates"
        )
    if len(prior) == copro.depth:
        _run_step(
            services,
            _request(
                control,
                index=copro.depth,
                adapter_key=Optimizer.IDENTITY.value,
                mode=StepMode.PURE,
                kind=StepKind.IDENTITY,
                kind_label="copro_finalize",
                candidates=winners,
                budget=_budget(control, prior),
                prior=prior[-1],
                output_count=len(winners),
                pools=_copro_finalization_pools(state, final),
            ),
            prior,
        )
    return _terminalize(control, services, prior)


def _state_delta(
    store: ObjectStore, result: OptimizationStepResult
) -> dict[str, Any]:
    if result.state_ref is None:
        return {}
    raw = store.get(result.state_ref.reference)
    if not isinstance(raw, dict):
        raise OptimizationControllerError("state snapshot is not an object")
    return raw


def _miprov2(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> OptimizationExecution:
    driver = Miprov2Driver(services.store)
    state = dict(control.pools)
    for result, _ in prior:
        state = driver.advance(
            state,
            AdapterOutput(state_delta=_state_delta(services.store, result)),
            result.resolved_intents,
        )
    while True:
        if prior and prior[-1][0].status is StepStatus.FAILED:
            return _terminalize(control, services, prior)
        plan = driver.next_plan(state, dict(control.hyperparameters))
        index = len(prior)
        hyperparameters = {
            **dict(control.hyperparameters),
            "bootstrap_eval_config": control.eval_configs[
                "bootstrap"
            ].model_dump(mode="json"),
            "minibatch_eval_config": control.eval_configs[
                "minibatch"
            ].model_dump(mode="json"),
            "full_eval_config": control.eval_configs["full"].model_dump(
                mode="json"
            ),
            "returned_proposal_count": control.output_count,
        }
        result = _run_step(
            services,
            _request(
                control,
                index=index,
                adapter_key=MIPROV2_ADAPTER_KEY,
                mode=StepMode.PROPOSAL_ONLY,
                kind=StepKind.PROPOSAL,
                kind_label=plan.kind,
                candidates=control.candidates,
                budget=_budget(control, prior),
                prior=prior[-1] if prior else None,
                output_count=plan.returned_proposal_count,
                pools=state,
                hyperparameters=hyperparameters,
            ),
            prior,
        )
        state = driver.advance(
            state,
            AdapterOutput(state_delta=_state_delta(services.store, result)),
            result.resolved_intents,
        )
        if plan.kind == COMPLETION or result.status is StepStatus.FAILED:
            return _terminalize(control, services, prior)


def _gepa(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> OptimizationExecution:
    pools = dict(control.pools)
    for result, _ in prior:
        pools.update(_state_delta(services.store, result))
    while not prior or prior[-1][0].status is StepStatus.CONTINUE:
        result = _run_step(
            services,
            _request(
                control,
                index=len(prior),
                adapter_key=Optimizer.GEPA.value,
                mode=StepMode.TOOL_USING,
                kind=StepKind.TOOL,
                candidates=control.candidates,
                budget=_budget(control, prior),
                prior=prior[-1] if prior else None,
                output_count=control.output_count,
                pools=pools,
                hyperparameters={
                    **dict(control.hyperparameters),
                    "returned_proposal_count": control.output_count,
                },
                tool_configs=control.tool_configs,
            ),
            prior,
        )
        pools.update(_state_delta(services.store, result))
    return _terminalize(control, services, prior)


def _codex(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
    prior: list[tuple[OptimizationStepResult, TypedRef]],
) -> OptimizationExecution:
    if not prior:
        _run_step(
            services,
            _request(
                control,
                index=0,
                adapter_key=Optimizer.CODEX.value,
                mode=StepMode.TOOL_USING,
                kind=StepKind.TOOL,
                candidates=control.candidates,
                budget=control.budget,
                prior=None,
                output_count=control.output_count,
                pools=control.pools,
                hyperparameters=control.hyperparameters,
                tool_configs=control.tool_configs,
            ),
            prior,
        )
    return _terminalize(control, services, prior)


def run_optimization(
    control: OptimizationRunControl,
    services: OptimizationRunServices,
) -> OptimizationExecution:
    """Advance or resume one algorithm through canonical harness records."""
    control_ref = bind_optimization_control(control, services)
    prior = _prior_results(services, control, control_ref)
    existing = services.harness.resolve_optimization_result(control.run_id)
    if existing is not None:
        return _terminalize(control, services, prior)
    handlers = {
        Optimizer.IDENTITY: _identity,
        Optimizer.COPRO: _copro,
        Optimizer.MIPROV2: _miprov2,
        Optimizer.GEPA: _gepa,
        Optimizer.CODEX: _codex,
    }
    return handlers[control.optimizer](control, services, prior)


__all__ = [
    "OPTIMIZATION_RUN_CONTROL_SCHEMA",
    "OPTIMIZATION_TRACE_SCHEMA",
    "CanonicalOptimizationTrace",
    "OptimizationControllerError",
    "OptimizationExecution",
    "OptimizationRunControl",
    "OptimizationRunControlRecord",
    "OptimizationRunServices",
    "Optimizer",
    "TraceStep",
    "bind_optimization_control",
    "run_optimization",
]
