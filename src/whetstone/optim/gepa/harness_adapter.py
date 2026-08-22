from __future__ import annotations

from typing import Any

from whetstone.core.identity import ImmutableJsonObject, canonical_json
from whetstone.core.leasing import ReplayPolicy
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import (
    OptimStepRequest,
    SearchEvidence,
    StepMode,
    StepStatus,
)
from whetstone.optim.gepa.adapter import project_gepa_terminal
from whetstone.optim.gepa.authorities import (
    CanonicalGepaCandidateAssembler,
    GepaCandidateFieldBinding,
)
from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaDataInstance,
    GepaSkippedMutation,
)
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.engine import GepaEngineAdapter
from whetstone.coordination.eval_service import (
    EvalEngineService,
    EvalExecutionContext,
    EvalPlatformDeferred,
)
from whetstone.optim.gepa.step_engine import (
    GEPA_STATE_KEY,
    GepaStepCheckpoint,
    load_gepa_checkpoint,
    run_one_gepa_iteration,
)

GEPA_ADAPTER_KEY = "gepa"
GEPA_TERMINAL_ARTIFACT_KEY = "terminal_artifact_ref"
#: State-delta key holding the reflection responses this Step's search
#: rejected. Present on every Step, terminal or not, so a skip is durable on
#: the Step it happened rather than only in the terminal transcript.
GEPA_SKIPPED_MUTATIONS_KEY = "skipped_mutations"


def _prefix_skipped_mutations(
    request: OptimStepRequest,
) -> tuple[GepaSkippedMutation, ...]:
    raw = dict(request.pools).get(GEPA_SKIPPED_MUTATIONS_KEY, [])
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        GepaSkippedMutation.model_validate(
            item.to_json() if hasattr(item, "to_json") else item
        )
        for item in raw
    )


def _state_delta(
    *,
    checkpoint: GepaStepCheckpoint,
    skipped: tuple[GepaSkippedMutation, ...],
) -> ImmutableJsonObject:
    return ImmutableJsonObject(
        {
            GEPA_STATE_KEY: checkpoint.model_dump(mode="json"),
            # Durable on this Step, not only on the terminal transcript:
            # a process death after a non-terminal skip must not lose it.
            GEPA_SKIPPED_MUTATIONS_KEY: [
                item.model_dump(mode="json") for item in skipped
            ],
        }
    )


def _union_skipped_mutations(
    prefix: tuple[GepaSkippedMutation, ...],
    produced: tuple[GepaSkippedMutation, ...],
) -> tuple[GepaSkippedMutation, ...]:
    seen = {
        canonical_json(item.model_dump(mode="json")) for item in prefix
    }
    merged = list(prefix)
    for item in produced:
        key = canonical_json(item.model_dump(mode="json"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return tuple(merged)


def gepa_candidate_field_name(
    *,
    component_name: str,
    component_names: tuple[str, ...],
    mutation_field: str,
) -> str:
    if len(component_names) == 1:
        return mutation_field
    return component_name


def seed_components_from_candidate(
    candidate: Candidate,
    *,
    component_names: tuple[str, ...],
    mutation_field: str,
) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for name in component_names:
        field = gepa_candidate_field_name(
            component_name=name,
            component_names=component_names,
            mutation_field=mutation_field,
        )
        try:
            value = candidate.payload[field]
        except KeyError as exc:
            raise ValueError(
                "GEPA launch candidate is missing seed component "
                f"{name!r} at payload field {field!r}"
            ) from exc
        if not isinstance(value, str):
            raise ValueError(
                "GEPA launch candidate seed component "
                f"{name!r} must be a string"
            )
        mapped[name] = value
    return mapped


class GepaHarnessAdapterFactory:
    def __init__(self, *, factory: Any) -> None:
        self._factory = factory

    def create(self, *, control: GepaControl) -> GepaEngineAdapter:
        return self._factory.create(control=control)

    def begin_step(self, *, step_index: int) -> None:
        """Bind this Step and drop evidence from earlier Steps."""
        self._factory.begin_step(step_index=step_index)

    def search_evidence(
        self,
        *,
        run_id: str,
        step_index: int,
    ) -> tuple[SearchEvidence, ...]:
        """Evidence for every evaluation this Step's search drove."""
        return tuple(
            self._factory.search_evidence(
                run_id=run_id,
                step_index=step_index,
            )
        )

    def skipped_mutations(self) -> tuple[GepaSkippedMutation, ...]:
        """Reflection responses this Step's search rejected, in order."""
        return tuple(self._factory.skipped_mutations())

    def bind_eval_context(self, context: EvalExecutionContext) -> None:
        binder = getattr(self._factory, "bind_eval_context", None)
        if callable(binder):
            binder(context)

    def bind_evaluation_service(self, service: EvalEngineService) -> None:
        binder = getattr(self._factory, "bind_evaluation_service", None)
        if callable(binder):
            binder(service)

    def persist_result(
        self,
        *,
        control: GepaControl,
        adapter: GepaEngineAdapter,
        detailed_result: Any,
    ) -> Any:
        return self._factory.persist_result(
            control=control,
            adapter=adapter,
            detailed_result=detailed_result,
        )


class GepaHarnessAdapter:
    def __init__(
        self,
        *,
        control: GepaControl,
        seed_candidate: dict[str, str],
        trainset: tuple[GepaDataInstance, ...],
        valset: tuple[GepaDataInstance, ...] | None,
        adapter_factory: GepaHarnessAdapterFactory,
    ) -> None:
        ordered_seed = dict(seed_candidate)
        if tuple(ordered_seed) != control.component_names:
            raise ValueError(
                "seed_candidate component order conflicts with GepaControl"
            )
        self._control = control
        self._seed_candidate = ordered_seed
        self._trainset = trainset
        self._valset = valset
        self._adapter_factory = adapter_factory
        self.invocations = 0

    @property
    def key(self) -> str:
        return GEPA_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.DURABLE_WORKFLOW

    @property
    def control(self) -> GepaControl:
        return self._control

    @property
    def seed_candidate(self) -> dict[str, str]:
        return dict(self._seed_candidate)

    def bind_eval_context(self, context: EvalExecutionContext) -> None:
        self._adapter_factory.bind_eval_context(context)

    def bind_evaluation_service(self, service: EvalEngineService) -> None:
        self._adapter_factory.bind_evaluation_service(service)

    def invoke(
        self,
        request: OptimStepRequest,
        handles: tuple[Any, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        if handles:
            raise ValueError("GEPA receives no Runtime Tool Handles")
        if request.run.record.optimizer_config != self._control.reference():
            raise ValueError("GEPA run optimizer_config does not bind exact control")
        hyper = dict(request.hyperparameters)
        iteration = int(hyper.get("round_index", request.step_index))
        if iteration != request.step_index:
            raise ValueError("GEPA round_index must equal step_index")
        checkpoint = load_gepa_checkpoint(request)
        prefix_skipped = _prefix_skipped_mutations(request)
        self._adapter_factory.begin_step(step_index=int(request.step_index))
        engine_adapter = self._adapter_factory.create(control=self._control)
        try:
            detailed, checkpoint = run_one_gepa_iteration(
                control=self._control,
                seed_candidate=self._seed_candidate,
                trainset=self._trainset,
                valset=self._valset,
                adapter=engine_adapter,
                checkpoint=checkpoint,
            )
        except EvalPlatformDeferred as deferred:
            intent = deferred.intent
            return AdapterOutput(
                proposed_status=StepStatus.CONTINUE,
                optim_eval_requests=() if intent is None else (intent,),
                state_delta=_state_delta(
                    checkpoint=load_gepa_checkpoint(request),
                    skipped=prefix_skipped,
                ),
            )
        state_delta = _state_delta(
            checkpoint=checkpoint,
            skipped=_union_skipped_mutations(
                prefix_skipped,
                self._adapter_factory.skipped_mutations(),
            ),
        )
        search_evidence = self._adapter_factory.search_evidence(
            run_id=str(request.run_id),
            step_index=int(request.step_index),
        )
        if checkpoint.terminal:
            artifact_ref = self._adapter_factory.persist_result(
                control=self._control,
                adapter=engine_adapter,
                detailed_result=detailed,
            )
            terminal = project_gepa_terminal(
                control=self._control,
                detailed_result=detailed,
                artifact_ref=artifact_ref,
            )
            mutation_field = request.run.record.mutation_field
            base_ref = candidate_reference(request.candidates[0])
            field_bindings = tuple(
                GepaCandidateFieldBinding(
                    component_name=name,
                    candidate_field=gepa_candidate_field_name(
                        component_name=name,
                        component_names=self._control.component_names,
                        mutation_field=mutation_field,
                    ),
                )
                for name in self._control.component_names
            )
            assembler = CanonicalGepaCandidateAssembler(
                base_candidate=base_ref,
                fields=field_bindings,
            )
            history_delta = ImmutableJsonObject(
                {GEPA_TERMINAL_ARTIFACT_KEY: artifact_ref.model_dump(mode="json")}
            )
            if terminal.best_candidate == self._seed_candidate:
                # GEPA searched and accepted nothing better than the seed.
                # That is a clean completion, not a failure, and it proposes
                # no candidate the run can carry forward.
                return AdapterOutput(
                    proposed_status=StepStatus.COMPLETE,
                    seed_retained=True,
                    retained_candidate=request.candidates[0],
                    search_evidence=search_evidence,
                    state_delta=state_delta,
                    history_delta=history_delta,
                    budget_delta=checkpoint.budget_delta,
                )
            components = tuple(
                GepaCandidateComponent(name=name, text=terminal.best_candidate[name])
                for name in self._control.component_names
            )
            assembled = assembler.assemble(components).record
            candidate = assembled.model_copy(
                update={
                    "candidate_id": f"{request.run_id}:gepa:best",
                    "base_ref": base_ref.record_ref,
                }
            )
            return AdapterOutput(
                accepted_candidates=(candidate,),
                proposed_candidates=(candidate,),
                proposed_status=StepStatus.COMPLETE,
                search_evidence=search_evidence,
                state_delta=state_delta,
                history_delta=history_delta,
                budget_delta=checkpoint.budget_delta,
            )
        return AdapterOutput(
            proposed_status=StepStatus.CONTINUE,
            search_evidence=search_evidence,
            state_delta=state_delta,
            budget_delta=checkpoint.budget_delta,
        )


__all__ = [
    "GEPA_ADAPTER_KEY",
    "GEPA_SKIPPED_MUTATIONS_KEY",
    "GEPA_TERMINAL_ARTIFACT_KEY",
    "GepaHarnessAdapter",
    "GepaHarnessAdapterFactory",
    "gepa_candidate_field_name",
    "seed_components_from_candidate",
]
