from __future__ import annotations

from typing import Any

from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import ImmutableJsonObject
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import OptimStepRequest, StepMode, StepStatus
from whetstone.optim.gepa.adapter import project_gepa_terminal
from whetstone.optim.gepa.authorities import (
    CanonicalGepaCandidateAssembler,
    GepaCandidateFieldBinding,
)
from whetstone.optim.gepa.contracts import GepaCandidateComponent, GepaDataInstance
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.engine import GepaEngineAdapter
from whetstone.optim.gepa.step_engine import (
    GEPA_STATE_KEY,
    load_gepa_checkpoint,
    run_one_gepa_iteration,
)

GEPA_ADAPTER_KEY = "gepa"
GEPA_TERMINAL_ARTIFACT_KEY = "terminal_artifact_ref"


class GepaHarnessAdapterFactory:
    def __init__(self, *, factory: Any) -> None:
        self._factory = factory

    def create(self, *, control: GepaControl) -> GepaEngineAdapter:
        return self._factory.create(control=control, effect_broker="harness")

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
        self._control = control
        self._seed_candidate = dict(seed_candidate)
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
        engine_adapter = self._adapter_factory.create(control=self._control)
        detailed, checkpoint = run_one_gepa_iteration(
            control=self._control,
            seed_candidate=self._seed_candidate,
            trainset=self._trainset,
            valset=self._valset,
            adapter=engine_adapter,
            checkpoint=checkpoint,
        )
        state_delta = ImmutableJsonObject(
            {GEPA_STATE_KEY: checkpoint.model_dump(mode="json")}
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
                    candidate_field=(
                        mutation_field
                        if len(self._control.component_names) == 1
                        else name
                    ),
                )
                for name in self._control.component_names
            )
            assembler = CanonicalGepaCandidateAssembler(
                base_candidate=base_ref,
                fields=field_bindings,
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
                state_delta=state_delta,
                history_delta=ImmutableJsonObject(
                    {GEPA_TERMINAL_ARTIFACT_KEY: artifact_ref.model_dump(mode="json")}
                ),
                budget_delta=checkpoint.budget_delta,
            )
        return AdapterOutput(
            proposed_status=StepStatus.CONTINUE,
            state_delta=state_delta,
            budget_delta=checkpoint.budget_delta,
        )


__all__ = [
    "GEPA_ADAPTER_KEY",
    "GEPA_STATE_KEY",
    "GEPA_TERMINAL_ARTIFACT_KEY",
    "GepaHarnessAdapter",
    "GepaHarnessAdapterFactory",
]
