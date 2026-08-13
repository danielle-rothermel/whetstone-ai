from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_store import ObjectStore

from whetstone.coordination.run_workflow import RunRequest
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import TypedRef, require_full_hash
from whetstone.experiment.candidate import Candidate
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OPTIM_RUN_SCHEMA,
    OptimRun,
    OptimRunRef,
    OptimStepResult,
    OptimStepResultRef,
    StepStatus,
    optimization_run_reference,
    step_result_reference,
)
from whetstone.optim.harness import OptimHarness

if TYPE_CHECKING:
    from whetstone.optim.copro.control import CoproControl

RUN_LAUNCH_SCHEMA = "whetstone.optim_run_launch"
RUN_LAUNCH_SCHEMA_VERSION = 1
RUN_LAUNCH_BINDING_PREFIX = "whetstone.optim_run_launch:"


@dataclass(frozen=True, slots=True)
class OptimRunLaunch:
    run: OptimRun
    initial_candidate: Candidate
    control: CoproControl | None = None


class HarnessRunController:
    """Drive one optimization run through the shared harness loop."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        harness: OptimHarness,
        runtime_hash: str,
        step_builder: StepRequestBuilder | None = None,
    ) -> None:
        require_full_hash(runtime_hash, field="controller_runtime_hash")
        self._store = store
        self._harness = harness
        self._runtime_hash = runtime_hash
        self._step_builder = step_builder or StepRequestBuilder(store=store)

    @property
    def runtime_hash(self) -> str:
        return self._runtime_hash

    def bind_launch(self, launch: OptimRunLaunch) -> OptimRunRef:
        run_ref = optimization_run_reference(launch.run)
        self._store.put(OPTIM_RUN_SCHEMA, run_ref.record.record_content())
        payload_ref, _ = self._store.put(
            RUN_LAUNCH_SCHEMA,
            {
                "schema_version": RUN_LAUNCH_SCHEMA_VERSION,
                "run_id": launch.run.run_id,
                "initial_candidate": launch.initial_candidate.model_dump(
                    mode="json"
                ),
                "control": (
                    None
                    if launch.control is None
                    else launch.control.model_dump(mode="json")
                ),
            },
        )
        self._store.bind(
            f"{RUN_LAUNCH_BINDING_PREFIX}{launch.run.run_id}",
            run_ref.record_ref.reference,
        )
        self._store.bind(
            f"{RUN_LAUNCH_BINDING_PREFIX}{launch.run.run_id}:payload",
            payload_ref,
        )
        return run_ref

    def _load_launch(self, run_id: str) -> OptimRunLaunch:
        run_binding = self._store.resolve(
            f"{RUN_LAUNCH_BINDING_PREFIX}{run_id}"
        )
        if run_binding is None:
            raise ValueError(
                f"optimization run launch is not bound: {run_id!r}"
            )
        run = OptimRun.model_validate(self._store.get(run_binding))
        payload_binding = self._store.resolve(
            f"{RUN_LAUNCH_BINDING_PREFIX}{run_id}:payload"
        )
        if payload_binding is None:
            raise ValueError(
                f"optimization run launch payload is not bound: {run_id!r}"
            )
        record = self._store.get(payload_binding)
        candidate = Candidate.model_validate(record["initial_candidate"])
        control = None
        if record.get("control") is not None:
            from whetstone.optim.copro.control import CoproControl

            control = CoproControl.model_validate(record["control"])
        return OptimRunLaunch(
            run=run,
            initial_candidate=candidate,
            control=control,
        )

    def drive(self, request: RunRequest) -> TypedRef:
        require_full_hash(
            request.control_identity_hash,
            field="control_identity_hash",
        )
        launch = self._load_launch(request.run_id)
        if launch.control is not None:
            if launch.control.identity_hash() != request.control_identity_hash:
                raise ValueError(
                    "run control_identity_hash does not match launch control"
                )
        elif launch.run.optimizer_config.record_hash != request.control_identity_hash:
            raise ValueError(
                "run control_identity_hash does not match launch run config"
            )
        bound = self._harness.bind_run(launch.run)
        adapter_key = bound.record.adapter_key
        control = launch.control
        if control is None and adapter_key == "copro":
            raise ValueError("COPRO run launch requires the exact control")
        step_results: list[OptimStepResultRef] = []
        prior_results: list[OptimStepResult] = []
        step_request = self._step_builder.build_first(
            run=bound,
            adapter_key=adapter_key,
            initial_candidate=launch.initial_candidate,
            control=control,
        )
        while True:
            result, result_ref = self._harness.run_step(step_request)
            step_results.append(step_result_reference(result))
            if result.status is not StepStatus.CONTINUE:
                break
            prior_results.append(result)
            if control is None:
                raise ValueError(
                    "continuing run requires a bound optimizer control"
                )
            step_request = self._step_builder.build_next(
                prior=result,
                prior_ref=result_ref,
                prior_results=tuple(prior_results),
                control=control,
                mutation_field=str(bound.record.mutation_field),
            )
        _terminal, terminal_ref = self._harness.terminalize(
            run=bound,
            step_results=tuple(step_results),
        )
        if terminal_ref.schema_name != OPTIM_RESULT_SCHEMA:
            raise ValueError(
                "terminalize must return an Optimization Result ref"
            )
        return terminal_ref


__all__ = [
    "HarnessRunController",
    "OptimRunLaunch",
    "RUN_LAUNCH_BINDING_PREFIX",
    "RUN_LAUNCH_SCHEMA",
    "RUN_LAUNCH_SCHEMA_VERSION",
]
