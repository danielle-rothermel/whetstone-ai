from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.coordination.step_contracts import (
    resolve_step_contract_provider,
)
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import TypedRef, compute_identity_hash, require_full_hash
from whetstone.experiment.candidate import Candidate, candidate_reference
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
    from whetstone.optim.gepa.control import GepaControl
    from whetstone.optim.miprov2.control import Miprov2Control

RUN_LAUNCH_SCHEMA = "whetstone.optim_run_launch"
RUN_LAUNCH_SCHEMA_VERSION = 1
RUN_LAUNCH_BINDING_PREFIX = "whetstone.optim_run_launch:"
RUN_WORKFLOW_SCHEMA = "whetstone.coordination.parent_run"
RUN_WORKFLOW_SCHEMA_VERSION = 1


class RunRequest(BaseModel):
    """Complete serializable input identifying one optimization run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    controller_identity_hash: StrictStr
    run_id: StrictStr
    control_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RunRequest:
        require_full_hash(
            self.controller_identity_hash,
            field="controller_identity_hash",
        )
        require_full_hash(
            self.control_identity_hash,
            field="control_identity_hash",
        )
        if not self.run_id:
            raise ValueError("run request run_id must be non-empty")
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=RUN_WORKFLOW_SCHEMA,
            schema_version=RUN_WORKFLOW_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


@dataclass(frozen=True, slots=True)
class OptimRunLaunch:
    run: OptimRun
    initial_candidate: Candidate
    control: CoproControl | GepaControl | Miprov2Control | None = None


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
        seed_ref = launch.run.initial_candidate_ref
        if seed_ref is not None and seed_ref != candidate_reference(
            launch.initial_candidate
        ):
            raise ValueError(
                "run initial_candidate_ref must address the exact launch "
                "initial candidate"
            )
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

    def load_launch(self, run_id: str) -> OptimRunLaunch:
        return load_launch(self._store, run_id)

    def drive(self, request: RunRequest) -> TypedRef:
        require_full_hash(
            request.control_identity_hash,
            field="control_identity_hash",
        )
        launch = self.load_launch(request.run_id)
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
        provider = resolve_step_contract_provider(adapter_key)
        if control is None and provider.requires_control():
            raise ValueError(
                f"{adapter_key!r} run launch requires the exact control"
            )
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


def load_launch(store: ObjectStore, run_id: str) -> OptimRunLaunch:
    run_binding = store.resolve(f"{RUN_LAUNCH_BINDING_PREFIX}{run_id}")
    if run_binding is None:
        raise ValueError(f"optimization run launch is not bound: {run_id!r}")
    run = OptimRun.model_validate(store.get(run_binding))
    payload_binding = store.resolve(
        f"{RUN_LAUNCH_BINDING_PREFIX}{run_id}:payload"
    )
    if payload_binding is None:
        raise ValueError(
            f"optimization run launch payload is not bound: {run_id!r}"
        )
    record = store.get(payload_binding)
    candidate = Candidate.model_validate(record["initial_candidate"])
    control = None
    if record.get("control") is not None:
        control = resolve_step_contract_provider(
            run.adapter_key
        ).parse_control(record["control"])
    return OptimRunLaunch(
        run=run,
        initial_candidate=candidate,
        control=control,
    )


__all__ = [
    "HarnessRunController",
    "OptimRunLaunch",
    "RUN_LAUNCH_BINDING_PREFIX",
    "RUN_LAUNCH_SCHEMA",
    "RUN_LAUNCH_SCHEMA_VERSION",
    "RUN_WORKFLOW_SCHEMA",
    "RUN_WORKFLOW_SCHEMA_VERSION",
    "RunRequest",
    "load_launch",
]
