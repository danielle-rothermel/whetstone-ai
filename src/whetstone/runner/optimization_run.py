"""The durable run control and the controller that drives one optimizer.

One :class:`OptimizationRunControl` is the complete, content-hashed input to
one optimization run: the exact run envelope, the initial candidates, the
starting budget, and the adapter the harness invokes. Its
``identity_hash()`` is what the parent run workflow keys on, so a changed
control under a bound ``run_id`` is a different workflow and can never
silently resume a differently-configured run.

:class:`HarnessRunController` drives the run: ``bind_run`` once, then
``run_step`` until a step reports a non-continuing status, accumulating the
ordered step-result references that ``terminalize`` requires. It is
replay-safe by construction, because every one of those harness calls is
idempotent on identical content -- a recovered parent workflow re-enters
``drive`` from the top and resolves the already-durable results rather than
re-executing them.

**One harness per optimizer.** The harness requires its configured
``adapter_replay_policy`` to equal each adapter's ``required_replay_policy``
exactly, not merely to be compatible. Codex is ``NO_REDRIVE`` and the identity
adapter is ``IDEMPOTENT``, so one harness cannot host both. The controller
therefore owns exactly one harness configured for exactly one adapter's
policy, and the runner constructs one controller per optimizer.

**Codex spend, stated honestly.** Codex steps evaluate inside a sandboxed MCP
child process. That child enforces capacity through its own admission authority
against the capacity binding passed to it, and it owns the Tool Call Store its
calls are admitted into. Harness-side ledger and step evidence therefore do not
reflect subprocess MCP spend: ``harness_store_accepted_call_count`` reads 0 for
such a run regardless of how many calls the agent made. Canonical evaluation
downstream is the authority for what a proposal is worth; per-proposal MCP
evidence is not matched against proposals.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_store import ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.coordination.run_workflow import RunRequest
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import TypedRef, compute_identity_hash
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.adapters import AdapterRegistry
from whetstone.optimization.contracts import (
    OPTIMIZATION_RESULT_SCHEMA,
    BudgetState,
    OptimizationResult,
    OptimizationRunRef,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationStepResultRef,
    OutputContract,
    StepKind,
    StepStatus,
    step_result_reference,
)
from whetstone.optimization.harness import OptimizationHarness

__all__ = [
    "RUN_CONTROL_SCHEMA",
    "RUN_CONTROL_SCHEMA_VERSION",
    "STEP_CEILING",
    "HarnessRunController",
    "OptimizationRunControl",
    "RunControlError",
    "StepRequestBuilder",
]

RUN_CONTROL_SCHEMA = "whetstone.runner.run_control"
RUN_CONTROL_SCHEMA_VERSION = 1

#: A run that never reports a terminal status is a controller or adapter
#: defect, not a long run. The ceiling turns that into a loud, bounded failure
#: instead of an unbounded paid loop.
STEP_CEILING = 512


class RunControlError(RuntimeError):
    """The run control or the run it drives is invalid."""


class OptimizationRunControl(BaseModel):
    """The complete durable control for one optimization run.

    ``run`` is the exact run envelope the harness binds, and it already
    carries the adapter key, step mode, terminal output contract, template
    render contract, reward policy, and tool configs.
    ``initial_candidates`` are the candidates the first step starts from --
    for a restartable optimizer this is the anchor a restart must reproduce
    exactly, because the algorithm layer treats the initial candidate as
    controller input and will not detect a substitution. ``initial_budget``
    is the starting budget the first step carries.

    ``identity_hash()`` covers all of it, so any change to the candidates,
    the budget, or the run envelope produces a different control identity and
    therefore a different parent workflow.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: OptimizationRunRef
    initial_candidates: tuple[Candidate, ...] = ()
    initial_budget: BudgetState = Field(default_factory=BudgetState)
    step_kind: StepKind
    adapter_replay_policy: ReplayPolicy
    owner_id: StrictStr
    #: The output contract each individual step must satisfy. The run's own
    #: ``terminal_output_contract`` governs the terminal result.
    step_output_contract: OutputContract
    step_ceiling: StrictInt = STEP_CEILING

    @model_validator(mode="after")
    def _validate(self) -> OptimizationRunControl:
        if not self.owner_id:
            raise ValueError("run control owner_id must be non-empty")
        if self.step_ceiling < 1:
            raise ValueError("run control step_ceiling must be positive")
        if self.run.record.adapter_key != self.adapter_key:
            raise ValueError("run control adapter key drifted from its run")
        identities = [
            candidate.identity_hash() for candidate in self.initial_candidates
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("initial candidates must be identity-unique")
        return self

    @property
    def adapter_key(self) -> str:
        return str(self.run.record.adapter_key)

    @property
    def run_id(self) -> str:
        return str(self.run.record.run_id)

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=RUN_CONTROL_SCHEMA,
            schema_version=RUN_CONTROL_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )

    def run_request(self, *, controller_identity_hash: str) -> RunRequest:
        """The parent-workflow request identifying this exact run."""
        return RunRequest(
            controller_identity_hash=controller_identity_hash,
            run_id=self.run_id,
            control_identity_hash=self.identity_hash(),
        )


class StepRequestBuilder:
    """Builds each step request for an adapter that does not build its own.

    Some adapters own their request construction, because only they can read
    their durable algorithm state to decide the next step's shape. This
    builder serves the rest: it carries the control's candidates and contract
    forward and threads the prior step's exact references and debited budget,
    which is the carry-forward contract the harness enforces.
    """

    def __init__(self, control: OptimizationRunControl) -> None:
        self._control = control

    def build(
        self,
        *,
        step_index: int,
        prior_result: OptimizationStepResult | None,
        prior_result_ref: object | None,
    ) -> OptimizationStepRequest:
        control = self._control
        if (prior_result is None) != (prior_result_ref is None):
            raise RunControlError(
                "a continuation requires both the prior result and its ref"
            )
        if step_index == 0:
            if prior_result is not None:
                raise RunControlError(
                    "the initial step carries no prior result"
                )
            return OptimizationStepRequest(
                run=control.run,
                step_id=f"{control.run_id}:{step_index}",
                kind=control.step_kind,
                step_index=step_index,
                candidates=control.initial_candidates,
                budget=control.initial_budget,
                step_output_contract=control.step_output_contract,
            )
        if prior_result is None:
            raise RunControlError(
                "a noninitial step requires its prior result"
            )
        return OptimizationStepRequest(
            run=control.run,
            step_id=f"{control.run_id}:{step_index}",
            kind=control.step_kind,
            step_index=step_index,
            prior_step_result_ref=prior_result_ref,  # ty: ignore[invalid-argument-type]
            prior_state_ref=prior_result.state_ref,
            prior_history_ref=prior_result.history_ref,
            candidates=tuple(
                reference.record
                for reference in prior_result.accepted_candidates
            )
            or control.initial_candidates,
            budget=prior_result.budget,
            step_output_contract=control.step_output_contract,
        )


@dataclass(frozen=True, slots=True)
class _DrivenStep:
    """One completed step and the exact reference the harness bound it to."""

    result: OptimizationStepResult
    reference: OptimizationStepResultRef


class HarnessRunController:
    """Drive one optimizer's whole run against one harness.

    The controller owns exactly one harness, configured for exactly one
    adapter replay policy, and one control. :meth:`drive` is the whole run and
    is safe to call again after a recovery: ``bind_run``, ``run_step``, and
    ``terminalize`` are each idempotent on identical content, so a re-driven
    run resolves its already-durable results instead of paying for them twice.
    """

    def __init__(
        self,
        *,
        control: OptimizationRunControl,
        harness: OptimizationHarness,
        adapter_registry: AdapterRegistry,
        store: ObjectStore,
        request_builder: StepRequestBuilder | None = None,
    ) -> None:
        adapter = adapter_registry.resolve(control.adapter_key)
        if adapter.required_replay_policy is not control.adapter_replay_policy:
            raise RunControlError(
                f"adapter {control.adapter_key!r} requires replay policy "
                f"{adapter.required_replay_policy.value!r}; this controller's "
                f"harness is configured for "
                f"{control.adapter_replay_policy.value!r}. One harness hosts "
                "exactly one replay policy, so this optimizer needs its own "
                "controller."
            )
        if adapter.mode is not control.run.record.mode:
            raise RunControlError(
                "adapter mode does not match the run's declared step mode"
            )
        self._control = control
        self._harness = harness
        self._store = store
        self._builder = request_builder or StepRequestBuilder(control)

    @property
    def control(self) -> OptimizationRunControl:
        return self._control

    @property
    def runtime_hash(self) -> str:
        """The identity the parent workflow registry binds this under."""
        return self._control.identity_hash()

    def resolve_result(self, reference: TypedRef) -> OptimizationResult:
        """Load the terminal Optimization Result a run reference names."""
        if reference.schema_name != OPTIMIZATION_RESULT_SCHEMA:
            raise RunControlError(
                "reference does not name an Optimization Result"
            )
        return OptimizationResult.model_validate(
            self._store.get(reference.reference)
        )

    def drive(self, request: RunRequest) -> TypedRef:
        """Bind the run, drive steps to a terminal status, and terminalize.

        Returns the exact reference of the terminal Optimization Result. The
        record itself is already durable in the ObjectStore, so the reference
        is the authoritative handle; resolve it with
        :meth:`OptimizationHarness.resolve_optimization_result` or by reading
        the store.
        """
        control = self._control
        if request.run_id != control.run_id:
            raise RunControlError(
                "run request names a different run than this control"
            )
        if request.control_identity_hash != control.identity_hash():
            raise RunControlError(
                "run request control identity does not match this control"
            )
        self._harness.bind_run(control.run)

        driven: list[_DrivenStep] = []
        prior: _DrivenStep | None = None
        for step_index in range(control.step_ceiling):
            step_request = self._builder.build(
                step_index=step_index,
                prior_result=prior.result if prior is not None else None,
                prior_result_ref=(
                    prior.reference.record_ref if prior is not None else None
                ),
            )
            result, _reference = self._harness.run_step(step_request)
            step = _DrivenStep(
                result=result, reference=step_result_reference(result)
            )
            driven.append(step)
            if result.status is not StepStatus.CONTINUE:
                _terminal, terminal_ref = self._harness.terminalize(
                    run=control.run,
                    step_results=tuple(item.reference for item in driven),
                )
                return terminal_ref
            prior = step
        raise RunControlError(
            f"run {control.run_id!r} did not reach a terminal status within "
            f"its {control.step_ceiling} step ceiling"
        )
