from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dr_store import BindingConflictError, ObjectStore

from whetstone.core.leasing import (
    ReplayPolicy,
)
from whetstone.core.identity import (
    ImmutableJsonObject,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.experiment.binding import EVAL_CONFIG_RECORD_SCHEMA
from whetstone.experiment.candidate import (
    CANDIDATE_RECORD_SCHEMA,
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.optim.adapters import (
    AdapterCheckpoint,
    AdapterRegistry,
    AdapterReplayPolicyMismatchError,
    OptimizerAdapter,
)
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OPTIM_RUN_SCHEMA,
    STEP_REQUEST_SCHEMA,
    STEP_RESULT_SCHEMA,
    OptimEvalRequest,
    OptimResult,
    OptimRun,
    OptimRunRef,
    OptimStepRequest,
    OptimStepResult,
    StepMode,
    StepStatus,
    optimization_result_reference,
    optimization_run_reference,
    step_result_reference,
)
from whetstone.optim.tools.contracts import (
    TOOL_CONFIG_SCHEMA,
    TOOL_DEFINITION_SCHEMA,
)

ADAPTER_CHECKPOINT_SCHEMA = "whetstone.optim_adapter_checkpoint"
STATE_SNAPSHOT_SCHEMA = "whetstone.optim_state_snapshot"
HISTORY_SNAPSHOT_SCHEMA = "whetstone.optim_history_snapshot"
STEP_RESULT_BINDING_PREFIX = "whetstone.optim_step_result:v2:"
OPTIM_RESULT_BINDING_PREFIX = "whetstone.optim_result:v2:"


class StepResultConflictError(Exception):
    def __init__(
        self,
        *,
        run_id: str,
        step_index: int,
        existing: TypedRef,
        requested: TypedRef,
    ) -> None:
        self.run_id = run_id
        self.step_index = step_index
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Step ({run_id}, index {step_index}) already has result "
            f"{existing.content_hash}; refusing {requested.content_hash}"
        )


class OptimRunConflictError(Exception):
    def __init__(
        self, *, run_id: str, existing: TypedRef, requested: TypedRef
    ) -> None:
        self.run_id = run_id
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Optimization run {run_id!r} is already bound to "
            f"{existing.content_hash}; refusing {requested.content_hash}"
        )


class OptimResultConflictError(Exception):
    def __init__(
        self, *, run_id: str, existing: TypedRef, requested: TypedRef
    ) -> None:
        self.run_id = run_id
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Optimization run {run_id!r} already has terminal result "
            f"{existing.content_hash}; refusing {requested.content_hash}"
        )


@dataclass(frozen=True, slots=True)
class _ResolvedAdapter:
    adapter: OptimizerAdapter
    replay_policy: ReplayPolicy


class OptimRunStore:
    _store: ObjectStore
    _adapter_registry: AdapterRegistry
    _adapter_replay_policy: ReplayPolicy

    @staticmethod
    def _result_binding_key(run_id: str, step_index: int) -> str:
        return f"{STEP_RESULT_BINDING_PREFIX}{run_id}#{step_index}"

    @staticmethod
    def _terminal_binding_key(run_id: str) -> str:
        return f"{OPTIM_RESULT_BINDING_PREFIX}{run_id}"

    @staticmethod
    def _run_binding_key(run_id: str) -> str:
        return f"whetstone.optim_run:{run_id}"

    def _resolve_binding(self, key: str) -> TypedRef | None:
        reference = self._store.resolve(key)
        if reference is None:
            return None
        return TypedRef(
            schema_name=reference.schema, content_hash=reference.content_hash
        )

    def _resolve_result_binding(
        self, run_id: str, step_index: int
    ) -> TypedRef | None:
        return self._resolve_binding(
            self._result_binding_key(run_id, step_index)
        )

    def _put(self, schema: str, content: dict[str, Any]) -> TypedRef:
        reference, _status = self._store.put(schema, content)
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def _put_request(self, request: OptimStepRequest) -> TypedRef:
        return self._put(STEP_REQUEST_SCHEMA, request.record_content())

    def _put_result(self, result: OptimStepResult) -> TypedRef:
        return self._put(STEP_RESULT_SCHEMA, result.record_content())

    def _load_run(self, ref: TypedRef) -> OptimRunRef:
        if ref.schema_name != OPTIM_RUN_SCHEMA:
            raise ValueError("bound run ref has the wrong schema")
        run = OptimRun.model_validate(self._store.get(ref.reference))
        exact = optimization_run_reference(run)
        if exact.record_ref != ref:
            raise ValueError("bound Optimization Run ref is not exact")
        return exact

    def bind_run(
        self, run: OptimRun | OptimRunRef
    ) -> OptimRunRef:
        if isinstance(run, OptimRunRef):
            exact = OptimRunRef.model_validate(
                run.model_dump(mode="json")
            )
        else:
            validated = OptimRun.model_validate(
                run.model_dump(mode="json")
            )
            exact = optimization_run_reference(validated)
        persisted = self._put(
            OPTIM_RUN_SCHEMA, exact.record.record_content()
        )
        if persisted != exact.record_ref:
            raise ValueError(
                "persisted Optimization Run ref failed content validation"
            )
        self._persist_tool_configs(exact)
        key = self._run_binding_key(str(exact.record.run_id))
        try:
            self._store.bind(key, exact.record_ref.reference)
        except BindingConflictError as conflict:
            existing = TypedRef(
                schema_name=conflict.existing.schema,
                content_hash=conflict.existing.content_hash,
            )
            if existing == exact.record_ref:
                self._bound_run = exact
                return exact
            raise OptimRunConflictError(
                run_id=str(exact.record.run_id),
                existing=existing,
                requested=exact.record_ref,
            ) from conflict
        existing = self._resolve_binding(key)
        if existing is None:
            raise RuntimeError("Optimization Run binding disappeared")
        if existing != exact.record_ref:
            raise OptimRunConflictError(
                run_id=str(exact.record.run_id),
                existing=existing,
                requested=exact.record_ref,
            )
        loaded = self._load_run(existing)
        if loaded != exact:
            raise ValueError("bound Optimization Run is not its exact record")
        self._bound_run = exact
        return exact

    def _persist_tool_configs(self, run: OptimRunRef) -> None:
        for config in run.record.tool_configs:
            definition = config.record.definition
            if (
                self._put(
                    TOOL_DEFINITION_SCHEMA,
                    definition.record.record_content(),
                )
                != definition.record_ref
            ):
                raise ValueError(
                    "persisted Tool Definition ref failed validation"
                )
            if (
                self._put(TOOL_CONFIG_SCHEMA, config.record.record_content())
                != config.record_ref
            ):
                raise ValueError("persisted Tool Config ref failed validation")

    def _validate_bound_run(self, request: OptimStepRequest) -> None:
        if self._bound_run is None:
            raise ValueError("bind_run must be called before run_step")
        if request.run != self._bound_run:
            raise ValueError(
                "Step Request belongs to a different exact Optimization Run"
            )
        actual = self._resolve_binding(
            self._run_binding_key(str(request.run_id))
        )
        if actual is None:
            raise ValueError("Optimization Run binding is absent")
        if actual != request.run.record_ref:
            raise ValueError(
                "Step Request run is not the durably bound exact run"
            )
        if self._load_run(actual) != request.run:
            raise ValueError("Step Request run ref is not exact")

    def _persist_candidate(self, candidate: Candidate) -> CandidateRef:
        expected = candidate_reference(candidate)
        persisted = self._put(
            CANDIDATE_RECORD_SCHEMA, candidate.record_content()
        )
        if persisted != expected.record_ref:
            raise ValueError(
                "persisted Candidate ref failed content validation"
            )
        return expected

    def _persist_intent_records(
        self, optim_eval_request: OptimEvalRequest
    ) -> None:
        candidate = self._persist_candidate(
            optim_eval_request.eval_request.candidate
        )
        expected = candidate_reference(optim_eval_request.eval_request.candidate)
        if candidate != expected:
            raise ValueError(
                "Optim Eval Request candidate ref is not its exact record"
            )

    def _persist_snapshot(
        self, schema: str, delta: ImmutableJsonObject
    ) -> TypedRef | None:
        if not delta:
            return None
        content = delta.to_json()
        expected = typed_ref_for_record(schema, content)
        persisted = self._put(schema, content)
        if persisted != expected:
            raise ValueError(f"persisted {schema} ref failed validation")
        return persisted

    def _load_result(self, ref: TypedRef) -> OptimStepResult:
        if ref.schema_name != STEP_RESULT_SCHEMA:
            raise ValueError("Step Result ref has the wrong schema")
        result = OptimStepResult.model_validate(
            self._store.get(ref.reference)
        )
        if step_result_reference(result).record_ref != ref:
            raise ValueError("persisted Step Result ref is not exact")
        return result

    def _load_checkpoint(self, ref: TypedRef) -> AdapterCheckpoint:
        if ref.schema_name != ADAPTER_CHECKPOINT_SCHEMA:
            raise ValueError("Adapter Checkpoint ref has the wrong schema")
        checkpoint = AdapterCheckpoint.model_validate(
            self._store.get(ref.reference)
        )
        if (
            typed_ref_for_record(
                ADAPTER_CHECKPOINT_SCHEMA, checkpoint.record_content()
            )
            != ref
        ):
            raise ValueError("persisted Adapter Checkpoint ref is not exact")
        return checkpoint

    def _load_terminal(self, ref: TypedRef) -> OptimResult:
        if ref.schema_name != OPTIM_RESULT_SCHEMA:
            raise ValueError("Optimization Result ref has the wrong schema")
        result = OptimResult.model_validate(
            self._store.get(ref.reference)
        )
        if optimization_result_reference(result) != ref:
            raise ValueError("persisted Optimization Result ref is not exact")
        return result

    def resolve_step_result(
        self, run_id: str, step_index: int
    ) -> TypedRef | None:
        return self._resolve_result_binding(run_id, step_index)

    def resolve_optimization_result(self, run_id: str) -> TypedRef | None:
        return self._resolve_binding(self._terminal_binding_key(run_id))

    def resolve_adapter(self, adapter_key: str) -> OptimizerAdapter:
        return self._resolve_compatible_adapter(adapter_key).adapter

    def _resolve_compatible_adapter(
        self,
        adapter_key: str,
        *,
        expected_mode: StepMode | None = None,
    ) -> _ResolvedAdapter:
        adapter = self._adapter_registry.resolve(adapter_key)
        resolved_key = adapter.key
        if type(resolved_key) is not str:
            raise TypeError("adapter key must be an actual string")
        if resolved_key != adapter_key:
            raise ValueError(
                "registry returned an adapter under the wrong key"
            )
        resolved_mode = adapter.mode
        if type(resolved_mode) is not StepMode:
            raise TypeError("adapter mode must be an actual StepMode enum")
        if expected_mode is not None and resolved_mode is not expected_mode:
            raise ValueError(
                f"adapter mode {resolved_mode.value!r} does not match request "
                f"mode {expected_mode.value!r}"
            )
        required_policy = adapter.required_replay_policy
        if type(required_policy) is not ReplayPolicy:
            raise TypeError(
                "adapter required_replay_policy must be an actual "
                "ReplayPolicy enum"
            )
        if required_policy is not self._adapter_replay_policy:
            raise AdapterReplayPolicyMismatchError(
                adapter_key=adapter_key,
                configured_policy=self._adapter_replay_policy,
                required_policy=required_policy,
            )
        return _ResolvedAdapter(
            adapter=adapter,
            replay_policy=required_policy,
        )

    def _validate_prior_binding(
        self, request: OptimStepRequest
    ) -> None:
        if request.step_index == 0:
            if request.prior_step_result_ref is not None:
                raise ValueError(
                    "initial Step Request carries no prior Step Result"
                )
            if request.prior_state_ref is not None:
                raise ValueError("initial Step Request carries no prior state")
            if request.prior_history_ref is not None:
                raise ValueError(
                    "initial Step Request carries no prior history"
                )
            return
        actual = self._resolve_result_binding(
            request.run_id, request.step_index - 1
        )
        if actual is None:
            raise ValueError(
                "noninitial Step Request references no durably bound "
                "preceding Step Result"
            )
        if actual != request.prior_step_result_ref:
            raise ValueError(
                "prior_step_result_ref does not match the actual preceding "
                "Step Result binding"
            )
        preceding = self._load_result(actual)
        if preceding.run_id != request.run_id:
            raise ValueError("preceding Step Result belongs to another run")
        if preceding.request.record.run != request.run:
            raise ValueError(
                "preceding Step Result belongs to another exact Optimization "
                "Run"
            )
        if preceding.step_index != request.step_index - 1:
            raise ValueError("preceding Step Result has the wrong step index")
        if preceding.status is not StepStatus.CONTINUE:
            raise ValueError(
                "a new Step may follow only a continuing Step Result"
            )
        if request.budget != preceding.budget:
            raise ValueError(
                "a new Step must carry forward the preceding durable budget"
            )
        if request.prior_state_ref != preceding.state_ref:
            raise ValueError(
                "a new Step must cite the preceding exact state, including "
                "its absence"
            )
        if request.prior_history_ref != preceding.history_ref:
            raise ValueError(
                "a new Step must cite the preceding exact history, including "
                "its absence"
            )


__all__ = [
    "ADAPTER_CHECKPOINT_SCHEMA",
    "OptimResultConflictError",
    "OptimRunConflictError",
    "OptimRunStore",
    "StepResultConflictError",
]
