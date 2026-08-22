from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeGuard, runtime_checkable

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, Field

from whetstone.core.identity import IdentityRef, ImmutableJsonObject, NonEmptyId, TypedRef
from whetstone.eval.schema import EvalEvidence, EvalFailureEvidence
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.candidate import Candidate
from whetstone.provider.policy import ProviderExecutionPolicy

if TYPE_CHECKING:
    from whetstone.experiment.sampling import EvalSplit
    from whetstone.optim.contracts import ResolutionDetail


class EvalRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: NonEmptyId
    candidate: Candidate
    metadata: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )


@dataclass(frozen=True, slots=True)
class EvalRejected:
    detail: ResolutionDetail


@dataclass(frozen=True, slots=True)
class EvalEvidenceWithRef:
    evidence: EvalEvidence | EvalFailureEvidence
    evidence_ref: TypedRef


EvalResult = EvalRejected | EvalEvidenceWithRef


def eval_is_rejected(result: EvalResult) -> TypeGuard[EvalRejected]:
    return isinstance(result, EvalRejected)


def eval_is_success(result: EvalResult) -> TypeGuard[EvalEvidenceWithRef]:
    return isinstance(result, EvalEvidenceWithRef) and isinstance(
        result.evidence, EvalEvidence
    )


@dataclass(frozen=True, slots=True)
class EvalPlanSnapshot:
    graph_hash: str
    dataset_hash: str
    task_hashes: tuple[str, ...]
    num_seeds: int
    split_role: str


@runtime_checkable
class EvalTaskView(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def task_hash(self) -> str: ...

    @property
    def prompt_inputs(self) -> dict[str, str]: ...


@runtime_checkable
class EvalSplitView(Protocol):
    @property
    def task_hashes(self) -> tuple[str, ...]: ...

    @property
    def num_seeds(self) -> int: ...

    @property
    def split_role(self) -> str: ...

    @property
    def tasks(self) -> tuple[EvalTaskView, ...]: ...


@runtime_checkable
class EvalEngine(Protocol):
    @property
    def eval_config_ref(self) -> EvalConfigRef: ...

    @property
    def provider_execution_policy_ref(self) -> IdentityRef: ...

    @property
    def provider_execution_policy_record(self) -> dict[str, Any]: ...

    @property
    def plan_snapshot(self) -> EvalPlanSnapshot: ...

    @property
    def sampling(self) -> EvalSplitView: ...

    @property
    def sampling_split(self) -> EvalSplit: ...

    def task_model_identity_hash(self) -> str: ...

    def execution_policy_identity_hash(self) -> str: ...

    def reward_policy_identity_hash(self) -> str: ...

    def expected_model_route(self) -> str: ...

    def evaluate(self, request: EvalRequest) -> EvalResult: ...

    def for_task_ids(self, task_ids: tuple[str, ...]) -> EvalEngine: ...


class EvalRuntimeConfig(Protocol):
    partial_log_path: str | None
    prompt_cache_path: str | None
    row_job_entrypoint: str

    @classmethod
    def model_validate_json(cls, data: str | bytes) -> Self: ...

    @property
    def execution_policy(self) -> ProviderExecutionPolicy: ...

    def build_engine(self, store: ObjectStore) -> EvalEngine: ...


def load_runtime_config(*, class_path: str, raw: bytes) -> EvalRuntimeConfig:
    import importlib

    module_name, separator, class_name = class_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("runtime config class_path must be module:Class")
    module = importlib.import_module(module_name)
    config_class = getattr(module, class_name)
    validated = config_class.model_validate_json(raw)
    return validated


__all__ = [
    "EvalEvidenceWithRef",
    "EvalRejected",
    "EvalRequest",
    "EvalResult",
    "EvalEngine",
    "EvalPlanSnapshot",
    "EvalRuntimeConfig",
    "EvalSplitView",
    "EvalTaskView",
    "eval_is_rejected",
    "eval_is_success",
    "load_runtime_config",
]
