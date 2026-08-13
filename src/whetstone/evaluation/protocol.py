from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable

from dr_store import ObjectStore

from whetstone.core.identity import IdentityRef, TypedRef
from whetstone.evaluation.schema import EvaluationEvidence
from whetstone.experiment.binding import EvalConfigRef, EvaluationBinding
from whetstone.experiment.candidate import Candidate
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    candidate: Candidate
    evaluation_binding: EvaluationBinding
    purpose: str


@dataclass(frozen=True, slots=True)
class EngineEvaluation:
    evidence: EvaluationEvidence
    evidence_ref: TypedRef

    @property
    def reward_value(self) -> float | None:
        if self.evidence.reward_ref is None:
            return None
        return self.evidence.reward_ref.record.value


@dataclass(frozen=True, slots=True)
class EvaluationPlanSnapshot:
    graph_hash: str
    dataset_hash: str
    task_hashes: tuple[str, ...]
    num_samples: int
    split_role: str


@runtime_checkable
class EvaluationTaskView(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def task_hash(self) -> str: ...

    @property
    def prompt_inputs(self) -> dict[str, str]: ...


@runtime_checkable
class EvaluationSamplingView(Protocol):
    @property
    def task_hashes(self) -> tuple[str, ...]: ...

    @property
    def num_samples(self) -> int: ...

    @property
    def split_role(self) -> str: ...

    @property
    def tasks(self) -> tuple[EvaluationTaskView, ...]: ...


@runtime_checkable
class EvaluationEngine(Protocol):
    @property
    def eval_config_ref(self) -> EvalConfigRef: ...

    @property
    def provider_execution_policy_ref(self) -> IdentityRef: ...

    @property
    def provider_execution_policy_record(self) -> dict[str, Any]: ...

    @property
    def plan_snapshot(self) -> EvaluationPlanSnapshot: ...

    @property
    def sampling(self) -> EvaluationSamplingView: ...

    def task_model_identity_hash(self) -> str: ...

    def execution_policy_identity_hash(self) -> str: ...

    def reward_policy_identity_hash(self) -> str: ...

    def expected_model_route(self) -> str: ...

    def validate_request(self, request: EvaluationRequest) -> None: ...

    def evaluate(self, request: EvaluationRequest) -> EngineEvaluation: ...

    def for_task_ids(self, task_ids: tuple[str, ...]) -> EvaluationEngine: ...


class EvaluationRuntimeConfig(Protocol):
    partial_log_path: str | None
    prompt_cache_path: str | None
    row_job_entrypoint: str

    @classmethod
    def model_validate_json(cls, data: str | bytes) -> Self: ...

    @property
    def execution_policy(self) -> ProviderExecutionPolicy: ...

    def build_engine(self, store: ObjectStore) -> EvaluationEngine: ...


def load_runtime_config(*, class_path: str, raw: bytes) -> EvaluationRuntimeConfig:
    import importlib

    module_name, separator, class_name = class_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("runtime config class_path must be module:Class")
    module = importlib.import_module(module_name)
    config_class = getattr(module, class_name)
    validated = config_class.model_validate_json(raw)
    return validated


__all__ = [
    "EngineEvaluation",
    "EvaluationEngine",
    "EvaluationPlanSnapshot",
    "EvaluationRequest",
    "EvaluationRuntimeConfig",
    "EvaluationSamplingView",
    "EvaluationTaskView",
    "load_runtime_config",
]
