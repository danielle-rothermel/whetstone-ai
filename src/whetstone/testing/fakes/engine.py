from __future__ import annotations

from typing import Any

from whetstone.evaluation.protocol import (
    EvaluationEngine,
    EvaluationPlanSnapshot,
    EvaluationRequest,
    EvaluationSamplingView,
)
from whetstone.experiment.binding import EvalConfigRef

__all__ = ["FakeEvaluationEngine"]


class FakeEvaluationEngine:
    """Minimal EvaluationEngine stub; use ReferenceEvaluationRuntimeConfig for real runs."""

    def __init__(
        self,
        *,
        eval_config_ref: EvalConfigRef,
        provider_execution_policy_ref: Any,
        provider_execution_policy_record: dict[str, Any],
        plan_snapshot: EvaluationPlanSnapshot,
        sampling: EvaluationSamplingView,
        model_route: str = "openai/chat_completions/fake-model",
    ) -> None:
        self._eval_config_ref = eval_config_ref
        self._provider_execution_policy_ref = provider_execution_policy_ref
        self._provider_execution_policy_record = provider_execution_policy_record
        self._plan_snapshot = plan_snapshot
        self._sampling = sampling
        self._model_route = model_route

    @property
    def eval_config_ref(self) -> EvalConfigRef:
        return self._eval_config_ref

    @property
    def provider_execution_policy_ref(self) -> Any:
        return self._provider_execution_policy_ref

    @property
    def provider_execution_policy_record(self) -> dict[str, Any]:
        return self._provider_execution_policy_record

    @property
    def plan_snapshot(self) -> EvaluationPlanSnapshot:
        return self._plan_snapshot

    @property
    def sampling(self) -> EvaluationSamplingView:
        return self._sampling

    def task_model_identity_hash(self) -> str:
        return "fake-task-model"

    def execution_policy_identity_hash(self) -> str:
        return str(self._provider_execution_policy_ref.record_hash)

    def reward_policy_identity_hash(self) -> str:
        return "fake-reward-policy"

    def expected_model_route(self) -> str:
        return self._model_route

    def validate_request(self, request: EvaluationRequest) -> None:
        _ = request

    def evaluate(self, request: EvaluationRequest) -> Any:
        raise NotImplementedError(
            "FakeEvaluationEngine is a protocol stub; use "
            "ReferenceEvaluationRuntimeConfig.build_engine() for evaluation"
        )

    def for_task_ids(self, task_ids: tuple[str, ...]) -> EvaluationEngine:
        if not task_ids:
            raise ValueError("task_ids must be non-empty")
        return self
