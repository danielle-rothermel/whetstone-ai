from __future__ import annotations

from typing import Any

from whetstone.eval.protocol import (
    EvalRequest,
    EvalResult,
    EvalEngine,
    EvalPlanSnapshot,
    EvalSplitView,
)
from whetstone.experiment.binding import EvalConfigRef
from whetstone.experiment.sampling import EvalSplit

__all__ = ["FakeEvalEngine"]


class FakeEvalEngine:
    """Minimal EvalEngine stub; use ReferenceEvalRuntimeConfig for real runs."""

    def __init__(
        self,
        *,
        eval_config_ref: EvalConfigRef,
        provider_execution_policy_ref: Any,
        provider_execution_policy_record: dict[str, Any],
        plan_snapshot: EvalPlanSnapshot,
        sampling: EvalSplitView,
        sampling_split: EvalSplit | None = None,
        model_route: str = "openai/chat_completions/fake-model",
    ) -> None:
        self._eval_config_ref = eval_config_ref
        self._provider_execution_policy_ref = provider_execution_policy_ref
        self._provider_execution_policy_record = provider_execution_policy_record
        self._plan_snapshot = plan_snapshot
        self._sampling = sampling
        self._sampling_split = sampling_split
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
    def plan_snapshot(self) -> EvalPlanSnapshot:
        return self._plan_snapshot

    @property
    def sampling(self) -> EvalSplitView:
        return self._sampling

    @property
    def sampling_split(self) -> EvalSplit:
        if self._sampling_split is None:
            raise TypeError(
                "FakeEvalEngine was constructed without sampling_split"
            )
        return self._sampling_split

    def task_model_identity_hash(self) -> str:
        return "fake-task-model"

    def execution_policy_identity_hash(self) -> str:
        return str(self._provider_execution_policy_ref.record_hash)

    def reward_policy_identity_hash(self) -> str:
        return "fake-reward-policy"

    def expected_model_route(self) -> str:
        return self._model_route

    def evaluate(self, request: EvalRequest) -> EvalResult:
        raise NotImplementedError(
            "FakeEvalEngine is a protocol stub; use "
            "ReferenceEvalRuntimeConfig.build_engine() for evaluation"
        )

    def for_task_ids(self, task_ids: tuple[str, ...]) -> EvalEngine:
        if not task_ids:
            raise ValueError("task_ids must be non-empty")
        return self
