from __future__ import annotations

from typing import Protocol, runtime_checkable

from whetstone.evaluation.drivers.eval_result import InternalEvalResult
from whetstone.evaluation.drivers.row_common import GenerationRowOutput
from whetstone.evaluation.protocol import EvalRequest
from whetstone.evaluation.schema import SubmissionResultRecord
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.env import Experiment
from whetstone.experiment.sampling import SplitSampling
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = ["EvaluationDriver"]


@runtime_checkable
class EvaluationDriver(Protocol):
    """Env-specific evaluation flow: sampling loop, rollout, and batch score."""

    def preflight(self, candidate: Candidate) -> None: ...

    def run(
        self,
        *,
        experiment: Experiment,
        sampling: SplitSampling,
        request: EvalRequest,
        eval_config_hash: str,
        execution_policy: ProviderExecutionPolicy,
        concurrency: int,
        max_wall_seconds: float | None,
        partial_log: PartialLog | None,
        prompt_cache: PromptResultCache | None,
    ) -> InternalEvalResult: ...

    def rendered_prompt(
        self,
        candidate: Candidate,
        task: object,
        *,
        max_budget: int | None,
    ) -> str: ...

    def submission_result_record(
        self, submission_result: object | None
    ) -> SubmissionResultRecord | None: ...

    def task_model_identity_hash(self, experiment: Experiment) -> str: ...

    def expected_model_route(self, experiment: Experiment) -> str: ...
