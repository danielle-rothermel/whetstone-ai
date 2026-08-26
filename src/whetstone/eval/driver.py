from __future__ import annotations

from typing import Protocol, runtime_checkable

from whetstone.eval.drivers.eval_result import InternalEvalResult
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.schema import SubmissionResultRecord
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.env import Experiment
from whetstone.experiment.sampling import EvalSplit
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = ["ClosableEvalDriver", "EvalDriver"]


@runtime_checkable
class EvalDriver(Protocol):
    """Env-specific evaluation flow: sampling loop, rollout, and batch score."""

    def preflight(self, candidate: Candidate) -> None: ...

    def run(
        self,
        *,
        experiment: Experiment,
        sampling: EvalSplit,
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


@runtime_checkable
class ClosableEvalDriver(Protocol):
    """A driver holding resources that outlive a single ``run``.

    Only some drivers own anything to release: the in-process driver owns
    nothing, while the worker-pool driver owns worker processes. Keeping the
    capability off :class:`EvalDriver` is what lets a caller ask, with a
    runtime check, whether this particular driver has a lifetime to end,
    instead of obliging every driver to carry an empty ``close``.
    """

    def close(self) -> None: ...
