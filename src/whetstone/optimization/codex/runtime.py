from __future__ import annotations

from pathlib import Path

from dr_store import ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.envs.factory import build_env_experiment
from whetstone.envs.sampling import Completeness
from whetstone.evaluation.drivers.internal import InternalRowRequest
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.execution.fanout import ProcessJob
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.provider.policy import ProviderExecutionPolicy


class EvaluationRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    env_name: StrictStr
    model: StrictStr
    pool_n_per_stratum: StrictInt | None = None
    split_sizes: tuple[int, int, int] | None = None
    repeats: StrictInt
    completeness: Completeness = Completeness.PROPAGATE
    max_skip_fraction: float = 0.0
    expected_eval_config_hash: StrictStr
    execution_policy: ProviderExecutionPolicy
    row_job_entrypoint: StrictStr
    concurrency: StrictInt = 5
    max_wall_seconds: float | None = None
    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> EvaluationRuntimeConfig:
        if ":" not in self.row_job_entrypoint:
            raise ValueError(
                "row_job_entrypoint must be 'importable.module:callable'"
            )
        if self.repeats < 1 or self.concurrency < 1:
            raise ValueError("repeats and concurrency must be positive")
        return self

    def _row_job(self, request: InternalRowRequest) -> ProcessJob:
        return ProcessJob(
            entrypoint=self.row_job_entrypoint,
            payload=request.model_dump(mode="json"),
        )

    def build_engine(self, store: ObjectStore) -> EvaluationEngine:
        experiment = build_env_experiment(
            self.env_name,
            model=self.model,
            pool_n_per_stratum=self.pool_n_per_stratum,
            completeness=self.completeness,
            max_skip_fraction=self.max_skip_fraction,
            repeats=self.repeats,
            split_sizes=self.split_sizes,
        )
        split = experiment.eval_configs.internal
        sampling = split.eval_config
        if sampling.config_identity_hash != self.expected_eval_config_hash:
            raise ValueError(
                "reconstructed runtime produced a different Eval Config"
            )
        return EvaluationEngine(
            store=store,
            experiment=experiment,
            sampling=split,
            execution_policy=self.execution_policy,
            row_job_factory=self._row_job,
            concurrency=self.concurrency,
            max_wall_seconds=self.max_wall_seconds,
            partial_log=PartialLog(Path(self.partial_log_path))
            if self.partial_log_path
            else None,
            prompt_cache=PromptResultCache(Path(self.prompt_cache_path))
            if self.prompt_cache_path
            else None,
        )


__all__ = ["EvaluationRuntimeConfig"]
