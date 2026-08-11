from __future__ import annotations

from pathlib import Path
from typing import Literal

from dr_store import ObjectStore
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.envs.code_comp.config import CodeCompExperimentConfig
from whetstone.envs.sampling import Completeness
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.execution.fanout import ProcessJob
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.provider.policy import ProviderExecutionPolicy


class CodeCompEvaluationRuntimeConfig(BaseModel):
    """Runtime binding that rebuilds an engine from one experiment config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_config: CodeCompExperimentConfig
    split_role: Literal["internal_eval", "official"] = "internal_eval"
    execution_policy: ProviderExecutionPolicy
    expected_eval_config_hash: StrictStr
    row_job_entrypoint: StrictStr
    concurrency: StrictInt = 5
    num_samples: StrictInt | None = None
    completeness: Completeness = Completeness.PROPAGATE
    max_skip_fraction: float = 0.0
    max_wall_seconds: float | None = None
    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> CodeCompEvaluationRuntimeConfig:
        if ":" not in self.row_job_entrypoint:
            raise ValueError(
                "row_job_entrypoint must be 'importable.module:callable'"
            )
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError("num_samples must be positive when set")
        return self

    def _row_job(self, request: BaseModel) -> ProcessJob:
        return ProcessJob(
            entrypoint=self.row_job_entrypoint,
            payload=request.model_dump(mode="json"),
        )

    def build_engine(self, store: ObjectStore) -> EvaluationEngine:
        config = self.experiment_config
        if self.num_samples is not None:
            config = config.model_copy(
                update={
                    "sampling": config.sampling.model_copy(
                        update={"num_samples": self.num_samples}
                    )
                }
            )
        if (
            self.completeness is not config.sampling.completeness
            or self.max_skip_fraction != config.sampling.max_skip_fraction
        ):
            config = config.model_copy(
                update={
                    "sampling": config.sampling.model_copy(
                        update={
                            "completeness": self.completeness,
                            "max_skip_fraction": self.max_skip_fraction,
                        }
                    )
                }
            )
        experiment = config.build_experiment()
        eval_configs = experiment.eval_configs
        split = (
            eval_configs.internal
            if self.split_role == "internal_eval"
            else eval_configs.official
        )
        sampling = split.eval_config
        if sampling.config_hash != self.expected_eval_config_hash:
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


__all__ = ["CodeCompEvaluationRuntimeConfig"]
