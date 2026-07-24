"""Serializable composition for rebuilding evaluation in an MCP child."""

from __future__ import annotations

import importlib
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
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.execution.fanout import FanoutConfig
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


class EvaluationRuntimeConfig(BaseModel):
    """Complete JSON boundary for reconstructing the canonical engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    env_name: StrictStr
    model: StrictStr
    sampling_role: StrictStr = "internal_eval"
    pool_n_per_stratum: StrictInt | None = None
    split_sizes: tuple[int, int, int] | None = None
    repeats: StrictInt
    completeness: Completeness = Completeness.PROPAGATE
    max_skip_fraction: float = 0.0
    expected_eval_config_hash: StrictStr
    execution_policy: ProviderExecutionPolicy
    transport_factory: StrictStr
    concurrency: StrictInt = 5
    max_wall_seconds: float | None = None
    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> EvaluationRuntimeConfig:
        if ":" not in self.transport_factory:
            raise ValueError("transport_factory must be 'module:callable'")
        if self.repeats < 1 or self.concurrency < 1:
            raise ValueError("repeats and concurrency must be positive")
        return self

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
        sampling = experiment.eval_configs.eval_config_for(self.sampling_role)
        split = (
            experiment.eval_configs.internal
            if self.sampling_role == "internal_eval"
            else experiment.eval_configs.official
        )
        if sampling.config_identity_hash != self.expected_eval_config_hash:
            raise ValueError(
                "reconstructed runtime produced a different Eval Config"
            )
        module_name, attr = self.transport_factory.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attr)
        transport: TransportCall = factory()
        return EvaluationEngine(
            store=store,
            experiment=experiment,
            sampling=split,
            execution_policy=self.execution_policy,
            transport=transport,
            fanout=FanoutConfig(
                concurrency=self.concurrency,
                max_wall_seconds=self.max_wall_seconds,
            ),
            partial_log=PartialLog(Path(self.partial_log_path))
            if self.partial_log_path
            else None,
            prompt_cache=PromptResultCache(Path(self.prompt_cache_path))
            if self.prompt_cache_path
            else None,
        )


__all__ = ["EvaluationRuntimeConfig"]
