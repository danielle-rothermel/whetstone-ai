from __future__ import annotations

from typing import Any

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.evaluation.protocol import EvaluationEngine
from whetstone.evaluation.runtime_engine import RuntimeEvaluationEngine
from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
from whetstone.testing.fakes.driver import FakeEvaluationDriver
from whetstone.testing.toy.experiment import build_toy_experiment

REFERENCE_RUNTIME_SCHEMA = "whetstone.reference_evaluation_runtime"
REFERENCE_RUNTIME_SCHEMA_VERSION = 1


class ReferenceEvaluationRuntimeConfig(BaseModel):
    """In-memory SQLite + toy experiment + FakeEvaluationDriver."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None
    row_job_entrypoint: StrictStr = (
        "whetstone.testing.fakes.driver:FakeEvaluationDriver"
    )
    env_name: StrictStr = "whetstone.toy"
    split_role: StrictStr = "internal_eval"
    transport_api_key_env: StrictStr = "WHETSTONE_TOY_API_KEY"

    @property
    def execution_policy(self) -> ProviderExecutionPolicy:
        transport = default_transport_policy(
            api_key_env=self.transport_api_key_env,
        )
        return ProviderExecutionPolicy(transport_policy=transport)

    def build_engine(self, store: ObjectStore) -> EvaluationEngine:
        _ = self.env_name
        experiment = build_toy_experiment()
        if self.split_role == "internal_eval":
            sampling = experiment.eval_configs.internal
        elif self.split_role == "official":
            sampling = experiment.eval_configs.official
        else:
            raise ValueError(f"unknown split role {self.split_role!r}")
        return RuntimeEvaluationEngine(
            store=store,
            experiment=experiment,
            sampling=sampling,
            execution_policy=self.execution_policy,
            driver=FakeEvaluationDriver(),
        )


__all__ = [
    "REFERENCE_RUNTIME_SCHEMA",
    "REFERENCE_RUNTIME_SCHEMA_VERSION",
    "ReferenceEvaluationRuntimeConfig",
]
