from __future__ import annotations

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.protocol import EvalEngine
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.transport import FakeLlmTransport
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

REFERENCE_RUNTIME_SCHEMA = "whetstone.reference_evaluation_runtime"
REFERENCE_RUNTIME_SCHEMA_VERSION = 1


class ReferenceEvalRuntimeConfig(BaseModel):
    """In-memory SQLite + toy experiment + GraphRolloutEvalDriver."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None
    row_job_entrypoint: StrictStr = (
        "whetstone.eval.drivers.graph_rollout:GraphRolloutEvalDriver"
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

    def build_engine(self, store: ObjectStore) -> EvalEngine:
        _ = self.env_name
        experiment = build_toy_experiment()
        if self.split_role == "internal_eval":
            sampling = experiment.eval_configs.internal
        elif self.split_role == "official":
            sampling = experiment.eval_configs.official
        else:
            raise ValueError(f"unknown split role {self.split_role!r}")
        execution_policy = self.execution_policy

        def transport_factory(
            policy: ProviderExecutionPolicy,
        ) -> FakeLlmTransport:
            return FakeLlmTransport(transport_policy=policy.transport_policy)

        driver = GraphRolloutEvalDriver(
            eval_runner=FakeEvalProcedureRunner(),
            mutation_field=TOY_MUTATION_FIELD,
            render_contract=toy_template_render_contract(),
            transport_factory=transport_factory,
        )
        return RuntimeEvalEngine(
            store=store,
            experiment=experiment,
            sampling=sampling,
            execution_policy=execution_policy,
            driver=driver,
        )


__all__ = [
    "REFERENCE_RUNTIME_SCHEMA",
    "REFERENCE_RUNTIME_SCHEMA_VERSION",
    "ReferenceEvalRuntimeConfig",
]
