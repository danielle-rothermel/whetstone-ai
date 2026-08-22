from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from dr_providers import ProviderKind
from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, PositiveFloat, StrictStr

from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.drivers.subprocess_graph_rollout import (
    SubprocessGraphRolloutEvalDriver,
)
from whetstone.eval.eval_procedure import EvalProcedureRunner
from whetstone.eval.protocol import EvalEngine
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from dr_store.localfs import ensure_private_directory
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.candidate import TemplateRenderContract
from whetstone.experiment.env import Experiment
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import (
    ProviderExecutionPolicy,
    default_transport_policy,
)
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.transport import fake_llm_transport_factory
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

REFERENCE_RUNTIME_SCHEMA = "whetstone.reference_evaluation_runtime"
REFERENCE_RUNTIME_SCHEMA_VERSION = 1


class ReferenceEvalRuntimeConfig(BaseModel):
    """In-memory SQLite + toy experiment + graph rollout driver."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partial_log_path: StrictStr | None = None
    prompt_cache_path: StrictStr | None = None
    row_job_entrypoint: StrictStr = "whetstone.eval.drivers.graph_worker:run_row"
    driver_mode: Literal["in_process", "subprocess"] = "in_process"
    #: Per-row wall budget for the subprocess driver. A row outrunning it is
    #: killed in its worker and reported as a unit timeout.
    unit_deadline_seconds: PositiveFloat = 86_400.0
    env_name: StrictStr = "whetstone.toy"
    split_role: StrictStr = "internal_eval"
    transport_api_key_env: StrictStr = "WHETSTONE_TOY_API_KEY"
    provider_kind: ProviderKind = ProviderKind.OPENAI

    @property
    def execution_policy(self) -> ProviderExecutionPolicy:
        transport = default_transport_policy(
            api_key_env=self.transport_api_key_env,
            provider_kind=self.provider_kind,
        )
        return ProviderExecutionPolicy(transport_policy=transport)

    def build_engine(
        self,
        store: ObjectStore,
        *,
        experiment: Experiment | None = None,
        eval_runner: EvalProcedureRunner | None = None,
        mutation_field: str | None = None,
        render_contract: TemplateRenderContract | None = None,
        transport_factory: (
            Callable[[ProviderExecutionPolicy], TransportCall] | None
        ) = None,
    ) -> EvalEngine:
        _ = self.env_name
        resolved_experiment = experiment or build_toy_experiment()
        try:
            sampling = resolved_experiment.eval_configs.split_for(
                self.split_role
            )
        except KeyError:
            raise ValueError(
                f"unknown split role {self.split_role!r}"
            ) from None
        execution_policy = self.execution_policy
        runner = eval_runner or FakeEvalProcedureRunner()
        field = mutation_field or TOY_MUTATION_FIELD
        contract = render_contract or toy_template_render_contract()
        factory = transport_factory or fake_llm_transport_factory

        if self.driver_mode == "subprocess":
            driver = SubprocessGraphRolloutEvalDriver(
                row_job_entrypoint=self.row_job_entrypoint,
                transport_api_key_env=self.transport_api_key_env,
                unit_deadline_seconds=self.unit_deadline_seconds,
                eval_runner=runner,
                mutation_field=field,
                render_contract=contract,
                transport_factory=factory,
            )
        else:
            driver = GraphRolloutEvalDriver(
                eval_runner=runner,
                mutation_field=field,
                render_contract=contract,
                transport_factory=factory,
            )
        partial_log = None
        if self.partial_log_path is not None:
            partial_path = Path(self.partial_log_path).resolve()
            ensure_private_directory(partial_path.parent)
            partial_log = PartialLog(partial_path)
        prompt_cache = None
        if self.prompt_cache_path is not None:
            cache_root = Path(self.prompt_cache_path).resolve()
            ensure_private_directory(cache_root)
            prompt_cache = PromptResultCache(root=cache_root)
        return RuntimeEvalEngine(
            store=store,
            experiment=resolved_experiment,
            sampling=sampling,
            execution_policy=execution_policy,
            driver=driver,
            partial_log=partial_log,
            prompt_cache=prompt_cache,
        )


__all__ = [
    "REFERENCE_RUNTIME_SCHEMA",
    "REFERENCE_RUNTIME_SCHEMA_VERSION",
    "ReferenceEvalRuntimeConfig",
]
