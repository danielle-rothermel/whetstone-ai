from __future__ import annotations

try:
    from dbos import DBOS, SetWorkflowID
except ImportError as exc:
    raise ImportError(
        "DBOS coordination requires the optional dbos extra: "
        "pip install 'whetstone-ai[dbos]'"
    ) from exc
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optim.gepa.adapter import GepaOptimizer, GepaPersistedRun
from whetstone.optim.gepa.contracts import GepaDataInstance
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.factory import CanonicalGepaAdapterFactory

GEPA_PARENT_RUN_SCHEMA = "whetstone.gepa.parent_run"
GEPA_PARENT_RUN_SCHEMA_VERSION = 1

_GEPA_FACTORIES: dict[str, CanonicalGepaAdapterFactory] = {}


class GepaParentRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    factory_identity_hash: StrictStr
    control: GepaControl
    seed_candidate: dict[StrictStr, StrictStr] = Field(min_length=1)
    trainset: tuple[GepaDataInstance, ...]
    valset: tuple[GepaDataInstance, ...] | None = None

    @model_validator(mode="after")
    def _validate(self) -> GepaParentRunRequest:
        require_full_hash(
            self.factory_identity_hash,
            field="factory_identity_hash",
        )
        if not self.trainset:
            raise ValueError("GEPA parent workflow trainset cannot be empty")
        if tuple(self.seed_candidate) != self.control.component_names:
            raise ValueError("GEPA parent workflow seed component order drift")
        if tuple(item.data_id for item in self.trainset) != (
            self.control.trainset_task_hashes
        ):
            raise ValueError("GEPA parent workflow trainset identity drift")

        if self.valset is None:
            if self.control.source_valset_task_hashes is not None:
                raise ValueError(
                    "GEPA parent workflow omitted its bound valset"
                )
        elif self.control.source_valset_task_hashes is None:
            raise ValueError("GEPA parent workflow supplied an unbound valset")
        elif tuple(item.data_id for item in self.valset) != (
            self.control.valset_task_hashes
        ):
            raise ValueError("GEPA parent workflow valset identity drift")
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_PARENT_RUN_SCHEMA,
            schema_version=GEPA_PARENT_RUN_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


def register_gepa_adapter_factory(
    factory: CanonicalGepaAdapterFactory,
) -> None:

    identity_hash = factory.runtime_hash
    require_full_hash(identity_hash, field="GEPA factory identity")
    existing = _GEPA_FACTORIES.get(identity_hash)
    if existing is not None and existing is not factory:
        raise ValueError("GEPA factory identity is already bound")
    _GEPA_FACTORIES[identity_hash] = factory


def _registered_factory(
    request: GepaParentRunRequest,
) -> CanonicalGepaAdapterFactory:
    try:
        factory = _GEPA_FACTORIES[request.factory_identity_hash]
    except KeyError:
        raise RuntimeError(
            "GEPA adapter factory is not registered before DBOS launch"
        ) from None
    if factory.runtime_hash != request.factory_identity_hash:
        raise RuntimeError("registered GEPA factory identity drifted")
    return factory


@DBOS.workflow()
def _gepa_parent_workflow(
    request: GepaParentRunRequest,
) -> GepaPersistedRun:
    factory = _registered_factory(request)
    return GepaOptimizer(
        control=request.control,
        adapter_factory=factory,
    ).run_detailed(
        seed_candidate=request.seed_candidate,
        trainset=request.trainset,
        valset=request.valset,
    )


class DbosGepaRunner:
    def run(self, request: GepaParentRunRequest) -> GepaPersistedRun:
        import warnings

        warnings.warn(
            "DbosGepaRunner is deprecated; use HarnessRunController or "
            "whetstone.platform.submit.submit_optim_run instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        workflow_id = f"whetstone-gepa-run-{request.identity_hash()}"
        with SetWorkflowID(workflow_id):
            result = _gepa_parent_workflow(request)
        return GepaPersistedRun.model_validate(result)


__all__ = [
    "GEPA_PARENT_RUN_SCHEMA",
    "GEPA_PARENT_RUN_SCHEMA_VERSION",
    "DbosGepaRunner",
    "GepaParentRunRequest",
    "register_gepa_adapter_factory",
]
