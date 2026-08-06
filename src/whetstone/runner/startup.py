"""The one place the runner registers durable capabilities before DBOS launch.

Everything a recovered workflow may need must already be registered when
``DBOS.launch()`` returns, because recovery starts immediately and resolves its
dependencies by identity. This module is that registration site -- the single
one -- and :func:`register_runtime` is the only function in the runner that
calls a ``register_*`` function.

**Why exactly one site.** Registration keyed by identity is safe only while a
given identity maps to a single object. Spreading registration across the
optimizer factories that happen to need it makes that invariant unenforceable:
two factories can each register a transport under the same key, and whichever
ran last wins silently. The proposal transport is registered here, once, for
every optimizer that proposes -- COPRO, MIPROv2, and GEPA alike -- and the GEPA
adapter factory is registered here too rather than inside
``CanonicalGepaAdapterFactory.create``.

``RunnerLaunch`` supplies controllers and GEPA factories that are already
constructed. :func:`register_runtime` registers the proposal transport, mints
and returns a durable proposal executor for that transport, then registers
those preconstructed capabilities. The public zero-argument CLI factory path
constructs the whole launch before registration and does not feed the returned
executor back into controller or factory construction. The registration site
therefore does not currently establish that those supplied capabilities use
the executor it minted.

**Two parent-workflow paths, deliberately.** Harness-driven optimizers run
through :class:`HarnessRunController` under the optimizer-agnostic parent run
workflow. GEPA instead owns ``DbosGepaRunner``, its own stable parent workflow
that replays a frozen engine run from ordinal 0; adapting it onto the harness
controller would mean reimplementing that replay contract. Both are registered
here, so the registration invariant holds across both paths.

**Registration is not launch.** This module never constructs a DBOS app and
never launches one. It runs strictly before launch and returns the handles the
lifecycle needs, which keeps the ordering rule -- register, then launch -- a
property of the call site rather than a comment.
"""

from __future__ import annotations

from dataclasses import dataclass

from whetstone.coordination.proposal_provider import (
    DbosProposalExecutor,
    register_proposal_transport,
)
from whetstone.coordination.run_workflow import (
    RunController,
    register_run_controller,
)
from whetstone.optimization.gepa.factory import CanonicalGepaAdapterFactory
from whetstone.optimization.gepa.runner import register_gepa_adapter_factory
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    ProviderProposerTransport,
)

__all__ = ["RegisteredRuntime", "register_runtime"]


@dataclass(frozen=True, slots=True)
class RegisteredRuntime:
    """What the single registration site bound, and under which identities.

    ``proposal_executor`` is the executor minted for the registered transport;
    ``transport_registry_key`` is the key it resolves at call time. Controllers
    and GEPA factories arrive preconstructed, so this result records what was
    bound but does not establish which executor those objects use. The identity
    hashes let a caller assert the registrations rather than infer them.
    """

    transport_registry_key: str
    proposal_executor: DurableProposalExecutor
    controller_identity_hashes: tuple[str, ...] = ()
    gepa_factory_identity_hashes: tuple[str, ...] = ()


def register_runtime(
    *,
    transport: ProviderProposerTransport,
    controllers: tuple[RunController, ...] = (),
    gepa_factories: tuple[CanonicalGepaAdapterFactory, ...] = (),
) -> RegisteredRuntime:
    """Register supplied durable capabilities and mint a proposal executor.

    ``controllers`` and ``gepa_factories`` are preconstructed. This function
    registers them but neither constructs nor rewires them; the returned
    result exposes the executor but does not apply it to those supplied
    objects.

    Call once, before ``DBOS.launch()``. Registering the identical object again
    is a no-op; binding a different object to an already-bound identity is
    refused by each registry, so a second call with drifted inputs fails loudly
    instead of silently replacing a capability a recovered workflow depends on.
    """
    registry_key = register_proposal_transport(transport)
    executor = DbosProposalExecutor(transport_registry_key=registry_key)
    controller_identities = tuple(
        register_run_controller(controller) for controller in controllers
    )
    factory_identities: list[str] = []
    for factory in gepa_factories:
        register_gepa_adapter_factory(factory)
        factory_identities.append(factory.runtime_identity_hash)
    return RegisteredRuntime(
        transport_registry_key=registry_key,
        proposal_executor=executor,
        controller_identity_hashes=controller_identities,
        gepa_factory_identity_hashes=tuple(factory_identities),
    )
