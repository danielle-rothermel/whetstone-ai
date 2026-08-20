from __future__ import annotations

from dr_store import ObjectStore

from whetstone.optim.contracts import OptimEvalRequest

PLATFORM_DEFERRED_INTENTS_PREFIX = "whetstone.platform_deferred_intents:"
PLATFORM_DEFERRED_INTENTS_SCHEMA = "whetstone.platform_deferred_intents"


def deferred_intents_binding_key(run_id: str, step_index: int) -> str:
    return f"{PLATFORM_DEFERRED_INTENTS_PREFIX}{run_id}:{step_index}"


def persist_deferred_intents(
    store: ObjectStore,
    *,
    run_id: str,
    step_index: int,
    intents: tuple[OptimEvalRequest, ...],
) -> None:
    payload = [intent.model_dump(mode="json") for intent in intents]
    reference, _ = store.put(
        PLATFORM_DEFERRED_INTENTS_SCHEMA,
        {"intents": payload},
    )
    store.bind(
        deferred_intents_binding_key(run_id, step_index),
        reference,
    )


def load_persisted_deferred_intents(
    store: ObjectStore,
    *,
    run_id: str,
    step_index: int,
) -> tuple[OptimEvalRequest, ...]:
    binding = store.resolve(deferred_intents_binding_key(run_id, step_index))
    if binding is None:
        return ()
    record = store.get(binding)
    if not isinstance(record, dict):
        return ()
    return tuple(
        OptimEvalRequest.model_validate(item)
        for item in record.get("intents", ())
    )


def evict_deferred_intents(
    store: ObjectStore,
    *,
    run_id: str,
    step_index: int,
) -> None:
    store.evict_bindings([deferred_intents_binding_key(run_id, step_index)])


__all__ = [
    "PLATFORM_DEFERRED_INTENTS_PREFIX",
    "PLATFORM_DEFERRED_INTENTS_SCHEMA",
    "deferred_intents_binding_key",
    "evict_deferred_intents",
    "load_persisted_deferred_intents",
    "persist_deferred_intents",
]
