from __future__ import annotations

from dr_store import ObjectStore
from dr_store.content_addressing import format_object_reference, parse_object_reference

OPTIM_WORK_STATE_HEAD_PREFIX = "whetstone.optim_work_state_head:"


def work_state_head_binding_key(run_id: str, work_key: str) -> str:
    return f"{OPTIM_WORK_STATE_HEAD_PREFIX}{run_id}:{work_key}"


def bind_work_state_head(
    store: ObjectStore,
    *,
    run_id: str,
    work_key: str,
    work_state_ref: str,
) -> None:
    key = work_state_head_binding_key(run_id, work_key)
    store.evict_bindings([key])
    store.bind(key, parse_object_reference(work_state_ref))


def resolve_work_state_head(
    store: ObjectStore,
    *,
    run_id: str,
    work_key: str,
) -> str | None:
    binding = store.resolve(work_state_head_binding_key(run_id, work_key))
    if binding is None:
        return None
    return format_object_reference(binding)


def evict_work_state_head(
    store: ObjectStore,
    *,
    run_id: str,
    work_key: str,
) -> None:
    store.evict_bindings([work_state_head_binding_key(run_id, work_key)])


__all__ = [
    "OPTIM_WORK_STATE_HEAD_PREFIX",
    "bind_work_state_head",
    "evict_work_state_head",
    "resolve_work_state_head",
    "work_state_head_binding_key",
]
