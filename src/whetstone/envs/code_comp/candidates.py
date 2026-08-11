from __future__ import annotations

from whetstone.core.identity import TypedRef, typed_ref_for_record
from whetstone.envs.code_comp.constants import CODE_COMP_ENV_NAME

ENV_CANDIDATE_BASE_SCHEMA = "whetstone.env_candidate_base"


def env_candidate_base_ref(env_name: str = CODE_COMP_ENV_NAME) -> TypedRef:
    """Address the immutable synthetic base binding for one environment."""
    return typed_ref_for_record(
        ENV_CANDIDATE_BASE_SCHEMA,
        {"env_name": env_name},
    )


__all__ = [
    "ENV_CANDIDATE_BASE_SCHEMA",
    "env_candidate_base_ref",
]
