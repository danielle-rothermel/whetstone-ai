"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.input_arms.
"""

from whetstone.envs.code_comp.input_arms import (
    DEFAULT_RENAME_TOKEN,
    DIRECT_ARMS,
    DIRECT_PROMPT_INSTRUCTION,
    NAME_ONLY_WRAPPER,
    PromptParts,
    direct_body,
    direct_prompt,
    rename_identifier,
    renamed_task,
    split_prompt,
)

__all__ = [
    "DEFAULT_RENAME_TOKEN",
    "DIRECT_ARMS",
    "DIRECT_PROMPT_INSTRUCTION",
    "NAME_ONLY_WRAPPER",
    "PromptParts",
    "direct_body",
    "direct_prompt",
    "rename_identifier",
    "renamed_task",
    "split_prompt",
]
