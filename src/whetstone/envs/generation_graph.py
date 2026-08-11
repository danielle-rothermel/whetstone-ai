from __future__ import annotations

from dr_providers import ProviderCallConfig, openrouter_chat_config

#: The Provider Call Config schema name (referenced by the LLM Call Node's
#: static Variable typed reference).
PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"

#: The single Graph External Input the LLM Call Node's prompt binds to: the
#: rendered prompt for the selected candidate against a task's external
#: inputs. The env task's ``task.<field>`` prompt inputs feed the render.
PROMPT_EXTERNAL_INPUT = "task.prompt"

#: Node ids for the two-node graph.
LLM_NODE_ID = "generate"
EVAL_NODE_ID = "evaluate"


def build_provider_call_config(model: str) -> ProviderCallConfig:
    """The native OpenRouter Provider Call Config for a task model."""
    return openrouter_chat_config(model=model)


__all__ = [
    "EVAL_NODE_ID",
    "LLM_NODE_ID",
    "PROMPT_EXTERNAL_INPUT",
    "PROVIDER_CALL_CONFIG_SCHEMA",
    "build_provider_call_config",
]
