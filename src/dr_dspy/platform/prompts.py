from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr

from dr_dspy.eval_failures import PermanentFailureError
from dr_dspy.graph import NodeSpec
from dr_dspy.lm.boundary import PlainPromptAdapter, PromptMessage

SYSTEM_PROMPT_KEY = "system_prompt"
USER_PROMPT_TEMPLATE_KEY = "user_prompt_template"
PROVIDER_CONFIG_ID_KEY = "provider_config_id"


class NodePromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_prompt_template: StrictStr
    system_prompt: StrictStr | None = None
    provider_config_id: StrictStr | None = None


def node_prompt_spec(node: NodeSpec) -> NodePromptSpec:
    metadata = node.config.metadata
    user_prompt_template = metadata.get(USER_PROMPT_TEMPLATE_KEY)
    if not isinstance(user_prompt_template, str):
        raise PermanentFailureError(
            "node prompt metadata missing user_prompt_template",
            metadata={
                "node_id": node.id,
                "metadata_key": USER_PROMPT_TEMPLATE_KEY,
            },
        )

    system_prompt = metadata.get(SYSTEM_PROMPT_KEY)
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise PermanentFailureError(
            "node prompt metadata system_prompt must be a string",
            metadata={"node_id": node.id, "metadata_key": SYSTEM_PROMPT_KEY},
        )

    provider_config_id = metadata.get(PROVIDER_CONFIG_ID_KEY)
    if provider_config_id is not None and not isinstance(
        provider_config_id, str
    ):
        raise PermanentFailureError(
            "node prompt metadata provider_config_id must be a string",
            metadata={
                "node_id": node.id,
                "metadata_key": PROVIDER_CONFIG_ID_KEY,
            },
        )

    return NodePromptSpec(
        user_prompt_template=user_prompt_template,
        system_prompt=system_prompt,
        provider_config_id=provider_config_id,
    )


def build_node_messages(
    *,
    node: NodeSpec,
    node_inputs: Mapping[str, Any],
) -> tuple[PromptMessage, ...]:
    prompt_spec = node_prompt_spec(node)
    adapter = PlainPromptAdapter(output_field=node.config.output_field)
    return adapter.messages(
        system_content=prompt_spec.system_prompt,
        user_content=_format_user_prompt(
            node=node,
            template=prompt_spec.user_prompt_template,
            node_inputs=node_inputs,
        ),
    )


def _format_user_prompt(
    *,
    node: NodeSpec,
    template: str,
    node_inputs: Mapping[str, Any],
) -> str:
    try:
        return template.format_map(_InputFormatMapping(node_inputs))
    except KeyError as error:
        missing = str(error)
        raise PermanentFailureError(
            "node prompt template references a missing input",
            underlying=error,
            metadata={"node_id": node.id, "missing_input": missing},
        ) from error


class _InputFormatMapping(dict[str, Any]):
    def __init__(self, values: Mapping[str, Any]) -> None:
        super().__init__(values)

    def __missing__(self, key: str) -> Any:
        raise KeyError(key)
