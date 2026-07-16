from __future__ import annotations

from typing import Any

import pytest
from dr_graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    NodeConfig,
    NodeSpec,
)

from whetstone.eval_failures import PermanentFailureError
from whetstone.platform.prompts import (
    USER_PROMPT_TEMPLATE_KEY,
    NodePromptSpec,
    build_node_messages,
    node_prompt_spec,
)


def _node(
    *,
    parameters: dict[str, Any] | None = None,
    user_prompt_template: str | None = "{prompt}",
) -> NodeSpec:
    resolved_parameters: dict[str, Any] = dict(parameters or {})
    if user_prompt_template is not None:
        resolved_parameters.setdefault(
            USER_PROMPT_TEMPLATE_KEY,
            user_prompt_template,
        )
    return NodeSpec(
        id="direct",
        op="llm_call",
        config=NodeConfig(
            fields=(
                FieldSpec(name="prompt", role=FieldRole.INPUT),
                FieldSpec(name="output", role=FieldRole.OUTPUT),
            ),
            input_bindings={
                "prompt": BindingRef.model_validate("task.prompt"),
            },
            output_field="output",
            parameters=resolved_parameters,
        ),
    )


def test_node_prompt_spec_rejects_missing_user_prompt_template() -> None:
    with pytest.raises(PermanentFailureError) as error:
        node_prompt_spec(_node(parameters={}, user_prompt_template=None))

    assert "missing user_prompt_template" in str(error.value)
    assert error.value.metadata["metadata_key"] == USER_PROMPT_TEMPLATE_KEY


@pytest.mark.parametrize("bad_value", [123, [], {}])
def test_node_prompt_spec_rejects_non_string_user_prompt_template(
    bad_value: object,
) -> None:
    with pytest.raises(PermanentFailureError) as error:
        node_prompt_spec(
            _node(
                parameters={USER_PROMPT_TEMPLATE_KEY: bad_value},
                user_prompt_template=None,
            )
        )

    assert "missing user_prompt_template" in str(error.value)


def test_node_prompt_spec_rejects_non_string_system_prompt() -> None:
    with pytest.raises(PermanentFailureError) as error:
        node_prompt_spec(
            _node(
                parameters={
                    USER_PROMPT_TEMPLATE_KEY: "{prompt}",
                    "system_prompt": 123,
                },
                user_prompt_template=None,
            )
        )

    assert "system_prompt must be a string" in str(error.value)
    assert error.value.metadata["metadata_key"] == "system_prompt"


def test_node_prompt_spec_rejects_non_string_provider_config_id() -> None:
    with pytest.raises(PermanentFailureError) as error:
        node_prompt_spec(
            _node(
                parameters={
                    USER_PROMPT_TEMPLATE_KEY: "{prompt}",
                    "provider_config_id": [],
                },
                user_prompt_template=None,
            )
        )

    assert "provider_config_id must be a string" in str(error.value)
    assert error.value.metadata["metadata_key"] == "provider_config_id"


def test_node_prompt_spec_happy_path() -> None:
    spec = node_prompt_spec(
        _node(
            parameters={
                USER_PROMPT_TEMPLATE_KEY: "Hello {prompt}",
                "system_prompt": "You are helpful.",
                "provider_config_id": "main",
            },
            user_prompt_template=None,
        )
    )

    assert spec == NodePromptSpec(
        user_prompt_template="Hello {prompt}",
        system_prompt="You are helpful.",
        provider_config_id="main",
    )


def test_build_node_messages_rejects_missing_template_input() -> None:
    with pytest.raises(PermanentFailureError) as error:
        build_node_messages(
            node=_node(user_prompt_template="{missing}"),
            node_inputs={},
        )

    assert "missing input" in str(error.value)
    assert error.value.metadata["missing_input"] == "'missing'"


def test_build_node_messages_rejects_malformed_format_string() -> None:
    with pytest.raises(PermanentFailureError) as error:
        build_node_messages(
            node=_node(user_prompt_template="{unclosed"),
            node_inputs={},
        )

    assert "malformed" in str(error.value)
    assert isinstance(error.value.underlying, ValueError)
