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
from whetstone.platform.spec_builder import (
    HUMANEVAL_DECODER_USER_PROMPT_TEMPLATE,
    HUMANEVAL_ENCODER_USER_PROMPT_TEMPLATE,
    humaneval_encdec_graph,
)


def _node(
    *,
    metadata: dict[str, Any] | None = None,
    user_prompt_template: str | None = "{prompt}",
) -> NodeSpec:
    resolved_metadata: dict[str, Any] = dict(metadata or {})
    if user_prompt_template is not None:
        resolved_metadata.setdefault(
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
            metadata=resolved_metadata,
        ),
    )


def test_node_prompt_spec_rejects_missing_user_prompt_template() -> None:
    with pytest.raises(PermanentFailureError) as error:
        node_prompt_spec(_node(metadata={}, user_prompt_template=None))

    assert "missing user_prompt_template" in str(error.value)
    assert error.value.metadata["metadata_key"] == USER_PROMPT_TEMPLATE_KEY


@pytest.mark.parametrize("bad_value", [123, [], {}])
def test_node_prompt_spec_rejects_non_string_user_prompt_template(
    bad_value: object,
) -> None:
    with pytest.raises(PermanentFailureError) as error:
        node_prompt_spec(
            _node(
                metadata={USER_PROMPT_TEMPLATE_KEY: bad_value},
                user_prompt_template=None,
            )
        )

    assert "missing user_prompt_template" in str(error.value)


def test_node_prompt_spec_rejects_non_string_system_prompt() -> None:
    with pytest.raises(PermanentFailureError) as error:
        node_prompt_spec(
            _node(
                metadata={
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
                metadata={
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
            metadata={
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


def test_humaneval_encdec_prompts_format_without_missing_inputs() -> None:
    graph = humaneval_encdec_graph()
    encoder = graph.node("encoder")
    decoder = graph.node("decoder")

    encoder_messages = build_node_messages(
        node=encoder,
        node_inputs={
            "instructions_start": "Describe this code briefly.",
            "budget": 42,
            "gt_code": "def add_one(x):\n    return x + 1\n",
            "instructions_end": "",
        },
    )
    encoder_user = encoder_messages[-1].content
    assert "Describe this code briefly." in encoder_user
    assert "Use at most 42 characters." in encoder_user
    assert "```python" in encoder_user
    assert "def add_one(x):" in encoder_user
    assert (
        encoder.config.metadata["user_prompt_template"]
        == HUMANEVAL_ENCODER_USER_PROMPT_TEMPLATE
    )

    decoder_messages = build_node_messages(
        node=decoder,
        node_inputs={"encoded_desc": "increment input by one"},
    )
    decoder_user = decoder_messages[-1].content
    assert "Write functional code in Python" in decoder_user
    assert "increment input by one" in decoder_user
    assert (
        decoder.config.metadata["user_prompt_template"]
        == HUMANEVAL_DECODER_USER_PROMPT_TEMPLATE
    )
