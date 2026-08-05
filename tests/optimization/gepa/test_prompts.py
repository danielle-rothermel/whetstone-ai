# ruff: noqa: E501

from __future__ import annotations

import pytest
from gepa.strategies.instruction_proposal import (
    InstructionProposalSignature,
)

from whetstone.optimization.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    GepaReflectionRequest,
    MappingGepaPromptRegistry,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _component(
    name: str = "user_prompt_template",
    *,
    schema_hash: str = _HASH_A,
) -> GepaComponentFormat:
    return GepaComponentFormat(
        component_name=name,
        component_schema_identity_hash=schema_hash,
        allowed_placeholders=("question", "context"),
        required_placeholders=("question",),
        placeholder_semantics={
            "question": "The literal user question.",
            "context": "Optional retrieved context.",
        },
        rendering_rules=(
            "Preserve every placeholder byte-for-byte.",
            "Use braces only for declared placeholders.",
        ),
        output_contract="Return only the replacement component text.",
    )


def _descriptor(
    *components: GepaComponentFormat,
) -> GepaPromptFormatDescriptor:
    return GepaPromptFormatDescriptor(
        format_name="native-chat-template",
        components=components or (_component(),),
    )


def _request(
    *,
    component_name: str = "user_prompt_template",
    examples: tuple[dict, ...] | None = None,
    components: tuple[str, ...] | None = None,
) -> GepaReflectionRequest:
    active_examples = examples or (
        {
            "Inputs": {"question": "What is 2 + 2?"},
            "Generated Outputs": {"answer": "5"},
            "Feedback": "The arithmetic is incorrect.",
        },
    )
    selected = components or (component_name,)
    candidate = {
        name: (
            "Answer {question} using {context}."
            if name == "user_prompt_template"
            else "Be concise."
        )
        for name in selected
    }
    return GepaReflectionRequest(
        candidate=candidate,
        reflective_dataset={name: active_examples for name in selected},
        components_to_update=selected,
        component_name=component_name,
    )


def test_text_reflection_prompt_snapshot_preserves_gepa_semantics() -> None:
    rendered = NativeGepaReflectionPromptBuilder().render(
        _descriptor(),
        _request(),
    )

    assert rendered.messages is None
    assert (
        rendered.text
        == """I provided an assistant with the following instructions to perform a task for me:
```
Answer {question} using {context}.
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
# Example 1
## Inputs
### question
What is 2 + 2?

## Generated Outputs
### answer
5

## Feedback
The arithmetic is incorrect.


```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Native prompt-format constraints:
- Component: user_prompt_template
- Allowed placeholders: {question}, {context}
- Required placeholders: {question}
- {question}: The literal user question.
- {context}: Optional retrieved context.
- Rendering rule: Preserve every placeholder byte-for-byte.
- Rendering rule: Use braces only for declared placeholders.
- Output contract: Return only the replacement component text.
- Do not add DSPy Signature field descriptions or output-prefix formatting.

Provide the new instructions within ``` blocks."""
    )
    assert "InputField" not in rendered.text
    assert "OutputField" not in rendered.text

    upstream = InstructionProposalSignature.prompt_renderer(
        {
            "current_instruction_doc": ("Answer {question} using {context}."),
            "dataset_with_feedback": (
                {
                    "Inputs": {"question": "What is 2 + 2?"},
                    "Generated Outputs": {"answer": "5"},
                    "Feedback": "The arithmetic is incorrect.",
                },
            ),
            "prompt_template": None,
        }
    )
    assert isinstance(upstream, str)
    prefix, native_tail = rendered.text.split(
        "\n\nNative prompt-format constraints:",
        maxsplit=1,
    )
    _, suffix = native_tail.split(
        "\n\nProvide the new instructions within ``` blocks.",
        maxsplit=1,
    )
    assert (
        prefix + "\n\nProvide the new instructions within ``` blocks." + suffix
        == upstream
    )


def test_format_failure_reflection_and_parser_snapshots() -> None:
    examples = (
        {
            "Inputs": {"question": "State the answer."},
            "Generated Outputs": (
                "Couldn't parse the output as per the expected output format. "
                "The raw response was:\n```\nunstructured\n```"
            ),
            "Feedback": (
                "Your output failed to parse. Return a JSON object with an "
                "`answer` key."
            ),
        },
    )
    rendered = NativeGepaReflectionPromptBuilder().render(
        _descriptor(),
        _request(examples=examples),
    )
    parser = NativeGepaReflectionResponseParser()

    assert "Couldn't parse the output" in rendered.text
    assert "Your output failed to parse" in rendered.text
    valid_responses = (
        "analysis\n```text\nUse {question} exactly.\n```",
        "```text\nUse {question} exactly.",
        "Use {question} exactly.```",
        "Use {question} exactly.",
    )
    for raw_response in valid_responses:
        expected = InstructionProposalSignature.output_extractor(raw_response)[
            "new_instruction"
        ]
        assert parser.parse(raw_response) == expected
    with pytest.raises(ValueError, match="empty component"):
        parser.parse("   ")
    with pytest.raises(ValueError, match="empty component"):
        parser.parse("```\n```")


def test_multimodal_reflection_snapshot_preserves_structured_part() -> None:
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    examples = (
        {
            "Inputs": {
                "question": "What shape is shown?",
                "image": image,
            },
            "Generated Outputs": {"answer": "square"},
            "Feedback": "The image contains a circle.",
        },
    )
    rendered = NativeGepaReflectionPromptBuilder().render(
        _descriptor(),
        _request(examples=examples),
    )

    assert rendered.messages == (
        {
            "role": "user",
            "content": (
                {"type": "text", "text": rendered.text},
                image,
            ),
        },
    )
    assert "structured multimodal content (1 media part(s))" in rendered.text
    assert "[MEDIA-1 — see structured content]" in rendered.text
    assert "data:image/png" not in rendered.text


def test_typed_but_textual_part_renders_its_text_instead_of_media() -> None:
    """A media ``type`` alone must not divert text out of the prompt.

    Classifying on ``type`` alone projected textual dicts into structured
    content, so their text never reached the reflection prompt.
    """

    textual = {
        "type": "image",
        "label": "diagram-7",
        "caption": "A circle inscribed in a square.",
    }
    examples = (
        {
            "Inputs": {"question": "What shape is shown?", "image": textual},
            "Generated Outputs": {"answer": "square"},
            "Feedback": "The image contains a circle.",
        },
    )
    rendered = NativeGepaReflectionPromptBuilder().render(
        _descriptor(),
        _request(examples=examples),
    )

    # No structured content part is emitted, so the prompt stays plain text.
    assert rendered.messages is None
    assert "[MEDIA-1 — see structured content]" not in rendered.text
    assert "diagram-7" in rendered.text
    assert "A circle inscribed in a square." in rendered.text


def test_multi_component_registry_and_prompt_order_snapshot() -> None:
    descriptor = _descriptor(
        _component(),
        GepaComponentFormat(
            component_name="system_instruction",
            component_schema_identity_hash=_HASH_B,
            rendering_rules=("Do not introduce template variables.",),
            output_contract="Return one system instruction.",
        ),
    )
    builder = NativeGepaReflectionPromptBuilder()
    parser = NativeGepaReflectionResponseParser()
    services = GepaPromptServices(
        descriptor=descriptor,
        reflection_builder=builder,
        reflection_parser=parser,
    )
    registry = MappingGepaPromptRegistry((services,))
    request = _request(
        component_name="system_instruction",
        components=("user_prompt_template", "system_instruction"),
    )

    resolved = registry.resolve(descriptor.identity_hash())
    rendered = resolved.reflection_builder.render(descriptor, request)

    assert resolved is services
    assert [item.component_name for item in descriptor.components] == [
        "user_prompt_template",
        "system_instruction",
    ]
    assert services.binding.prompt_format_identity_hash == (
        descriptor.identity_hash()
    )
    assert "```\nBe concise.\n```" in rendered.text
    assert "- Component: system_instruction" in rendered.text
    assert "- Allowed placeholders: (none)" in rendered.text
    assert "- Required placeholders: (none)" in rendered.text
    assert (
        "- Rendering rule: Do not introduce template variables."
        in rendered.text
    )


def test_prompt_identity_changes_with_format_builder_or_parser() -> None:
    first = GepaPromptServices(
        descriptor=_descriptor(),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )
    changed_format = GepaPromptServices(
        descriptor=_descriptor(
            _component().model_copy(
                update={
                    "rendering_rules": (
                        "Escape braces before provider rendering.",
                    )
                }
            )
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )

    assert first.descriptor.identity_hash() != (
        changed_format.descriptor.identity_hash()
    )
    assert first.binding.identity_hash() != (
        changed_format.binding.identity_hash()
    )
    assert (
        first.parse_replacement(
            "user_prompt_template",
            "```text\nUse {question} and optional {context}.\n```",
        )
        == "Use {question} and optional {context}."
    )
    parsed_with_literal_fence = (
        "Use {question}, then emit this example:\n```json\n{{}}\n```"
    )
    assert (
        first.validate_replacement(
            "user_prompt_template",
            parsed_with_literal_fence,
        )
        == parsed_with_literal_fence
    )
    with pytest.raises(ValueError, match="undeclared placeholders"):
        first.parse_replacement(
            "user_prompt_template",
            "Use {question} and {invented}.",
        )
    assert (
        first.parse_replacement(
            "user_prompt_template",
            "Use {question!r} within {context:>20}.",
        )
        == "Use {question!r} within {context:>20}."
    )
    with pytest.raises(ValueError, match="malformed braces"):
        first.parse_replacement(
            "user_prompt_template",
            "Use {question.",
        )
    with pytest.raises(ValueError, match="omitted required"):
        first.parse_replacement(
            "user_prompt_template",
            "Answer concisely.",
        )
    with pytest.raises(ValueError, match="empty component"):
        first.parse_replacement("user_prompt_template", "")

    repeated_required = GepaComponentFormat(
        component_name="repeated",
        component_schema_identity_hash=_HASH_A,
        allowed_placeholders=("question",),
        required_placeholders=("question", "question"),
    )
    with pytest.raises(ValueError, match="omitted required"):
        repeated_required.validate_replacement("Use {question}.")
    assert (
        repeated_required.validate_replacement(
            "Compare {question} with {question!r}."
        )
        == "Compare {question} with {question!r}."
    )

    source_semantics = {"question": "Original meaning."}
    frozen_component = GepaComponentFormat(
        component_name="immutable",
        component_schema_identity_hash=_HASH_A,
        allowed_placeholders=("question",),
        placeholder_semantics=source_semantics,
    )
    identity_before = frozen_component.identity_hash()
    source_semantics["question"] = "Mutated after construction."
    assert frozen_component.identity_hash() == identity_before
    assert frozen_component.placeholder_semantics == (
        ("question", "Original meaning."),
    )
