from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.proposal.mutation import (
    template_placeholder_fields,
)

GEPA_PROMPT_FORMAT_SCHEMA = "whetstone.gepa.prompt_format"
GEPA_PROMPT_FORMAT_SCHEMA_VERSION = 1
GEPA_PROMPT_BINDING_SCHEMA = "whetstone.gepa.prompt_binding"
GEPA_PROMPT_BINDING_SCHEMA_VERSION = 1
GEPA_REFLECTION_PROMPT_SCHEMA = "whetstone.gepa.reflection_prompt"
GEPA_REFLECTION_PROMPT_SCHEMA_VERSION = 1
GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA = (
    "whetstone.gepa.reflection_response_parser"
)
GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA_VERSION = 1

GEPA_REFLECTION_PROMPT_ROLE = (
    "I provided an assistant with the following instructions to perform a "
    "task for me:"
)
GEPA_REFLECTION_EXAMPLES_ROLE = (
    "The following are examples of different task inputs provided to the "
    "assistant along with the assistant's response for each of them, and "
    "some feedback on how the assistant's response could be better:"
)
GEPA_REFLECTION_TASK = """Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well."""


class GepaComponentFormat(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_name: StrictStr
    component_schema_identity_hash: StrictStr
    allowed_placeholders: tuple[StrictStr, ...] = ()
    required_placeholders: tuple[StrictStr, ...] = ()
    placeholder_semantics: tuple[tuple[StrictStr, StrictStr], ...] = Field(
        default_factory=tuple
    )
    rendering_rules: tuple[StrictStr, ...] = ()
    output_contract: StrictStr = "Return one replacement component."

    @field_validator("placeholder_semantics", mode="before")
    @classmethod
    def _freeze_placeholder_semantics(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, Mapping):
            return tuple(value.items())
        return value

    @model_validator(mode="after")
    def _validate(self) -> GepaComponentFormat:
        if not self.component_name:
            raise ValueError("component_name must be non-empty")
        require_full_hash(
            self.component_schema_identity_hash,
            field="component_schema_identity_hash",
        )
        if len(set(self.allowed_placeholders)) != len(
            self.allowed_placeholders
        ):
            raise ValueError("allowed_placeholders must be unique")
        if any(not name for name in self.allowed_placeholders):
            raise ValueError("allowed placeholder names must be non-empty")
        if any(not name for name in self.required_placeholders):
            raise ValueError("required placeholder names must be non-empty")
        if not set(self.required_placeholders).issubset(
            self.allowed_placeholders
        ):
            raise ValueError(
                "required_placeholders must be allowed placeholders"
            )
        semantic_names = [
            placeholder for placeholder, _ in self.placeholder_semantics
        ]
        if len(set(semantic_names)) != len(semantic_names):
            raise ValueError("placeholder semantics must be unique")
        if set(semantic_names) - set(self.allowed_placeholders):
            raise ValueError(
                "placeholder semantics must name allowed placeholders"
            )
        if not self.output_contract:
            raise ValueError("output_contract must be non-empty")
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema="whetstone.gepa.component_format",
            schema_version=1,
            payload=self.model_dump(mode="json"),
        )

    def validate_replacement(self, text: str) -> str:

        if not text:
            raise ValueError("GEPA replacement component must be non-empty")
        try:
            observed = Counter(template_placeholder_fields(text))
        except ValueError as exc:
            raise ValueError(
                "GEPA replacement component has malformed braces"
            ) from exc
        undeclared = set(observed) - set(self.allowed_placeholders)
        if undeclared:
            raise ValueError(
                "GEPA replacement component introduced undeclared "
                f"placeholders {sorted(undeclared)}"
            )
        required = Counter(self.required_placeholders)
        missing = sorted(
            name for name, count in required.items() if observed[name] < count
        )
        if missing:
            raise ValueError(
                "GEPA replacement component omitted required placeholders "
                f"{missing}"
            )
        return text


class GepaPromptFormatDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format_name: StrictStr
    components: tuple[GepaComponentFormat, ...]

    @model_validator(mode="after")
    def _validate(self) -> GepaPromptFormatDescriptor:
        if not self.format_name:
            raise ValueError("format_name must be non-empty")
        if not self.components:
            raise ValueError("GEPA prompt format requires a component")
        names = [component.component_name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("GEPA component names must be unique")
        return self

    def component(self, name: str) -> GepaComponentFormat:
        for component in self.components:
            if component.component_name == name:
                return component
        raise KeyError(f"unknown GEPA prompt component {name!r}")

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_PROMPT_FORMAT_SCHEMA,
            schema_version=GEPA_PROMPT_FORMAT_SCHEMA_VERSION,
            payload={
                "format_name": self.format_name,
                "components": [
                    {
                        **component.model_dump(mode="json"),
                        "component_format_identity_hash": (
                            component.identity_hash()
                        ),
                    }
                    for component in self.components
                ],
            },
        )


class GepaReflectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: dict[StrictStr, StrictStr]
    reflective_dataset: dict[StrictStr, tuple[dict[str, Any], ...]]
    components_to_update: tuple[StrictStr, ...]
    component_name: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> GepaReflectionRequest:
        if self.component_name not in self.components_to_update:
            raise ValueError(
                "component_name must be selected by components_to_update"
            )
        if self.component_name not in self.candidate:
            raise ValueError("selected component is missing from candidate")
        if self.component_name not in self.reflective_dataset:
            raise ValueError(
                "selected component is missing from reflective dataset"
            )
        if not self.reflective_dataset[self.component_name]:
            raise ValueError("selected component has no reflective examples")
        return self


class GepaRenderedPrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr
    messages: tuple[dict[str, Any], ...] | None = None


class GepaPromptBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_format_identity_hash: StrictStr
    reflection_builder_identity_hash: StrictStr
    reflection_parser_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> GepaPromptBinding:
        for field_name in (
            "prompt_format_identity_hash",
            "reflection_builder_identity_hash",
            "reflection_parser_identity_hash",
        ):
            require_full_hash(getattr(self, field_name), field=field_name)
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_PROMPT_BINDING_SCHEMA,
            schema_version=GEPA_PROMPT_BINDING_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


class GepaReflectionPromptBuilder(Protocol):
    @property
    def identity_hash(self) -> str: ...

    def render(
        self,
        descriptor: GepaPromptFormatDescriptor,
        request: GepaReflectionRequest,
    ) -> GepaRenderedPrompt: ...


class GepaReflectionResponseParser(Protocol):
    @property
    def identity_hash(self) -> str: ...

    def parse(self, raw_response: str) -> str: ...


_MEDIA_PAYLOAD_KEYS = frozenset({"image_url", "url", "data", "input_audio"})
_MEDIA_PART_TYPES = frozenset(
    {"image", "image_url", "input_image", "input_audio", "video"}
)


def _structured_content_part(value: Any) -> dict[str, Any] | None:

    if not isinstance(value, dict):
        return None
    if value.get("type") not in _MEDIA_PART_TYPES:
        return None
    if not _MEDIA_PAYLOAD_KEYS.intersection(value):
        return None
    return dict(value)


def _format_examples(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...]]:

    content_parts: list[dict[str, Any]] = []

    def render_value(value: Any, level: int = 3) -> str:
        content_part = _structured_content_part(value)
        if content_part is not None:
            content_parts.append(content_part)
            return f"[MEDIA-{len(content_parts)} — see structured content]\n\n"
        if isinstance(value, Mapping):
            rendered = ""
            for key, item in value.items():
                rendered += f"{'#' * level} {key}\n"
                rendered += render_value(item, min(level + 1, 6))
            return rendered or "\n"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rendered = ""
            for index, item in enumerate(value, start=1):
                rendered += f"{'#' * level} Item {index}\n"
                rendered += render_value(item, min(level + 1, 6))
            return rendered or "\n"
        return f"{str(value).strip()}\n\n"

    rendered_examples: list[str] = []
    for index, example in enumerate(examples, start=1):
        rendered = f"# Example {index}\n"
        for key, value in example.items():
            rendered += f"## {key}\n"
            rendered += render_value(value)
        rendered_examples.append(rendered)
    text = "\n\n".join(rendered_examples)
    if content_parts:
        text = (
            "The evaluation data below includes structured multimodal content "
            f"({len(content_parts)} media part(s)). Analyze both the text and "
            "media when suggesting improvements.\n\n" + text
        )
    return text, tuple(content_parts)


@dataclass(frozen=True, slots=True)
class NativeGepaReflectionPromptBuilder:
    @property
    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_REFLECTION_PROMPT_SCHEMA,
            schema_version=GEPA_REFLECTION_PROMPT_SCHEMA_VERSION,
            payload={
                "upstream_semantics": "gepa==0.1.1.InstructionProposalSignature",
                "native_format_constraints": True,
                "dspy_signature_field_formatting": False,
                "multimodal_projection": "ordered_structured_content_parts/v2",
            },
        )

    def render(
        self,
        descriptor: GepaPromptFormatDescriptor,
        request: GepaReflectionRequest,
    ) -> GepaRenderedPrompt:
        component = descriptor.component(request.component_name)
        examples, content_parts = _format_examples(
            request.reflective_dataset[request.component_name]
        )
        constraints = [
            "Native prompt-format constraints:",
            f"- Component: {component.component_name}",
            (
                "- Allowed placeholders: "
                + (
                    ", ".join(
                        f"{{{name}}}"
                        for name in component.allowed_placeholders
                    )
                    if component.allowed_placeholders
                    else "(none)"
                )
            ),
            (
                "- Required placeholders: "
                + (
                    ", ".join(
                        f"{{{name}}}"
                        for name in component.required_placeholders
                    )
                    if component.required_placeholders
                    else "(none)"
                )
            ),
        ]
        placeholder_semantics = dict(component.placeholder_semantics)
        for placeholder_name in component.allowed_placeholders:
            semantic = placeholder_semantics.get(placeholder_name)
            if semantic is not None:
                constraints.append(f"- {{{placeholder_name}}}: {semantic}")
        constraints.extend(
            f"- Rendering rule: {rule}" for rule in component.rendering_rules
        )
        constraints.append(f"- Output contract: {component.output_contract}")
        constraints.append(
            "- Do not add DSPy Signature field descriptions or output-prefix "
            "formatting."
        )
        constraints_text = "\n".join(constraints)
        text = (
            f"{GEPA_REFLECTION_PROMPT_ROLE}\n"
            f"```\n{request.candidate[request.component_name]}\n```\n\n"
            f"{GEPA_REFLECTION_EXAMPLES_ROLE}\n"
            f"```\n{examples}\n```\n\n"
            f"{GEPA_REFLECTION_TASK}\n\n"
            f"{constraints_text}\n\n"
            "Provide the new instructions within ``` blocks."
        )
        messages = None
        if content_parts:
            messages = (
                {
                    "role": "user",
                    "content": (
                        {"type": "text", "text": text},
                        *content_parts,
                    ),
                },
            )
        return GepaRenderedPrompt(text=text, messages=messages)


@dataclass(frozen=True, slots=True)
class NativeGepaReflectionResponseParser:
    @property
    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA,
            schema_version=GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA_VERSION,
            payload={
                "algorithm": "gepa==0.1.1 first-last-fence extraction",
                "native_candidate_constraint": "reject_empty_response",
            },
        )

    def parse(self, raw_response: str) -> str:
        start = raw_response.find("```") + 3
        end = raw_response.rfind("```")
        if start >= end:
            stripped = raw_response.strip()
            if stripped.startswith("```"):
                match = re.match(r"^```\S*\n?", raw_response)
                parsed = (
                    raw_response[match.end() :].strip()
                    if match is not None
                    else stripped
                )
            elif stripped.endswith("```"):
                parsed = stripped[:-3].strip()
            else:
                parsed = stripped
        else:
            parsed = raw_response[start:end]
            match = re.match(r"^\S*\n", parsed)
            if match is not None:
                parsed = parsed[match.end() :]
            parsed = parsed.strip()
        if not parsed:
            raise ValueError(
                "GEPA reflection response produced an empty component"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class GepaPromptServices:
    descriptor: GepaPromptFormatDescriptor
    reflection_builder: GepaReflectionPromptBuilder
    reflection_parser: GepaReflectionResponseParser

    @property
    def binding(self) -> GepaPromptBinding:
        return GepaPromptBinding(
            prompt_format_identity_hash=self.descriptor.identity_hash(),
            reflection_builder_identity_hash=(
                self.reflection_builder.identity_hash
            ),
            reflection_parser_identity_hash=(
                self.reflection_parser.identity_hash
            ),
        )

    def parse_replacement(
        self,
        component_name: str,
        raw_response: str,
    ) -> str:
        parsed = self.reflection_parser.parse(raw_response)
        return self.validate_replacement(component_name, parsed)

    def validate_replacement(
        self,
        component_name: str,
        parsed_replacement: str,
    ) -> str:

        return self.descriptor.component(component_name).validate_replacement(
            parsed_replacement
        )


class MappingGepaPromptRegistry:
    def __init__(self, services: Sequence[GepaPromptServices]) -> None:
        self._services: dict[str, GepaPromptServices] = {}
        for item in services:
            format_hash = item.descriptor.identity_hash()
            if format_hash in self._services:
                raise ValueError(
                    "duplicate GEPA prompt-format registry identity "
                    f"{format_hash}"
                )
            self._services[format_hash] = item

    def resolve(self, prompt_format_identity_hash: str) -> GepaPromptServices:
        require_full_hash(
            prompt_format_identity_hash,
            field="prompt_format_identity_hash",
        )
        try:
            return self._services[prompt_format_identity_hash]
        except KeyError:
            raise KeyError(
                "no GEPA prompt services registered for format "
                f"{prompt_format_identity_hash}"
            ) from None


__all__ = [
    "GEPA_PROMPT_BINDING_SCHEMA",
    "GEPA_PROMPT_BINDING_SCHEMA_VERSION",
    "GEPA_PROMPT_FORMAT_SCHEMA",
    "GEPA_PROMPT_FORMAT_SCHEMA_VERSION",
    "GEPA_REFLECTION_EXAMPLES_ROLE",
    "GEPA_REFLECTION_PROMPT_ROLE",
    "GEPA_REFLECTION_PROMPT_SCHEMA",
    "GEPA_REFLECTION_PROMPT_SCHEMA_VERSION",
    "GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA",
    "GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA_VERSION",
    "GEPA_REFLECTION_TASK",
    "GepaComponentFormat",
    "GepaPromptBinding",
    "GepaPromptFormatDescriptor",
    "GepaPromptServices",
    "GepaReflectionPromptBuilder",
    "GepaReflectionRequest",
    "GepaReflectionResponseParser",
    "GepaRenderedPrompt",
    "MappingGepaPromptRegistry",
    "NativeGepaReflectionPromptBuilder",
    "NativeGepaReflectionResponseParser",
]
