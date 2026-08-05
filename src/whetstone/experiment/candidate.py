"""Candidate identity and template-rendering contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from enum import UNIQUE, StrEnum, verify
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    IdentityHash,
    ImmutableJsonObject,
    NonEmptyId,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)

CANDIDATE_RECORD_SCHEMA = "whetstone.optimization_candidate"
CANDIDATE_IDENTITY_SCHEMA = "whetstone.optimization_candidate"
CANDIDATE_IDENTITY_SCHEMA_VERSION = 1


def _require_ordered_sequence(value: Any, info: ValidationInfo) -> Any:
    if type(value) not in (list, tuple):
        raise ValueError(
            f"{info.field_name} must be an ordered tuple or JSON array"
        )
    return value


__all__ = [
    "CANDIDATE_IDENTITY_SCHEMA",
    "CANDIDATE_IDENTITY_SCHEMA_VERSION",
    "CANDIDATE_RECORD_SCHEMA",
    "Candidate",
    "CandidateRef",
    "TemplateRenderContract",
    "TemplateRenderKind",
    "candidate_reference",
]


@verify(UNIQUE)
class TemplateRenderKind(StrEnum):
    """Pinned renderer semantics for an exact optimization run."""

    PYTHON_FORMAT_V1 = "python_format/v1"
    LITERAL_REPLACE_V1 = "literal_replace/v1"
    LITERAL_BODY_V1 = "literal_body/v1"


class TemplateRenderContract(BaseModel):
    """Frozen rendering authority composed once into an exact run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: TemplateRenderKind
    available_fields: tuple[StrictStr, ...]
    required_fields: tuple[StrictStr, ...] = ()

    @field_validator("available_fields", "required_fields", mode="before")
    @classmethod
    def _validate_ordered_fields(cls, value: Any, info: ValidationInfo) -> Any:
        return _require_ordered_sequence(value, info)

    @model_validator(mode="after")
    def _validate(self) -> TemplateRenderContract:
        invalid_fields = tuple(
            field
            for field in (*self.available_fields, *self.required_fields)
            if not field.isidentifier()
        )
        if invalid_fields:
            raise ValueError(
                "template render fields must be non-empty identifiers: "
                + ", ".join(repr(field) for field in invalid_fields)
            )
        if len(set(self.available_fields)) != len(self.available_fields):
            raise ValueError("available_fields must be unique")
        unavailable_required = tuple(
            dict.fromkeys(
                field
                for field in self.required_fields
                if field not in self.available_fields
            )
        )
        if unavailable_required:
            raise ValueError(
                "required_fields must be available: "
                + ", ".join(unavailable_required)
            )
        if (
            self.kind is TemplateRenderKind.LITERAL_REPLACE_V1
            and len(self.available_fields) != 1
        ):
            raise ValueError(
                "literal_replace/v1 requires exactly one available field"
            )
        if self.kind is TemplateRenderKind.LITERAL_BODY_V1 and (
            self.available_fields or self.required_fields
        ):
            raise ValueError("literal_body/v1 has no active fields")
        return self

    @staticmethod
    def _validate_template_text(template: object) -> str:
        if type(template) is not str:
            raise ValueError("template must be a strict string")
        if not template:
            raise ValueError("template must be non-empty")
        return template

    def placeholder_fields(self, template: object) -> tuple[str, ...]:
        """Extract active fields under this contract's pinned syntax."""
        text = self._validate_template_text(template)
        if self.kind is TemplateRenderKind.LITERAL_BODY_V1:
            return ()
        if self.kind is TemplateRenderKind.LITERAL_REPLACE_V1:
            field = self.available_fields[0]
            return (field,) * text.count(f"{{{field}}}")

        fields: list[str] = []
        index = 0
        while index < len(text):
            character = text[index]
            if character == "{":
                if index + 1 < len(text) and text[index + 1] == "{":
                    index += 2
                    continue
                closing = text.find("}", index + 1)
                if closing < 0:
                    raise ValueError(
                        "template has malformed placeholders: unmatched '{'"
                    )
                field_name = text[index + 1 : closing]
                if "{" in field_name:
                    raise ValueError(
                        "template has malformed placeholders: nested '{'"
                    )
                if not field_name.isidentifier():
                    raise ValueError(
                        "template has unsupported field expression "
                        f"{field_name!r}; only simple named fields are allowed"
                    )
                fields.append(field_name)
                index = closing + 1
                continue
            if character == "}":
                if index + 1 < len(text) and text[index + 1] == "}":
                    index += 2
                    continue
                raise ValueError(
                    "template has malformed placeholders: unmatched '}'"
                )
            index += 1
        return tuple(fields)

    def validate_template(self, template: object) -> tuple[str, ...]:
        """Validate availability and required multiplicity."""
        observed = self.placeholder_fields(template)
        unknown = tuple(
            dict.fromkeys(
                field
                for field in observed
                if field not in self.available_fields
            )
        )
        if unknown:
            raise ValueError(
                "template contains unavailable fields: " + ", ".join(unknown)
            )
        observed_counts = Counter(observed)
        required_counts = Counter(self.required_fields)
        missing = tuple(
            field
            for field in dict.fromkeys(self.required_fields)
            if observed_counts[field] < required_counts[field]
        )
        if missing:
            raise ValueError(
                "template does not satisfy required field multiplicity: "
                + ", ".join(
                    f"{field} ({observed_counts[field]}/"
                    f"{required_counts[field]})"
                    for field in missing
                )
            )
        return observed

    def render(self, template: object, values: Mapping[str, object]) -> str:
        """Render text without selecting semantics at the call site."""
        observed = self.validate_template(template)
        text = self._validate_template_text(template)
        if self.kind is TemplateRenderKind.LITERAL_BODY_V1:
            return text
        missing_values = tuple(
            dict.fromkeys(field for field in observed if field not in values)
        )
        if missing_values:
            raise ValueError(
                "render values are missing fields: "
                + ", ".join(missing_values)
            )
        render_values: dict[str, str] = {}
        non_string_values: list[str] = []
        for field in dict.fromkeys(observed):
            value = values[field]
            if type(value) is not str:
                non_string_values.append(field)
            else:
                render_values[field] = value
        if non_string_values:
            raise ValueError(
                "render values must be strings: "
                + ", ".join(non_string_values)
            )
        if self.kind is TemplateRenderKind.LITERAL_REPLACE_V1:
            if not observed:
                return text
            field = self.available_fields[0]
            return text.replace(f"{{{field}}}", render_values[field])
        return text.format_map(render_values)


class Candidate(BaseModel):
    """Identity-bearing candidate with exact ancestry and immutable payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: NonEmptyId
    base_ref: TypedRef
    payload: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    def identity_payload(self) -> dict[str, Any]:
        # These persisted identity keys are an explicit wire contract. Never
        # derive them by iterating over model fields.
        return {
            "candidate_id": self.candidate_id,
            "base_ref": self.base_ref.model_dump(mode="json"),
            "payload": self.payload.to_json(),
        }

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=CANDIDATE_IDENTITY_SCHEMA,
            schema_version=CANDIDATE_IDENTITY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CandidateRef(BaseModel):
    """Exact typed candidate record and its Identity Hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: Candidate
    record_ref: TypedRef
    identity_hash: IdentityHash

    @model_validator(mode="after")
    def _validate(self) -> CandidateRef:
        expected_ref = typed_ref_for_record(
            CANDIDATE_RECORD_SCHEMA, self.record.record_content()
        )
        if self.record_ref != expected_ref:
            raise ValueError(
                "candidate record_ref must address the exact candidate record"
            )
        if self.identity_hash != self.record.identity_hash():
            raise ValueError(
                "candidate identity_hash must match the exact candidate record"
            )
        return self


def candidate_reference(candidate: Candidate) -> CandidateRef:
    return CandidateRef(
        record=candidate,
        record_ref=typed_ref_for_record(
            CANDIDATE_RECORD_SCHEMA, candidate.record_content()
        ),
        identity_hash=candidate.identity_hash(),
    )
