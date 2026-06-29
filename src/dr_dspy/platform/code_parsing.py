from __future__ import annotations

import ast
import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_dspy.humaneval.code_extraction import (
    apply_cleaning,
    validate_python_source,
)

BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID = "humaneval-best-effort"
STRICT_FIELD_MARKER_PARSER_PROFILE_ID = "humaneval-field-marker"
PARSER_PROFILE_VERSION = "v1"
DEFAULT_CODE_FIELD = "code"
FIELD_MARKER_RE = re.compile(
    r"\[\[\s*##\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]"
)


class ExtractionMethod(StrEnum):
    DSPY_CODE_FIELD = "dspy_code_field"
    JSON_CODE_FIELD = "json_code_field"
    JSON_STRING = "json_string"
    FENCED_CODE = "fenced_code"
    CLEANED_CANDIDATE = "cleaned_candidate"
    BARE_PYTHON = "bare_python"
    FIELD_MARKER = "field_marker"


class CodeParserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: StrictStr
    version: StrictStr
    code_field: StrictStr = DEFAULT_CODE_FIELD


class CodeExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_generation: StrictStr | None
    extracted_code: StrictStr | None
    extraction_method: ExtractionMethod | None
    candidate_count: StrictInt
    selected_candidate_index: StrictInt | None = None
    compile_ok: bool
    compile_error: StrictStr | None = None
    extraction_error: StrictStr | None = None
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.extracted_code is not None


BEST_EFFORT_HUMANEVAL_PARSER_PROFILE = CodeParserProfile(
    profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    version=PARSER_PROFILE_VERSION,
)
STRICT_FIELD_MARKER_PARSER_PROFILE = CodeParserProfile(
    profile_id=STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    version=PARSER_PROFILE_VERSION,
)


def resolve_parser_profile(
    *,
    parser_profile_id: str,
    parser_version: str,
    code_field: str = DEFAULT_CODE_FIELD,
) -> CodeParserProfile:
    if parser_version != PARSER_PROFILE_VERSION:
        raise ValueError(
            f"unsupported parser profile version: {parser_version}"
        )
    if parser_profile_id not in {
        BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    }:
        raise ValueError(f"unsupported parser profile id: {parser_profile_id}")
    return CodeParserProfile(
        profile_id=parser_profile_id,
        version=parser_version,
        code_field=code_field,
    )


def extract_code_with_profile(
    raw_generation: Any,
    *,
    profile: CodeParserProfile,
) -> CodeExtractionResult:
    if profile.profile_id == BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID:
        return extract_best_effort_code(raw_generation, profile=profile)
    if profile.profile_id == STRICT_FIELD_MARKER_PARSER_PROFILE_ID:
        return extract_strict_field_marker_code(
            raw_generation,
            profile=profile,
        )
    raise ValueError(f"unsupported parser profile id: {profile.profile_id}")


def extract_best_effort_code(
    raw_generation: Any,
    *,
    profile: CodeParserProfile = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
) -> CodeExtractionResult:
    unwrapped, unwrap_method, unwrap_metadata = unwrap_generation(
        raw_generation,
        code_field=profile.code_field,
    )
    if unwrapped is None:
        return extraction_failure(
            raw_generation=None,
            candidate_count=0,
            error="generation is not a supported code-bearing value",
            metadata=unwrap_metadata,
        )
    if not unwrapped.strip():
        return extraction_failure(
            raw_generation=unwrapped,
            candidate_count=0,
            error="empty raw generation",
            metadata=unwrap_metadata,
        )

    candidates = apply_cleaning(unwrapped, apply_dedent=True)
    if not candidates:
        return extraction_failure(
            raw_generation=unwrapped,
            candidate_count=0,
            error="no code candidates extracted",
            metadata=unwrap_metadata,
        )

    first_compile_error: str | None = None
    for index, candidate in enumerate(candidates):
        validation = validate_python_source(candidate)
        if not validation.compile_ok:
            first_compile_error = (
                first_compile_error or validation.compile_error
            )
            continue
        if is_plain_literal_module(candidate):
            first_compile_error = first_compile_error or (
                "plain literal modules are not valid HumanEval code"
            )
            continue
        method = selected_method(
            raw_generation=unwrapped,
            candidate=candidate,
            unwrap_method=unwrap_method,
        )
        return CodeExtractionResult(
            raw_generation=unwrapped,
            extracted_code=candidate,
            extraction_method=method,
            candidate_count=len(candidates),
            selected_candidate_index=index,
            compile_ok=True,
            compile_error=None,
            metadata={
                **unwrap_metadata,
                "candidate_count": len(candidates),
                "selected_candidate_index": index,
            },
        )

    return extraction_failure(
        raw_generation=unwrapped,
        candidate_count=len(candidates),
        error="no compilable extracted candidate",
        compile_error=first_compile_error,
        metadata={
            **unwrap_metadata,
            "candidate_count": len(candidates),
        },
    )


def extract_strict_field_marker_code(
    raw_generation: Any,
    *,
    profile: CodeParserProfile = STRICT_FIELD_MARKER_PARSER_PROFILE,
) -> CodeExtractionResult:
    if not isinstance(raw_generation, str):
        return extraction_failure(
            raw_generation=None,
            candidate_count=0,
            error="strict parser requires string generation",
        )
    field_value = field_marker_value(
        raw_generation,
        field_name=profile.code_field,
    )
    if field_value is None:
        return extraction_failure(
            raw_generation=raw_generation,
            candidate_count=0,
            error=f"missing field marker for {profile.code_field!r}",
        )
    candidate = field_value.strip()
    if not candidate:
        return extraction_failure(
            raw_generation=raw_generation,
            candidate_count=1,
            error="empty field-marker code",
        )
    validation = validate_python_source(candidate)
    if not validation.compile_ok:
        return extraction_failure(
            raw_generation=raw_generation,
            candidate_count=1,
            error="field-marker code is not compilable",
            compile_error=validation.compile_error,
        )
    if is_plain_literal_module(candidate):
        return extraction_failure(
            raw_generation=raw_generation,
            candidate_count=1,
            error="plain literal modules are not valid HumanEval code",
        )
    return CodeExtractionResult(
        raw_generation=raw_generation,
        extracted_code=candidate,
        extraction_method=ExtractionMethod.FIELD_MARKER,
        candidate_count=1,
        selected_candidate_index=0,
        compile_ok=True,
        metadata={
            "candidate_count": 1,
            "selected_candidate_index": 0,
            "field_name": profile.code_field,
        },
    )


def unwrap_generation(
    raw_generation: Any,
    *,
    code_field: str,
) -> tuple[str | None, ExtractionMethod | None, dict[str, Any]]:
    code_field_value = getattr(raw_generation, code_field, None)
    if code_field_value is not None:
        inner_code = getattr(code_field_value, "code", None)
        if isinstance(inner_code, str):
            return inner_code, ExtractionMethod.DSPY_CODE_FIELD, {
                "code_field": code_field,
            }
        if isinstance(code_field_value, str):
            return code_field_value, ExtractionMethod.DSPY_CODE_FIELD, {
                "code_field": code_field,
            }
    inner_code = getattr(raw_generation, "code", None)
    if isinstance(inner_code, str):
        return inner_code, ExtractionMethod.DSPY_CODE_FIELD, {
            "code_field": "code",
        }
    if isinstance(raw_generation, dict):
        return unwrap_mapping(raw_generation, code_field=code_field)
    if isinstance(raw_generation, str):
        return unwrap_string(raw_generation, code_field=code_field)
    return None, None, {"raw_type": type(raw_generation).__name__}


def unwrap_mapping(
    value: dict[Any, Any],
    *,
    code_field: str,
) -> tuple[str | None, ExtractionMethod | None, dict[str, Any]]:
    code_value = value.get(code_field)
    if isinstance(code_value, str):
        return code_value, ExtractionMethod.JSON_CODE_FIELD, {
            "code_field": code_field,
        }
    return None, None, {
        "raw_type": "dict",
        "code_field": code_field,
        "available_fields": sorted(str(key) for key in value),
    }


def unwrap_string(
    value: str,
    *,
    code_field: str,
) -> tuple[str, ExtractionMethod | None, dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value, None, {}
    if isinstance(parsed, dict):
        code_value = parsed.get(code_field)
        if isinstance(code_value, str):
            return code_value, ExtractionMethod.JSON_CODE_FIELD, {
                "code_field": code_field,
                "json_unwrapped": True,
            }
    if isinstance(parsed, str):
        return parsed, ExtractionMethod.JSON_STRING, {"json_unwrapped": True}
    return value, None, {"json_value_type": type(parsed).__name__}


def field_marker_value(raw_generation: str, *, field_name: str) -> str | None:
    matches = list(FIELD_MARKER_RE.finditer(raw_generation))
    for index, match in enumerate(matches):
        if match.group("field") != field_name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        return raw_generation[start:end]
    return None


def selected_method(
    *,
    raw_generation: str,
    candidate: str,
    unwrap_method: ExtractionMethod | None,
) -> ExtractionMethod:
    if unwrap_method is not None:
        return unwrap_method
    if "```" in raw_generation or "~~~" in raw_generation:
        return ExtractionMethod.FENCED_CODE
    if raw_generation.strip() == candidate.strip():
        return ExtractionMethod.BARE_PYTHON
    return ExtractionMethod.CLEANED_CANDIDATE


def is_plain_literal_module(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    if len(tree.body) != 1:
        return False
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr):
        return False
    return isinstance(stmt.value, ast.Dict | ast.List | ast.Set | ast.Tuple)


def extraction_failure(
    *,
    raw_generation: str | None,
    candidate_count: int,
    error: str,
    compile_error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CodeExtractionResult:
    return CodeExtractionResult(
        raw_generation=raw_generation,
        extracted_code=None,
        extraction_method=None,
        candidate_count=candidate_count,
        selected_candidate_index=None,
        compile_ok=False,
        compile_error=compile_error,
        extraction_error=error,
        metadata={
            **(metadata or {}),
            "candidate_count": candidate_count,
        },
    )
