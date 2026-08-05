"""Direct template rendering, parsing, and serialization contracts."""

import pytest
from pydantic import ValidationError

from whetstone.experiment.candidate import (
    TemplateRenderContract,
    TemplateRenderKind,
)


def python_format_contract(
    *,
    available_fields: tuple[str, ...] = ("query",),
    required_fields: tuple[str, ...] = (),
) -> TemplateRenderContract:
    return TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=available_fields,
        required_fields=required_fields,
    )


def test_python_format_extracts_only_simple_fields_with_multiplicity() -> None:
    contract = python_format_contract()
    assert contract.placeholder_fields("{query} then {query}") == (
        "query",
        "query",
    )


def test_required_fields_preserve_order_and_multiplicity() -> None:
    contract = python_format_contract(
        available_fields=("query", "answer"),
        required_fields=("query", "answer", "query"),
    )
    assert contract.validate_template("{query} {answer} {query}") == (
        "query",
        "answer",
        "query",
    )
    with pytest.raises(ValueError, match=r"query \(1/2\)"):
        contract.validate_template("{answer} {query}")


def test_template_render_contract_rejects_unknown_and_invalid_fields() -> None:
    contract = python_format_contract(available_fields=("query",))
    with pytest.raises(ValueError, match="unavailable fields: answer"):
        contract.validate_template("{answer}")

    for payload in (
        {
            "kind": "python_format/v1",
            "available_fields": ["query", "query"],
        },
        {
            "kind": "python_format/v1",
            "available_fields": [""],
        },
        {
            "kind": "python_format/v1",
            "available_fields": [1],
        },
        {
            "kind": "python_format/v1",
            "available_fields": ["query"],
            "required_fields": ["answer"],
        },
    ):
        with pytest.raises(ValidationError):
            TemplateRenderContract.model_validate(payload)


def test_template_render_contract_exact_json_roundtrip() -> None:
    contract = python_format_contract(
        available_fields=("query", "answer"),
        required_fields=("query", "query"),
    )
    payload = contract.model_dump(mode="json")

    assert payload == {
        "kind": "python_format/v1",
        "available_fields": ["query", "answer"],
        "required_fields": ["query", "query"],
    }
    assert TemplateRenderContract.model_validate(payload) == contract


@pytest.mark.parametrize("template", ("", None, 1))
def test_all_render_kinds_require_nonempty_strict_template_text(
    template: object,
) -> None:
    contracts = (
        python_format_contract(),
        TemplateRenderContract(
            kind=TemplateRenderKind.LITERAL_REPLACE_V1,
            available_fields=("input",),
        ),
        TemplateRenderContract(
            kind=TemplateRenderKind.LITERAL_BODY_V1,
            available_fields=(),
        ),
    )

    for contract in contracts:
        with pytest.raises(ValueError, match=r"strict string|non-empty"):
            contract.validate_template(template)


def test_literal_replace_treats_json_and_unmatched_braces_as_literal() -> None:
    contract = TemplateRenderContract(
        kind=TemplateRenderKind.LITERAL_REPLACE_V1,
        available_fields=("input",),
        required_fields=("input",),
    )
    template = (
        '{"object": {"key": 1}, "prompt": "{input}", '
        '"partial": "{input", "other": "{query}"}'
    )

    assert contract.placeholder_fields(template) == ("input",)
    assert contract.render(template, {"input": "exact {replacement}"}) == (
        '{"object": {"key": 1}, "prompt": "exact {replacement}", '
        '"partial": "{input", "other": "{query}"}'
    )


def test_literal_replace_requires_exactly_one_available_field() -> None:
    for available_fields in ((), ("input", "answer")):
        with pytest.raises(
            ValidationError, match="exactly one available field"
        ):
            TemplateRenderContract(
                kind=TemplateRenderKind.LITERAL_REPLACE_V1,
                available_fields=available_fields,
            )


def test_literal_replace_without_optional_token_needs_no_value() -> None:
    contract = TemplateRenderContract(
        kind=TemplateRenderKind.LITERAL_REPLACE_V1,
        available_fields=("input",),
    )
    template = '{"literal": "{other}", "unmatched": "{"}'

    assert contract.render(template, {}) == template


def test_literal_body_has_no_active_fields_and_returns_text_unchanged() -> (
    None
):
    contract = TemplateRenderContract(
        kind=TemplateRenderKind.LITERAL_BODY_V1,
        available_fields=(),
    )
    template = '{"unmatched": "{", "looks_active": "{query}"}'

    assert contract.placeholder_fields(template) == ()
    assert contract.validate_template(template) == ()
    assert contract.render(template, {"query": "ignored"}) == template

    with pytest.raises(ValidationError, match="no active fields"):
        TemplateRenderContract(
            kind=TemplateRenderKind.LITERAL_BODY_V1,
            available_fields=("query",),
        )


def test_render_requires_string_values_for_observed_fields() -> None:
    contract = python_format_contract()
    with pytest.raises(ValueError, match="missing fields: query"):
        contract.render("{query}", {})
    with pytest.raises(ValueError, match="must be strings: query"):
        contract.render("{query}", {"query": 1})
