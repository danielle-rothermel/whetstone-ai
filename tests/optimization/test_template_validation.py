"""Unit tests for the candidate-template placeholder validation helper.

The proposer's candidate templates are untrusted LLM output. Before any eval
spend, :func:`invalid_template_placeholders` must catch a template that names a
field the env render cannot fill (the live c22 crash: ``{question}`` against
c22's ``{constraints_block}`` inputs). These tests pin the helper's contract
directly (no network, no eval).
"""

from __future__ import annotations

from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import valid_prompt_input_keys
from whetstone.optimization.mutation import (
    POSITIONAL_FIELD_TOKEN,
    invalid_template_placeholders,
    template_placeholder_fields,
)

VALID = {"constraints_block", "query"}


def test_exact_match_keys_are_valid() -> None:
    assert invalid_template_placeholders("{constraints_block}", VALID) == ()
    assert (
        invalid_template_placeholders("a {query} b {constraints_block}", VALID)
        == ()
    )


def test_unknown_named_field_is_rejected() -> None:
    # The live c22 shape.
    assert invalid_template_placeholders(
        "Question: {question}\n\nAnswer:", VALID
    ) == ("question",)


def test_positional_field_is_rejected() -> None:
    assert invalid_template_placeholders("{} tail", VALID) == (
        POSITIONAL_FIELD_TOKEN,
    )


def test_index_field_is_rejected() -> None:
    assert invalid_template_placeholders("{0} {1}", VALID) == (
        POSITIONAL_FIELD_TOKEN,
    )


def test_escaped_braces_are_ok() -> None:
    # Escaped ``{{`` / ``}}`` are literals, not fields.
    assert (
        invalid_template_placeholders("literal {{not_a_field}} text", VALID)
        == ()
    )
    assert (
        invalid_template_placeholders("{{a}} {constraints_block} {{b}}", VALID)
        == ()
    )


def test_offending_fields_deduped_in_order() -> None:
    assert invalid_template_placeholders(
        "{q} {constraints_block} {q} {z}", VALID
    ) == ("q", "z")


def test_attribute_and_index_suffix_uses_head_key() -> None:
    # A ``{grid[0]}`` / ``{grid.x}`` reference is valid iff ``grid`` is a key.
    assert invalid_template_placeholders("{grid[0]}", {"grid"}) == ()
    assert invalid_template_placeholders("{grid.x}", {"grid"}) == ()
    assert invalid_template_placeholders("{missing[0]}", {"grid"}) == (
        "missing",
    )


def test_placeholder_fields_lists_all_fields_in_order() -> None:
    assert template_placeholder_fields("{a} {b} {a}") == ("a", "b", "a")
    assert template_placeholder_fields("no fields here") == ()
    assert template_placeholder_fields("{{escaped}}") == ()


def test_valid_keys_derived_from_env_reject_c22_question() -> None:
    # Non-hardcoded: c22's valid keys come from its instance prompt_inputs +
    # its own probe templates. ``{question}`` is not among them; the crash
    # placeholder is caught, the real key is accepted.
    env = env_spec("c22")
    inst = env.generate_pool().instances[0]
    keys = valid_prompt_input_keys(env, inst)
    assert "constraints_block" in keys
    assert "question" not in keys
    assert invalid_template_placeholders("{question}", keys) == ("question",)
    assert invalid_template_placeholders("{constraints_block}", keys) == ()


def test_valid_keys_cover_c19_translated_fact_line() -> None:
    # c19's render translates the public ``fact_type`` input into a
    # ``{fact_line}`` slot; the env's own templates use ``{fact_line}``, so a
    # candidate mimicking them must NOT be spuriously rejected.
    env = env_spec("c19")
    inst = env.generate_pool().instances[0]
    keys = valid_prompt_input_keys(env, inst)
    assert {"grid", "command", "fact_line"} <= keys
    assert invalid_template_placeholders(
        env.surface.ceiling_template, keys
    ) == ()
