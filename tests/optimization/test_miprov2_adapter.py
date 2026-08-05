from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from dr_store import MemoryBackend, ObjectStore

import whetstone.optimization as optimization
from tests.optimization.support import candidate, python_format_contract
from tests.optimization.test_miprov2_control import _defaults
from whetstone.optimization.effect_authority import ReplayPolicy
from whetstone.optimization.miprov2 import Miprov2Adapter
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvalConfigResolver,
)
from whetstone.optimization.miprov2_render import (
    candidate_from_components,
    compose_user_prompt_template,
)
from whetstone.optimization.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    _durable_proposal_executor,
)
from whetstone.optimization.schema import (
    StepMode,
    TemplateRenderContract,
    TemplateRenderKind,
    candidate_reference,
)


class _UnusedResolver:
    def resolve(
        self, _request: Miprov2EvalConfigBindingRequest
    ) -> Miprov2EvalConfigBinding:
        raise AssertionError("test does not execute evaluation config effects")


def _unused_executor() -> DurableProposalExecutor:
    """Mint the canonical capability over a never-invoked execution."""

    def execute(*, config, request, transport, count):
        raise AssertionError("test does not execute proposal effects")

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash="c" * 64,
        ),
        execute=execute,
    )


def _adapter() -> Miprov2Adapter:
    defaults = _defaults()
    transport = FakeProposerTransport(
        {},
        execution_policy_hash="a" * 64,
        prompt_adapter_identity_hash="b" * 64,
    )
    return Miprov2Adapter(
        store=ObjectStore(MemoryBackend()),
        proposer_config=defaults.prompt_model,
        transport=transport,
        eval_config_resolver=cast(
            Miprov2EvalConfigResolver, _UnusedResolver()
        ),
        proposal_executor=_unused_executor(),
    )


def test_adapter_exposes_only_proposal_only_harness_contract() -> None:
    adapter = _adapter()

    assert adapter.key == "miprov2"
    assert adapter.mode is StepMode.PROPOSAL_ONLY
    assert adapter.required_replay_policy is ReplayPolicy.DURABLE_WORKFLOW


def test_adapter_requires_the_executor_durable_workflow_replay() -> None:
    adapter = _adapter()

    assert (
        adapter.required_replay_policy
        is adapter.proposal_executor.recovery_policy
    )
    assert adapter.proposal_executor.policy_identity_hash == "c" * 64


def test_adapter_rejects_a_structural_proposal_executor() -> None:
    class _StructuralExecutor:
        policy_identity_hash = "c" * 64
        recovery_policy = ReplayPolicy.DURABLE_WORKFLOW

        def execute(self, **_kwargs):
            raise AssertionError("test does not execute proposal effects")

    defaults = _defaults()
    transport = FakeProposerTransport(
        {},
        execution_policy_hash="a" * 64,
        prompt_adapter_identity_hash="b" * 64,
    )

    with pytest.raises(TypeError, match="DurableProposalExecutor"):
        Miprov2Adapter(
            store=ObjectStore(MemoryBackend()),
            proposer_config=defaults.prompt_model,
            transport=transport,
            eval_config_resolver=cast(
                Miprov2EvalConfigResolver, _UnusedResolver()
            ),
            proposal_executor=cast(
                DurableProposalExecutor, _StructuralExecutor()
            ),
        )


def test_package_facade_is_minimal_and_current() -> None:
    expected = {
        "MIPROV2_ADAPTER_KEY",
        "MIPROV2_ALGORITHM_VERSION",
        "MIPROV2_OPTUNA_VERSION",
        "MIPROV2_PROMPT_FORMAT_ADAPTER_VERSION",
        "MIPROV2_REFERENCE_COMMIT",
        "Miprov2Adapter",
        "Miprov2AutoMode",
        "Miprov2Control",
        "Miprov2InjectedDefaults",
        "configure_miprov2",
    }
    public_miprov2 = {
        name
        for name in optimization.__all__
        if name.startswith("Miprov2")
        or name.startswith("MIPROV2")
        or name == "configure_miprov2"
    }

    assert public_miprov2 == expected


def test_candidate_rendering_mutates_only_user_prompt_template() -> None:
    base = candidate("base", text="Initial {query}.")
    base = type(base).model_validate(
        {
            **base.model_dump(mode="json"),
            "payload": {
                "user_prompt_template": "Initial {query}.",
                "fixed": {"nested": [1, 2]},
            },
        }
    )
    base_ref = candidate_reference(base)

    rendered = candidate_from_components(
        base=base_ref,
        candidate_id="trial-1",
        components=(
            {
                "component_id": "generate",
                "instruction": "Improved {query}.",
                "instruction_index": 1,
                "instruction_identity_hash": "a" * 64,
                "demo_index": 0,
                "demo_identity_hash": "b" * 64,
                "demo_set": {"examples": [{"query": "q", "answer": "a"}]},
            },
        ),
        template_render_contract=python_format_contract(),
    )

    assert rendered.base_ref == base_ref.record_ref
    assert rendered.payload["fixed"] == {"nested": [1, 2]}
    assert (
        rendered.payload["user_prompt_template"]
        != (base.payload["user_prompt_template"])
    )


def test_composition_is_deterministic_and_json_is_format_literal() -> None:
    component = {
        "component_id": "encode",
        "instruction": "Encode {query}.",
        "demo_set": {"query": "literal", "answer": "value"},
    }

    contract = python_format_contract()
    first = compose_user_prompt_template(
        (component,), template_render_contract=contract
    )
    second = compose_user_prompt_template(
        (component,), template_render_contract=contract
    )

    assert first == second
    assert "Encode {query}." in first
    assert '{{"component_id":"encode"' in first
    contract.validate_template(first)


@pytest.mark.parametrize(
    "kind",
    (
        TemplateRenderKind.PYTHON_FORMAT_V1,
        TemplateRenderKind.LITERAL_REPLACE_V1,
        TemplateRenderKind.LITERAL_BODY_V1,
    ),
)
def test_composed_json_survives_rendering_under_every_contract(
    kind: TemplateRenderKind,
) -> None:
    """Rendered metadata and demonstrations must reach the task model as JSON.

    ``{{``/``}}`` are brace escapes only under ``python_format/v1``; the
    literal contracts pass them through verbatim, so escaping there would
    deliver malformed JSON.
    """

    literal_body = kind is TemplateRenderKind.LITERAL_BODY_V1
    contract = TemplateRenderContract(
        kind=kind,
        available_fields=() if literal_body else ("query",),
    )
    instruction = (
        "Answer plainly."
        if literal_body
        else (
            "Answer {{style}} for {query}."
            if kind is TemplateRenderKind.PYTHON_FORMAT_V1
            else "Answer {style} for {query}."
        )
    )
    template = compose_user_prompt_template(
        (
            {
                "component_id": "encode",
                "instruction": instruction,
                "instruction_index": 0,
                "instruction_identity_hash": "a" * 64,
                "demo_index": 0,
                "demo_identity_hash": "b" * 64,
                "demo_set": [{"answer": "ok"}],
            },
        ),
        template_render_contract=contract,
    )

    rendered = contract.render(
        template, {} if literal_body else {"query": "blue"}
    )
    metadata, _, remainder = rendered.split("### Metadata\n")[1].partition(
        "\n### Instruction\n"
    )
    demonstrations = remainder.split("### Demonstrations\n")[1]

    assert json.loads(metadata)["component_id"] == "encode"
    assert json.loads(demonstrations) == [{"answer": "ok"}]
    assert "{{" not in rendered and "}}" not in rendered


def test_rendering_rejects_empty_or_multiple_components() -> None:
    base = candidate_reference(candidate("base", text="Initial {query}."))

    with pytest.raises(ValueError, match="exactly one"):
        candidate_from_components(
            base=base,
            candidate_id="empty",
            components=(),
            template_render_contract=python_format_contract(),
        )
    with pytest.raises(ValueError, match="exactly one"):
        candidate_from_components(
            base=base,
            candidate_id="multiple",
            components=(
                {"component_id": "generate", "instruction": "One {query}"},
                {"component_id": "encode", "instruction": "Two {query}"},
            ),
            template_render_contract=python_format_contract(),
        )


@pytest.mark.parametrize(
    "forbidden",
    (
        "_seeded_tpe_choice",
        "_materialize_demonstrations",
        "DemoPair",
        "itertools.combinations",
        "combination_candidates",
        '"promotion": "noop"',
        "retry_until_distinct",
    ),
)
def test_adapter_source_has_no_approximation_markers(forbidden: str) -> None:
    root = Path(__file__).parents[2] / "src" / "whetstone" / "optimization"

    assert forbidden not in (root / "miprov2.py").read_text()
    assert forbidden not in (root / "miprov2_runtime.py").read_text()
