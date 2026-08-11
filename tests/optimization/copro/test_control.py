from __future__ import annotations

import inspect

import pytest

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    FULL_D,
    eval_config,
    evaluation_binding,
    internal_reward_policy,
)
from whetstone.core.identity import (
    IdentityRef,
    typed_ref_for_record,
)
from whetstone.experiment.binding import eval_config_reference
from whetstone.optimization.codex.proposer import CodexCliProposerConfig
from whetstone.optimization.copro.code_comp.contract import (
    encdec_copro_proposal_contract,
)
from whetstone.optimization.copro.control import (
    COPRO_ALGORITHM_VERSION,
    COPRO_PROPOSAL_PROMPT_SCHEMA_TAG,
    COPRO_REFERENCE_COMMIT,
    CoproInjectedDefaults,
    configure_copro,
)
from whetstone.optimization.proposal.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter


def _prompt_model(route: str = "provider://default") -> ProposerConfig:
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                {"route": route},
            ),
            identity_hash=FULL_A,
        ),
        temperature=None,
    )


def _defaults(
    *,
    prompt_model: ProposerConfig | None = None,
    policy_hash: str = FULL_B,
    expected_reward_policy_hash: str | None = None,
    prompt_adapter: PlainPromptAdapter | None = None,
) -> CoproInjectedDefaults:
    return CoproInjectedDefaults(
        prompt_model=prompt_model or _prompt_model(),
        proposal_contract=encdec_copro_proposal_contract(budget_ratio=None),
        evaluation_binding=evaluation_binding(),
        expected_reward_policy_hash=(
            expected_reward_policy_hash
            or internal_reward_policy().identity_hash()
        ),
        provider_execution_policy_hash=policy_hash,
        prompt_adapter=prompt_adapter or PlainPromptAdapter(),
    )


def test_public_signature_has_only_copro_owned_controls() -> None:
    signature = inspect.signature(configure_copro)

    assert signature.parameters["prompt_model"].default is None
    assert signature.parameters["metric"].default is None
    assert signature.parameters["breadth"].default == 10
    assert signature.parameters["depth"].default == 3
    assert "init_temperature" not in signature.parameters
    assert signature.parameters["track_stats"].default is False
    assert (
        signature.parameters["defaults"].kind is inspect.Parameter.KEYWORD_ONLY
    )


def test_none_resolves_only_through_explicit_injected_defaults() -> None:
    defaults = _defaults()

    control = configure_copro(defaults=defaults)

    assert control.prompt_model is defaults.prompt_model
    assert control.proposal_contract is defaults.proposal_contract
    assert control.evaluation_binding is defaults.evaluation_binding
    assert control.breadth == 10
    assert control.depth == 3
    assert control.track_stats is False


def test_explicit_prompt_model_and_metric_override_defaults() -> None:
    prompt_model = _prompt_model("provider://explicit")
    metric = eval_config_reference(eval_config(FULL_C))

    control = configure_copro(
        prompt_model=prompt_model,
        metric=metric,
        defaults=_defaults(),
    )

    assert control.prompt_model is prompt_model
    assert control.evaluation_binding.eval_config == metric


def test_codex_cli_can_be_the_copro_proposer_route() -> None:
    proposer = CodexCliProposerConfig(model="gpt-5.4")

    control = configure_copro(
        prompt_model=proposer,
        defaults=_defaults(),
    )

    assert control.prompt_model is proposer
    assert "temperature" not in control.prompt_model.identity_payload()


def test_copro_rejects_a_provider_owned_temperature() -> None:
    with pytest.raises(ValueError, match="leave temperature unset"):
        configure_copro(
            prompt_model=ProposerConfig(
                provider_call_config=_prompt_model().provider_call_config,
                temperature=0.7,
            ),
            defaults=_defaults(),
        )


def test_identity_binds_all_algorithm_and_execution_controls() -> None:
    defaults = _defaults()
    control = configure_copro(defaults=defaults)
    payload = control.identity_payload()

    assert payload["algorithm_version"] == COPRO_ALGORITHM_VERSION
    assert payload["reference_commit"] == COPRO_REFERENCE_COMMIT
    assert (
        payload["proposal_prompt_schema_tag"]
        == COPRO_PROPOSAL_PROMPT_SCHEMA_TAG
    )
    assert payload["provider_execution_policy_hash"] == FULL_B
    assert payload["expected_reward_policy_hash"] == (
        internal_reward_policy().identity_hash()
    )
    assert payload["prompt_adapter_identity_hash"] == (
        prompt_adapter_identity_hash(defaults.prompt_adapter)
    )
    assert payload["prompt_model"]["identity_hash"] == (
        defaults.prompt_model.identity_hash()
    )
    assert payload["proposal_contract"] == (
        defaults.proposal_contract.model_dump(mode="json")
    )
    assert payload["evaluation_binding"]["record"] == (
        defaults.evaluation_binding.model_dump(mode="json")
    )


def test_policy_and_prompt_adapter_change_optimizer_identity() -> None:
    base = configure_copro(defaults=_defaults())
    other_policy = configure_copro(defaults=_defaults(policy_hash=FULL_C))
    other_reward = configure_copro(
        defaults=_defaults(expected_reward_policy_hash=FULL_D)
    )
    other_adapter = configure_copro(
        defaults=_defaults(
            prompt_adapter=PlainPromptAdapter(output_field="instruction")
        )
    )
    budgeted_defaults = _defaults().model_copy(
        update={
            "proposal_contract": encdec_copro_proposal_contract(
                budget_ratio=0.5
            )
        }
    )
    other_contract = configure_copro(defaults=budgeted_defaults)

    assert base.identity_hash() != other_policy.identity_hash()
    assert base.identity_hash() != other_reward.identity_hash()
    assert base.identity_hash() != other_adapter.identity_hash()
    assert base.identity_hash() != other_contract.identity_hash()


def test_step_controls_repeat_identity_bindings_and_round_index() -> None:
    control = configure_copro(defaults=_defaults())

    hyperparameters = control.step_hyperparameters(iteration=1)

    assert hyperparameters["round_index"] == 1
    assert "copro_iteration" not in hyperparameters
    assert hyperparameters["algorithm_version"] == COPRO_ALGORITHM_VERSION
    assert (
        hyperparameters["proposal_prompt_schema_tag"]
        == COPRO_PROPOSAL_PROMPT_SCHEMA_TAG
    )
    assert hyperparameters["proposal_contract"] == (
        control.proposal_contract.model_dump(mode="json")
    )
    assert hyperparameters["provider_execution_policy_hash"] == FULL_B
    assert hyperparameters["prompt_adapter_identity_hash"] == (
        control.prompt_adapter_identity_hash
    )


def test_persisted_optimizer_identity_conflict_is_rejected() -> None:
    control = configure_copro(defaults=_defaults())

    control.require_identity_hash(control.identity_hash())
    with pytest.raises(ValueError, match="conflicts"):
        control.require_identity_hash(FULL_C)
