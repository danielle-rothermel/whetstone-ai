from __future__ import annotations

import pytest
from dr_providers import ProviderTransportPolicy
from pydantic import ValidationError

from whetstone.core.identity import (
    IdentityRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation import DefinitionRef, EvalConfig
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA,
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvalConfigRef,
    EvaluationBinding,
    ExecutionEnvironmentFingerprint,
    eval_config_reference,
)
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64
FULL_D = "d" * 64


def _eval_config(identity_hash: str = FULL_B) -> EvalConfig:
    return EvalConfig(
        definition_ref=DefinitionRef(
            definition_id="eval",
            version="1",
            schema_name="whetstone.eval.definition",
            identity_hash=FULL_A,
        ),
        sampling_config_hash=FULL_A,
        evaluation_procedure_config_hash=FULL_C,
        aggregation_config_hash=FULL_D,
        config_hash=identity_hash,
    )


def _provider_execution_policy_ref() -> IdentityRef:
    policy = ProviderExecutionPolicy(
        transport_policy=ProviderTransportPolicy(
            api_key_env="TEST_PROVIDER_API_KEY",
            base_url="https://provider.test/v1",
        ),
        max_attempts=2,
    )
    return IdentityRef(
        record_ref=typed_ref_for_record(
            PROVIDER_EXECUTION_POLICY_SCHEMA,
            policy.identity_payload(),
        ),
        record_hash=policy.identity_hash,
    )


def _binding(
    *,
    role: EvaluationRole = EvaluationRole.INTERNAL,
    authority_principal: str | None = None,
    config: EvalConfigRef | None = None,
) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=config or eval_config_reference(_eval_config()),
        role=role,
        authority_principal=authority_principal,
        campaign="schema-tests",
        provider_execution_policy_ref=_provider_execution_policy_ref(),
        retry_policy_ref=typed_ref_for_record(
            "whetstone.test.retry_policy",
            {"max_retries": 1},
        ),
        operational_policy_refs=(
            typed_ref_for_record(
                "whetstone.test.accounting_policy",
                {"currency": "usd"},
            ),
        ),
        environment_fingerprint=ExecutionEnvironmentFingerprint(
            dependency_versions=(("dr-code", "0.1.0"),),
            code_revision="deadbeef",
            runtime_hash="linux-x86_64",
        ),
        provenance_note="schema test",
        provenance_ordinal=1,
    )


def test_eval_config_ref_round_trips_exact_json() -> None:
    ref = eval_config_reference(_eval_config())
    dumped = ref.model_dump(mode="json")

    assert EvalConfigRef.model_validate(dumped) == ref
    assert EvalConfigRef.model_validate_json(ref.model_dump_json()) == ref


def test_eval_config_ref_rejects_tampered_record_ref() -> None:
    ref = eval_config_reference(_eval_config())
    payload = ref.model_dump(mode="json")
    payload["record_ref"]["schema_name"] = "whetstone.test.wrong"

    with pytest.raises(
        ValidationError, match=r"record_ref.*exact typed record"
    ):
        EvalConfigRef.model_validate(payload)


def test_eval_config_ref_rejects_tampered_identity_hash() -> None:
    ref = eval_config_reference(_eval_config())
    payload = ref.model_dump(mode="json")
    payload["config_hash"] = FULL_A

    with pytest.raises(
        ValidationError, match=r"identity_hash.*exact typed record"
    ):
        EvalConfigRef.model_validate(payload)


def test_evaluation_binding_identity_contract_literals_are_pinned() -> None:
    binding = _binding()

    assert EVALUATION_BINDING_SCHEMA == "whetstone.evaluation_binding"
    assert EVALUATION_BINDING_SCHEMA_VERSION == 3
    assert binding.schema_version == EVALUATION_BINDING_SCHEMA_VERSION
    assert binding.record_content()["schema_version"] == 3
    assert tuple(binding.record_content()) == tuple(binding.identity_payload())
    assert tuple(binding.identity_payload()) == (
        "schema_version",
        "eval_config",
        "role",
        "authority_principal",
        "campaign",
        "provider_execution_policy_ref",
        "retry_policy_ref",
        "operational_policy_refs",
        "environment_fingerprint",
        "provenance_note",
        "provenance_ordinal",
    )
    assert binding.identity_payload()["provider_execution_policy_ref"] == {
        "record_ref": {
            "schema_name": "whetstone.provider_execution_policy",
            "content_hash": (
                "6282ea0fd226f31de13386807a249566"
                "ed066b2a437fa5ae5fb7fb2792267f40"
            ),
        },
        "record_hash": (
            "dc3b0bb25a7e0e87d461cf376518310daa7ffe318450e85733203e13e7eb1e42"
        ),
    }
    assert (
        binding.identity_hash()
        == "75e1a5fce887b36da046c8b5fc5aa05838ac17413d4b5403ed4041eab4bd4313"
    )
    assert (
        EvaluationBinding.model_validate(binding.model_dump(mode="json"))
        == binding
    )
    assert EvaluationBinding.model_validate_json(
        binding.model_dump_json()
    ) == (binding)


def test_evaluation_binding_rejects_wrong_provider_policy_schema() -> None:
    payload = _binding().model_dump(mode="json")
    payload["provider_execution_policy_ref"]["record_ref"]["schema_name"] = (
        "whetstone.test.wrong_policy"
    )

    with pytest.raises(
        ValidationError,
        match="provider_execution_policy_ref must use schema",
    ):
        EvaluationBinding.model_validate(payload)


@pytest.mark.parametrize(
    "provider_ref_present",
    [True, False],
    ids=["provider-ref-present", "provider-ref-absent"],
)
def test_evaluation_binding_v1_wire_is_partitioned_and_rejected(
    provider_ref_present: bool,
) -> None:
    current_payload = _binding().model_dump(mode="json")
    if not provider_ref_present:
        current_payload["provider_execution_policy_ref"] = None
    current_binding = EvaluationBinding.model_validate(current_payload)

    legacy_wire = current_binding.model_dump(mode="json")
    legacy_wire.pop("schema_version")
    if provider_ref_present:
        policy_ref = current_binding.provider_execution_policy_ref
        assert policy_ref is not None
        legacy_wire["provider_execution_policy_ref"] = (
            policy_ref.record_ref.model_dump(mode="json")
        )

    legacy_identity_hash = compute_identity_hash(
        schema=EVALUATION_BINDING_SCHEMA,
        schema_version=1,
        payload=legacy_wire,
    )
    assert (
        legacy_identity_hash
        == {
            True: (
                "eb8c90e15fb6147ce7359958ce7fbd8cd152c8170803a150effcf5fa19804b4e"
            ),
            False: (
                "bfd74fbb8fe543f797e2901d68697c621d8b382027498bf5e6320d7369d4a75c"
            ),
        }[provider_ref_present]
    )
    assert legacy_identity_hash != current_binding.identity_hash()

    with pytest.raises(ValidationError, match="Field required"):
        EvaluationBinding.model_validate(legacy_wire)
    with pytest.raises(ValidationError, match="Input should be 3"):
        EvaluationBinding.model_validate({"schema_version": 1, **legacy_wire})


@pytest.mark.parametrize(
    ("role", "authority_principal", "message"),
    [
        (EvaluationRole.OFFICIAL, None, "required for official"),
        (EvaluationRole.INTERNAL, "official-publisher", "absent for internal"),
    ],
    ids=["official-without-principal", "internal-with-principal"],
)
@pytest.mark.precheck
def test_evaluation_binding_rejects_invalid_role_authority_pair(
    role: EvaluationRole,
    authority_principal: str | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _binding(role=role, authority_principal=authority_principal)


def test_evaluation_binding_defensively_copies_nested_identity_data() -> None:
    source = _binding().model_dump(mode="json")
    binding = EvaluationBinding.model_validate(source)
    before = binding.identity_hash()

    source["environment_fingerprint"]["dependency_versions"][0][1] = "9.9.9"
    source["operational_policy_refs"][0]["schema_name"] = "whetstone.wrong"

    assert binding.environment_fingerprint.dependency_versions == (
        ("dr-code", "0.1.0"),
    )
    assert (
        binding.operational_policy_refs[0].schema_name
        == "whetstone.test.accounting_policy"
    )
    assert binding.identity_hash() == before
    with pytest.raises(ValidationError, match="frozen"):
        binding.environment_fingerprint.__setattr__("code_revision", "changed")


def test_evaluation_binding_identity_is_sensitive_to_exact_content() -> None:
    internal = _binding()
    changed_campaign_payload = internal.model_dump(mode="json")
    changed_campaign_payload["campaign"] = "another-campaign"
    changed_environment_payload = internal.model_dump(mode="json")
    changed_environment_payload["environment_fingerprint"]["runtime_hash"] = (
        "darwin"
    )
    official = _binding(
        role=EvaluationRole.OFFICIAL,
        authority_principal="official-publisher",
    )

    assert (
        EvaluationBinding.model_validate(
            changed_campaign_payload
        ).identity_hash()
        != internal.identity_hash()
    )
    assert (
        EvaluationBinding.model_validate(
            changed_environment_payload
        ).identity_hash()
        != internal.identity_hash()
    )
    assert official.identity_hash() != internal.identity_hash()


def test_environment_dependencies_are_unique_and_canonical_by_package() -> (
    None
):
    fingerprint = ExecutionEnvironmentFingerprint(
        dependency_versions=(
            ("whetstone-envs", "2"),
            ("dr-code", "1"),
        )
    )
    assert fingerprint.dependency_versions == (
        ("dr-code", "1"),
        ("whetstone-envs", "2"),
    )
    with pytest.raises(ValidationError, match="package names must be unique"):
        ExecutionEnvironmentFingerprint(
            dependency_versions=(("dr-code", "1"), ("dr-code", "2"))
        )


@pytest.mark.parametrize("unordered", [set(), frozenset(), {}])
def test_top_level_dependency_versions_reject_unordered_containers(
    unordered: object,
) -> None:
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        ExecutionEnvironmentFingerprint(dependency_versions=unordered)


def test_dependency_pairs_reject_unordered_containers() -> None:
    with pytest.raises(ValidationError, match="package/version pairs"):
        ExecutionEnvironmentFingerprint(dependency_versions=[{"dr-code", "1"}])


@pytest.mark.parametrize("unordered", [set(), frozenset(), {}])
def test_operational_policy_refs_reject_unordered_containers(
    unordered: object,
) -> None:
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        EvaluationBinding.model_validate(
            {
                **_binding().model_dump(mode="json"),
                "operational_policy_refs": unordered,
            }
        )
