from __future__ import annotations

from typing import Any

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    FULL_D,
    eval_config,
)
from whetstone.core.identity import IdentityRef, typed_ref_for_record
from whetstone.experiment.binding import eval_config_reference
from whetstone.optimization.gepa.contracts import (
    GepaCandidateComponent,
    GepaDataInstance,
    GepaEffectContext,
    GepaEffectSlot,
    GepaEvaluationAuthorityBinding,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaEvaluationRow,
    GepaProposalAuthorityBinding,
)
from whetstone.optimization.gepa.control import configure_gepa
from whetstone.optimization.gepa.engine import GepaDetailedResult
from whetstone.optimization.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optimization.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
)
from whetstone.optimization.proposal.proposer import ProposerConfig

_FULL_E = "e" * 64
_FULL_F = "f" * 64


def prompt_services() -> GepaPromptServices:
    return GepaPromptServices(
        descriptor=GepaPromptFormatDescriptor(
            format_name="test",
            components=(
                GepaComponentFormat(
                    component_name="alpha",
                    component_schema_identity_hash=FULL_A,
                ),
                GepaComponentFormat(
                    component_name="beta",
                    component_schema_identity_hash=FULL_B,
                ),
            ),
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )


def effect_context() -> GepaEffectContext:
    return GepaEffectContext(
        run_id="gepa:test",
        control_identity_hash=FULL_A,
        source_manifest_identity_hash=FULL_B,
        adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    )


def data_instance(index: int) -> GepaDataInstance:
    return GepaDataInstance(
        upstream_position=index,
        data_id=f"{index + 1:064x}",
        data_ref=typed_ref_for_record(
            "test.gepa.data",
            {"index": index},
        ),
        loader_identity_hash=FULL_C,
    )


def evaluation_authority_binding(
    *,
    failure_score: float = 0.0,
    add_format_failure_as_feedback: bool = False,
    warn_on_score_mismatch: bool = True,
    selection_seed: int = 0,
) -> GepaEvaluationAuthorityBinding:
    return GepaEvaluationAuthorityBinding(
        authority_identity_hash=FULL_A,
        evaluation_config_identity_hash=FULL_B,
        reward_policy_identity_hash=FULL_C,
        provider_route_identity_hash=FULL_D,
        execution_policy_identity_hash=_FULL_E,
        prompt_adapter_identity_hash=_FULL_F,
        response_parser_identity_hash=FULL_A,
        data_registry_identity_hash=FULL_B,
        failure_score=failure_score,
        add_format_failure_as_feedback=add_format_failure_as_feedback,
        warn_on_score_mismatch=warn_on_score_mismatch,
        selection_seed=selection_seed,
    )


def proposal_authority_binding(
    services: GepaPromptServices | None = None,
) -> GepaProposalAuthorityBinding:
    active = services or prompt_services()
    return GepaProposalAuthorityBinding(
        authority_identity_hash=FULL_B,
        proposer_transport_identity_hash=FULL_C,
        prompt_binding_identity_hash=active.binding.identity_hash(),
        execution_policy_identity_hash=FULL_D,
        prompt_adapter_identity_hash=_FULL_E,
        durability_policy_identity_hash=_FULL_F,
        proposer_config=ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    {"provider_call_config_ref": "provider://gepa-reflection"},
                ),
                identity_hash=FULL_D,
            ),
        ),
    )


def evaluation_request() -> GepaEvaluationEffectRequest:
    return GepaEvaluationEffectRequest(
        slot=GepaEffectSlot(context=effect_context(), invocation_ordinal=0),
        candidate=(
            GepaCandidateComponent(name="alpha", text="alpha-0"),
            GepaCandidateComponent(name="beta", text="beta-0"),
        ),
        data=(data_instance(0), data_instance(1)),
        capture_traces=False,
        authority=evaluation_authority_binding(),
    )


def evaluation_result(
    request: GepaEvaluationEffectRequest,
) -> GepaEvaluationEffectResult:
    return GepaEvaluationEffectResult(
        request_identity_hash=request.identity_hash(),
        rows=tuple(
            GepaEvaluationRow(
                data=item,
                output={"data_id": item.data_id},
                score=float(index),
                evidence_refs=(item.data_ref,),
            )
            for index, item in enumerate(request.data)
        ),
        logical_metric_calls=len(request.data),
    )


def gepa_control(**overrides: Any):
    values: dict[str, Any] = {
        "reflection_model": ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    {"provider_call_config_ref": "provider://reflection"},
                ),
                identity_hash=FULL_A,
            ),
        ),
        "metric": eval_config_reference(eval_config()),
        "reward_policy_hash": FULL_B,
        "evaluation_execution_policy_hash": FULL_C,
        "proposal_execution_policy_hash": FULL_A,
        "proposal_prompt_adapter_identity_hash": FULL_B,
        "proposal_durability_policy_identity_hash": FULL_D,
        "task_model_identity_hash": FULL_D,
        "prompt_format_identity_hash": FULL_A,
        "prompt_binding_identity_hash": FULL_B,
        "trainset_task_identities": (FULL_A,),
        "valset_task_identities": None,
        "component_names": ("prompt",),
        "num_predictors": 1,
        "max_metric_calls": 1,
    }
    values.update(overrides)
    return configure_gepa(**values)


def make_gepa_detailed_result(control) -> GepaDetailedResult:
    return GepaDetailedResult(
        candidates=({"prompt": "seed"}, {"prompt": "best"}),
        parents=((None,), (0,)),
        val_aggregate_scores=(0.0, 1.0),
        val_subscores=({FULL_A: 0.0}, {FULL_A: 1.0}),
        per_val_instance_best_candidates={FULL_A: (1,)},
        discovery_eval_counts=(0, 1),
        total_metric_calls=1,
        num_full_val_evals=2,
        seed=control.seed,
        best_idx=1,
        control_identity_hash=control.identity_hash(),
    )
