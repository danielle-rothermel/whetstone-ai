from __future__ import annotations

try:
    from dbos import DBOS, SetWorkflowID
except ImportError as exc:
    raise ImportError(
        "DBOS coordination requires the optional dbos extra: "
        "pip install 'whetstone-ai[dbos]'"
    ) from exc
from dr_store import ObjectStore

from whetstone.core.identity import require_full_hash
from whetstone.optimization.gepa.contracts import (
    GEPA_EVALUATION_REQUEST_RECORD_SCHEMA,
    GEPA_EVALUATION_RESULT_RECORD_SCHEMA,
    GEPA_PROPOSAL_REQUEST_RECORD_SCHEMA,
    GEPA_PROPOSAL_RESULT_RECORD_SCHEMA,
    GepaEffectRecorder,
    GepaEvaluationEffectAuthority,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaProposalEffectAuthority,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)


class GepaEffectDurabilityError(RuntimeError):
    pass


_EVALUATION_AUTHORITIES: dict[str, GepaEvaluationEffectAuthority] = {}
_PROPOSAL_AUTHORITIES: dict[str, GepaProposalEffectAuthority] = {}


def register_gepa_evaluation_authority(
    identity_hash: str,
    authority: GepaEvaluationEffectAuthority,
) -> None:

    require_full_hash(
        identity_hash,
        field="GEPA evaluation authority identity",
    )
    if authority.runtime_hash != identity_hash:
        raise GepaEffectDurabilityError(
            "GEPA evaluation authority identity does not match its registry "
            "key"
        )
    existing = _EVALUATION_AUTHORITIES.get(identity_hash)
    if existing is not None and existing is not authority:
        raise GepaEffectDurabilityError(
            "GEPA evaluation authority identity is already bound"
        )
    _EVALUATION_AUTHORITIES[identity_hash] = authority


def register_gepa_proposal_authority(
    identity_hash: str,
    authority: GepaProposalEffectAuthority,
) -> None:

    require_full_hash(identity_hash, field="GEPA proposal authority identity")
    if authority.runtime_hash != identity_hash:
        raise GepaEffectDurabilityError(
            "GEPA proposal authority identity does not match its registry key"
        )
    existing = _PROPOSAL_AUTHORITIES.get(identity_hash)
    if existing is not None and existing is not authority:
        raise GepaEffectDurabilityError(
            "GEPA proposal authority identity is already bound"
        )
    _PROPOSAL_AUTHORITIES[identity_hash] = authority


def _evaluation_authority(
    request: GepaEvaluationEffectRequest,
) -> GepaEvaluationEffectAuthority:
    identity_hash = request.authority.authority_identity_hash
    try:
        authority = _EVALUATION_AUTHORITIES[identity_hash]
    except KeyError:
        raise GepaEffectDurabilityError(
            "GEPA evaluation authority is not registered before DBOS launch"
        ) from None
    if authority.runtime_hash != identity_hash:
        raise GepaEffectDurabilityError(
            "registered GEPA evaluation authority identity drifted"
        )
    return authority


def _proposal_authority(
    request: GepaProposalEffectRequest,
) -> GepaProposalEffectAuthority:
    identity_hash = request.authority.authority_identity_hash
    try:
        authority = _PROPOSAL_AUTHORITIES[identity_hash]
    except KeyError:
        raise GepaEffectDurabilityError(
            "GEPA proposal authority is not registered before DBOS launch"
        ) from None
    if authority.runtime_hash != identity_hash:
        raise GepaEffectDurabilityError(
            "registered GEPA proposal authority identity drifted"
        )
    return authority


def _gepa_evaluation_effect_implementation(
    request: GepaEvaluationEffectRequest,
) -> GepaEvaluationEffectResult:
    result = _evaluation_authority(request).evaluate(request)
    if result.request_hash != request.identity_hash():
        raise GepaEffectDurabilityError(
            "GEPA evaluation authority returned another request's result"
        )
    return result


@DBOS.workflow()
def _gepa_evaluation_effect_workflow(
    request: GepaEvaluationEffectRequest,
) -> GepaEvaluationEffectResult:

    return _gepa_evaluation_effect_implementation(request)


def _gepa_proposal_effect_implementation(
    request: GepaProposalEffectRequest,
) -> GepaProposalEffectResult:
    result = _proposal_authority(request).propose(request)
    if result.request_hash != request.identity_hash():
        raise GepaEffectDurabilityError(
            "GEPA proposal authority returned another request's result"
        )
    return result


@DBOS.workflow()
def _gepa_proposal_effect_workflow(
    request: GepaProposalEffectRequest,
) -> GepaProposalEffectResult:
    return _gepa_proposal_effect_implementation(request)


class DbosGepaEffectBroker:
    def __init__(self, store: ObjectStore) -> None:
        self._recorder = GepaEffectRecorder(store)

    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult:
        self._recorder.record_request(request)
        workflow_id = f"whetstone-gepa-evaluate-{request.identity_hash()}"
        with SetWorkflowID(workflow_id):
            result = _gepa_evaluation_effect_workflow(request)
        return self._recorder.record_evaluation_result(
            request,
            GepaEvaluationEffectResult.model_validate(result),
        )

    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult:
        self._recorder.record_request(request)
        workflow_id = f"whetstone-gepa-propose-{request.identity_hash()}"
        with SetWorkflowID(workflow_id):
            result = _gepa_proposal_effect_workflow(request)
        return self._recorder.record_proposal_result(
            request,
            GepaProposalEffectResult.model_validate(result),
        )


__all__ = [
    "GEPA_EVALUATION_REQUEST_RECORD_SCHEMA",
    "GEPA_EVALUATION_RESULT_RECORD_SCHEMA",
    "GEPA_PROPOSAL_REQUEST_RECORD_SCHEMA",
    "GEPA_PROPOSAL_RESULT_RECORD_SCHEMA",
    "DbosGepaEffectBroker",
    "GepaEffectDurabilityError",
    "GepaEffectRecorder",
    "register_gepa_evaluation_authority",
    "register_gepa_proposal_authority",
]
