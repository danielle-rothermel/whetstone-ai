from __future__ import annotations

from dr_store import ObjectStore

from whetstone.optim.gepa.contracts import (
    GepaEffectBroker,
    GepaEffectRecorder,
    GepaEvaluationEffectAuthority,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaProposalEffectAuthority,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
)


class HarnessGepaEffectBroker(GepaEffectBroker):
    """In-process GEPA effect broker backed by registered authorities."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        evaluation_authority: GepaEvaluationEffectAuthority,
        proposal_authority: GepaProposalEffectAuthority,
    ) -> None:
        self._recorder = GepaEffectRecorder(store)
        self._evaluation_authority = evaluation_authority
        self._proposal_authority = proposal_authority

    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult:
        self._recorder.record_request(request)
        cached = self._recorder.load_evaluation_result(request)
        if cached is not None:
            # A replayed effect costs nothing and mints no intent, but the
            # evaluation it stands for is still evidence this Step must
            # report -- otherwise a retry after a mid-step crash silently
            # drops it from the Step Result.
            self._evaluation_authority.collect_replayed(cached)
            return cached
        result = self._evaluation_authority.evaluate(request)
        return self._recorder.record_evaluation_result(request, result)

    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult:
        self._recorder.record_request(request)
        cached = self._recorder.load_proposal_result(request)
        if cached is not None:
            return cached
        result = self._proposal_authority.propose(request)
        return self._recorder.record_proposal_result(request, result)


__all__ = ["HarnessGepaEffectBroker"]
