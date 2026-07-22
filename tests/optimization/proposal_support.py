"""Shared builders + a deterministic driver for the proposal-optimizer tests.

These exercise the COPRO and MIPROv2 adapters against the real durable harness
with a FAKE proposer transport (scripted responses, no network). The driver
plays Whetstone's role *outside* the optimizer invocation: it threads the
Attempt History / pools / study state forward through immutable Step Request
references, resolving each Evaluation Intent through the harness's
EvaluationService, exactly as the real harness would.
"""

from __future__ import annotations

from typing import Any

from dr_store import MemoryBackend, ObjectStore

from whetstone.optimization import (
    Candidate,
    EvaluationIntent,
    FakeProposerTransport,
    IntentResolution,
    OptimizationHarness,
    OptimizationStepRequest,
    OutputContract,
    ProposerConfig,
    StepKind,
    StepMode,
    typed_ref_for_record,
)

FULL_A = "a" * 64
FULL_MINIBATCH = "1" * 64
FULL_FULL = "2" * 64
EVIDENCE_SCHEMA = "whetstone.test.proposal_evidence"


def make_store() -> ObjectStore:
    return ObjectStore(MemoryBackend())


def proposer_config(
    *, route: str = "pcc://openai/gpt-5.4", temperature: float = 1.0
) -> ProposerConfig:
    """A proposer route whose Provider Call Config hash is distinct from any
    encoder/decoder route hash used inside a graph."""
    # A stand-in full hash for the pinned Provider Call Config identity.
    return ProposerConfig(
        provider_call_config_ref=route,
        provider_call_config_hash="f" * 64,
        temperature=temperature,
    )


class ScriptedEvaluationService:
    """Resolves each Evaluation Intent under its exact target Eval Config.

    Records every resolved Intent (so restart tests can prove a completed
    proposal invocation is not rerun) and returns a caller-scripted internal
    score per intent purpose/candidate, stored as stand-in evidence.
    """

    def __init__(
        self,
        store: ObjectStore,
        *,
        scores: dict[str, float] | None = None,
    ) -> None:
        self._store = store
        self._scores = scores or {}
        self.resolved: list[EvaluationIntent] = []

    def score_for(self, intent: EvaluationIntent) -> float:
        return self._scores.get(
            intent.candidate_id, self._scores.get(intent.purpose, 1.0)
        )

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        self.resolved.append(intent)
        evidence: dict[str, Any] = {
            "intent_id": intent.intent_id,
            "candidate_id": intent.candidate_id,
            "eval_config_hash": intent.target_eval_config_hash,
            "role": intent.context_role.value,
            "score": self.score_for(intent),
        }
        self._store.put(EVIDENCE_SCHEMA, evidence)
        evidence_ref = typed_ref_for_record(EVIDENCE_SCHEMA, evidence)
        return IntentResolution(
            intent=intent,
            evaluation_evidence_refs=(evidence_ref,),
            resolved_eval_config_hash=intent.target_eval_config_hash,
        )


def bootstrap_demo_sets(n: int = 3) -> list[dict[str, Any]]:
    """``n`` distinct non-empty demo sets as bootstrap would carry them.

    Combined with the always-present empty set, ``n=3`` yields the
    ``num_demo_set_candidates=4`` frozen demo-set pool the brief pins.
    """
    return [
        {
            "pairs": [
                {
                    "ground_truth_code": f"gt-{i}",
                    "encoded_representation": f"enc-{i}",
                }
            ]
        }
        for i in range(n)
    ]


def seed_candidates() -> tuple[Candidate, ...]:
    return (
        Candidate(
            candidate_id="A",
            base_ref="encoder-A",
            payload={"user_prompt_template": "describe concisely"},
        ),
        Candidate(
            candidate_id="B",
            base_ref="encoder-A",
            payload={"user_prompt_template": "compress for reconstruction"},
        ),
    )


def copro_request(
    *,
    run_id: str,
    step_index: int,
    candidates: tuple[Candidate, ...],
    hyper: dict[str, Any],
    pools: dict[str, Any],
    prior_step_result_ref=None,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s{step_index}",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PROPOSAL_ONLY,
        kind=StepKind.PROPOSAL,
        kind_label="seed_proposal" if step_index == 0 else "history_proposal",
        step_index=step_index,
        candidates=candidates,
        hyperparameters=hyper,
        pools=pools,
        output_contract=OutputContract(returned_proposal_count=3),
        prior_step_result_ref=prior_step_result_ref,
    )


def copro_hyper(
    *, breadth: int = 4, depth: int = 2
) -> dict[str, Any]:
    return {
        "copro_variant": "whetstone_multi_seed/v1",
        "breadth": breadth,
        "depth": depth,
        "mutation_field": "user_prompt_template",
        "step_eval_config_ref": "evalcfg://internal/20task",
        "step_eval_config_hash": FULL_MINIBATCH,
    }


def miprov2_request(
    *,
    run_id: str,
    step_index: int,
    kind_label: str,
    candidates: tuple[Candidate, ...] = (),
    hyper: dict[str, Any] | None = None,
    pools: dict[str, Any] | None = None,
    prior_step_result_ref=None,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s{step_index}",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PROPOSAL_ONLY,
        kind=StepKind.PROPOSAL,
        kind_label=kind_label,
        step_index=step_index,
        candidates=candidates,
        hyperparameters=hyper or miprov2_hyper(),
        pools=pools or {},
        output_contract=OutputContract(returned_proposal_count=3),
        prior_step_result_ref=prior_step_result_ref,
    )


def miprov2_hyper(
    *,
    num_trials: int = 12,
    promote_every: int = 5,
    num_demo_sets: int = 4,
    num_instructions: int = 6,
    attempt_cap: int = 12,
    returned: int = 3,
    seed: int = 9,
) -> dict[str, Any]:
    return {
        "num_trials": num_trials,
        "minibatch_full_eval_steps": promote_every,
        "num_demo_set_candidates": num_demo_sets,
        "num_instruction_candidates": num_instructions,
        "instruction_attempt_cap": attempt_cap,
        "returned_proposal_count": returned,
        "seed": seed,
        "mutation_field": "user_prompt_template",
        "minibatch_eval_config_ref": "evalcfg://internal/minibatch",
        "minibatch_eval_config_hash": FULL_MINIBATCH,
        "full_eval_config_ref": "evalcfg://internal/full",
        "full_eval_config_hash": FULL_FULL,
    }


def make_harness(
    store: ObjectStore, evaluator: ScriptedEvaluationService
) -> OptimizationHarness:
    return OptimizationHarness(store=store, evaluation_service=evaluator)


def fake_transport(
    script: dict[tuple[str, int], tuple[str, ...]],
    *,
    default: tuple[str, ...] = (),
) -> FakeProposerTransport:
    return FakeProposerTransport(script, default=default)
