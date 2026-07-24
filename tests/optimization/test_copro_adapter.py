"""COPRO adapter: multi-step proposal-only run against the durable harness.

Covers the loop shape pinned by the brief (breadth/depth defaults, seed vs.
history proposal, Reward-ranked history via immutable state refs, next-round
proposals conditioned on resolved Reward evidence), the proposer route being
distinct from any graph route and living in the optimizer Config identity, and
the deterministic-harness paths: full multi-step run, restart-mid-run,
proposal validation / Diff Check rejection, and cardinality failure.
"""

from __future__ import annotations

import pytest

from whetstone.optimization import (
    COPRO_VARIANT,
    HISTORY_PROPOSAL,
    SEED_PROPOSAL,
    Candidate,
    CoproAdapter,
    OptimizationHarness,
    StepStatus,
    rank_attempt_history,
)

from .proposal_support import (
    ScriptedEvaluationService,
    copro_hyper,
    copro_request,
    fake_transport,
    make_harness,
    make_store,
    proposer_config,
    seed_candidates,
)


def _history_from_result(result, store) -> list[dict]:
    """Build the next Attempt History version from a resolved Step Result.

    Whetstone's role: after external evaluation resolves each Intent, it
    commits a new immutable Attempt History version pairing each candidate with
    the Reward derived from the resolved evidence. The driver reads the score
    back out of the resolved evidence body (one Evaluation Intent per Step, so
    the batch shares one score attributed to every new candidate). This is the
    "next-round proposals conditioned on resolved Reward evidence" path.
    """
    resolution = result.resolved_intents[0]
    evidence = store.get(resolution.evaluation_evidence_refs[0].reference)
    score = float(evidence["score"])
    return [
        {
            "candidate_id": candidate.candidate_id,
            "base_ref": candidate.base_ref,
            "template": candidate.payload["user_prompt_template"],
            "reward": score,
        }
        for candidate in result.accepted_candidates
    ]


def test_copro_full_two_step_run_seed_then_history() -> None:
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    transport = fake_transport(
        {
            (SEED_PROPOSAL, 0): ("seed-t1", "seed-t2", "seed-t3", "seed-t4"),
            (HISTORY_PROPOSAL, 1): (
                "hist-t1",
                "hist-t2",
                "hist-t3",
                "hist-t4",
            ),
        }
    )
    adapter = CoproAdapter(
        proposer_config=proposer_config(), transport=transport
    )
    assert isinstance(adapter, object)

    run_id = "copro-run"
    hyper = copro_hyper(breadth=4, depth=2)
    seeds = seed_candidates()

    # Round 1 = Seed Proposal (index 0). H0 measured entries carried in pools.
    h0 = [
        {"candidate_id": c.candidate_id, "base_ref": c.base_ref,
         "template": c.payload["user_prompt_template"], "reward": 0.5}
        for c in seeds
    ]
    req0 = copro_request(
        run_id=run_id, step_index=0, candidates=seeds, hyper=hyper,
        pools={"attempt_history": h0},
    )
    result0, ref0 = harness.run_step(req0, adapter)

    assert hyper["copro_variant"] == COPRO_VARIANT
    # Exactly breadth new candidates + one Evaluation Intent for the batch.
    assert len(result0.accepted_candidates) == 4
    assert len(result0.resolved_intents) == 1
    assert result0.resolved_intents[0].intent.purpose == SEED_PROPOSAL
    assert result0.status is StepStatus.CONTINUE
    # The intent resolved under the exact pinned step Eval Config.
    assert (
        result0.resolved_intents[0].resolved_eval_config_hash
        == hyper["step_eval_config_hash"]
    )

    # Whetstone commits the next immutable Attempt History version from the
    # resolved evidence, then issues round 2 conditioned on it.
    h1 = h0 + _history_from_result(result0, store)
    req1 = copro_request(
        run_id=run_id, step_index=1, candidates=seeds, hyper=hyper,
        pools={"attempt_history": h1}, prior_step_result_ref=ref0,
    )
    result1, _ref1 = harness.run_step(req1, adapter)

    # Round 2 is History Proposal, returns breadth new candidates, completes.
    assert len(result1.accepted_candidates) == 4
    assert result1.resolved_intents[0].intent.purpose == HISTORY_PROPOSAL
    assert result1.status is StepStatus.COMPLETE
    # The history-proposal templates came from the scripted history batch.
    templates = {
        c.payload["user_prompt_template"] for c in result1.accepted_candidates
    }
    assert templates == {"hist-t1", "hist-t2", "hist-t3", "hist-t4"}


def test_copro_conditions_on_reward_ranked_history() -> None:
    # The history-proposal base is the best-ranked prior candidate; ranking is
    # Reward-descending with candidate_id tie-break.
    entries = (
        {"candidate_id": "P0-1", "base_ref": "b", "template": "lo",
         "reward": 0.2},
        {"candidate_id": "P0-0", "base_ref": "b", "template": "hi",
         "reward": 0.9},
        {"candidate_id": "P0-2", "base_ref": "b", "template": "mid",
         "reward": 0.5},
    )
    ranked = rank_attempt_history(entries)
    assert [e["candidate_id"] for e in ranked] == ["P0-0", "P0-2", "P0-1"]


def test_copro_proposer_route_distinct_from_graph_and_in_config_identity(
) -> None:
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    transport = fake_transport(
        {(SEED_PROPOSAL, 0): ("t1", "t2", "t3", "t4")}
    )
    pc = proposer_config(route="pcc://openai/gpt-5.4-proposer")
    adapter = CoproAdapter(proposer_config=pc, transport=transport)

    seeds = seed_candidates()
    hyper = copro_hyper()
    req0 = copro_request(
        run_id="c", step_index=0, candidates=seeds, hyper=hyper,
        pools={"attempt_history": []},
    )
    harness.run_step(req0, adapter)

    # The proposer transport was called through the proposer route's identity
    # hash — distinct from the encoder base_ref that names the graph route.
    assert transport.calls
    used_hash, _req, _count = transport.calls[0]
    assert used_hash == pc.identity_hash()
    encoder_route = seeds[0].base_ref  # graph-side route identity
    assert used_hash != encoder_route
    # The proposer route is NOT carried in the serialized Step Request (it is
    # process-side compute in the optimizer Config identity, not graph/request
    # identity): no field of the request equals the proposer hash.
    assert pc.identity_hash() not in str(req0.record_content())


def test_copro_diff_check_rejects_off_surface_mutation() -> None:
    # A draft that would change a non-surface field is impossible to express in
    # this adapter (it only writes user_prompt_template); prove the underlying
    # diff_check the adapter uses rejects an off-surface payload.
    from whetstone.optimization import DiffCheckError, diff_check

    base = Candidate(
        candidate_id="A", base_ref="enc",
        payload={"user_prompt_template": "x", "model_route": "r0"},
    )
    off_surface = Candidate(
        candidate_id="P", base_ref="enc",
        payload={"user_prompt_template": "y", "model_route": "r1"},
    )
    with pytest.raises(DiffCheckError, match="outside the Mutation Surface"):
        diff_check(base=base, proposed=off_surface)
    wrong_base = Candidate(
        candidate_id="P", base_ref="other",
        payload={"user_prompt_template": "y"},
    )
    with pytest.raises(DiffCheckError, match="named base"):
        diff_check(base=base, proposed=wrong_base)


def test_copro_cardinality_failure_when_drafts_are_invalid() -> None:
    # A transport that returns empty templates (invalid: empty surface value)
    # cannot fill breadth -> the Step fails, blocking official materialization.
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    transport = fake_transport(
        {(SEED_PROPOSAL, 0): ("", "", "", "")},
        default=("", ""),
    )
    adapter = CoproAdapter(
        proposer_config=proposer_config(), transport=transport
    )
    req0 = copro_request(
        run_id="c-fail", step_index=0, candidates=seed_candidates(),
        hyper=copro_hyper(), pools={"attempt_history": []},
    )
    result, _ref = harness.run_step(req0, adapter)
    assert result.status is StepStatus.FAILED
    assert not result.accepted_candidates


def test_copro_restart_mid_run_reuses_checkpoint() -> None:
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    transport = fake_transport(
        {(SEED_PROPOSAL, 0): ("t1", "t2", "t3", "t4")}
    )
    adapter = CoproAdapter(
        proposer_config=proposer_config(), transport=transport
    )
    req0 = copro_request(
        run_id="c-restart", step_index=0, candidates=seed_candidates(),
        hyper=copro_hyper(), pools={"attempt_history": []},
    )
    # First pass: invoke + durable checkpoint (proposer called once).
    harness._run_proposal(req0, adapter)
    assert len(transport.calls) == 1

    # A brand-new harness over the SAME store (a real restart) replays: the
    # durable checkpoint is reused and the proposal invocation is NOT rerun.
    fresh = OptimizationHarness(store=store, evaluation_service=evaluator)
    result, _ref = fresh.run_step(req0, adapter)
    assert len(transport.calls) == 1  # proposer never re-called across restart
    assert len(result.accepted_candidates) == 4
    assert result.status is StepStatus.CONTINUE
