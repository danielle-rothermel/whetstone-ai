"""MIPROv2 adapter: trial + promotion steps against the durable harness.

Covers the step structure pinned by the brief (bootstrap, pool construction,
baseline full, minibatch trials, promotion, completion), pools/study/budgets
advancing only through immutable refs, the proposer route distinct from graph
routes, the cardinality-failure gate, budget exhaustion mid-loop, restart, and
the no-op promotion path.
"""

from __future__ import annotations

from typing import Any

from whetstone.optimization import (
    BASELINE_FULL,
    BOOTSTRAP,
    COMPLETION,
    MINIBATCH,
    POOL_CONSTRUCTION,
    PROMOTION_FULL,
    BudgetState,
    Miprov2Adapter,
    OptimizationHarness,
    StepStatus,
    TrialCombinationIdentity,
)

from .proposal_support import (
    FULL_FULL,
    ScriptedEvaluationService,
    bootstrap_demo_sets,
    fake_transport,
    make_harness,
    make_store,
    miprov2_hyper,
    miprov2_request,
    proposer_config,
    seed_candidates,
)


def _instruction_script(count: int) -> dict[tuple[str, int], tuple[str, ...]]:
    # Pool construction drafts one instruction per attempt; distinct texts.
    return {
        (POOL_CONSTRUCTION, i): (f"instruction-{i}",) for i in range(count)
    }


def _run(harness, req, adapter):
    return harness.run_step(req, adapter)


def test_miprov2_full_run_bootstrap_to_completion() -> None:
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    transport = fake_transport(_instruction_script(8))
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(temperature=1.0),
        transport=transport,
    )
    run_id = "mip-run"
    hyper = miprov2_hyper(num_instructions=6, num_demo_sets=4, returned=3)

    # Step 0: bootstrap -> emits num_demo_set_candidates bootstrap Intents.
    r0, ref0 = _run(
        harness,
        miprov2_request(
            run_id=run_id, step_index=0, kind_label=BOOTSTRAP,
            candidates=seed_candidates(), hyper=hyper,
        ),
        adapter,
    )
    assert r0.state_ref is not None
    assert len(r0.resolved_intents) == 4
    assert all(
        ir.intent.purpose == BOOTSTRAP for ir in r0.resolved_intents
    )

    # Step 1: pool construction -> freeze instruction x demo-set combination
    # pool. The proposal LM drafted the remaining distinct instructions.
    r1, ref1 = _run(
        harness,
        miprov2_request(
            run_id=run_id, step_index=1, kind_label=POOL_CONSTRUCTION,
            candidates=seed_candidates(), hyper=hyper,
            pools={"demo_set_pool": bootstrap_demo_sets(3)},
            prior_step_result_ref=ref0,
        ),
        adapter,
    )
    assert r1.status is StepStatus.CONTINUE
    state1 = _state(harness, r1)
    assert state1["pool_frozen"] is True
    # 6 instructions x 4 demo sets = 24 combinations.
    assert state1["combination_pool_size"] == 24
    assert len(state1["combination_pool"]) == 24
    combination_pool = state1["combination_pool"]
    templates = state1["instruction_texts"]

    # Step 2: baseline full -> one baseline_full Intent.
    r2, ref2 = _run(
        harness,
        miprov2_request(
            run_id=run_id, step_index=2, kind_label=BASELINE_FULL, hyper=hyper,
            pools={"combination_pool": combination_pool},
            prior_step_result_ref=ref1,
        ),
        adapter,
    )
    assert len(r2.resolved_intents) == 1
    assert r2.resolved_intents[0].intent.purpose == BASELINE_FULL

    # Steps 3..: minibatch trials. The seeded sampler picks pool combinations.
    means: dict[str, float] = {}
    templates_by_combo: dict[str, str] = {}
    prior = ref2
    trial_indices = [3, 4, 5]
    for score, idx in zip((0.4, 0.9, 0.6), trial_indices, strict=True):
        evaluator._scores.clear()  # reset per-trial scoring
        r, prior = _run(
            harness,
            miprov2_request(
                run_id=run_id, step_index=idx, kind_label=MINIBATCH,
                hyper=hyper,
                pools={"combination_pool": combination_pool},
                prior_step_result_ref=prior,
            ),
            adapter,
        )
        combo = _state(harness, r)["trial_combination_hash"]
        # Whetstone groups minibatch observations by combination and tracks the
        # arithmetic mean; here one observation per trial.
        means[combo] = score
        templates_by_combo.setdefault(combo, templates[0])
        assert r.resolved_intents[0].intent.purpose == MINIBATCH

    # Promotion: highest-mean combination not yet fully evaluated.
    study = {
        "combination_means": means,
        "fully_evaluated": [],
        "full_scores": {},
        "combination_templates": templates_by_combo,
    }
    r_promo, prior = _run(
        harness,
        miprov2_request(
            run_id=run_id, step_index=6, kind_label=PROMOTION_FULL,
            hyper=hyper,
            pools={
                "combination_pool": combination_pool,
                "study_state": study,
            },
            prior_step_result_ref=prior,
        ),
        adapter,
    )
    assert len(r_promo.resolved_intents) == 1
    assert r_promo.resolved_intents[0].intent.purpose == PROMOTION_FULL
    promoted = _state(harness, r_promo)["promoted_combination_hash"]
    # Highest mean (0.9) is chosen.
    assert means[promoted] == max(means.values())

    # Completion: three ordered distinct measured proposals.
    full_scores = {promoted: 0.9}
    study_final = {
        "combination_means": means,
        "full_scores": full_scores,
        "combination_templates": {c: templates[0] for c in means},
    }
    r_done, ref_done = _run(
        harness,
        miprov2_request(
            run_id=run_id, step_index=7, kind_label=COMPLETION, hyper=hyper,
            pools={"study_state": study_final},
            prior_step_result_ref=prior,
        ),
        adapter,
    )
    assert r_done.status is StepStatus.COMPLETE
    assert len(r_done.accepted_candidates) == 3
    # The full-scored combination ranks first (tier 1 before tier 2).
    ordered = _state(harness, r_done)["ordered_combination_hashes"]
    assert ordered[0] == promoted

    terminal = harness.terminalize(
        run_id=run_id,
        step_result_refs=(ref0, ref1, ref2, ref_done),
    )
    assert len(terminal.proposals) == 3


def test_miprov2_combination_pool_reflects_supplied_demo_sets() -> None:
    # The frozen pool = instructions x (empty set + bootstrap demo sets); the
    # adapter never fabricates demo sets. With 1 bootstrap set -> 2 demo sets
    # total -> 6 x 2 = 12 combinations.
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(),
        transport=fake_transport(_instruction_script(8)),
    )
    r, _ref = harness.run_step(
        miprov2_request(
            run_id="mip-demos", step_index=0, kind_label=POOL_CONSTRUCTION,
            candidates=seed_candidates(), hyper=miprov2_hyper(),
            pools={"demo_set_pool": bootstrap_demo_sets(1)},
        ),
        adapter,
    )
    assert _state(harness, r)["combination_pool_size"] == 12


def test_miprov2_proposer_route_distinct_and_in_config_identity() -> None:
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    transport = fake_transport(_instruction_script(8))
    pc = proposer_config(route="pcc://openai/gpt-5.4-miprov2-proposer")
    adapter = Miprov2Adapter(proposer_config=pc, transport=transport)
    hyper = miprov2_hyper()

    req = miprov2_request(
        run_id="mip", step_index=0, kind_label=POOL_CONSTRUCTION,
        candidates=seed_candidates(), hyper=hyper, pools={"demo_set_pool": []},
    )
    harness.run_step(req, adapter)
    assert transport.calls
    used_hash, _r, _c = transport.calls[0]
    assert used_hash == pc.identity_hash()
    # Distinct from the encoder graph route (candidate base_ref).
    assert used_hash != seed_candidates()[0].base_ref
    # Not serialized into the request identity (optimizer Config, not graph).
    assert pc.identity_hash() not in str(req.record_content())


def test_miprov2_instruction_cardinality_failure() -> None:
    # Every drafted instruction is a duplicate of seed A -> the pool cannot
    # reach num_instruction_candidates within the attempt cap -> failed.
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    seed_text = seed_candidates()[0].payload["user_prompt_template"]
    transport = fake_transport({}, default=(seed_text,))
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(), transport=transport
    )
    r, _ref = harness.run_step(
        miprov2_request(
            run_id="mip-card", step_index=0, kind_label=POOL_CONSTRUCTION,
            candidates=seed_candidates(),
            hyper=miprov2_hyper(num_instructions=6, attempt_cap=12),
            pools={"demo_set_pool": []},
        ),
        adapter,
    )
    assert r.status is StepStatus.FAILED
    assert "cardinality" in _state(harness, r)["reason"]


def test_miprov2_completion_cardinality_failure_blocks_official() -> None:
    # Fewer than returned_proposal_count distinct measured combinations ->
    # explicit failed terminal Step Result; terminalize claims no proposals.
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(),
        transport=fake_transport({}),
    )
    study = {
        "combination_means": {"c1": 0.5},
        "full_scores": {},
        "combination_templates": {"c1": "t"},
    }
    r, ref = harness.run_step(
        miprov2_request(
            run_id="mip-done-fail", step_index=0, kind_label=COMPLETION,
            hyper=miprov2_hyper(returned=3),
            pools={"study_state": study},
        ),
        adapter,
    )
    assert r.status is StepStatus.FAILED
    terminal = harness.terminalize(
        run_id="mip-done-fail", step_result_refs=(ref,)
    )
    assert terminal.status is StepStatus.FAILED
    assert terminal.proposals == ()


def test_miprov2_promotion_noop_when_all_fully_evaluated() -> None:
    # Every trial-observed combination already fully evaluated -> deterministic
    # no-op Step Result, no Evaluation Intent.
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(),
        transport=fake_transport({}),
    )
    study = {
        "combination_means": {"c1": 0.5, "c2": 0.7},
        "fully_evaluated": ["c1", "c2"],
        "full_scores": {"c1": 0.5, "c2": 0.7},
        "combination_templates": {"c1": "t1", "c2": "t2"},
    }
    r, _ref = harness.run_step(
        miprov2_request(
            run_id="mip-noop", step_index=0, kind_label=PROMOTION_FULL,
            hyper=miprov2_hyper(),
            pools={"combination_pool": ["c1", "c2"], "study_state": study},
        ),
        adapter,
    )
    assert r.status is StepStatus.CONTINUE
    assert not r.resolved_intents  # no Intent emitted for a no-op
    assert _state(harness, r)["promotion"] == "noop"


def test_miprov2_budget_exhaustion_mid_loop_fails_with_accounting() -> None:
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(),
        transport=fake_transport({}),
    )
    # A trial carrying an exhausted search budget: the harness copies the
    # immutable request budget onto the Step Result, so the FAILED accounting
    # is durable. (Index 0 keeps the request valid without a prior ref.)
    req = miprov2_request(
        run_id="mip-budget", step_index=0, kind_label=MINIBATCH,
        hyper=miprov2_hyper(),
        pools={"combination_pool": ["c1", "c2", "c3"]},
    )
    req = req.model_copy(
        update={
            "budget": BudgetState(
                consumed={"search_rollouts": 236},
                remaining={"search_rollouts": 0},
            )
        }
    )
    r, ref = harness.run_step(req, adapter)
    assert r.status is StepStatus.FAILED
    reason = _state(harness, r)
    assert reason["reason"] == "search budget exhausted mid-loop"
    assert reason["remaining"] == 0
    assert reason["consumed"] == 236
    # Budget accounting rides on the immutable Step Result budget.
    assert r.budget.remaining["search_rollouts"] == 0
    terminal = harness.terminalize(
        run_id="mip-budget", step_result_refs=(ref,)
    )
    assert terminal.status is StepStatus.FAILED


def test_miprov2_restart_mid_run_reuses_checkpoint() -> None:
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    transport = fake_transport(_instruction_script(8))
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(), transport=transport
    )
    req = miprov2_request(
        run_id="mip-restart", step_index=0, kind_label=POOL_CONSTRUCTION,
        candidates=seed_candidates(), hyper=miprov2_hyper(),
        pools={"demo_set_pool": []},
    )
    # First pass: invoke + durable checkpoint.
    harness._run_proposal(req, adapter)
    calls_after_first = len(transport.calls)
    assert calls_after_first > 0

    # A brand-new harness over the same store replays without re-drafting.
    fresh = OptimizationHarness(store=store, evaluation_service=evaluator)
    r, _ref = fresh.run_step(req, adapter)
    assert len(transport.calls) == calls_after_first  # no re-draft on restart
    assert r.status is StepStatus.CONTINUE


def test_minibatch_trials_group_by_combination_identity() -> None:
    # Two trials that sample the same (instruction, demo set) share a
    # combination identity hash; the study averages their observations.
    ih = "a" * 64
    dh = "b" * 64
    c = TrialCombinationIdentity(instruction_hash=ih, demo_set_hash=dh)
    again = TrialCombinationIdentity(instruction_hash=ih, demo_set_hash=dh)
    assert c.identity_hash() == again.identity_hash()
    assert FULL_FULL != c.identity_hash()


def test_miprov2_state_snapshot_resolvable_after_restart() -> None:
    # The frozen pools/study state a Step commits are persisted as
    # content-addressed snapshot objects, so a fresh harness (a real restart)
    # resolves the exact state body its Step Result references — the immutable
    # state ref through which pools/study advance across Steps.
    store = make_store()
    evaluator = ScriptedEvaluationService(store)
    harness = make_harness(store, evaluator)
    adapter = Miprov2Adapter(
        proposer_config=proposer_config(),
        transport=fake_transport(_instruction_script(8)),
    )
    req = miprov2_request(
        run_id="mip-snap", step_index=0, kind_label=POOL_CONSTRUCTION,
        candidates=seed_candidates(), hyper=miprov2_hyper(),
        pools={"demo_set_pool": bootstrap_demo_sets(3)},
    )
    result, _ref = harness.run_step(req, adapter)
    assert result.state_ref is not None

    fresh = OptimizationHarness(store=store, evaluation_service=evaluator)
    body = fresh._store.get(result.state_ref.reference)
    assert isinstance(body, dict)
    assert body["pool_frozen"] is True
    assert body["combination_pool_size"] == 24


def _state(harness: OptimizationHarness, result) -> dict[str, Any]:
    assert result.state_ref is not None
    body = harness._store.get(result.state_ref.reference)
    assert isinstance(body, dict)
    return body
