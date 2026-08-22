"""Proposal plan / fold / resume without the harness loop."""

from __future__ import annotations

from dr_store.sync import open_sqlite

from whetstone.testing.runtime import (
    build_miprov2_adapter,
    build_toy_copro_control,
    prepare_toy_miprov2_run,
    register_toy_runtime,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY, MIPROV2_STATE_KEY
from whetstone.optim.miprov2.control import Miprov2DemoMode
from whetstone.optim.miprov2.proposal import (
    Miprov2ProposalResponse,
    Miprov2ProposalState,
    fold_proposal_response,
    plan_next_proposal_request,
    start_miprov2_proposal,
)
from whetstone.optim.miprov2.runtime import Miprov2State
from whetstone.testing.toy.miprov2 import build_toy_miprov2_control


def _opening_state(store, *, run_id: str) -> Miprov2State:
    engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_miprov2_control(
        engine=engine, demo_mode=Miprov2DemoMode.ZEROSHOT
    )
    adapter = build_miprov2_adapter(
        store=store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    launch = prepare_toy_miprov2_run(
        runtime, run_id=run_id, control=control, engine=engine
    )
    assert launch.extra_pools is not None
    return Miprov2State.model_validate(launch.extra_pools[MIPROV2_STATE_KEY])


def test_start_plan_fold_and_resume_keep_the_pending_request(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "proposal.sqlite")) as store:
        opening = _opening_state(store, run_id="proposal-unit")

    state = start_miprov2_proposal(
        bindings=opening.bindings,
        optimization_run_identity_hash=opening.run.config_hash,
        components=opening.proposal_components,
        trainset=opening.proposal_trainset,
        demo_candidates=None,
        num_candidates=opening.control.num_instruct_candidates,
        view_data_batch_size=opening.control.view_data_batch_size,
        init_temperature=opening.control.init_temperature,
        data_aware=opening.control.data_aware_proposer,
        program_aware=opening.control.program_aware_proposer,
        tip_aware=opening.control.tip_aware_proposer,
        fewshot_aware=opening.control.fewshot_aware_proposer,
        rng_checkpoint=opening.rng_checkpoint,
    )
    planned = plan_next_proposal_request(state)
    assert planned.request is not None
    pending = planned.state

    restored = Miprov2ProposalState.model_validate(
        pending.model_dump(mode="json")
    )
    replayed = plan_next_proposal_request(restored)
    assert replayed.request == planned.request
    assert replayed.state.pending_request == pending.pending_request

    folded = fold_proposal_response(
        pending,
        Miprov2ProposalResponse(
            request_hash=planned.request.identity_hash,
            text="A short description of the toy greeting tasks.",
        ),
    )
    assert folded.pending_request is None
    resumed = Miprov2ProposalState.model_validate(
        folded.model_dump(mode="json")
    )
    assert plan_next_proposal_request(resumed).request == (
        plan_next_proposal_request(folded).request
    )
