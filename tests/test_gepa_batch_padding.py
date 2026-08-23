"""Upstream's padded minibatches survive the position-unique eval contract.

``EpochShuffledBatchSampler`` pads a shuffled epoch up to a multiple of the
minibatch size by repeating its least-frequent ids, so a minibatch repeats an
instance whenever ``len(trainset) % reflection_minibatch_size != 0``. A
Whetstone evaluation request is position-unique by contract, so the repeat used
to raise "GEPA evaluation positions must be unique" and kill the run.

The adapter now evaluates the distinct instances once and expands the rows back
to the upstream batch shape, so GEPA still sees a repeated instance's score,
output, and trajectory once per occurrence -- which its accept/reject sum
depends on -- while providers are billed once per distinct instance.
"""

from __future__ import annotations

import random

import pytest

from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

from whetstone.core.identity import compute_identity_hash, typed_ref_for_record
from whetstone.optim.gepa.contracts import (
    GepaCandidateComponent,
    GepaDataInstance,
    GepaEffectContext,
    GepaEvalAuthorityBinding,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaEvaluationRow,
    GepaProposalAuthorityBinding,
)
from whetstone.optim.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optim.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    WhetstoneGepaAdapter,
)

COMPONENT = "generate"
SEED = "Reply briefly to: {prompt}"
LOADER_HASH = "9" * 64

#: (trainset size, reflection minibatch size). The first two are the shapes
#: that failed; (6, 3) and (1, 1) divide evenly and must stay unchanged.
SHAPES = [(4, 3), (44, 3), (6, 3), (5, 2), (1, 1)]


def _services() -> GepaPromptServices:
    return GepaPromptServices(
        descriptor=GepaPromptFormatDescriptor(
            format_name="padding_prompt_template",
            components=(
                GepaComponentFormat(
                    component_name=COMPONENT,
                    component_schema_identity_hash=compute_identity_hash(
                        schema="whetstone.testing.gepa_component",
                        schema_version=1,
                        payload={"field": "user_prompt_template"},
                    ),
                    allowed_placeholders=("prompt",),
                    required_placeholders=("prompt",),
                ),
            ),
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )


def _proposer_config():
    from whetstone.core.identity import IdentityRef
    from whetstone.optim.proposal.proposer import ProposerConfig

    record_ref = typed_ref_for_record(
        "whetstone.testing.provider_call_config", {"model": "fake"}
    )
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=record_ref,
            record_hash=record_ref.content_hash,
        ),
    )


def _instance(position: int) -> GepaDataInstance:
    return GepaDataInstance(
        upstream_position=position,
        data_id=f"task-{position}",
        task_hash=compute_identity_hash(
            schema="whetstone.testing.gepa_task",
            schema_version=1,
            payload={"position": position},
        ),
        data_ref=typed_ref_for_record(
            "whetstone.testing.gepa_data", {"position": position}
        ),
        loader_identity_hash=LOADER_HASH,
    )


class RecordingEvalBroker:
    """Scores each instance by position and records every request it saw."""

    def __init__(self) -> None:
        self.requests: list[GepaEvaluationEffectRequest] = []

    @property
    def provider_rows(self) -> int:
        """Rows a provider was actually billed for."""
        return sum(len(request.data) for request in self.requests)

    def evaluate(
        self, request: GepaEvaluationEffectRequest
    ) -> GepaEvaluationEffectResult:
        self.requests.append(request)
        rows = tuple(
            GepaEvaluationRow(
                data=instance,
                output={"text": f"out-{instance.upstream_position}"},
                score=float(instance.upstream_position),
                evidence_refs=(
                    typed_ref_for_record(
                        "whetstone.testing.gepa_evidence",
                        {"position": instance.upstream_position},
                    ),
                ),
            )
            for instance in request.data
        )
        return GepaEvaluationEffectResult(
            request_hash=request.identity_hash(),
            rows=rows,
            logical_metric_calls=len(rows),
        )

    def propose(self, request):  # pragma: no cover - unused here
        raise AssertionError("this broker only serves evaluations")


def _adapter(broker: RecordingEvalBroker) -> WhetstoneGepaAdapter:
    services = _services()
    return WhetstoneGepaAdapter(
        context=GepaEffectContext(
            run_id="gepa-padding-run",
            control_identity_hash="f" * 64,
            source_manifest_identity_hash="0" * 64,
            adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
        ),
        broker=broker,
        evaluation_authority=GepaEvalAuthorityBinding(
            authority_identity_hash="1" * 64,
            evaluation_config_hash="2" * 64,
            reward_policy_identity_hash="3" * 64,
            provider_route_identity_hash="4" * 64,
            execution_policy_identity_hash="5" * 64,
            prompt_adapter_identity_hash="6" * 64,
            response_parser_identity_hash="7" * 64,
            data_registry_identity_hash="8" * 64,
        ),
        proposal_authority=GepaProposalAuthorityBinding(
            authority_identity_hash="a" * 64,
            proposer_transport_identity_hash="b" * 64,
            prompt_binding_identity_hash=services.binding.identity_hash(),
            execution_policy_identity_hash="c" * 64,
            prompt_adapter_identity_hash="d" * 64,
            durability_policy_identity_hash="e" * 64,
            proposer_config=_proposer_config(),
        ),
        prompt_services=services,
    )


class _Loader:
    """The subset of the upstream DataLoader surface the sampler touches."""

    def __init__(self, size: int) -> None:
        self._size = size

    def all_ids(self) -> list[int]:
        return list(range(self._size))

    def __len__(self) -> int:
        return self._size


class _State:
    def __init__(self) -> None:
        self.i = 0


def _upstream_batches(
    trainset_size: int, minibatch_size: int, *, steps: int, seed: int = 0
) -> list[list[int]]:
    """Drive the real pinned sampler, so padding is upstream's, not a mock."""
    sampler = EpochShuffledBatchSampler(
        minibatch_size=minibatch_size, rng=random.Random(seed)
    )
    loader = _Loader(trainset_size)
    state = _State()
    batches: list[list[int]] = []
    for step in range(steps):
        state.i = step
        batches.append(list(sampler.next_minibatch_ids(loader, state)))
    return batches


def test_upstream_sampler_still_pads_with_duplicates() -> None:
    """Pin the upstream behavior this fix accommodates rather than removes.

    If a future bump makes the sampler stop repeating ids, the adapter's
    expansion becomes dead weight and this test says so.
    """
    batches = _upstream_batches(4, 3, steps=6)
    assert any(len(set(batch)) != len(batch) for batch in batches)
    # Evenly divisible shapes never pad.
    assert all(
        len(set(batch)) == len(batch)
        for batch in _upstream_batches(6, 3, steps=6)
    )


@pytest.mark.parametrize(("trainset_size", "minibatch_size"), SHAPES)
def test_padded_batch_evaluates_and_expands(
    trainset_size: int, minibatch_size: int
) -> None:
    """Every shape yields unique requests and upstream-shaped score vectors."""
    steps = 20
    batches = _upstream_batches(trainset_size, minibatch_size, steps=steps)
    broker = RecordingEvalBroker()
    adapter = _adapter(broker)

    for batch_ids in batches:
        batch = [_instance(position) for position in batch_ids]
        result = adapter.evaluate(batch, {COMPONENT: SEED})

        # GEPA receives exactly the batch shape it asked for, duplicates
        # included, because it sums these scores when accepting a mutation.
        assert len(result.scores) == minibatch_size
        assert result.scores == [float(position) for position in batch_ids]
        assert result.outputs == [
            {"text": f"out-{position}"} for position in batch_ids
        ]

    # Every request the contract saw was position-unique: the original defect.
    for request in broker.requests:
        positions = [item.upstream_position for item in request.data]
        assert len(positions) == len(set(positions))

    # Providers are billed once per distinct instance, never per duplicate.
    assert broker.provider_rows == sum(
        len(set(batch_ids)) for batch_ids in batches
    )


@pytest.mark.parametrize(("trainset_size", "minibatch_size"), SHAPES)
def test_logical_metric_calls_follow_upstream(
    trainset_size: int, minibatch_size: int
) -> None:
    """Budget accounting stays upstream's padded count, duplicates included.

    Upstream charges ``len(subsample_ids)`` per reflection minibatch at
    ``reflective_mutation.py`` line 198, so ``max_metric_calls`` must keep
    meaning what it means upstream even though providers are billed less.
    """
    steps = 20
    batches = _upstream_batches(trainset_size, minibatch_size, steps=steps)
    upstream_charge = sum(len(batch_ids) for batch_ids in batches)
    assert upstream_charge == steps * minibatch_size

    broker = RecordingEvalBroker()
    adapter = _adapter(broker)
    returned = 0
    for batch_ids in batches:
        batch = [_instance(position) for position in batch_ids]
        returned += len(adapter.evaluate(batch, {COMPONENT: SEED}).scores)

    # What GEPA counts against its budget is the padded length it passed in.
    assert returned == upstream_charge
    # Provider spend is never more than the logical charge, and is strictly
    # less exactly when the sampler padded.
    assert broker.provider_rows <= upstream_charge
    pads = upstream_charge != sum(
        len(set(batch_ids)) for batch_ids in batches
    )
    assert (broker.provider_rows < upstream_charge) is pads


@pytest.mark.parametrize(("trainset_size", "minibatch_size"), SHAPES)
def test_expansion_is_replay_identical(
    trainset_size: int, minibatch_size: int
) -> None:
    """The expansion adds no state, so a replayed run scores identically."""
    # (44, 3) does not pad until step 14, so a short probe would pass even
    # unfixed; every non-divisible shape must actually reach a duplicate.
    steps = 20
    batches = _upstream_batches(trainset_size, minibatch_size, steps=steps)

    def run() -> list[list[float]]:
        adapter = _adapter(RecordingEvalBroker())
        return [
            list(
                adapter.evaluate(
                    [_instance(position) for position in batch_ids],
                    {COMPONENT: SEED},
                ).scores
            )
            for batch_ids in batches
        ]

    assert run() == run()
    # The sampler itself is seed-deterministic, so the whole sequence repeats.
    assert batches == _upstream_batches(
        trainset_size, minibatch_size, steps=steps
    )


def test_repeated_position_with_conflicting_instance_is_rejected() -> None:
    """A repeat must be a genuine repeat, not two instances colliding."""
    broker = RecordingEvalBroker()
    adapter = _adapter(broker)
    first = _instance(0)
    conflicting = first.model_copy(update={"data_id": "task-other"})
    with pytest.raises(ValueError, match="different instance"):
        adapter.evaluate([first, conflicting], {COMPONENT: SEED})


def _reflection_broker(broker: RecordingEvalBroker):
    """Wrap the eval broker with a scripted, provider-free reflection path."""
    from whetstone.optim.gepa.contracts import GepaProposalEffectResult

    class _Broker:
        def __init__(self) -> None:
            self.evaluate = broker.evaluate
            self.proposals = 0

        def propose(self, request):
            self.proposals += 1
            text = f"Reply in {self.proposals} words to: {{prompt}}"
            return (
                GepaProposalEffectResult(
                    request_hash=request.identity_hash(),
                    raw_response=text,
                    parsed_components=(
                        GepaCandidateComponent(
                            name=request.component_name, text=text
                        ),
                    ),
                    request_evidence={"scripted": True},
                    response_evidence={"scripted": True},
                    provider_attempt_refs=(
                        typed_ref_for_record(
                            "whetstone.gepa.proposal_provider_attempt/v2",
                            {"scripted": True},
                        ),
                    ),
                ),
                False,
            )

    return _Broker()


class _TracingEvalBroker(RecordingEvalBroker):
    """Adds the trajectories the reflection path requires."""

    def evaluate(self, request):
        from whetstone.optim.gepa.contracts import GepaTrajectoryProjection

        result = super().evaluate(request)
        if not request.capture_traces:
            return result
        rows = tuple(
            row.model_copy(
                update={
                    "trajectory": GepaTrajectoryProjection(
                        data_id=row.data.data_id,
                        inputs={"prompt": row.data.data_id},
                        generated_outputs=row.output,
                        feedback=f"score {row.score}",
                        component_records={},
                        module_score=row.score,
                    )
                }
            )
            for row in result.rows
        )
        return result.model_copy(update={"rows": rows})


@pytest.mark.parametrize(
    ("trainset_size", "minibatch_size", "max_metric_calls"),
    [(4, 3, 24), (44, 3, 420)],
)
def test_gepa_toy_e2e_survives_padded_minibatches(
    trainset_size: int, minibatch_size: int, max_metric_calls: int
) -> None:
    """The real upstream engine drives a non-divisible shape to completion.

    Before the adapter collapsed duplicate positions this raised
    "GEPA evaluation positions must be unique" partway through the run. The
    (44, 3) shape is the whetstone-envs protocol shape, which does not pad
    until step 14, so the budget is sized to get well past that.
    """
    from gepa import optimize

    broker = _TracingEvalBroker()
    adapter = _adapter(_reflection_broker(broker))

    trainset = [_instance(position) for position in range(trainset_size)]
    result = optimize(
        seed_candidate={COMPONENT: SEED},
        trainset=trainset,
        valset=trainset,
        adapter=adapter,
        reflection_lm=None,
        custom_candidate_proposer=None,
        logger=None,
        callbacks=None,
        batch_sampler="epoch_shuffled",
        reflection_minibatch_size=minibatch_size,
        max_metric_calls=max_metric_calls,
        run_dir=None,
        use_wandb=False,
        use_mlflow=False,
        display_progress_bar=False,
        seed=0,
        raise_on_exception=True,
    )

    assert result is not None
    # The run really exercised the padded path, and every request the
    # position-unique contract saw stayed unique.
    assert broker.requests
    # Assert the run actually reached a padded batch, otherwise this passes
    # even unfixed and proves nothing.
    assert any(
        len(request.data) < minibatch_size and request.capture_traces
        for request in broker.requests
    )
    for request in broker.requests:
        positions = [item.upstream_position for item in request.data]
        assert len(positions) == len(set(positions))
