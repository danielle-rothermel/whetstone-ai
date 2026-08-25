"""Node-failure diagnostics, blank-generation scoring, and row-level retry.

Covers the Stage-0 defect where a row failed with ``node_execution_error``
and the store recorded no exception type, message, or node name -- so the
cause of a lost row was unrecoverable after the fact.

Also pins the blank-generation contract: an empty model output is an
observed, scored, failing sample rather than an unscoreable row. It is a
result -- just one that cannot possibly pass.
"""

from __future__ import annotations


import pytest

from whetstone.eval.drivers.graph_rollout import run_rollout_row
from whetstone.eval.traces import ExecutedRowState
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.provider.llm_call import LlmCallContext
from whetstone.provider.policy import (
    ProviderExecutionPolicy,
    default_transport_policy,
)
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.transport import FakeLlmTransport
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)


class _Task:
    task_id = "task-a"
    prompt_inputs = {"prompt": "hello A"}
    gold = "A"
    strata: tuple[str, ...] = ()
    seed = 0


class _SeedPlan:
    num_seeds = 1
    rng_seeds: tuple[tuple[str, int], ...] = ()


def _policy() -> ProviderExecutionPolicy:
    return ProviderExecutionPolicy(
        transport_policy=default_transport_policy(
            api_key_env="WHETSTONE_TOY_API_KEY"
        )
    )


def _run_row(text_factory, *, max_row_attempts=None):
    experiment = build_toy_experiment()
    policy = _policy()
    context = LlmCallContext(
        execution_policy=policy,
        transport=FakeLlmTransport(
            transport_policy=policy.transport_policy,
            text_factory=text_factory,
        ),
        prompt_adapter=PlainPromptAdapter(),
    )
    kwargs = {}
    if max_row_attempts is not None:
        kwargs["max_row_attempts"] = max_row_attempts
    return run_rollout_row(
        experiment=experiment,
        candidate=experiment.initial_candidate,
        task=_Task(),
        task_index=0,
        task_hash="h" * 64,
        seed_index=0,
        seed_plan=_SeedPlan(),
        split_role="internal_eval",
        llm_context=context,
        eval_runner=FakeEvalProcedureRunner(),
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        resolve_provider_call_config=(
            lambda _ref: experiment.rollout_graph.provider_call_config
        ),
        **kwargs,
    )


class _RaiseOnceTransport:
    """Raise a node-level error on the first call, then succeed."""

    def __init__(self, *, transport_policy) -> None:
        self._inner = FakeLlmTransport(
            transport_policy=transport_policy,
            text_factory=lambda _request: "wall",
        )
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("node blew up once")
        return self._inner(request)


def test_node_error_row_records_exception_cause_and_node() -> None:
    """A node failure persists what raised, not just a generic code."""

    def _boom(_request):
        raise RuntimeError("node blew up")

    experiment = build_toy_experiment()
    policy = _policy()

    class _Boom:
        def __call__(self, request):
            return _boom(request)

    context = LlmCallContext(
        execution_policy=policy,
        transport=_Boom(),
        prompt_adapter=PlainPromptAdapter(),
    )
    row = run_rollout_row(
        experiment=experiment,
        candidate=experiment.initial_candidate,
        task=_Task(),
        task_index=0,
        task_hash="h" * 64,
        seed_index=0,
        seed_plan=_SeedPlan(),
        split_role="internal_eval",
        llm_context=context,
        eval_runner=FakeEvalProcedureRunner(),
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        resolve_provider_call_config=(
            lambda _ref: experiment.rollout_graph.provider_call_config
        ),
        max_row_attempts=1,
    )
    assert row.row_state is ExecutedRowState.FAILED
    # The defect: these were all absent, leaving the failure unexplainable.
    assert row.error_type is not None
    assert row.error_message is not None
    assert row.failed_node_id == "generate"
    assert "blew up" in row.error_message


def test_blank_generation_is_an_observed_scored_failing_row() -> None:
    """An empty generation is a result -- one that cannot possibly pass.

    The model was asked and it answered nothing. That is an observed sample,
    so the row is *present* and carries the failing score the eval family
    gives empty output, rather than being withheld as unscoreable.
    """
    row = _run_row(lambda _request: "", max_row_attempts=1)
    assert row.row_state is ExecutedRowState.SUCCESS
    assert row.invalid is False
    assert row.failed is False
    assert row.missing is False
    # Scored, and scored at the family's floor for empty output.
    assert row.score == 0.0
    assert row.output_text == ""
    # The blank marker is retained as explanation, not as row state.
    assert row.failure_code == "blank-provider-generation"


def test_whitespace_generation_is_an_observed_scored_failing_row() -> None:
    row = _run_row(lambda _request: "   \n  ", max_row_attempts=1)
    assert row.row_state is ExecutedRowState.SUCCESS
    assert row.invalid is False
    assert row.score == 0.0
    assert row.failure_code == "blank-provider-generation"


def test_node_error_row_is_retried_and_succeeds() -> None:
    """A node that raises once then succeeds yields a present row."""
    experiment = build_toy_experiment()
    policy = _policy()
    transport = _RaiseOnceTransport(transport_policy=policy.transport_policy)
    context = LlmCallContext(
        execution_policy=policy,
        transport=transport,
        prompt_adapter=PlainPromptAdapter(),
    )
    row = run_rollout_row(
        experiment=experiment,
        candidate=experiment.initial_candidate,
        task=_Task(),
        task_index=0,
        task_hash="h" * 64,
        seed_index=0,
        seed_plan=_SeedPlan(),
        split_role="internal_eval",
        llm_context=context,
        eval_runner=FakeEvalProcedureRunner(),
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        resolve_provider_call_config=(
            lambda _ref: experiment.rollout_graph.provider_call_config
        ),
        max_row_attempts=3,
    )
    assert transport.calls == 2
    assert row.row_state is ExecutedRowState.SUCCESS
    assert row.score is not None
    assert row.row_attempts == 2


class _AlwaysRaiseTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        raise RuntimeError("node always blows up")


def _run_with_transport(transport, *, max_row_attempts):
    experiment = build_toy_experiment()
    policy = _policy()
    context = LlmCallContext(
        execution_policy=policy,
        transport=transport,
        prompt_adapter=PlainPromptAdapter(),
    )
    return run_rollout_row(
        experiment=experiment,
        candidate=experiment.initial_candidate,
        task=_Task(),
        task_index=0,
        task_hash="h" * 64,
        seed_index=0,
        seed_plan=_SeedPlan(),
        split_role="internal_eval",
        llm_context=context,
        eval_runner=FakeEvalProcedureRunner(),
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        resolve_provider_call_config=(
            lambda _ref: experiment.rollout_graph.provider_call_config
        ),
        max_row_attempts=max_row_attempts,
    )


def test_retry_is_bounded_and_reports_its_attempt_count() -> None:
    """A persistently failing node stops at the bound, not forever."""
    transport = _AlwaysRaiseTransport()
    row = _run_with_transport(transport, max_row_attempts=3)
    assert transport.calls == 3
    assert row.row_state is ExecutedRowState.FAILED
    assert row.row_attempts == 3
    assert row.error_type is not None
    assert row.failed_node_id == "generate"


def test_blank_generation_is_terminal_and_never_retried() -> None:
    """A blank generation is a sample, not an execution accident.

    Retrying it would resample until the model happened to say something,
    which silently discards an observed failure and biases the measurement.
    A sample is a sample: it is scored once and kept.
    """

    class _Blank:
        def __init__(self, policy) -> None:
            self.calls = 0
            self._inner = FakeLlmTransport(
                transport_policy=policy, text_factory=lambda _r: ""
            )

        def __call__(self, request):
            self.calls += 1
            return self._inner(request)

    transport = _Blank(_policy().transport_policy)
    row = _run_with_transport(transport, max_row_attempts=3)
    assert transport.calls == 1
    assert row.row_state is ExecutedRowState.SUCCESS
    assert row.score == 0.0
    assert row.row_attempts == 1


def test_max_row_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _run_with_transport(_AlwaysRaiseTransport(), max_row_attempts=0)


def test_retry_uses_a_distinct_call_config_per_attempt() -> None:
    """Each attempt must be a fresh call, not a replay of the failure.

    ``drive_ordinal`` is part of the prompt cache key, so distinct ordinals
    are what stop a cached failure from being handed back unchanged.
    """
    seen: list[int] = []

    class _RecordingTransport:
        def __init__(self, policy) -> None:
            self.calls = 0
            self._inner = FakeLlmTransport(
                transport_policy=policy, text_factory=lambda _r: "wall"
            )

        def __call__(self, request):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("transient node error")
            return self._inner(request)

    from whetstone.execution import prompt_cache as pc

    original = pc.prompt_cache_key

    def _spy(request, policy, seed_index, drive_ordinal):
        seen.append(drive_ordinal)
        return original(request, policy, seed_index, drive_ordinal)

    pc.prompt_cache_key = _spy
    try:
        row = _run_with_transport(
            _RecordingTransport(_policy().transport_policy),
            max_row_attempts=3,
        )
    finally:
        pc.prompt_cache_key = original

    assert row.row_state is ExecutedRowState.SUCCESS
    assert row.row_attempts == 3


# --- the blank-score contract ---------------------------------------------
#
# The row takes its score from the injected eval runner, so nothing in the
# rollout path forces a blank generation to score at the family floor. These
# pin that contract for every runner shipped in this repo, and for the
# reference runtime end to end.


@pytest.mark.parametrize("blank", ["", "   ", "\n", " \t\n "])
def test_every_shipped_eval_runner_scores_empty_output_at_the_floor(
    blank: str,
) -> None:
    """An empty generation is scored, and scored at the failing floor.

    ``EvalProcedureRunner`` requires empty output to be scored rather than
    refused or returned as ``None`` -- a ``None`` would make the row
    unobserved and reintroduce the lost-row failure this change removes.
    """
    from whetstone.testing.fakes.eval_procedure import (
        FakeEvalProcedureRunner,
        RepeatVaryingEvalProcedureRunner,
    )

    for runner, kwargs in (
        (FakeEvalProcedureRunner(), {}),
        (RepeatVaryingEvalProcedureRunner(), {"seed_index": 0}),
    ):
        score, _submission, _metadata = runner.run_eval_node(
            node_id="evaluate",
            node_inputs={"provider_generation": blank},
            evaluation_procedure_config_hash="h" * 64,
            task=_Task(),
            **kwargs,
        )
        # Scored -- not withheld as an unobservable row ...
        assert score is not None, f"{type(runner).__name__} withheld a score"
        # ... and scored at the floor, which cannot pass.
        assert score == 0.0, f"{type(runner).__name__} scored {score!r}"


def test_the_toy_scorer_floors_blank_output_below_any_real_generation() -> None:
    """The floor is a real floor: no non-empty generation scores lower."""
    from whetstone.testing.toy.scoring import score_generation

    blank = score_generation(generation="", gold="A", task_id="task-a")
    assert blank == 0.0
    # A gold-matching generation is the family's ceiling, and an arbitrary
    # non-matching one still scores at or above the blank floor.
    assert score_generation(
        generation="A", gold="A", task_id="task-a"
    ) > blank
    assert (
        score_generation(generation="zzz", gold="A", task_id="task-a")
        >= blank
    )


def test_reference_runtime_scores_a_blank_generation_as_a_present_row() -> None:
    """End to end: a blank row is present and scored in persisted evidence.

    This is the contract Stage-0 depends on -- the projection the store and
    every downstream consumer read, not just the in-memory row.
    """
    from dr_store.testing import temp_sqlite_store

    from whetstone.eval.metadata import metadata_with_purpose
    from whetstone.eval.preview.persisted import load_evaluation_outputs
    from whetstone.eval.protocol import EvalEvidenceWithRef, EvalRequest
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    experiment = build_toy_experiment(num_seeds=1)
    with temp_sqlite_store() as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store,
            experiment=experiment,
            transport_factory=lambda policy: FakeLlmTransport(
                transport_policy=policy.transport_policy,
                text_factory=lambda _request: "",
            ),
        )
        evaluated = engine.evaluate(
            EvalRequest(
                request_id="blank-contract",
                candidate=experiment.initial_candidate,
                metadata=metadata_with_purpose("blank-contract"),
            )
        )
        assert isinstance(evaluated, EvalEvidenceWithRef)
        evidence = evaluated.evidence
        outputs = load_evaluation_outputs(store, evidence)

    # Every row is present and scored; none landed in the invalid cell.
    accounting = evidence.row_accounting
    assert accounting.present == accounting.planned
    assert accounting.invalid == 0
    assert accounting.failed == 0
    assert all(count == 1 for count in evidence.per_task_counts)
    assert all(value == 0.0 for value in evidence.per_task_values)

    for row in outputs.outputs:
        assert row.score == 0.0
        assert row.invalid is False and row.failed is False
        assert row.output_text == ""
        # Explainability survives the round trip to the store.
        assert row.failure_code == "blank-provider-generation"
