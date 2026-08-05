"""Differential oracle against unmodified frozen GEPA 0.1.1."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.upstream_oracle import run_oracle
from tests.optimization.support import eval_config
from whetstone.core.identity import (
    IdentityRef,
    typed_ref_for_record,
)
from whetstone.experiment.binding import eval_config_reference
from whetstone.optimization.gepa.contracts import (
    GepaCandidateComponent,
    GepaComponentTraceProjection,
    GepaDataInstance,
    GepaEffectContext,
    GepaEffectRecorder,
    GepaEvaluationAuthorityBinding,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaEvaluationRow,
    GepaProposalAuthorityBinding,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
    GepaTrajectoryProjection,
)
from whetstone.optimization.gepa.control import (
    GepaComponentSelector,
    configure_gepa,
)
from whetstone.optimization.gepa.engine import run_gepa_engine
from whetstone.optimization.gepa.prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optimization.gepa.source import GEPA_SOURCE_MANIFEST_HASH
from whetstone.optimization.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    WhetstoneGepaAdapter,
)
from whetstone.optimization.proposal.proposer import ProposerConfig

_ORACLE = Path(__file__).with_name("upstream_oracle.py")
# Harness watchdog only; oracle assertions do not depend on elapsed time.
_ORACLE_TIMEOUT_SECONDS = 120.0
_FIXTURE = Path(__file__).parent / "fixtures" / "gepa_replay_spike_v1.json"
_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64
_G = "1" * 64
_H = "2" * 64
_I = "3" * 64


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _invoke(
    *,
    component_selector: str,
    merge: bool,
    hash_seed: int,
    effect_log: Path,
    crash_after: int,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(hash_seed)
    command = [
        sys.executable,
        str(_ORACLE),
        "--component-selector",
        component_selector,
        "--effect-log",
        str(effect_log),
        "--crash-after",
        str(crash_after),
        *(["--merge"] if merge else []),
    ]
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=_ORACLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            "frozen GEPA oracle subprocess exceeded its "
            f"{_ORACLE_TIMEOUT_SECONDS:.0f}-second watchdog "
            f"(component_selector={component_selector!r}, merge={merge!r}, "
            f"hash_seed={hash_seed}, crash_after={crash_after}); "
            f"captured stdout={error.stdout!r}; "
            f"captured stderr={error.stderr!r}"
        ) from error


def _replay_to_completion(
    *,
    root: Path,
    component_selector: str,
    merge: bool,
    hash_seed: int,
    expected_effect_count: int,
) -> dict:
    root.mkdir(parents=True)
    effect_log = root / "effects.json"
    crashes = 0
    for crash_index in range(expected_effect_count):
        before = (
            len(json.loads(effect_log.read_text()))
            if effect_log.exists()
            else 0
        )
        completed = _invoke(
            component_selector=component_selector,
            merge=merge,
            hash_seed=hash_seed,
            effect_log=effect_log,
            crash_after=before,
        )
        assert completed.returncode == 86, (
            "frozen GEPA oracle did not crash after the next expected effect "
            f"({crash_index=}, {expected_effect_count=}, {before=}); "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
        after = len(json.loads(effect_log.read_text()))
        assert after == before + 1, (
            "frozen GEPA oracle crash did not record exactly one effect "
            f"({crash_index=}, {expected_effect_count=}, {before=}, {after=})"
        )
        crashes += 1

    before = len(json.loads(effect_log.read_text()))
    assert before == expected_effect_count
    completed = _invoke(
        component_selector=component_selector,
        merge=merge,
        hash_seed=hash_seed,
        effect_log=effect_log,
        crash_after=before,
    )
    assert completed.returncode == 0, (
        "frozen GEPA oracle did not terminate immediately after replaying its "
        f"{expected_effect_count} frozen effects; "
        f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
    )
    records = json.loads(effect_log.read_text())
    assert crashes == len(records) == expected_effect_count
    payload = json.loads(completed.stdout)
    assert payload["effect_identities"] == [
        record["identity"] for record in records
    ]
    return payload


def _assert_merge_has_no_proposal(timeline: list[dict[str, Any]]) -> None:
    accepted_positions = [
        index
        for index, event in enumerate(timeline)
        if event["kind"] == "merge_accepted"
    ]
    assert accepted_positions
    for accepted_position in accepted_positions:
        accepted = timeline[accepted_position]
        iteration = accepted["iteration"]
        start_position = max(
            index
            for index, event in enumerate(timeline[:accepted_position])
            if event
            == {
                "kind": "iteration_start",
                "iteration": iteration,
            }
        )
        kinds = [
            event["kind"]
            for event in timeline[start_position : accepted_position + 1]
        ]
        assert "merge_attempted" in kinds
        assert "evaluate" in kinds
        assert "propose" not in kinds


def _native_prompt_services() -> GepaPromptServices:
    return GepaPromptServices(
        descriptor=GepaPromptFormatDescriptor(
            format_name="gepa-differential",
            components=(
                GepaComponentFormat(
                    component_name="alpha",
                    component_schema_identity_hash=_A,
                ),
                GepaComponentFormat(
                    component_name="beta",
                    component_schema_identity_hash=_B,
                ),
            ),
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )


def _native_data(
    label: str,
    *,
    position: int,
    loader_identity_hash: str,
) -> GepaDataInstance:
    return GepaDataInstance(
        upstream_position=position,
        data_id=_digest({"label": label}),
        data_ref=typed_ref_for_record(
            "test.gepa.differential_data",
            {"label": label},
        ),
        loader_identity_hash=loader_identity_hash,
    )


class _RecorderBackedOracleBroker:
    """Native boundary double with real semantic request/result persistence."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        labels: dict[str, str],
        groups: dict[str, str],
    ) -> None:
        self._recorder = GepaEffectRecorder(store)
        self._labels = labels
        self._groups = groups
        self.events: list[dict[str, Any]] = []
        self.evaluation_executions = 0
        self.proposal_executions = 0

    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult:
        self._recorder.record_request(request)
        self.events.append(
            {
                "kind": "evaluate",
                "candidate": [
                    [item.name, item.text] for item in request.candidate
                ],
                "data_ids": [
                    self._labels[item.data_id] for item in request.data
                ],
                "capture_traces": request.capture_traces,
            }
        )
        completed = self._recorder.load_evaluation_result(request)
        if completed is not None:
            return completed

        self.evaluation_executions += 1
        candidate = {item.name: item.text for item in request.candidate}
        alpha_level = int(candidate["alpha"][1:])
        beta_level = int(candidate["beta"][1:])
        rows: list[GepaEvaluationRow] = []
        for item in request.data:
            label = self._labels[item.data_id]
            group = self._groups[item.data_id]
            score = 0.5
            score += (0.2 if group == "A" else -0.2) * alpha_level
            score += (0.2 if group == "B" else -0.2) * beta_level
            output = {
                "id": label,
                "candidate": [
                    [component.name, component.text]
                    for component in request.candidate
                ],
                "score": score,
            }
            trajectory = None
            if request.capture_traces:
                component_records = {
                    name: (
                        GepaComponentTraceProjection(
                            inputs={"id": label, "group": group},
                            generated_outputs=output,
                            feedback=f"score={score}",
                            feedback_score=score,
                            source_refs=(item.data_ref,),
                        ),
                    )
                    for name in ("alpha", "beta")
                }
                trajectory = GepaTrajectoryProjection(
                    data_id=item.data_id,
                    inputs={"id": label, "group": group},
                    generated_outputs=output,
                    feedback=f"score={score}",
                    component_records=component_records,
                    module_score=score,
                    source_refs=(item.data_ref,),
                )
            rows.append(
                GepaEvaluationRow(
                    data=item,
                    output=output,
                    score=score,
                    trajectory=trajectory,
                    evidence_refs=(item.data_ref,),
                )
            )
        result = GepaEvaluationEffectResult(
            request_identity_hash=request.identity_hash(),
            rows=tuple(rows),
            logical_metric_calls=len(rows),
        )
        return self._recorder.record_evaluation_result(request, result)

    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult:
        self._recorder.record_request(request)
        self.events.append(
            {
                "kind": "propose_component",
                "candidate": [
                    [item.name, item.text] for item in request.candidate
                ],
                "components_to_update": list(request.components_to_update),
                "component_name": request.component_name,
            }
        )
        completed = self._recorder.load_proposal_result(request)
        if completed is not None:
            return completed

        self.proposal_executions += 1
        prompt = request.rendered_prompt.text
        alpha_count = prompt.count("### group\nA\n\n")
        beta_count = prompt.count("### group\nB\n\n")
        favored_count = (
            alpha_count if request.component_name == "alpha" else beta_count
        )
        direction = (
            1
            if favored_count >= alpha_count + beta_count - favored_count
            else -1
        )
        candidate = {item.name: item.text for item in request.candidate}
        current = candidate[request.component_name]
        replacement = current[0] + str(int(current[1:]) + direction)
        provider_attempt_ref = typed_ref_for_record(
            "test.gepa.differential_provider_attempt",
            {
                "request_identity_hash": request.identity_hash(),
                "raw_response": f"```\n{replacement}\n```",
            },
        )
        result = GepaProposalEffectResult(
            request_identity_hash=request.identity_hash(),
            raw_response=f"```\n{replacement}\n```",
            parsed_components=(
                GepaCandidateComponent(
                    name=request.component_name,
                    text=replacement,
                ),
            ),
            request_evidence={"rendered_prompt": prompt},
            response_evidence={"raw_response": f"```\n{replacement}\n```"},
            provider_attempt_refs=(provider_attempt_ref,),
            usage={"output_tokens": 1},
            cost=0.0,
        )
        return self._recorder.record_proposal_result(request, result)


def _normalized_direct_effects(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized = []
    for event in payload["timeline"]:
        if event["kind"] == "evaluate":
            semantic = event["semantic"]
            normalized.append(
                {
                    "kind": "evaluate",
                    "candidate": semantic["candidate"],
                    "data_ids": semantic["data_ids"],
                    "capture_traces": semantic["capture_traces"],
                }
            )
        elif event["kind"] == "propose":
            semantic = event["semantic"]
            normalized.append(
                {
                    "kind": "propose",
                    "candidate": semantic["candidate"],
                    "components_to_update": semantic["components_to_update"],
                }
            )
    return normalized


def _normalized_native_effects(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] == "evaluate":
            normalized.append(event)
            continue
        proposal = {
            "kind": "propose",
            "candidate": event["candidate"],
            "components_to_update": event["components_to_update"],
        }
        if not normalized or normalized[-1] != proposal:
            normalized.append(proposal)
    return normalized


def _normalized_native_result(
    result: Any,
    *,
    val_identities: tuple[str, ...],
) -> dict[str, Any]:
    val_positions = {
        identity: position for position, identity in enumerate(val_identities)
    }
    return {
        "candidates": [
            list(candidate.items()) for candidate in result.candidates
        ],
        "parents": result.parents,
        "val_aggregate_scores": result.val_aggregate_scores,
        "val_subscores": [
            {
                val_positions[identity]: score
                for identity, score in subscores.items()
            }
            for subscores in result.val_subscores
        ],
        "frontier": {
            val_positions[identity]: sorted(candidates)
            for identity, candidates in (
                result.per_val_instance_best_candidates.items()
            )
        },
        "discovery_eval_counts": result.discovery_eval_counts,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "best_idx": result.best_idx,
    }


@pytest.mark.parametrize(
    ("component_selector", "merge"),
    [
        ("round_robin", False),
        ("round_robin", True),
        ("all", False),
        ("all", True),
    ],
)
@pytest.mark.process_integration
def test_direct_upstream_oracle_matches_frozen_crash_fixture_across_hash_seeds(
    component_selector: str,
    merge: bool,
    tmp_path: Path,
) -> None:
    fixture = json.loads(_FIXTURE.read_text())
    scenario = f"{component_selector}-merge-{str(merge).lower()}"
    expected = fixture["scenarios"][scenario]

    seed_one = _replay_to_completion(
        root=tmp_path / "1",
        component_selector=component_selector,
        merge=merge,
        hash_seed=1,
        expected_effect_count=expected["effect_count"],
    )
    seed_777 = _replay_to_completion(
        root=tmp_path / "777",
        component_selector=component_selector,
        merge=merge,
        hash_seed=777,
        expected_effect_count=expected["effect_count"],
    )

    # Exact equality preserves every candidate component pair in upstream
    # insertion order.  Do not sort or coerce candidate mappings here.
    assert seed_one == seed_777
    effect_kinds = seed_one["effect_kinds"]
    result = seed_one["result"]
    merge_parents = [
        parents for parents in result["parents"] if len(parents) > 1
    ]

    assert len(effect_kinds) == expected["effect_count"]
    assert effect_kinds.count("evaluate") == expected["evaluation_count"]
    assert effect_kinds.count("propose") == expected["proposal_count"]
    assert merge_parents == expected["merge_parents"]
    assert (
        _digest(seed_one["effect_identities"])
        == expected["effect_identities_sha256"]
    )
    assert _digest(result) == expected["result_sha256"]

    if merge:
        _assert_merge_has_no_proposal(seed_one["timeline"])
        assert merge_parents
        assert all(
            list(dict(candidate).keys()) == ["alpha", "beta"]
            for candidate in result["candidates"]
        )
    else:
        assert not any(
            event["kind"].startswith("merge_")
            for event in seed_one["timeline"]
        )


@pytest.mark.parametrize(
    ("component_selector", "merge"),
    [
        ("round_robin", False),
        ("round_robin", True),
        ("all", False),
        ("all", True),
    ],
)
def test_native_adapter_matches_independent_upstream_oracle_and_replays(
    component_selector: GepaComponentSelector,
    merge: bool,
    tmp_path: Path,
) -> None:
    direct = run_oracle(
        component_selector=component_selector,
        use_merge=merge,
    )
    services = _native_prompt_services()
    reflection_model = ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                {"provider_call_config_ref": "provider://gepa-differential"},
            ),
            identity_hash=_D,
        ),
    )
    train_labels = (
        "train-A-0",
        "train-B-0",
        "train-A-1",
        "train-B-1",
    )
    val_labels = (
        "val-A-0",
        "val-B-0",
        "val-A-1",
        "val-B-1",
        "val-A-2",
        "val-B-2",
    )
    train = tuple(
        _native_data(label, position=index, loader_identity_hash=_E)
        for index, label in enumerate(train_labels)
    )
    val = tuple(
        _native_data(label, position=index, loader_identity_hash=_F)
        for index, label in enumerate(val_labels)
    )
    control = configure_gepa(
        reflection_model=reflection_model,
        metric=eval_config_reference(eval_config()),
        reward_policy_hash=_A,
        evaluation_execution_policy_hash=_B,
        proposal_execution_policy_hash=_G,
        proposal_prompt_adapter_identity_hash=_H,
        proposal_durability_policy_identity_hash=_I,
        task_model_identity_hash=_C,
        prompt_format_identity_hash=services.descriptor.identity_hash(),
        prompt_binding_identity_hash=services.binding.identity_hash(),
        trainset_task_identities=tuple(item.data_id for item in train),
        valset_task_identities=tuple(item.data_id for item in val),
        component_names=("alpha", "beta"),
        num_predictors=2,
        max_metric_calls=100,
        reflection_minibatch_size=2,
        candidate_selection_strategy="pareto",
        skip_perfect_score=False,
        component_selector=component_selector,
        use_merge=merge,
        max_merge_invocations=3,
        merge_val_overlap_floor=1,
        seed=0,
    )
    context = GepaEffectContext(
        run_id=f"gepa-differential:{component_selector}:{merge}",
        control_identity_hash=control.identity_hash(),
        source_manifest_identity_hash=GEPA_SOURCE_MANIFEST_HASH,
        adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    )
    evaluation_authority = GepaEvaluationAuthorityBinding(
        authority_identity_hash=_A,
        evaluation_config_identity_hash=control.metric.identity_hash,
        reward_policy_identity_hash=control.reward_policy_hash,
        provider_route_identity_hash=control.task_model_identity_hash,
        execution_policy_identity_hash=(
            control.evaluation_execution_policy_hash
        ),
        prompt_adapter_identity_hash=control.prompt_format_identity_hash,
        response_parser_identity_hash=services.reflection_parser.identity_hash,
        data_registry_identity_hash=_E,
        failure_score=control.failure_score,
        add_format_failure_as_feedback=(
            control.add_format_failure_as_feedback
        ),
        warn_on_score_mismatch=control.warn_on_score_mismatch,
        selection_seed=control.seed,
    )
    proposal_authority = GepaProposalAuthorityBinding(
        authority_identity_hash=_B,
        proposer_transport_identity_hash=reflection_model.identity_hash(),
        prompt_binding_identity_hash=services.binding.identity_hash(),
        execution_policy_identity_hash=(
            control.proposal_execution_policy_hash
        ),
        prompt_adapter_identity_hash=(
            control.proposal_prompt_adapter_identity_hash
        ),
        durability_policy_identity_hash=(
            control.proposal_durability_policy_identity_hash
        ),
        proposer_config=reflection_model,
    )
    labels = {
        item.data_id: label
        for item, label in zip(
            (*train, *val),
            (*train_labels, *val_labels),
            strict=True,
        )
    }
    groups = {
        identity: ("A" if "-A-" in label else "B")
        for identity, label in labels.items()
    }
    store = ObjectStore(SqliteBackend(tmp_path / "native-differential.sqlite"))

    def execute() -> tuple[Any, _RecorderBackedOracleBroker]:
        broker = _RecorderBackedOracleBroker(
            store,
            labels=labels,
            groups=groups,
        )
        adapter = WhetstoneGepaAdapter(
            context=context,
            broker=broker,
            evaluation_authority=evaluation_authority,
            proposal_authority=proposal_authority,
            prompt_services=services,
        )
        result = run_gepa_engine(
            control=control,
            seed_candidate={"alpha": "A0", "beta": "B0"},
            trainset=train,
            valset=val,
            adapter=adapter,
        )
        return result, broker

    native, first_broker = execute()
    replay, replay_broker = execute()

    assert _canonical(
        _normalized_native_result(
            native,
            val_identities=control.valset_task_identities,
        )
    ) == _canonical(direct["result"])
    assert _canonical(
        _normalized_native_effects(first_broker.events)
    ) == _canonical(_normalized_direct_effects(direct))
    assert replay == native
    assert replay_broker.events == first_broker.events
    assert replay_broker.evaluation_executions == 0
    assert replay_broker.proposal_executions == 0
