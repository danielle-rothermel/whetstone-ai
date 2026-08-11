from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from itertools import product
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    ImmutableJsonObject,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.envs.ed1 import (
    ENCODER_FRAME,
    ENCODER_FRAME_NO_BUDGET,
    ed1_initial_candidate,
    render_encoder_frame,
    validate_ed1_body,
)
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.experiment.graph.character_budget import (
    CharacterBudgetRule,
    derive_character_bound,
)
from whetstone.optimization.codex.proposer import (
    CodexCliProposerConfig,
    CodexCliProposerTransport,
)
from whetstone.optimization.copro.adapter import (
    CoproConfig,
    CoproDriver,
    CoproRoundPlan,
    CoproState,
)
from whetstone.optimization.copro.ed1_contract import (
    Ed1CoproProposalContract,
    ed1_copro_proposal_contract,
)
from whetstone.optimization.proposal.mutation import (
    MUTATION_FIELD,
    diff_check,
)
from whetstone.optimization.proposal.prompts import (
    COPRO_INSTRUCTION_CONTRACT_KEY,
    COPRO_INSTRUCTION_HISTORY_KEY,
    copro_proposal_prompt,
)
from whetstone.optimization.proposal.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerRouteConfig,
    ProposerTransport,
)

DUMMY_COPRO_PROPOSER_CONFIG_SCHEMA = "whetstone.dummy_copro_proposer_config"
DUMMY_COPRO_PROPOSER_CONFIG_SCHEMA_VERSION = 1
DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA = (
    "whetstone.dummy_copro_proposer_transport"
)
DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA_VERSION = 1


def _ordered_tuple(value: Any, info: ValidationInfo) -> Any:
    if type(value) not in (list, tuple):
        raise ValueError(
            f"{info.field_name} must be an ordered tuple or JSON array"
        )
    return value


class Ed1CoproSweepPoint(BaseModel):
    """One concrete ED1 prompt-only COPRO configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sweep_ordinal: StrictInt
    budget_ratio: float | None
    copro: CoproConfig

    @model_validator(mode="after")
    def _validate(self) -> Ed1CoproSweepPoint:
        if self.sweep_ordinal < 0:
            raise ValueError("sweep_ordinal cannot be negative")
        if self.budget_ratio is not None:
            CharacterBudgetRule(ratio=self.budget_ratio)
        return self


class Ed1CoproSweepRanges(BaseModel):
    """Ordered experiment-setting axes for the ED1 COPRO sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_ratios: tuple[float | None, ...] = (None, 0.5)
    breadths: tuple[StrictInt, ...] = (3,)
    depths: tuple[StrictInt, ...] = (1,)

    @field_validator(
        "budget_ratios",
        "breadths",
        "depths",
        mode="before",
    )
    @classmethod
    def _validate_ordered_axes(cls, value: Any, info: ValidationInfo) -> Any:
        return _ordered_tuple(value, info)

    @model_validator(mode="after")
    def _validate(self) -> Ed1CoproSweepRanges:
        for name in (
            "budget_ratios",
            "breadths",
            "depths",
        ):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must contain at least one value")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        for ratio in self.budget_ratios:
            if ratio is not None:
                CharacterBudgetRule(ratio=ratio)
        for breadth, depth in product(
            self.breadths,
            self.depths,
        ):
            CoproConfig(
                breadth=breadth,
                depth=depth,
            )
        return self

    def expand(self) -> tuple[Ed1CoproSweepPoint, ...]:
        """Expand axes in declared order into concrete sweep points."""

        points: list[Ed1CoproSweepPoint] = []
        for ordinal, (ratio, breadth, depth) in enumerate(
            product(
                self.budget_ratios,
                self.breadths,
                self.depths,
            )
        ):
            points.append(
                Ed1CoproSweepPoint(
                    sweep_ordinal=ordinal,
                    budget_ratio=ratio,
                    copro=CoproConfig(
                        breadth=breadth,
                        depth=depth,
                    ),
                )
            )
        return tuple(points)


class Ed1CoproPreviewTask(BaseModel):
    """One named HumanEval prompt input used only for rendered previews."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr
    input_code: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> Ed1CoproPreviewTask:
        if not self.task_id:
            raise ValueError("preview task_id must be non-empty")
        if not self.input_code:
            raise ValueError("preview input_code must be non-empty")
        return self


class DummyCoproProposerConfig(BaseModel):
    """Identity-bearing scripted outputs for the dummy proposer transport."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bodies: tuple[StrictStr, ...]

    @field_validator("bodies", mode="before")
    @classmethod
    def _validate_ordered_bodies(cls, value: Any, info: ValidationInfo) -> Any:
        return _ordered_tuple(value, info)

    @model_validator(mode="after")
    def _validate(self) -> DummyCoproProposerConfig:
        if not self.bodies:
            raise ValueError("dummy proposer requires at least one body")
        for body in self.bodies:
            if not body:
                raise ValueError("dummy proposer bodies must be non-empty")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {"kind": "dummy", "bodies": list(self.bodies)}

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=DUMMY_COPRO_PROPOSER_CONFIG_SCHEMA,
            schema_version=DUMMY_COPRO_PROPOSER_CONFIG_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


@dataclass(frozen=True, slots=True)
class DummyCoproProposerTransport:
    """Deterministic transport implementing the production draft boundary."""

    @property
    def execution_policy_hash(self) -> str:
        return compute_identity_hash(
            schema=DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA,
            schema_version=DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA_VERSION,
            payload={"effect": "none", "retry": "none"},
        )

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return compute_identity_hash(
            schema=DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA,
            schema_version=DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA_VERSION,
            payload={"prompt_projection": "recorded_proposal_prompt"},
        )

    @property
    def durability_identity_hash(self) -> str:
        return compute_identity_hash(
            schema=DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA,
            schema_version=DUMMY_COPRO_PROPOSER_TRANSPORT_SCHEMA_VERSION,
            payload={
                "execution_policy_hash": self.execution_policy_hash,
                "prompt_adapter_identity_hash": (
                    self.prompt_adapter_identity_hash
                ),
            },
        )

    def draft(
        self,
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        if not isinstance(config, DummyCoproProposerConfig):
            raise TypeError(
                "dummy COPRO transport requires DummyCoproProposerConfig"
            )
        if type(count) is not int or count < 1:
            raise ValueError("dummy proposal count must be positive")
        prompt = request.context.get("proposal_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "dummy COPRO proposer requires one nonblank proposal_prompt"
            )

        drafts: list[ProposalDraft] = []
        for index in range(count):
            request_evidence = {
                "proposal_request_identity_hash": request.identity_hash(),
                "proposer_config_identity_hash": config.identity_hash(),
                "proposer": "dummy",
                "draft_index": index,
            }
            if index >= len(config.bodies):
                drafts.append(
                    ProposalDraft.failure(
                        detail=(
                            "dummy proposer has fewer bodies than the "
                            "requested proposal count"
                        ),
                        request_evidence=request_evidence,
                        response_evidence={"scripted": True},
                        usage={"proposer_calls": 0},
                        cost=0.0,
                    )
                )
                continue
            drafts.append(
                ProposalDraft(
                    template=config.bodies[index],
                    request_evidence=request_evidence,
                    response_evidence={
                        "scripted": True,
                        "draft_index": index,
                    },
                    usage={"proposer_calls": 0},
                    cost=0.0,
                )
            )
        return tuple(drafts)


class Ed1PromptFill(BaseModel):
    """The exact values inserted into one ED1 encoder frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_code: StrictStr
    body: StrictStr
    max_budget: StrictInt | None


class Ed1PromptPreview(BaseModel):
    """Literal body, frame, fill, and exact model-visible prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr
    body_literal: StrictStr
    frame_template: StrictStr
    fill: Ed1PromptFill
    rendered_prompt: StrictStr


class Ed1CoproCandidateMutation(BaseModel):
    """One body-only proposal mutation plus its rendered prompt preview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_ordinal: StrictInt
    candidate: CandidateRef
    previous_body: StrictStr
    proposed_body: StrictStr
    prompt: Ed1PromptPreview

    @model_validator(mode="after")
    def _validate(self) -> Ed1CoproCandidateMutation:
        if self.proposal_ordinal < 0:
            raise ValueError("proposal_ordinal cannot be negative")
        return self


class Ed1CoproProposalCall(BaseModel):
    """The exact config, request, and drafts at one proposer boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposer_kind: StrictStr
    proposer_config: ImmutableJsonObject
    proposer_config_identity_hash: StrictStr
    transport_execution_policy_hash: StrictStr
    transport_prompt_adapter_identity_hash: StrictStr
    transport_durability_identity_hash: StrictStr
    request: ProposalRequest
    requested_count: StrictInt
    drafts: tuple[ProposalDraft, ...]

    @model_validator(mode="after")
    def _validate(self) -> Ed1CoproProposalCall:
        if not self.proposer_kind:
            raise ValueError("proposer_kind must be non-empty")
        for field_name in (
            "proposer_config_identity_hash",
            "transport_execution_policy_hash",
            "transport_prompt_adapter_identity_hash",
            "transport_durability_identity_hash",
        ):
            require_full_hash(getattr(self, field_name), field=field_name)
        if self.requested_count < 1:
            raise ValueError("requested_count must be positive")
        if len(self.drafts) != self.requested_count:
            raise ValueError(
                "proposal call must record exactly the requested draft count"
            )
        _ = self.instruction_contract
        history = self.request.context.get(COPRO_INSTRUCTION_HISTORY_KEY)
        if type(history) is not tuple:
            raise ValueError(
                "proposal call requires ordered instruction history"
            )
        prompt = self.request.context.get("proposal_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("proposal call requires a nonblank prompt")
        return self

    @property
    def instruction_contract(self) -> Ed1CoproProposalContract:
        raw = self.request.context.get(COPRO_INSTRUCTION_CONTRACT_KEY)
        if not isinstance(raw, ImmutableJsonObject):
            raise ValueError(
                "proposal call instruction contract must be a record"
            )
        return Ed1CoproProposalContract.model_validate(raw.to_json())


@verify(UNIQUE)
class Ed1CoproProposalRejectionKind(StrEnum):
    """Why one requested proposal slot could not become a candidate."""

    PROVIDER_FAILED = "provider_failed"
    REJECTED = "rejected"


class Ed1CoproProposalRejection(BaseModel):
    """One failed proposal slot with its returned body and exact reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_ordinal: StrictInt
    proposed_body: StrictStr
    kind: Ed1CoproProposalRejectionKind
    reason: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> Ed1CoproProposalRejection:
        if self.proposal_ordinal < 0:
            raise ValueError("proposal_ordinal cannot be negative")
        if not self.reason:
            raise ValueError("proposal rejection reason must be non-empty")
        return self


class Ed1CoproRoundAttempt(BaseModel):
    """A complete proposal call with every accepted or rejected slot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    starting_state: CoproState
    round_plan: CoproRoundPlan
    proposal_call: Ed1CoproProposalCall
    candidate_mutations: tuple[Ed1CoproCandidateMutation, ...]
    rejections: tuple[Ed1CoproProposalRejection, ...]
    terminal_failure: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> Ed1CoproRoundAttempt:
        accepted = {
            mutation.proposal_ordinal for mutation in self.candidate_mutations
        }
        rejected = {
            rejection.proposal_ordinal for rejection in self.rejections
        }
        if len(accepted) != len(self.candidate_mutations):
            raise ValueError("accepted proposal ordinals must be unique")
        if len(rejected) != len(self.rejections):
            raise ValueError("rejected proposal ordinals must be unique")
        if accepted & rejected:
            raise ValueError("a proposal slot cannot be accepted and rejected")
        expected = set(range(self.proposal_call.requested_count))
        if accepted | rejected != expected:
            raise ValueError("every requested proposal slot must be accounted")
        if bool(self.rejections) != (self.terminal_failure is not None):
            raise ValueError(
                "proposal rejections and terminal failure must occur together"
            )
        return self

    @property
    def succeeded(self) -> bool:
        return self.terminal_failure is None

    def require_preview(self) -> Ed1CoproRoundPreview:
        """Return the successful preview or raise its intake error."""

        if self.rejections:
            first = self.rejections[0]
            if first.kind is Ed1CoproProposalRejectionKind.PROVIDER_FAILED:
                raise ValueError(
                    "COPRO proposer returned a failed draft: " + first.reason
                )
            raise ValueError(first.reason)
        return Ed1CoproRoundPreview(
            starting_state=self.starting_state,
            round_plan=self.round_plan,
            proposal_call=self.proposal_call,
            candidate_mutations=self.candidate_mutations,
        )


class Ed1CoproSweepTranscript(BaseModel):
    """The initialized lifecycle and seed mutations for one sweep point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    settings: Ed1CoproSweepPoint
    initial_state: CoproState
    round_plan: CoproRoundPlan
    baseline_candidate: CandidateRef
    baseline_prompt: Ed1PromptPreview
    proposal_call: Ed1CoproProposalCall
    candidate_mutations: tuple[Ed1CoproCandidateMutation, ...]


class Ed1CoproRoundPreview(BaseModel):
    """One proposal boundary projected from an exact COPRO state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    starting_state: CoproState
    round_plan: CoproRoundPlan
    proposal_call: Ed1CoproProposalCall
    candidate_mutations: tuple[Ed1CoproCandidateMutation, ...]


class Ed1CoproDryRunTranscript(BaseModel):
    """JSON-serializable record of a proposal-only ED1 COPRO sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sweep: Ed1CoproSweepRanges
    preview_task: Ed1CoproPreviewTask
    proposer: ImmutableJsonObject
    points: tuple[Ed1CoproSweepTranscript, ...]


def _prompt_preview(
    *,
    task: Ed1CoproPreviewTask,
    body: str,
    budget_ratio: float | None,
) -> Ed1PromptPreview:
    max_budget = (
        None
        if budget_ratio is None
        else derive_character_bound(
            CharacterBudgetRule(ratio=budget_ratio),
            task_length=len(task.input_code),
        )
    )
    frame = ENCODER_FRAME_NO_BUDGET if budget_ratio is None else ENCODER_FRAME
    fill = Ed1PromptFill(
        input_code=task.input_code,
        body=body,
        max_budget=max_budget,
    )
    return Ed1PromptPreview(
        task_id=task.task_id,
        body_literal=body,
        frame_template=frame,
        fill=fill,
        rendered_prompt=render_encoder_frame(
            body,
            input_code=task.input_code,
            max_budget=max_budget,
        ),
    )


def _candidate_from_body(
    *,
    base: Candidate,
    candidate_id: str,
    body: str,
) -> Candidate:
    validate_ed1_body(body)
    payload = base.payload.to_json()
    payload[MUTATION_FIELD] = body
    candidate = Candidate(
        candidate_id=candidate_id,
        base_ref=candidate_reference(base).record_ref,
        payload=payload,
    )
    diff_check(base=base, proposed=candidate)
    return candidate


def attempt_ed1_copro_round(
    *,
    settings: Ed1CoproSweepPoint,
    state: CoproState,
    preview_task: Ed1CoproPreviewTask,
    proposer_kind: str,
    proposer_config: ProposerRouteConfig,
    transport: ProposerTransport,
    request_ordinal: int,
) -> Ed1CoproRoundAttempt:
    """Run one proposer call and account for every returned proposal slot."""

    driver = CoproDriver(settings.copro)
    plan = driver.advance(state)
    baseline = state.initial_candidate
    baseline_body = baseline.payload[MUTATION_FIELD]
    assert isinstance(baseline_body, str)
    contract = ed1_copro_proposal_contract(budget_ratio=settings.budget_ratio)
    context: dict[str, Any] = {
        COPRO_INSTRUCTION_CONTRACT_KEY: contract.model_dump(mode="json"),
        COPRO_INSTRUCTION_HISTORY_KEY: [
            item.to_json() for item in plan.instruction_history
        ],
    }
    proposal_request = ProposalRequest(
        proposal_mode=plan.proposal_mode,
        request_ordinal=request_ordinal,
        proposal_authority_identity_hash=compute_identity_hash(
            schema="whetstone.ed1_copro_preview_authority",
            schema_version=1,
            payload=settings.model_dump(mode="json"),
        ),
        base_candidate=candidate_reference(baseline),
        context=context,
    )
    proposal_request = proposal_request.model_copy(
        update={
            "context": ImmutableJsonObject(
                {
                    **context,
                    "proposal_prompt": copro_proposal_prompt(proposal_request),
                }
            )
        }
    )
    drafts = transport.draft(
        proposer_config,
        proposal_request,
        plan.proposal_count,
    )
    proposal_call = Ed1CoproProposalCall(
        proposer_kind=proposer_kind,
        proposer_config=proposer_config.identity_payload(),
        proposer_config_identity_hash=proposer_config.identity_hash(),
        transport_execution_policy_hash=transport.execution_policy_hash,
        transport_prompt_adapter_identity_hash=(
            transport.prompt_adapter_identity_hash
        ),
        transport_durability_identity_hash=(
            transport.durability_identity_hash
        ),
        request=proposal_request,
        requested_count=plan.proposal_count,
        drafts=drafts,
    )
    mutations: list[Ed1CoproCandidateMutation] = []
    rejections: list[Ed1CoproProposalRejection] = []
    for proposal_ordinal, draft in enumerate(drafts):
        body = draft.template.strip('"').strip()
        if draft.failed:
            detail = draft.terminal_failure
            rejections.append(
                Ed1CoproProposalRejection(
                    proposal_ordinal=proposal_ordinal,
                    proposed_body=body,
                    kind=Ed1CoproProposalRejectionKind.PROVIDER_FAILED,
                    reason=(
                        detail.message
                        if detail is not None
                        else "unknown proposal failure"
                    ),
                )
            )
            continue
        try:
            contract.validate_instruction(body)
            candidate = _candidate_from_body(
                base=baseline,
                candidate_id=(
                    f"copro-preview:{settings.sweep_ordinal}:"
                    f"{state.completed_rounds}:{proposal_ordinal}"
                ),
                body=body,
            )
        except ValueError as exc:
            rejections.append(
                Ed1CoproProposalRejection(
                    proposal_ordinal=proposal_ordinal,
                    proposed_body=body,
                    kind=Ed1CoproProposalRejectionKind.REJECTED,
                    reason=str(exc),
                )
            )
            continue
        mutations.append(
            Ed1CoproCandidateMutation(
                proposal_ordinal=proposal_ordinal,
                candidate=candidate_reference(candidate),
                previous_body=baseline_body,
                proposed_body=body,
                prompt=_prompt_preview(
                    task=preview_task,
                    body=body,
                    budget_ratio=settings.budget_ratio,
                ),
            )
        )
    terminal_failure = None
    if rejections:
        terminal_failure = (
            f"{len(rejections)} of {proposal_call.requested_count} proposal "
            "slots failed validation"
        )
    return Ed1CoproRoundAttempt(
        starting_state=state,
        round_plan=plan,
        proposal_call=proposal_call,
        candidate_mutations=tuple(mutations),
        rejections=tuple(rejections),
        terminal_failure=terminal_failure,
    )


def preview_ed1_copro_round(
    *,
    settings: Ed1CoproSweepPoint,
    state: CoproState,
    preview_task: Ed1CoproPreviewTask,
    proposer_kind: str,
    proposer_config: ProposerRouteConfig,
    transport: ProposerTransport,
    request_ordinal: int,
) -> Ed1CoproRoundPreview:
    """Preview one successful proposer call from the lifecycle state."""

    return attempt_ed1_copro_round(
        settings=settings,
        state=state,
        preview_task=preview_task,
        proposer_kind=proposer_kind,
        proposer_config=proposer_config,
        transport=transport,
        request_ordinal=request_ordinal,
    ).require_preview()


def run_ed1_copro_dry_run(
    *,
    sweep: Ed1CoproSweepRanges,
    preview_task: Ed1CoproPreviewTask,
    dummy_proposer: DummyCoproProposerConfig,
    log: Callable[[str], None] | None = None,
) -> Ed1CoproDryRunTranscript:
    """Start every sweep point, preview seed proposals, and do no evaluation.

    When ``log`` is provided, it receives the exact indented JSON transcript
    once. The returned typed transcript is identical to those logged bytes.
    """

    return _run_ed1_copro_preview(
        sweep=sweep,
        preview_task=preview_task,
        proposer_kind="dummy",
        proposer_config=dummy_proposer,
        transport=DummyCoproProposerTransport(),
        log=log,
    )


def run_ed1_copro_codex_preview(
    *,
    sweep: Ed1CoproSweepRanges,
    preview_task: Ed1CoproPreviewTask,
    proposer_config: CodexCliProposerConfig,
    transport: CodexCliProposerTransport,
    log: Callable[[str], None] | None = None,
) -> Ed1CoproDryRunTranscript:
    """Use Codex CLI to propose seed mutations without evaluating them."""

    return _run_ed1_copro_preview(
        sweep=sweep,
        preview_task=preview_task,
        proposer_kind="codex_cli",
        proposer_config=proposer_config,
        transport=transport,
        log=log,
    )


def _run_ed1_copro_preview(
    *,
    sweep: Ed1CoproSweepRanges,
    preview_task: Ed1CoproPreviewTask,
    proposer_kind: str,
    proposer_config: ProposerRouteConfig,
    transport: ProposerTransport,
    log: Callable[[str], None] | None,
) -> Ed1CoproDryRunTranscript:
    baseline = ed1_initial_candidate()
    baseline_body = baseline.payload[MUTATION_FIELD]
    assert isinstance(baseline_body, str)
    points: list[Ed1CoproSweepTranscript] = []
    for settings in sweep.expand():
        driver = CoproDriver(settings.copro)
        state = driver.initial_state(baseline)
        round_preview = preview_ed1_copro_round(
            settings=settings,
            state=state,
            preview_task=preview_task,
            proposer_kind=proposer_kind,
            proposer_config=proposer_config,
            transport=transport,
            request_ordinal=settings.sweep_ordinal,
        )
        points.append(
            Ed1CoproSweepTranscript(
                settings=settings,
                initial_state=state,
                round_plan=round_preview.round_plan,
                baseline_candidate=candidate_reference(baseline),
                baseline_prompt=_prompt_preview(
                    task=preview_task,
                    body=baseline_body,
                    budget_ratio=settings.budget_ratio,
                ),
                proposal_call=round_preview.proposal_call,
                candidate_mutations=round_preview.candidate_mutations,
            )
        )
    transcript = Ed1CoproDryRunTranscript(
        sweep=sweep,
        preview_task=preview_task,
        proposer=ImmutableJsonObject(
            {
                "kind": proposer_kind,
                "config": proposer_config.identity_payload(),
                "config_identity_hash": proposer_config.identity_hash(),
            }
        ),
        points=tuple(points),
    )
    if log is not None:
        log(transcript.model_dump_json(indent=2))
    return transcript


__all__ = [
    "DUMMY_COPRO_PROPOSER_CONFIG_SCHEMA",
    "DUMMY_COPRO_PROPOSER_CONFIG_SCHEMA_VERSION",
    "DummyCoproProposerConfig",
    "DummyCoproProposerTransport",
    "Ed1CoproCandidateMutation",
    "Ed1CoproDryRunTranscript",
    "Ed1CoproPreviewTask",
    "Ed1CoproProposalCall",
    "Ed1CoproProposalRejection",
    "Ed1CoproProposalRejectionKind",
    "Ed1CoproRoundAttempt",
    "Ed1CoproRoundPreview",
    "Ed1CoproSweepPoint",
    "Ed1CoproSweepRanges",
    "Ed1CoproSweepTranscript",
    "Ed1PromptFill",
    "Ed1PromptPreview",
    "attempt_ed1_copro_round",
    "preview_ed1_copro_round",
    "run_ed1_copro_codex_preview",
    "run_ed1_copro_dry_run",
]
