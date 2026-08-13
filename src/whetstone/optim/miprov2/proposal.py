from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    ImmutableJsonObject,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.experiment.candidate import TemplateRenderContract
from whetstone.optim.miprov2.demo import (
    ComponentDemo,
    ComponentDemoSet,
    DemoSourceKind,
)
from whetstone.optim.miprov2.rng import (
    MIPROV2_DEMO_BRIDGE_VERSION,
    Miprov2DurableBindings,
    Miprov2RandomState,
    Miprov2RngCheckpoint,
    Miprov2RngDraw,
)

MIPROV2_PROPOSAL_SCHEMA = "whetstone.miprov2_grounded_proposal"
MIPROV2_PROPOSAL_SCHEMA_VERSION = 2
DATASET_INITIAL_SCHEMA_TAG = "miprov2-dataset-initial/v1"
DATASET_FOLLOWUP_SCHEMA_TAG = "miprov2-dataset-followup/v1"
DATASET_FINAL_SCHEMA_TAG = "miprov2-dataset-final/v1"
PROGRAM_DESCRIPTION_SCHEMA_TAG = "miprov2-program-description/v1"
COMPONENT_DESCRIPTION_SCHEMA_TAG = "miprov2-component-description/v1"
INSTRUCTION_PROPOSAL_SCHEMA_TAG = "miprov2-instruction-proposal/v1"

NO_TASK_DEMOS = "No task demos provided."
PROGRAM_DESCRIPTION_UNAVAILABLE = "Not available"
COMPONENT_DESCRIPTION_UNAVAILABLE = "Not provided"

TIP_TEXTS: tuple[tuple[str, str], ...] = (
    ("none", ""),
    (
        "creative",
        "Don't be afraid to be creative when creating the new instruction!",
    ),
    ("simple", "Keep the instruction clear and concise."),
    (
        "description",
        "Make sure your instruction is very informative and descriptive.",
    ),
    (
        "high_stakes",
        "The instruction should include a high stakes scenario in which the "
        "LM must solve the task!",
    ),
    (
        "persona",
        "Include a persona that is relevant to the task in the instruction "
        '(ie. "You are a ...")',
    ),
)

Miprov2ProposalStage = Literal[
    "dataset_initial",
    "dataset_followup",
    "dataset_final",
    "proposal_select",
    "program_description",
    "component_description",
    "instruction_proposal",
    "complete",
    "failed",
]
Miprov2ProposalEffect = Literal[
    "dataset_initial",
    "dataset_followup",
    "dataset_final",
    "program_description",
    "component_description",
    "instruction_proposal",
]


class Miprov2PromptField(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    value: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> Miprov2PromptField:
        if not self.name:
            raise ValueError("proposal prompt field name must not be empty")
        return self


class Miprov2PromptComponent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    template: StrictStr
    template_render_contract: TemplateRenderContract
    rendering_rules: StrictStr
    example_execution: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> Miprov2PromptComponent:
        if not self.component_id or not self.template:
            raise ValueError("component id and template must not be empty")
        self.template_render_contract.validate_template(self.template)
        return self


class Miprov2DatasetExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_hash: StrictStr
    rendered_record: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> Miprov2DatasetExample:
        require_full_hash(
            self.task_hash,
            field="dataset task_hash",
        )
        return self


class Miprov2DemoField(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: StrictStr
    value: StrictStr


class Miprov2ProposalDemo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fields: tuple[Miprov2DemoField, ...]
    augmented_key_present: StrictBool
    augmented: StrictBool | None = None

    @model_validator(mode="after")
    def _validate_augmented(self) -> Miprov2ProposalDemo:
        if self.augmented_key_present != (self.augmented is not None):
            raise ValueError(
                "augmented key presence must match augmented value presence"
            )
        return self

    def render(self) -> str:
        return "\n".join(
            f"{field.name}: {field.value}" for field in self.fields
        )


class Miprov2DemoSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    examples: tuple[Miprov2ProposalDemo, ...] = ()


class Miprov2ComponentDemoCandidates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    demo_sets: tuple[Miprov2DemoSet, ...]
    bridge_version: Literal["whetstone_component_demo_bridge/v1"] = (
        MIPROV2_DEMO_BRIDGE_VERSION
    )


def proposal_candidates_from_demo_sets(
    demo_sets: tuple[ComponentDemoSet, ...],
    *,
    components: tuple[Miprov2PromptComponent, ...],
    component_field_order: dict[str, tuple[str, ...]],
) -> tuple[Miprov2ComponentDemoCandidates, ...]:

    component_ids = tuple(component.component_id for component in components)
    if set(component_field_order) != set(component_ids):
        raise ValueError(
            "proposal demo field-order mapping must match prompt components"
        )
    for component_id, order in component_field_order.items():
        if not order or any(not field for field in order):
            raise ValueError(
                f"proposal demo field order for {component_id!r} is empty"
            )
        if len(order) != len(set(order)):
            raise ValueError(
                "proposal demo field order for "
                f"{component_id!r} has duplicates"
            )
    for candidate in demo_sets:
        if (
            tuple(sequence.component_id for sequence in candidate.components)
            != component_ids
        ):
            raise ValueError(
                "bootstrap demo components do not match proposal "
                "component order"
            )

    return tuple(
        Miprov2ComponentDemoCandidates(
            component_id=component_id,
            demo_sets=tuple(
                Miprov2DemoSet(
                    examples=tuple(
                        _proposal_demo_from_component_demo(
                            demo,
                            field_order=component_field_order[component_id],
                        )
                        for demo in candidate.demos_for(component_id)
                    )
                )
                for candidate in demo_sets
            ),
        )
        for component_id in component_ids
    )


def _proposal_demo_from_component_demo(
    demo: ComponentDemo,
    *,
    field_order: tuple[str, ...],
) -> Miprov2ProposalDemo:
    values = {**demo.inputs.to_json(), **demo.outputs.to_json()}
    fields = tuple(
        Miprov2DemoField(
            name=name,
            value=_render_demo_value(values.get(name)),
        )
        for name in field_order
    )
    bootstrapped = demo.source_kind is DemoSourceKind.BOOTSTRAPPED
    return Miprov2ProposalDemo(
        fields=fields,
        augmented_key_present=bootstrapped,
        augmented=True if bootstrapped else None,
    )


def _render_demo_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class Miprov2ProposalRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bindings: Miprov2DurableBindings
    optimization_run_identity_hash: StrictStr
    effect_ordinal: StrictInt
    effect: Miprov2ProposalEffect
    schema_tag: StrictStr
    temperature: float
    generation_id: StrictInt | None = None
    component_index: StrictInt | None = None
    component_id: StrictStr | None = None
    proposal_index: StrictInt | None = None
    demo_set_index: StrictInt | None = None
    selected_tip_key: StrictStr | None = None
    fields: tuple[Miprov2PromptField, ...]
    prompt: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> Miprov2ProposalRequest:
        require_full_hash(
            self.optimization_run_identity_hash,
            field="optimization_run_identity_hash",
        )
        if self.effect_ordinal < 0:
            raise ValueError("proposal effect ordinal cannot be negative")
        if not math.isfinite(self.temperature):
            raise ValueError("proposal temperature must be finite")
        expected_schema = {
            "dataset_initial": DATASET_INITIAL_SCHEMA_TAG,
            "dataset_followup": DATASET_FOLLOWUP_SCHEMA_TAG,
            "dataset_final": DATASET_FINAL_SCHEMA_TAG,
            "program_description": PROGRAM_DESCRIPTION_SCHEMA_TAG,
            "component_description": COMPONENT_DESCRIPTION_SCHEMA_TAG,
            "instruction_proposal": INSTRUCTION_PROPOSAL_SCHEMA_TAG,
        }[self.effect]
        if self.schema_tag != expected_schema:
            raise ValueError(
                "proposal request schema tag does not match its effect"
            )
        per_proposal = self.effect in {
            "program_description",
            "component_description",
            "instruction_proposal",
        }
        indices = (
            self.component_index,
            self.component_id,
            self.proposal_index,
            self.demo_set_index,
        )
        if per_proposal and any(value is None for value in indices):
            raise ValueError(
                "per-proposal requests require component and proposal indices"
            )
        if per_proposal and self.generation_id is None:
            raise ValueError("per-proposal requests require a generation id")
        if (
            self.generation_id is not None
            and not 0 <= self.generation_id <= 10**9
        ):
            raise ValueError("proposal generation id is outside DSPy's range")
        if self.component_index is not None and self.component_index < 0:
            raise ValueError("proposal component index cannot be negative")
        if self.proposal_index is not None and self.proposal_index < 0:
            raise ValueError("proposal index cannot be negative")
        if self.demo_set_index is not None and self.demo_set_index < 0:
            raise ValueError("proposal demo-set index cannot be negative")
        tip_keys = {key for key, _ in TIP_TEXTS}
        if (
            self.selected_tip_key is not None
            and self.selected_tip_key not in tip_keys
        ):
            raise ValueError("proposal selected_tip_key is unknown")
        if not per_proposal and (
            any(value is not None for value in indices)
            or self.generation_id is not None
            or self.selected_tip_key is not None
        ):
            raise ValueError(
                "dataset-summary requests cannot carry proposal coordinates"
            )
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise ValueError("proposal prompt field names must be unique")
        if self.prompt != render_miprov2_prompt(
            effect=self.effect,
            fields=self.fields,
        ):
            raise ValueError("proposal prompt does not match semantic fields")
        return self

    def identity_payload(self) -> dict[str, Any]:

        return {
            "bindings": self.bindings.model_dump(mode="json"),
            "optimization_run_identity_hash": (
                self.optimization_run_identity_hash
            ),
            "effect_ordinal": self.effect_ordinal,
            "effect": self.effect,
            "schema_tag": self.schema_tag,
            "temperature": self.temperature,
            "generation_id": self.generation_id,
            "component_index": self.component_index,
            "component_id": self.component_id,
            "proposal_index": self.proposal_index,
            "demo_set_index": self.demo_set_index,
            "selected_tip_key": self.selected_tip_key,
            "fields": [field.model_dump(mode="json") for field in self.fields],
            "prompt": self.prompt,
        }

    @property
    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_PROPOSAL_SCHEMA,
            schema_version=MIPROV2_PROPOSAL_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class Miprov2ProposalResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_hash: StrictStr
    text: StrictStr = ""
    failed: StrictBool = False
    failure_detail: StrictStr | None = None
    evidence: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    def model_post_init(self, _context: Any) -> None:
        if not isinstance(self.evidence, ImmutableJsonObject):
            object.__setattr__(
                self,
                "evidence",
                ImmutableJsonObject(self.evidence),
            )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        payload = self.model_dump(mode="json")
        payload.update(update or {})
        return type(self).model_validate(payload)

    @model_validator(mode="after")
    def _validate(self) -> Miprov2ProposalResponse:
        require_full_hash(
            self.request_hash,
            field="proposal request_hash",
        )
        if self.failed and not self.failure_detail:
            raise ValueError("failed proposal response requires detail")
        if not self.failed and self.failure_detail is not None:
            raise ValueError(
                "successful proposal response cannot carry failure detail"
            )
        return self


class Miprov2ProposalEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: Miprov2ProposalRequest
    response: Miprov2ProposalResponse
    parsed_text: StrictStr | None = None
    accepted: StrictBool
    rejection_reason: StrictStr | None = None
    displaced_by_original: StrictBool = False

    @model_validator(mode="after")
    def _validate_evidence(self) -> Miprov2ProposalEvidence:
        _validate_proposal_evidence(self)
        return self


def _validate_proposal_evidence(item: Miprov2ProposalEvidence) -> None:
    Miprov2ProposalResponse.model_validate(
        item.response.model_dump(mode="json")
    )
    if item.response.request_hash != item.request.identity_hash:
        raise ValueError(
            "proposal evidence response belongs to another request"
        )
    if item.accepted and (
        item.response.failed or item.rejection_reason is not None
    ):
        raise ValueError(
            "accepted proposal evidence cannot be failed or rejected"
        )
    if not item.accepted and item.rejection_reason is None:
        raise ValueError("rejected proposal evidence requires a reason")
    if (
        item.displaced_by_original
        and item.request.effect != "instruction_proposal"
    ):
        raise ValueError("only instruction evidence can be displaced")


class Miprov2InstructionSlot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    proposal_index: StrictInt
    generated_instruction: StrictStr | None
    pool_instruction: StrictStr | None
    displaced_by_original: StrictBool = False
    rejection_reason: StrictStr | None = None


class Miprov2InstructionGenerationFailed(RuntimeError):
    def __init__(
        self,
        detail: str,
        *,
        state: Miprov2ProposalState | None = None,
    ) -> None:
        self.state = state
        super().__init__(detail)


class Miprov2ProposalState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    components: tuple[Miprov2PromptComponent, ...]
    trainset: tuple[Miprov2DatasetExample, ...]
    demo_candidates: tuple[Miprov2ComponentDemoCandidates, ...] | None
    num_candidates: StrictInt
    view_data_batch_size: StrictInt
    init_temperature: float
    initial_data_aware: StrictBool
    data_aware: StrictBool
    program_aware: StrictBool
    tip_aware: StrictBool
    fewshot_aware: StrictBool
    bindings: Miprov2DurableBindings
    optimization_run_identity_hash: StrictStr
    initial_rng_checkpoint: Miprov2RngCheckpoint
    rng_checkpoint: Miprov2RngCheckpoint

    stage: Miprov2ProposalStage
    pending_request: Miprov2ProposalRequest | None = None
    effect_count: StrictInt = 0

    dataset_descriptor_calls: StrictInt = 0
    next_dataset_start: StrictInt = 0
    dataset_complete_skips: StrictInt = 0
    dataset_observations: StrictStr = ""
    dataset_summary: StrictStr | None = None

    component_index: StrictInt = 0
    proposal_index: StrictInt = 0
    selected_tip_key: StrictStr | None = None
    selected_tip: StrictStr | None = None
    generation_id: StrictInt | None = None
    task_demos: StrictStr = NO_TASK_DEMOS
    program_description: StrictStr = PROGRAM_DESCRIPTION_UNAVAILABLE
    component_description: StrictStr = COMPONENT_DESCRIPTION_UNAVAILABLE

    evidence: tuple[Miprov2ProposalEvidence, ...] = ()
    instruction_slots: tuple[Miprov2InstructionSlot, ...] = ()
    instruction_pools: tuple[tuple[StrictStr, ...], ...] = ()
    terminal_failure_response: Miprov2ProposalResponse | None = None

    @model_validator(mode="after")
    def _validate(self) -> Miprov2ProposalState:
        require_full_hash(
            self.optimization_run_identity_hash,
            field="optimization_run_identity_hash",
        )
        if not self.components:
            raise ValueError(
                "MIPROv2 proposer requires at least one component"
            )
        if not self.trainset:
            raise ValueError("MIPROv2 proposer requires a nonempty trainset")
        if self.num_candidates < 1 or self.view_data_batch_size < 1:
            raise ValueError(
                "proposal and dataset batch counts must be positive"
            )
        if not math.isfinite(self.init_temperature):
            raise ValueError("proposal temperature must be finite")
        component_ids = tuple(item.component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("proposal component ids must be unique")
        if self.demo_candidates is not None:
            demo_ids = tuple(
                item.component_id for item in self.demo_candidates
            )
            if demo_ids != component_ids:
                raise ValueError(
                    "demo candidate components must match component order"
                )
            count = self.proposal_count
            if any(
                len(item.demo_sets) < count for item in self.demo_candidates
            ):
                raise ValueError(
                    "every component requires the reference proposal demo "
                    "count derived from component zero"
                )
        if self.pending_request is not None:
            expected = self.effect_count
            if self.pending_request.effect_ordinal != expected:
                raise ValueError(
                    "pending request ordinal must equal folded effect count"
                )
            if self.pending_request.bindings != self.bindings:
                raise ValueError("pending request bindings do not match state")
            if (
                self.pending_request.optimization_run_identity_hash
                != self.optimization_run_identity_hash
            ):
                raise ValueError(
                    "pending request run identity does not match state"
                )
        elif self.stage in {
            "program_description",
            "component_description",
            "instruction_proposal",
        }:
            raise ValueError(
                "effect stages require their corresponding pending request"
            )
        if self.stage == "failed":
            if self.pending_request is not None:
                raise ValueError("failed proposal state cannot be pending")
            if (
                self.terminal_failure_response is None
                or not self.terminal_failure_response.failed
            ):
                raise ValueError(
                    "failed proposal state requires its terminal response"
                )
        elif self.terminal_failure_response is not None:
            raise ValueError(
                "only failed proposal state can carry a terminal response"
            )
        if self.stage == "complete":
            if self.pending_request is not None:
                raise ValueError("complete proposal state cannot be pending")
            if len(self.instruction_pools) != len(self.components):
                raise ValueError(
                    "complete proposal state requires every instruction pool"
                )
        if self.effect_count != len(self.evidence):
            raise ValueError("effect count must equal folded evidence count")
        if tuple(
            item.request.effect_ordinal for item in self.evidence
        ) != tuple(range(self.effect_count)):
            raise ValueError("proposal evidence ordinals must be contiguous")
        if any(
            item.request.bindings != self.bindings for item in self.evidence
        ):
            raise ValueError("proposal evidence bindings do not match state")
        if any(
            item.request.optimization_run_identity_hash
            != self.optimization_run_identity_hash
            for item in self.evidence
        ):
            raise ValueError(
                "proposal evidence run identity does not match state"
            )
        for item in self.evidence:
            _validate_proposal_evidence(item)
        if self.pending_request is not None:
            _require_exact_pending_request(self)
        if (
            self.stage == "proposal_select"
            and self.pending_request is not None
        ):
            raise ValueError(
                "proposal-selection state cannot have a pending request"
            )
        if self.stage == "complete" and (
            self.component_index < len(self.components)
            or self.proposal_index != 0
        ):
            raise ValueError(
                "complete state has unfinished proposal coordinates"
            )
        expected_slots = sum(
            1
            for item in self.evidence
            if item.request.effect == "instruction_proposal"
        )
        if len(self.instruction_slots) != expected_slots:
            raise ValueError(
                "instruction slots must match folded instruction effects"
            )
        slot_coordinates = tuple(
            (slot.component_id, slot.proposal_index)
            for slot in self.instruction_slots
        )
        expected_coordinates = tuple(
            (
                item.request.component_id,
                item.request.proposal_index,
            )
            for item in self.evidence
            if item.request.effect == "instruction_proposal"
        )
        if slot_coordinates != expected_coordinates:
            raise ValueError(
                "instruction slots do not match proposal evidence"
            )
        _require_canonical_proposal_state(self)
        return self

    @property
    def proposal_count(self) -> int:
        if self.demo_candidates is None:
            return self.num_candidates
        return min(
            self.num_candidates,
            max(len(self.demo_candidates[0].demo_sets), 1),
        )


class Miprov2ProposalPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: Miprov2ProposalState
    request: Miprov2ProposalRequest | None


def start_miprov2_proposal(
    *,
    bindings: Miprov2DurableBindings,
    optimization_run_identity_hash: str,
    components: tuple[Miprov2PromptComponent, ...],
    trainset: tuple[Miprov2DatasetExample, ...],
    demo_candidates: tuple[Miprov2ComponentDemoCandidates, ...] | None,
    num_candidates: int,
    view_data_batch_size: int = 10,
    init_temperature: float = 1.0,
    data_aware: bool = True,
    program_aware: bool = True,
    tip_aware: bool = True,
    fewshot_aware: bool = True,
    rng_checkpoint: Miprov2RngCheckpoint,
) -> Miprov2ProposalState:

    return Miprov2ProposalState(
        bindings=bindings,
        optimization_run_identity_hash=optimization_run_identity_hash,
        components=components,
        trainset=trainset,
        demo_candidates=demo_candidates,
        num_candidates=num_candidates,
        view_data_batch_size=view_data_batch_size,
        init_temperature=init_temperature,
        initial_data_aware=data_aware,
        data_aware=data_aware,
        program_aware=program_aware,
        tip_aware=tip_aware,
        fewshot_aware=fewshot_aware,
        initial_rng_checkpoint=rng_checkpoint,
        rng_checkpoint=rng_checkpoint,
        stage="dataset_initial" if data_aware else "proposal_select",
    )


def plan_next_proposal_request(
    state: Miprov2ProposalState,
) -> Miprov2ProposalPlan:

    _require_canonical_proposal_state(state)
    planned, request = _plan_next_proposal_request_unchecked(state)
    return Miprov2ProposalPlan(state=planned, request=request)


def _plan_next_proposal_request_unchecked(
    state: Miprov2ProposalState,
) -> tuple[Miprov2ProposalState, Miprov2ProposalRequest | None]:

    if state.pending_request is not None:
        _require_exact_pending_request(state)
        return state, state.pending_request
    if state.stage == "complete":
        return state, None
    if state.stage == "failed":
        response = state.terminal_failure_response
        assert response is not None
        raise Miprov2InstructionGenerationFailed(
            response.failure_detail or "instruction proposal failed",
            state=state,
        )
    if state.stage == "dataset_initial":
        request = _dataset_initial_request(state)
        return _pending_unchecked(state, request)
    if state.stage == "dataset_followup":
        if (
            state.next_dataset_start >= len(state.trainset)
            or state.dataset_descriptor_calls >= 10
            or state.dataset_complete_skips >= 5
        ):
            return _plan_next_proposal_request_unchecked(
                state.model_copy(update={"stage": "dataset_final"})
            )
        request = _dataset_followup_request(state)
        return _pending_unchecked(state, request)
    if state.stage == "dataset_final":
        request = _dataset_final_request(state)
        return _pending_unchecked(state, request)
    if state.stage == "proposal_select":
        if state.component_index >= len(state.components):
            return _finalize_instruction_pools(state), None
        if state.proposal_index >= state.proposal_count:
            return _plan_next_proposal_request_unchecked(
                state.model_copy(
                    update={
                        "component_index": state.component_index + 1,
                        "proposal_index": 0,
                    }
                )
            )
        selected = _select_proposal_configuration(state)
        if selected.program_aware:
            request = _program_description_request(selected)
            return _pending_unchecked(
                selected.model_copy(update={"stage": "program_description"}),
                request,
            )
        request = _instruction_request(selected)
        return _pending_unchecked(
            selected.model_copy(update={"stage": "instruction_proposal"}),
            request,
        )
    raise ValueError(
        f"cannot plan from response-waiting stage {state.stage!r}"
    )


def fold_proposal_response(
    state: Miprov2ProposalState,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:

    _require_canonical_proposal_state(state)
    response = Miprov2ProposalResponse.model_validate(
        response.model_dump(mode="json")
    )
    request = state.pending_request
    if request is None:
        raise ValueError("proposal state has no pending request")
    if response.request_hash != request.identity_hash:
        raise ValueError("proposal response belongs to another request")
    _require_exact_pending_request(state)

    return _fold_proposal_response_unchecked(state, request, response)


def _fold_proposal_response_unchecked(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:
    if request.effect == "dataset_initial":
        return _fold_dataset_initial(state, request, response)
    if request.effect == "dataset_followup":
        return _fold_dataset_followup(state, request, response)
    if request.effect == "dataset_final":
        return _fold_dataset_final(state, request, response)
    if request.effect == "program_description":
        return _fold_program_description(state, request, response)
    if request.effect == "component_description":
        return _fold_component_description(state, request, response)
    if request.effect == "instruction_proposal":
        return _fold_instruction(state, request, response)
    raise AssertionError(f"unknown proposal effect {request.effect!r}")


def render_miprov2_prompt(
    *,
    effect: Miprov2ProposalEffect,
    fields: tuple[Miprov2PromptField, ...],
) -> str:

    roles = {
        "dataset_initial": (
            "Inspect these ordered dataset examples. Describe trends that "
            "hold for most or all samples, including likely task, content, "
            "syntax, and conciseness."
        ),
        "dataset_followup": (
            "Inspect the next ordered dataset examples and add observations "
            "not already covered. If the observations are comprehensive, "
            "respond with COMPLETE."
        ),
        "dataset_final": (
            "Summarize the accumulated dataset observations in two or three "
            "sentences containing only the most important details."
        ),
        "program_description": (
            "Describe what task this Whetstone prompt-component graph is "
            "designed to solve and how its components work together."
        ),
        "component_description": (
            "Describe the selected prompt component's role in the broader "
            "Whetstone prompt-component graph."
        ),
        "instruction_proposal": (
            "Use the information below to generate one complete replacement "
            "instruction for the selected Whetstone prompt component."
        ),
    }
    lines = [roles[effect]]
    for field in fields:
        lines.extend(["", f"## {field.name}", field.value])
    if effect == "instruction_proposal":
        lines.extend(
            [
                "",
                "Return only the complete replacement instruction. Preserve "
                "every required native {placeholder} occurrence.",
            ]
        )
    return "\n".join(lines)


def strip_dspy_prefix(text: str) -> str:

    pattern = r"^[\*\s]*(([\w\'\-]+\s+){0,4}[\w\'\-]+):\s*"
    return re.sub(pattern, "", text).strip('"')


def _pending(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
) -> Miprov2ProposalPlan:
    pending, request = _pending_unchecked(state, request)
    return Miprov2ProposalPlan(state=pending, request=request)


def _pending_unchecked(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
) -> tuple[Miprov2ProposalState, Miprov2ProposalRequest]:
    return state.model_copy(update={"pending_request": request}), request


def _request(
    state: Miprov2ProposalState,
    *,
    effect: Miprov2ProposalEffect,
    schema_tag: str,
    fields: tuple[Miprov2PromptField, ...],
    temperature: float,
    per_proposal: bool = False,
) -> Miprov2ProposalRequest:
    component = state.components[state.component_index]
    return Miprov2ProposalRequest(
        bindings=state.bindings,
        optimization_run_identity_hash=state.optimization_run_identity_hash,
        effect_ordinal=state.effect_count,
        effect=effect,
        schema_tag=schema_tag,
        temperature=temperature,
        generation_id=state.generation_id if per_proposal else None,
        component_index=state.component_index if per_proposal else None,
        component_id=component.component_id if per_proposal else None,
        proposal_index=state.proposal_index if per_proposal else None,
        demo_set_index=state.proposal_index if per_proposal else None,
        selected_tip_key=state.selected_tip_key if per_proposal else None,
        fields=fields,
        prompt=render_miprov2_prompt(effect=effect, fields=fields),
    )


def _require_exact_pending_request(state: Miprov2ProposalState) -> None:

    pending = state.pending_request
    if pending is None:
        raise ValueError("proposal state has no pending request")
    builders = {
        "dataset_initial": _dataset_initial_request,
        "dataset_followup": _dataset_followup_request,
        "dataset_final": _dataset_final_request,
        "program_description": _program_description_request,
        "component_description": _component_description_request,
        "instruction_proposal": _instruction_request,
    }
    builder = builders.get(state.stage)
    if builder is None:
        raise ValueError(
            f"proposal stage {state.stage!r} cannot carry a pending request"
        )
    expected = builder(state.model_copy(update={"pending_request": None}))
    if pending != expected:
        raise ValueError(
            "pending proposal request does not match reconstructed state"
        )


def _require_canonical_proposal_state(state: Miprov2ProposalState) -> None:

    replay = Miprov2ProposalState.model_construct(
        components=state.components,
        trainset=state.trainset,
        demo_candidates=state.demo_candidates,
        num_candidates=state.num_candidates,
        view_data_batch_size=state.view_data_batch_size,
        init_temperature=state.init_temperature,
        initial_data_aware=state.initial_data_aware,
        data_aware=state.initial_data_aware,
        program_aware=state.program_aware,
        tip_aware=state.tip_aware,
        fewshot_aware=state.fewshot_aware,
        bindings=state.bindings,
        optimization_run_identity_hash=state.optimization_run_identity_hash,
        initial_rng_checkpoint=state.initial_rng_checkpoint,
        rng_checkpoint=state.initial_rng_checkpoint,
        stage=(
            "dataset_initial"
            if state.initial_data_aware
            else "proposal_select"
        ),
        pending_request=None,
        effect_count=0,
        dataset_descriptor_calls=0,
        next_dataset_start=0,
        dataset_complete_skips=0,
        dataset_observations="",
        dataset_summary=None,
        component_index=0,
        proposal_index=0,
        selected_tip_key=None,
        selected_tip=None,
        generation_id=None,
        task_demos=NO_TASK_DEMOS,
        program_description=PROGRAM_DESCRIPTION_UNAVAILABLE,
        component_description=COMPONENT_DESCRIPTION_UNAVAILABLE,
        evidence=(),
        instruction_slots=(),
        instruction_pools=(),
        terminal_failure_response=None,
    )
    for expected_item in state.evidence:
        try:
            planned, request = _plan_next_proposal_request_unchecked(replay)
        except Miprov2InstructionGenerationFailed as exc:
            raise ValueError(
                "proposal evidence continues after terminal failure"
            ) from exc
        if request is None or request != expected_item.request:
            raise ValueError(
                "proposal evidence request does not match canonical replay"
            )
        replay = _fold_proposal_response_unchecked(
            planned,
            request,
            expected_item.response,
        )
        if not replay.evidence or replay.evidence[-1] != expected_item:
            raise ValueError(
                "proposal evidence decision does not match canonical replay"
            )

    allowed = [replay]
    if replay.stage != "failed":
        planned, _request = _plan_next_proposal_request_unchecked(replay)
        if planned != replay:
            allowed.append(planned)
    actual = state.model_dump(mode="json")
    if all(
        actual != candidate.model_dump(mode="json") for candidate in allowed
    ):
        raise ValueError(
            "proposal state does not match canonical evidence replay"
        )


def _dataset_batch(
    state: Miprov2ProposalState,
    start: int,
) -> str:
    stop = min(len(state.trainset), start + state.view_data_batch_size)
    return "\n\n".join(
        f"Task {example.task_hash}:\n{example.rendered_record}"
        for example in state.trainset[start:stop]
    )


def _dataset_initial_request(
    state: Miprov2ProposalState,
) -> Miprov2ProposalRequest:
    fields = (
        Miprov2PromptField(
            name="Ordered dataset examples",
            value=_dataset_batch(state, 0),
        ),
    )
    return _request(
        state,
        effect="dataset_initial",
        schema_tag=DATASET_INITIAL_SCHEMA_TAG,
        fields=fields,
        temperature=1.0,
    )


def _dataset_followup_request(
    state: Miprov2ProposalState,
) -> Miprov2ProposalRequest:
    fields = (
        Miprov2PromptField(
            name="Prior observations",
            value=state.dataset_observations,
        ),
        Miprov2PromptField(
            name="Ordered dataset examples",
            value=_dataset_batch(state, state.next_dataset_start),
        ),
    )
    return _request(
        state,
        effect="dataset_followup",
        schema_tag=DATASET_FOLLOWUP_SCHEMA_TAG,
        fields=fields,
        temperature=1.0,
    )


def _dataset_final_request(
    state: Miprov2ProposalState,
) -> Miprov2ProposalRequest:
    fields = (
        Miprov2PromptField(
            name="Accumulated observations",
            value=state.dataset_observations,
        ),
    )
    return _request(
        state,
        effect="dataset_final",
        schema_tag=DATASET_FINAL_SCHEMA_TAG,
        fields=fields,
        temperature=1.0,
    )


def _program_graph(state: Miprov2ProposalState) -> str:
    lines: list[str] = []
    for index, component in enumerate(state.components):
        lines.extend(
            [
                f"Component {index}: {component.component_id}",
                "Current complete template:",
                component.template,
                "Allowed placeholders: "
                + (
                    ", ".join(
                        f"{{{name}}}"
                        for name in (
                            component.template_render_contract.available_fields
                        )
                    )
                    or "(none)"
                ),
                f"Rendering rules: {component.rendering_rules}",
                f"Example execution: {component.example_execution}",
            ]
        )
    return "\n".join(lines)


def _component_spec(component: Miprov2PromptComponent) -> str:
    return "\n".join(
        [
            f"Component id: {component.component_id}",
            "Current complete template:",
            component.template,
            "Allowed placeholders: "
            + (
                ", ".join(
                    f"{{{name}}}"
                    for name in (
                        component.template_render_contract.available_fields
                    )
                )
                or "(none)"
            ),
            f"Rendering rules: {component.rendering_rules}",
            f"Example execution: {component.example_execution}",
        ]
    )


def _program_description_request(
    state: Miprov2ProposalState,
) -> Miprov2ProposalRequest:
    fields = (
        Miprov2PromptField(
            name="Whetstone prompt-component graph",
            value=_program_graph(state),
        ),
        Miprov2PromptField(
            name="Example component execution",
            value=state.task_demos,
        ),
    )
    return _request(
        state,
        effect="program_description",
        schema_tag=PROGRAM_DESCRIPTION_SCHEMA_TAG,
        fields=fields,
        temperature=state.init_temperature,
        per_proposal=True,
    )


def _component_description_request(
    state: Miprov2ProposalState,
) -> Miprov2ProposalRequest:
    component = state.components[state.component_index]
    fields = (
        Miprov2PromptField(
            name="Whetstone prompt-component graph",
            value=_program_graph(state),
        ),
        Miprov2PromptField(
            name="Example component execution",
            value=state.task_demos,
        ),
        Miprov2PromptField(
            name="Program description",
            value=state.program_description,
        ),
        Miprov2PromptField(
            name="Selected component",
            value=_component_spec(component),
        ),
    )
    return _request(
        state,
        effect="component_description",
        schema_tag=COMPONENT_DESCRIPTION_SCHEMA_TAG,
        fields=fields,
        temperature=state.init_temperature,
        per_proposal=True,
    )


def _instruction_request(
    state: Miprov2ProposalState,
) -> Miprov2ProposalRequest:
    component = state.components[state.component_index]
    fields: list[Miprov2PromptField] = []
    if state.data_aware and state.dataset_summary is not None:
        fields.append(
            Miprov2PromptField(
                name="Dataset description",
                value=state.dataset_summary,
            )
        )
    if state.program_aware:
        fields.extend(
            [
                Miprov2PromptField(
                    name="Whetstone prompt-component graph",
                    value=_program_graph(state),
                ),
                Miprov2PromptField(
                    name="Program description",
                    value=state.program_description,
                ),
                Miprov2PromptField(
                    name="Selected component",
                    value=_component_spec(component),
                ),
                Miprov2PromptField(
                    name="Component description",
                    value=state.component_description,
                ),
            ]
        )
    fields.extend(
        [
            Miprov2PromptField(
                name="Task demonstrations",
                value=state.task_demos,
            ),
            Miprov2PromptField(
                name="Basic instruction",
                value=component.template,
            ),
        ]
    )
    if state.selected_tip:
        fields.append(Miprov2PromptField(name="Tip", value=state.selected_tip))
    return _request(
        state,
        effect="instruction_proposal",
        schema_tag=INSTRUCTION_PROPOSAL_SCHEMA_TAG,
        fields=tuple(fields),
        temperature=state.init_temperature,
        per_proposal=True,
    )


def _select_proposal_configuration(
    state: Miprov2ProposalState,
) -> Miprov2ProposalState:
    checkpoint = state.rng_checkpoint
    rng = checkpoint.state.restore()
    selected_tip_key: str | None = None
    selected_tip: str | None = None
    if state.tip_aware:
        keys = tuple(key for key, _ in TIP_TEXTS)
        selected_tip_key = rng.choice(keys)
        selected_tip = dict(TIP_TEXTS)[selected_tip_key]
        checkpoint = checkpoint.append(
            rng=rng,
            phase="proposal",
            operation="choice",
            arguments=keys,
            result=selected_tip_key,
        )
    generation_id = rng.randint(0, 10**9)
    checkpoint = checkpoint.append(
        rng=rng,
        phase="proposal",
        operation="randint",
        arguments=(0, 10**9),
        result=generation_id,
    )
    return state.model_copy(
        update={
            "rng_checkpoint": checkpoint,
            "selected_tip_key": selected_tip_key,
            "selected_tip": selected_tip,
            "generation_id": generation_id,
            "task_demos": _task_demos(state),
            "program_description": PROGRAM_DESCRIPTION_UNAVAILABLE,
            "component_description": COMPONENT_DESCRIPTION_UNAVAILABLE,
        }
    )


def _task_demos(state: Miprov2ProposalState) -> str:
    candidates = state.demo_candidates
    if not state.fewshot_aware or candidates is None:
        return NO_TASK_DEMOS
    sets = candidates[state.component_index].demo_sets
    if state.proposal_index == 0 or not sets:
        return NO_TASK_DEMOS
    rotated = sets[state.proposal_index :] + sets[: state.proposal_index]
    rendered: list[str] = []
    for demo_set in rotated:
        for example in demo_set.examples:
            if not example.augmented_key_present:
                continue
            rendered.append(example.render())
            if len(rendered) >= 3:
                return "\n\n".join(rendered) + "\n\n"
    if not rendered:
        return NO_TASK_DEMOS
    return "\n\n".join(rendered) + "\n\n"


def _evidence(
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
    *,
    parsed_text: str | None,
    accepted: bool,
    rejection_reason: str | None = None,
    displaced: bool = False,
) -> Miprov2ProposalEvidence:
    return Miprov2ProposalEvidence(
        request=request,
        response=response,
        parsed_text=parsed_text,
        accepted=accepted,
        rejection_reason=rejection_reason,
        displaced_by_original=displaced,
    )


def _fold_dataset_initial(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:
    if response.failed:
        item = _evidence(
            request,
            response,
            parsed_text=None,
            accepted=False,
            rejection_reason=response.failure_detail,
        )
        return state.model_copy(
            update={
                "pending_request": None,
                "effect_count": state.effect_count + 1,
                "data_aware": False,
                "stage": "proposal_select",
                "evidence": (*state.evidence, item),
            }
        )
    item = _evidence(
        request,
        response,
        parsed_text=response.text,
        accepted=True,
    )
    return state.model_copy(
        update={
            "pending_request": None,
            "effect_count": state.effect_count + 1,
            "dataset_descriptor_calls": 1,
            "next_dataset_start": state.view_data_batch_size,
            "dataset_observations": response.text,
            "stage": "dataset_followup",
            "evidence": (*state.evidence, item),
        }
    )


def _fold_dataset_followup(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:
    updates: dict[str, Any] = {
        "pending_request": None,
        "effect_count": state.effect_count + 1,
        "dataset_descriptor_calls": state.dataset_descriptor_calls + 1,
        "next_dataset_start": (
            state.next_dataset_start + state.view_data_batch_size
        ),
        "stage": "dataset_followup",
    }
    if response.failed:
        accepted = False
        rejection = response.failure_detail
        parsed = None
        updates["stage"] = "dataset_final"
    elif len(response.text) >= 8 and response.text[:8].upper() == "COMPLETE":
        accepted = False
        rejection = "COMPLETE"
        parsed = response.text
        updates["dataset_complete_skips"] = state.dataset_complete_skips + 1
        if updates["dataset_complete_skips"] >= 5:
            updates["stage"] = "dataset_final"
    else:
        accepted = True
        rejection = None
        parsed = response.text

        updates["dataset_observations"] = (
            state.dataset_observations + response.text
        )
    item = _evidence(
        request,
        response,
        parsed_text=parsed,
        accepted=accepted,
        rejection_reason=rejection,
    )
    updates["evidence"] = (*state.evidence, item)
    return state.model_copy(update=updates)


def _fold_dataset_final(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:
    if response.failed:
        accepted = False
        rejection = response.failure_detail
        parsed = None
        summary = None
        data_aware = False
    else:
        accepted = True
        rejection = None
        parsed = strip_dspy_prefix(response.text)
        summary = parsed
        data_aware = True
    item = _evidence(
        request,
        response,
        parsed_text=parsed,
        accepted=accepted,
        rejection_reason=rejection,
    )
    return state.model_copy(
        update={
            "pending_request": None,
            "effect_count": state.effect_count + 1,
            "dataset_summary": summary,
            "data_aware": data_aware,
            "stage": "proposal_select",
            "evidence": (*state.evidence, item),
        }
    )


def _fold_program_description(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:
    if response.failed:
        item = _evidence(
            request,
            response,
            parsed_text=None,
            accepted=False,
            rejection_reason=response.failure_detail,
        )
        advanced = state.model_copy(
            update={
                "pending_request": None,
                "effect_count": state.effect_count + 1,
                "program_description": PROGRAM_DESCRIPTION_UNAVAILABLE,
                "component_description": COMPONENT_DESCRIPTION_UNAVAILABLE,
                "evidence": (*state.evidence, item),
            }
        )
        instruction = _instruction_request(advanced)
        return advanced.model_copy(
            update={
                "stage": "instruction_proposal",
                "pending_request": instruction,
            }
        )
    parsed = strip_dspy_prefix(response.text)
    item = _evidence(
        request,
        response,
        parsed_text=parsed,
        accepted=True,
    )
    advanced = state.model_copy(
        update={
            "pending_request": None,
            "effect_count": state.effect_count + 1,
            "program_description": parsed,
            "evidence": (*state.evidence, item),
        }
    )
    component_request = _component_description_request(advanced)
    return advanced.model_copy(
        update={
            "stage": "component_description",
            "pending_request": component_request,
        }
    )


def _fold_component_description(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:
    if response.failed:
        parsed = None
        accepted = False
        rejection = response.failure_detail
        description = COMPONENT_DESCRIPTION_UNAVAILABLE
    else:
        parsed = response.text
        accepted = True
        rejection = None
        description = response.text
    item = _evidence(
        request,
        response,
        parsed_text=parsed,
        accepted=accepted,
        rejection_reason=rejection,
    )
    advanced = state.model_copy(
        update={
            "pending_request": None,
            "effect_count": state.effect_count + 1,
            "component_description": description,
            "evidence": (*state.evidence, item),
        }
    )
    instruction = _instruction_request(advanced)
    return advanced.model_copy(
        update={
            "stage": "instruction_proposal",
            "pending_request": instruction,
        }
    )


def _validate_instruction(
    component: Miprov2PromptComponent,
    instruction: str,
) -> str | None:
    if not instruction:
        return "empty proposed instruction"
    contract = component.template_render_contract
    try:
        proposed = Counter(contract.validate_template(instruction))
        required = Counter(contract.placeholder_fields(component.template))
    except ValueError as exc:
        return f"violates template render contract: {exc}"
    missing = tuple(
        name for name, count in required.items() if proposed[name] < count
    )
    if missing:
        return "removes required placeholders: " + ", ".join(missing)
    return None


def _fold_instruction(
    state: Miprov2ProposalState,
    request: Miprov2ProposalRequest,
    response: Miprov2ProposalResponse,
) -> Miprov2ProposalState:
    component = state.components[state.component_index]
    parsed: str | None = None
    rejection: str | None
    if response.failed:
        rejection = response.failure_detail or "instruction proposal failed"
        item = _evidence(
            request,
            response,
            parsed_text=None,
            accepted=False,
            rejection_reason=rejection,
        )
        slot = Miprov2InstructionSlot(
            component_id=component.component_id,
            proposal_index=state.proposal_index,
            generated_instruction=None,
            pool_instruction=None,
            rejection_reason=rejection,
        )
        return state.model_copy(
            update={
                "pending_request": None,
                "effect_count": state.effect_count + 1,
                "stage": "failed",
                "terminal_failure_response": response,
                "evidence": (*state.evidence, item),
                "instruction_slots": (*state.instruction_slots, slot),
            }
        )

    parsed = strip_dspy_prefix(strip_dspy_prefix(response.text))
    rejection = _validate_instruction(component, parsed)
    displaced = state.proposal_index == 0
    accepted = rejection is None and not displaced
    pool_instruction = (
        component.template
        if displaced
        else (parsed if rejection is None else None)
    )
    item = _evidence(
        request,
        response,
        parsed_text=parsed,
        accepted=accepted,
        rejection_reason=(
            rejection
            or ("displaced by original instruction" if displaced else None)
        ),
        displaced=displaced,
    )
    slot = Miprov2InstructionSlot(
        component_id=component.component_id,
        proposal_index=state.proposal_index,
        generated_instruction=parsed,
        pool_instruction=pool_instruction,
        displaced_by_original=displaced,
        rejection_reason=rejection,
    )
    return state.model_copy(
        update={
            "pending_request": None,
            "effect_count": state.effect_count + 1,
            "proposal_index": state.proposal_index + 1,
            "stage": "proposal_select",
            "selected_tip_key": None,
            "selected_tip": None,
            "generation_id": None,
            "evidence": (*state.evidence, item),
            "instruction_slots": (*state.instruction_slots, slot),
        }
    )


def _finalize_instruction_pools(
    state: Miprov2ProposalState,
) -> Miprov2ProposalState:
    pools: list[tuple[str, ...]] = []
    evidence = list(state.evidence)
    for component in state.components:
        slots = tuple(
            item
            for item in state.instruction_slots
            if item.component_id == component.component_id
        )
        pool = tuple(
            item.pool_instruction
            for item in slots
            if item.pool_instruction is not None
        )
        if not pool or pool[0] != component.template:
            raise ValueError(
                "component instruction pool must start with original template"
            )
        pools.append(pool)
    return state.model_copy(
        update={
            "stage": "complete",
            "instruction_pools": tuple(pools),
            "evidence": tuple(evidence),
        }
    )


__all__ = [
    "COMPONENT_DESCRIPTION_SCHEMA_TAG",
    "DATASET_FINAL_SCHEMA_TAG",
    "DATASET_FOLLOWUP_SCHEMA_TAG",
    "DATASET_INITIAL_SCHEMA_TAG",
    "INSTRUCTION_PROPOSAL_SCHEMA_TAG",
    "MIPROV2_DEMO_BRIDGE_VERSION",
    "MIPROV2_PROPOSAL_SCHEMA",
    "MIPROV2_PROPOSAL_SCHEMA_VERSION",
    "NO_TASK_DEMOS",
    "PROGRAM_DESCRIPTION_SCHEMA_TAG",
    "TIP_TEXTS",
    "Miprov2ComponentDemoCandidates",
    "Miprov2DatasetExample",
    "Miprov2DemoField",
    "Miprov2DemoSet",
    "Miprov2InstructionGenerationFailed",
    "Miprov2InstructionSlot",
    "Miprov2PromptComponent",
    "Miprov2PromptField",
    "Miprov2ProposalDemo",
    "Miprov2ProposalEvidence",
    "Miprov2ProposalPlan",
    "Miprov2ProposalRequest",
    "Miprov2ProposalResponse",
    "Miprov2ProposalState",
    "Miprov2RandomState",
    "Miprov2RngDraw",
    "fold_proposal_response",
    "plan_next_proposal_request",
    "proposal_candidates_from_demo_sets",
    "render_miprov2_prompt",
    "start_miprov2_proposal",
    "strip_dspy_prefix",
]
