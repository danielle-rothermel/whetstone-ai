from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    IdentityRef,
    compute_identity_hash,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.core.roles import EvalRole
from whetstone.experiment.binding import EvalConfigRef
from whetstone.optim.codex.proposer import CodexCliProposerConfig
from whetstone.optim.copro.proposal_contract import (
    CoproProposalContractRecord,
)
from whetstone.optim.copro.prompts import (
    COPRO_PROPOSAL_PROMPT_SCHEMA_TAG,
)
from whetstone.optim.proposal.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter

COPRO_ALGORITHM_VERSION = "dspy_copro_single_prompt/v1"
COPRO_REFERENCE_COMMIT = "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
COPRO_CONTROL_SCHEMA = "whetstone.copro_optimizer_config"
COPRO_CONTROL_SCHEMA_VERSION = 2

type CoproProposerConfig = ProposerConfig | CodexCliProposerConfig


class CoproInjectedDefaults(BaseModel):
    """Explicit bindings used when conceptual arguments are ``None``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_model: CoproProposerConfig
    proposal_contract: CoproProposalContractRecord
    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
    provider_execution_policy_ref: IdentityRef | None = None
    expected_reward_policy_hash: StrictStr
    provider_execution_policy_hash: StrictStr
    prompt_adapter: PlainPromptAdapter

    @model_validator(mode="after")
    def _validate(self) -> CoproInjectedDefaults:
        require_full_hash(
            self.expected_reward_policy_hash,
            field="expected_reward_policy_hash",
        )
        require_full_hash(
            self.provider_execution_policy_hash,
            field="provider_execution_policy_hash",
        )
        if self.eval_role is not EvalRole.INTERNAL:
            raise ValueError("COPRO requires internal evaluation")
        return self


class CoproControl(BaseModel):
    """Fully resolved, identity-bearing COPRO construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_model: CoproProposerConfig
    proposal_contract: CoproProposalContractRecord
    eval_config_ref: EvalConfigRef
    eval_role: EvalRole
    provider_execution_policy_ref: IdentityRef | None = None
    expected_reward_policy_hash: StrictStr
    breadth: StrictInt = 10
    depth: StrictInt = 3
    track_stats: StrictBool = False
    provider_execution_policy_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    algorithm_version: StrictStr = COPRO_ALGORITHM_VERSION
    proposal_prompt_schema_tag: StrictStr = COPRO_PROPOSAL_PROMPT_SCHEMA_TAG

    @model_validator(mode="after")
    def _validate(self) -> CoproControl:
        if self.breadth <= 1:
            raise ValueError("COPRO breadth must be greater than 1")
        if self.depth < 1:
            raise ValueError("COPRO depth must be positive")
        if (
            isinstance(self.prompt_model, ProposerConfig)
            and self.prompt_model.temperature is not None
        ):
            raise ValueError(
                "COPRO provider proposer must leave temperature unset"
            )
        require_full_hash(
            self.expected_reward_policy_hash,
            field="expected_reward_policy_hash",
        )
        require_full_hash(
            self.provider_execution_policy_hash,
            field="provider_execution_policy_hash",
        )
        require_full_hash(
            self.prompt_adapter_identity_hash,
            field="prompt_adapter_identity_hash",
        )
        if self.eval_role is not EvalRole.INTERNAL:
            raise ValueError("COPRO requires internal evaluation")
        if self.algorithm_version != COPRO_ALGORITHM_VERSION:
            raise ValueError("COPRO algorithm_version is fixed")
        if self.proposal_prompt_schema_tag != COPRO_PROPOSAL_PROMPT_SCHEMA_TAG:
            raise ValueError("COPRO proposal prompt schema tag is fixed")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "algorithm": "copro",
            "algorithm_version": self.algorithm_version,
            "reference_commit": COPRO_REFERENCE_COMMIT,
            "proposal_prompt_schema_tag": self.proposal_prompt_schema_tag,
            "provider_execution_policy_hash": (
                self.provider_execution_policy_hash
            ),
            "prompt_adapter_identity_hash": (
                self.prompt_adapter_identity_hash
            ),
            "prompt_model": {
                "identity_hash": self.prompt_model.identity_hash(),
                "config": self.prompt_model.identity_payload(),
            },
            "proposal_contract": self.proposal_contract.model_dump(
                mode="json"
            ),
            "eval_config_ref": self.eval_config_ref.model_dump(mode="json"),
            "eval_role": self.eval_role.value,
            "provider_execution_policy_ref": (
                None
                if self.provider_execution_policy_ref is None
                else self.provider_execution_policy_ref.model_dump(mode="json")
            ),
            "expected_reward_policy_hash": self.expected_reward_policy_hash,
            "breadth": self.breadth,
            "depth": self.depth,
            "track_stats": self.track_stats,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=COPRO_CONTROL_SCHEMA,
            schema_version=COPRO_CONTROL_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def reference(self) -> IdentityRef:
        return IdentityRef(
            record_ref=typed_ref_for_record(
                COPRO_CONTROL_SCHEMA,
                self.record_content(),
            ),
            record_hash=self.identity_hash(),
        )

    def require_identity_hash(self, persisted_hash: str) -> None:
        require_full_hash(persisted_hash, field="optimizer_config_hash")
        if persisted_hash != self.identity_hash():
            raise ValueError(
                "optimizer_config_hash conflicts with resolved COPRO control"
            )

    def step_hyperparameters(self, *, iteration: int) -> dict[str, Any]:
        if iteration < 0 or iteration >= self.depth:
            raise ValueError("COPRO iteration exceeds configured depth")
        return {
            "breadth": self.breadth,
            "depth": self.depth,
            "track_stats": self.track_stats,
            "round_index": iteration,
            "eval_config_ref": self.eval_config_ref.model_dump(mode="json"),
            "eval_role": self.eval_role.value,
            "provider_execution_policy_ref": (
                None
                if self.provider_execution_policy_ref is None
                else self.provider_execution_policy_ref.model_dump(mode="json")
            ),
            "expected_reward_policy_hash": self.expected_reward_policy_hash,
            "algorithm_version": self.algorithm_version,
            "proposal_prompt_schema_tag": self.proposal_prompt_schema_tag,
            "proposal_contract": self.proposal_contract.model_dump(
                mode="json"
            ),
            "provider_execution_policy_hash": (
                self.provider_execution_policy_hash
            ),
            "prompt_adapter_identity_hash": (
                self.prompt_adapter_identity_hash
            ),
        }


def configure_copro(
    prompt_model: CoproProposerConfig | None = None,
    metric: EvalConfigRef | None = None,
    breadth: int = 10,
    depth: int = 3,
    track_stats: bool = False,
    *,
    defaults: CoproInjectedDefaults,
) -> CoproControl:
    resolved_prompt_model = (
        defaults.prompt_model if prompt_model is None else prompt_model
    )
    resolved_eval_config = (
        defaults.eval_config_ref if metric is None else metric
    )
    return CoproControl(
        prompt_model=resolved_prompt_model,
        proposal_contract=defaults.proposal_contract,
        eval_config_ref=resolved_eval_config,
        eval_role=defaults.eval_role,
        provider_execution_policy_ref=defaults.provider_execution_policy_ref,
        expected_reward_policy_hash=defaults.expected_reward_policy_hash,
        breadth=breadth,
        depth=depth,
        track_stats=track_stats,
        provider_execution_policy_hash=(
            defaults.provider_execution_policy_hash
        ),
        prompt_adapter_identity_hash=prompt_adapter_identity_hash(
            defaults.prompt_adapter
        ),
    )


__all__ = [
    "COPRO_ALGORITHM_VERSION",
    "COPRO_CONTROL_SCHEMA",
    "COPRO_CONTROL_SCHEMA_VERSION",
    "COPRO_REFERENCE_COMMIT",
    "CoproControl",
    "CoproInjectedDefaults",
    "CoproProposerConfig",
    "configure_copro",
]
