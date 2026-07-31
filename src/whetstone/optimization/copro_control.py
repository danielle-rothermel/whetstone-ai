"""Public COPRO construction and persisted optimizer-control identity."""

from __future__ import annotations

import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.evaluation_role import EvaluationRole
from whetstone.lm.boundary import PlainPromptAdapter
from whetstone.optimization.identity import (
    IdentityRef,
    compute_identity_hash,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.optimization.proposal_prompts import (
    COPRO_PROPOSAL_PROMPT_SCHEMA_TAG,
)
from whetstone.optimization.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.optimization.schema import EvalConfigRef, EvaluationBinding

COPRO_ALGORITHM_VERSION = "dspy_copro_single_prompt/v1"
COPRO_REFERENCE_COMMIT = "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
COPRO_CONTROL_SCHEMA = "whetstone.copro_optimizer_config"
COPRO_CONTROL_SCHEMA_VERSION = 1


class CoproInjectedDefaults(BaseModel):
    """Explicit bindings used when conceptual arguments are ``None``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_model: ProposerConfig
    evaluation_binding: EvaluationBinding
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
        if self.evaluation_binding.role is not EvaluationRole.INTERNAL:
            raise ValueError("COPRO requires an internal Evaluation Binding")
        return self


class CoproControl(BaseModel):
    """Fully resolved, identity-bearing COPRO construction.

    This is the persisted optimizer Config behind a COPRO run. It binds the
    algorithm version, its own prompt schema, provider attempt policy, prompt
    projection, resolved proposer route, resolved metric, and DSPy-compatible
    hyperparameters. Generic proposer protocol versions are intentionally not
    substitutes for the algorithm-specific prompt schema.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_model: ProposerConfig
    evaluation_binding: EvaluationBinding
    expected_reward_policy_hash: StrictStr
    breadth: StrictInt = 10
    depth: StrictInt = 3
    init_temperature: float = 1.4
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
        if not math.isfinite(self.init_temperature):
            raise ValueError("COPRO init_temperature must be finite")
        if self.prompt_model.temperature != self.init_temperature:
            raise ValueError(
                "prompt_model temperature conflicts with init_temperature"
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
        if self.evaluation_binding.role is not EvaluationRole.INTERNAL:
            raise ValueError("COPRO requires an internal Evaluation Binding")
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
            "evaluation_binding": {
                "identity_hash": self.evaluation_binding.identity_hash(),
                "record": self.evaluation_binding.model_dump(mode="json"),
            },
            "expected_reward_policy_hash": self.expected_reward_policy_hash,
            "breadth": self.breadth,
            "depth": self.depth,
            "init_temperature": self.init_temperature,
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
        """Return the exact content and identity binding for this control."""

        return IdentityRef(
            record_ref=typed_ref_for_record(
                COPRO_CONTROL_SCHEMA,
                self.record_content(),
            ),
            identity_hash=self.identity_hash(),
        )

    def require_identity_hash(self, persisted_hash: str) -> None:
        """Reject a run/request bound to conflicting persisted controls."""

        require_full_hash(persisted_hash, field="optimizer_config_hash")
        if persisted_hash != self.identity_hash():
            raise ValueError(
                "optimizer_config_hash conflicts with resolved COPRO control"
            )

    def step_hyperparameters(self, *, iteration: int) -> dict[str, Any]:
        """Project resolved controls onto the durable step protocol."""

        if iteration < 0 or iteration >= self.depth:
            raise ValueError("COPRO iteration exceeds configured depth")
        return {
            "breadth": self.breadth,
            "depth": self.depth,
            "init_temperature": self.init_temperature,
            "track_stats": self.track_stats,
            "round_index": iteration,
            "evaluation_binding": self.evaluation_binding.model_dump(
                mode="json"
            ),
            "expected_reward_policy_hash": self.expected_reward_policy_hash,
            "algorithm_version": self.algorithm_version,
            "proposal_prompt_schema_tag": self.proposal_prompt_schema_tag,
            "provider_execution_policy_hash": (
                self.provider_execution_policy_hash
            ),
            "prompt_adapter_identity_hash": (
                self.prompt_adapter_identity_hash
            ),
        }


def configure_copro(
    prompt_model: ProposerConfig | None = None,
    metric: EvalConfigRef | None = None,
    breadth: int = 10,
    depth: int = 3,
    init_temperature: float = 1.4,
    track_stats: bool = False,
    *,
    defaults: CoproInjectedDefaults,
) -> CoproControl:
    """Resolve DSPy's public COPRO arguments through explicit defaults.

    ``None`` means the corresponding binding from ``defaults``. There is no
    ambient model, metric, provider policy, or prompt adapter. An explicit
    prompt model whose generation temperature disagrees with
    ``init_temperature`` is rejected instead of silently choosing one source.
    """

    resolved_prompt_model = (
        defaults.prompt_model if prompt_model is None else prompt_model
    )
    resolved_binding = (
        defaults.evaluation_binding
        if metric is None
        else EvaluationBinding.model_validate(
            {
                **defaults.evaluation_binding.model_dump(mode="json"),
                "eval_config": metric.model_dump(mode="json"),
            }
        )
    )
    return CoproControl(
        prompt_model=resolved_prompt_model,
        evaluation_binding=resolved_binding,
        expected_reward_policy_hash=defaults.expected_reward_policy_hash,
        breadth=breadth,
        depth=depth,
        init_temperature=init_temperature,
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
    "configure_copro",
]
