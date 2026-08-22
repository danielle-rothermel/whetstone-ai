"""The persisted configuration of one Codex-direct optimizer run.

Codex runs exactly one opaque Step: the CLI is given a single Tool -- an
external MCP evaluation endpoint bound to the internal split -- and it
returns the ``call_id`` of the call whose candidate it selected. The
control pins everything that identity depends on: the model and its
reasoning effort, the containment posture, the eval binding, and the
per-run tool capacity that is simultaneously the Step's ``tool_calls``
budget.

No field shapes identity without shaping execution. They reach it by
different routes. ``codex_binary``, ``model``, ``reasoning_effort``, and
``denied_features`` reach the CLI argv (``build_codex_command``).
``wall_seconds`` and ``max_output_bytes`` bound the dr-exec job that runs
it. ``max_tool_calls`` is the per-run ``ToolCapacity`` the evaluation
server admits against, and ``mutation_field`` is the field the rebuilt
candidate is written to. The remainder -- ``eval_config_ref``,
``reward_policy_hash``, ``evaluation_execution_policy_hash``,
``task_model_identity_hash``, and ``internal_task_hashes`` -- pin what
each admitted evaluation is measured against, and the
``containment_profile``, ``network_policy``, and ``filesystem_policy``
literals record the posture the runner enforces. ``algorithm_version``
and ``adapter_schema_version`` name the code that reads all of it.

That is the standard a field must meet, which is why the control carries
no turn cap and no sampling seed: the Codex CLI exposes neither, so
either would have recorded a distinction the run never made. Evaluation
seeding is unaffected -- it lives in the Eval Config's ``SeedPlan``, and
every admitted Tool Call evaluates under the run's pinned seed plan.
"""

from __future__ import annotations

import math
from enum import UNIQUE, StrEnum, verify
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
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
from whetstone.experiment.binding import EvalConfigRef
from whetstone.optim.codex.containment import (
    CODEX_CONTAINMENT_PROFILE,
    CODEX_DEFAULT_MAX_OUTPUT_BYTES,
    CODEX_DENIED_FEATURES,
    CODEX_FILESYSTEM_POLICY,
    CODEX_NETWORK_POLICY,
)

CODEX_ALGORITHM = "codex"
CODEX_ALGORITHM_VERSION = "whetstone.codex_direct/v1"
CODEX_ADAPTER_SCHEMA_VERSION = "whetstone.codex_output_artifact/v1"
CODEX_CONTROL_SCHEMA = "whetstone.codex_optimizer_config"
CODEX_CONTROL_SCHEMA_VERSION = 1
CODEX_DEFAULT_MUTATION_FIELD = "user_prompt_template"
CODEX_DEFAULT_BINARY = "codex"
CODEX_DEFAULT_WALL_SECONDS = 600.0


@verify(UNIQUE)
class CodexReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CodexControl(BaseModel):
    """One Codex-direct run's complete, identity-bearing configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: StrictStr
    #: Passed to the CLI as ``-c model_reasoning_effort``. Like every
    #: control field it shapes execution as well as identity; a field
    #: that shaped identity alone would claim a distinction the run
    #: never made. See the module docstring for each field's route.
    reasoning_effort: CodexReasoningEffort = CodexReasoningEffort.MEDIUM
    wall_seconds: float = CODEX_DEFAULT_WALL_SECONDS
    max_output_bytes: StrictInt = CODEX_DEFAULT_MAX_OUTPUT_BYTES
    #: Both the Step's ``tool_calls`` budget label and the per-run
    #: ``ToolCapacity.max_accepted_calls``. Required: whetstone-ai holds no
    #: policy on how large an eval budget a caller should buy.
    max_tool_calls: StrictInt
    codex_binary: StrictStr = CODEX_DEFAULT_BINARY

    eval_config_ref: EvalConfigRef
    reward_policy_hash: StrictStr
    evaluation_execution_policy_hash: StrictStr
    task_model_identity_hash: StrictStr
    #: The exact internal split the Tool evaluates. The Tool grants no
    #: narrowing: one call scores this whole set.
    internal_task_hashes: tuple[StrictStr, ...]
    mutation_field: StrictStr = CODEX_DEFAULT_MUTATION_FIELD

    denied_features: tuple[StrictStr, ...] = CODEX_DENIED_FEATURES
    containment_profile: Literal["process_boundary_only"] = (
        CODEX_CONTAINMENT_PROFILE
    )
    network_policy: Literal["allowed"] = CODEX_NETWORK_POLICY
    filesystem_policy: Literal["scratch_only"] = CODEX_FILESYSTEM_POLICY

    algorithm_version: StrictStr = CODEX_ALGORITHM_VERSION
    adapter_schema_version: StrictStr = CODEX_ADAPTER_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate(self) -> CodexControl:
        for field in (
            "reward_policy_hash",
            "evaluation_execution_policy_hash",
            "task_model_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        if not self.internal_task_hashes:
            raise ValueError("internal_task_hashes must be non-empty")
        for identity in self.internal_task_hashes:
            require_full_hash(identity, field="internal_task_hashes")
        if len(set(self.internal_task_hashes)) != len(
            self.internal_task_hashes
        ):
            raise ValueError(
                "internal_task_hashes must contain unique identities"
            )
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.codex_binary.strip():
            raise ValueError("codex_binary must be non-empty")
        if not self.mutation_field.strip():
            raise ValueError("mutation_field must be non-empty")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be finite and positive")
        if self.max_output_bytes < 4:
            raise ValueError("max_output_bytes must leave room for retention")
        if not self.denied_features:
            raise ValueError("denied_features must be non-empty")
        if len(set(self.denied_features)) != len(self.denied_features):
            raise ValueError("denied_features must be unique")
        fixed_values = {
            "denied_features": CODEX_DENIED_FEATURES,
            "algorithm_version": CODEX_ALGORITHM_VERSION,
            "adapter_schema_version": CODEX_ADAPTER_SCHEMA_VERSION,
        }
        for field, expected in fixed_values.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} is fixed")
        return self

    def identity_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["algorithm"] = CODEX_ALGORITHM
        payload["eval_config_identity_hash"] = self.eval_config_ref.config_hash
        return payload

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=CODEX_CONTROL_SCHEMA,
            schema_version=CODEX_CONTROL_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def reference(self) -> IdentityRef:
        return IdentityRef(
            record_ref=typed_ref_for_record(
                CODEX_CONTROL_SCHEMA,
                self.record_content(),
            ),
            record_hash=self.identity_hash(),
        )

    def step_hyperparameters(self, *, iteration: int) -> dict[str, Any]:
        if iteration != 0:
            raise ValueError("Codex runs exactly one opaque step")
        return {
            "round_index": iteration,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort.value,
            "max_tool_calls": self.max_tool_calls,
            "eval_config_identity_hash": self.eval_config_ref.config_hash,
            "reward_policy_hash": self.reward_policy_hash,
        }

    def require_identity_hash(self, persisted_hash: str) -> None:
        require_full_hash(persisted_hash, field="optimizer_config_hash")
        if persisted_hash != self.identity_hash():
            raise ValueError(
                "optimizer_config_hash conflicts with resolved Codex control"
            )

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy through the complete validation boundary.

        Pydantic's default update path trusts replacement values and skips
        validators. A control copy is itself a persisted authority, so every
        field and cross-field invariant must be revalidated.
        """
        del deep
        payload = self.model_dump(mode="json")
        payload.update(update or {})
        return type(self).model_validate(payload)


def configure_codex(
    *,
    model: str,
    max_tool_calls: int,
    eval_config_ref: EvalConfigRef,
    reward_policy_hash: str,
    evaluation_execution_policy_hash: str,
    task_model_identity_hash: str,
    internal_task_hashes: tuple[str, ...],
    reasoning_effort: CodexReasoningEffort = CodexReasoningEffort.MEDIUM,
    wall_seconds: float = CODEX_DEFAULT_WALL_SECONDS,
    max_output_bytes: int = CODEX_DEFAULT_MAX_OUTPUT_BYTES,
    codex_binary: str = CODEX_DEFAULT_BINARY,
    mutation_field: str = CODEX_DEFAULT_MUTATION_FIELD,
) -> CodexControl:
    """Build one Codex control through the full validation boundary."""
    return CodexControl(
        model=model,
        reasoning_effort=reasoning_effort,
        wall_seconds=wall_seconds,
        max_output_bytes=max_output_bytes,
        max_tool_calls=max_tool_calls,
        codex_binary=codex_binary,
        eval_config_ref=eval_config_ref,
        reward_policy_hash=reward_policy_hash,
        evaluation_execution_policy_hash=evaluation_execution_policy_hash,
        task_model_identity_hash=task_model_identity_hash,
        internal_task_hashes=internal_task_hashes,
        mutation_field=mutation_field,
    )


__all__ = [
    "CODEX_ADAPTER_SCHEMA_VERSION",
    "CODEX_ALGORITHM",
    "CODEX_ALGORITHM_VERSION",
    "CODEX_CONTROL_SCHEMA",
    "CODEX_CONTROL_SCHEMA_VERSION",
    "CODEX_DEFAULT_BINARY",
    "CODEX_DEFAULT_MUTATION_FIELD",
    "CODEX_DEFAULT_WALL_SECONDS",
    "CodexControl",
    "CodexReasoningEffort",
    "configure_codex",
]
