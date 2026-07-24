"""The algorithm-neutral durable Optimization Step harness and its schemas.

This package owns Whetstone's Workstream 7 surface: the immutable Optimization
Run/Step protocol, the optimizer tool boundary lifecycle, the reusable Reward
contract, the algorithm-neutral harness engine, and the identity optimizer
adapter.

* :mod:`whetstone.optimization.identity` — the Identity/Content Hash helpers
  and the :class:`TypedRef` (typed Object Reference + Content Hash) primitive.
* :mod:`whetstone.optimization.reward` — the reusable Reward Policy/Reward
  contract; the sole Reward constructor refuses official-role evidence.
* :mod:`whetstone.optimization.tools` — the versioned Tool Definition -> Tool
  Config lifecycle, the non-serializable Runtime Tool Handle, and the Tool
  Call/Result/Refusal types.
* :mod:`whetstone.optimization.tool_store` — the authoritative Tool Call Store
  keyed by ``(tool_config_hash, call_id)``.
* :mod:`whetstone.optimization.schema` — the Optimization Run, immutable Step
  Request, Evaluation Intent, immutable Step Result, and terminal Optimization
  Result schemas + identities.
* :mod:`whetstone.optimization.adapters` — the narrow adapter surface and the
  pure identity optimizer adapter.
* :mod:`whetstone.optimization.harness` — the durable harness engine.
"""

from __future__ import annotations

from whetstone.optimization.adapters import (
    AdapterOutput,
    IdentityOptimizerAdapter,
    OptimizerAdapter,
    ToolCallRecord,
)
from whetstone.optimization.harness import (
    ADAPTER_CHECKPOINT_SCHEMA,
    EvaluationService,
    OptimizationHarness,
    StepResultConflictError,
    ToolExecutor,
)
from whetstone.optimization.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.optimization.reward import (
    REWARD_POLICY_SCHEMA,
    MissingDataPolicy,
    OfficialRewardError,
    Reward,
    RewardInputCitation,
    RewardPolicy,
    RewardTerm,
    apply_reward_policy,
)
from whetstone.optimization.schema import (
    OPTIMIZATION_RESULT_SCHEMA,
    OPTIMIZATION_RUN_SCHEMA,
    STEP_REQUEST_SCHEMA,
    STEP_RESULT_SCHEMA,
    BudgetState,
    Candidate,
    EvaluationIntent,
    IntentResolution,
    OptimizationProposal,
    OptimizationResult,
    OptimizationRun,
    OptimizationStepRequest,
    OptimizationStepResult,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
    ToolEvidence,
    step_request_reference,
    step_result_reference,
)
from whetstone.optimization.tool_store import (
    ToolCallState,
    ToolCallStore,
    ToolCallStoreConflictError,
    ToolCallStoreEntry,
    ToolCapacityExceededError,
)
from whetstone.optimization.tools import (
    TOOL_CONFIG_SCHEMA,
    TOOL_DEFINITION_SCHEMA,
    TOOL_RESULT_SCHEMA,
    RefusalClass,
    RuntimeToolHandle,
    ToolCall,
    ToolCapacity,
    ToolConfig,
    ToolDefinition,
    ToolRefusal,
    ToolResult,
    tool_result_reference,
)

__all__ = [
    "ADAPTER_CHECKPOINT_SCHEMA",
    "OPTIMIZATION_RESULT_SCHEMA",
    "OPTIMIZATION_RUN_SCHEMA",
    "REWARD_POLICY_SCHEMA",
    "STEP_REQUEST_SCHEMA",
    "STEP_RESULT_SCHEMA",
    "TOOL_CONFIG_SCHEMA",
    "TOOL_DEFINITION_SCHEMA",
    "TOOL_RESULT_SCHEMA",
    "AdapterOutput",
    "BudgetState",
    "Candidate",
    "EvaluationIntent",
    "EvaluationService",
    "IdentityOptimizerAdapter",
    "IntentResolution",
    "MissingDataPolicy",
    "OfficialRewardError",
    "OptimizationHarness",
    "OptimizationProposal",
    "OptimizationResult",
    "OptimizationRun",
    "OptimizationStepRequest",
    "OptimizationStepResult",
    "OptimizerAdapter",
    "OutputContract",
    "RefusalClass",
    "Reward",
    "RewardInputCitation",
    "RewardPolicy",
    "RewardTerm",
    "RuntimeToolHandle",
    "StepKind",
    "StepMode",
    "StepResultConflictError",
    "StepStatus",
    "ToolCall",
    "ToolCallRecord",
    "ToolCallState",
    "ToolCallStore",
    "ToolCallStoreConflictError",
    "ToolCallStoreEntry",
    "ToolCapacity",
    "ToolCapacityExceededError",
    "ToolConfig",
    "ToolDefinition",
    "ToolEvidence",
    "ToolExecutor",
    "ToolRefusal",
    "ToolResult",
    "TypedRef",
    "apply_reward_policy",
    "compute_identity_hash",
    "require_full_hash",
    "step_request_reference",
    "step_result_reference",
    "tool_result_reference",
    "typed_ref_for_record",
]
