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
from whetstone.optimization.codex import (
    CodexAdapter,
    CodexRunner,
    CodexRunResult,
    CodexToolCallLog,
    OpaqueStepError,
)
from whetstone.optimization.codex_runner import (
    FakeCodexRunner,
    ScriptedAgentCall,
    SubprocessCodexRunner,
    build_codex_command,
)
from whetstone.optimization.copro import (
    COPRO_VARIANT,
    HISTORY_PROPOSAL,
    SEED_PROPOSAL,
    CoproAdapter,
    attempt_history_entries,
    rank_attempt_history,
)
from whetstone.optimization.gepa import (
    ACCEPTANCE_POLICY,
    GEPA_VARIANT,
    TOOL_EVALUATE_MINIBATCH,
    TOOL_EVALUATE_SUBSET,
    GepaAdapter,
    GepaHyperparameters,
    strict_pareto_accepts,
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
from whetstone.optimization.mcp_bridge import (
    MCP_PROTOCOL_VERSION,
    EvaluateCandidateServer,
    McpError,
    ScriptedMcpClient,
    tool_result_to_mcp_content,
)
from whetstone.optimization.miprov2 import (
    BASELINE_FULL,
    BOOTSTRAP,
    COMPLETION,
    MINIBATCH,
    POOL_CONSTRUCTION,
    PROMOTION_FULL,
    Miprov2Adapter,
)
from whetstone.optimization.miprov2_identity import (
    DEMO_SET_SCHEMA,
    INSTRUCTION_SCHEMA,
    TRIAL_COMBINATION_SCHEMA,
    DemoPair,
    DemoSetIdentity,
    InstructionIdentity,
    TrialCombinationIdentity,
)
from whetstone.optimization.mutation import (
    MUTATION_FIELD,
    DiffCheckError,
    diff_check,
)
from whetstone.optimization.proposer import (
    PROPOSER_CONFIG_SCHEMA,
    FakeProposerTransport,
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
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
from whetstone.optimization.stub_evaluator import (
    StubToolEvaluator,
    make_stub_evaluator,
)
from whetstone.optimization.tool_eval import (
    EvaluatingToolExecutor,
    ToolEvaluation,
    ToolEvaluator,
    ToolValidationError,
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
    "ACCEPTANCE_POLICY",
    "ADAPTER_CHECKPOINT_SCHEMA",
    "BASELINE_FULL",
    "BOOTSTRAP",
    "COMPLETION",
    "COPRO_VARIANT",
    "DEMO_SET_SCHEMA",
    "GEPA_VARIANT",
    "HISTORY_PROPOSAL",
    "INSTRUCTION_SCHEMA",
    "MCP_PROTOCOL_VERSION",
    "MINIBATCH",
    "MUTATION_FIELD",
    "OPTIMIZATION_RESULT_SCHEMA",
    "OPTIMIZATION_RUN_SCHEMA",
    "POOL_CONSTRUCTION",
    "PROMOTION_FULL",
    "PROPOSER_CONFIG_SCHEMA",
    "REWARD_POLICY_SCHEMA",
    "SEED_PROPOSAL",
    "STEP_REQUEST_SCHEMA",
    "STEP_RESULT_SCHEMA",
    "TOOL_CONFIG_SCHEMA",
    "TOOL_DEFINITION_SCHEMA",
    "TOOL_EVALUATE_MINIBATCH",
    "TOOL_EVALUATE_SUBSET",
    "TOOL_RESULT_SCHEMA",
    "TRIAL_COMBINATION_SCHEMA",
    "AdapterOutput",
    "BudgetState",
    "Candidate",
    "CodexAdapter",
    "CodexRunResult",
    "CodexRunner",
    "CodexToolCallLog",
    "CoproAdapter",
    "DemoPair",
    "DemoSetIdentity",
    "DiffCheckError",
    "EvaluateCandidateServer",
    "EvaluatingToolExecutor",
    "EvaluationIntent",
    "EvaluationService",
    "FakeCodexRunner",
    "FakeProposerTransport",
    "GepaAdapter",
    "GepaHyperparameters",
    "IdentityOptimizerAdapter",
    "InstructionIdentity",
    "IntentResolution",
    "McpError",
    "Miprov2Adapter",
    "MissingDataPolicy",
    "OfficialRewardError",
    "OpaqueStepError",
    "OptimizationHarness",
    "OptimizationProposal",
    "OptimizationResult",
    "OptimizationRun",
    "OptimizationStepRequest",
    "OptimizationStepResult",
    "OptimizerAdapter",
    "OutputContract",
    "ProposalDraft",
    "ProposalRequest",
    "ProposerConfig",
    "ProposerTransport",
    "RefusalClass",
    "Reward",
    "RewardInputCitation",
    "RewardPolicy",
    "RewardTerm",
    "RuntimeToolHandle",
    "ScriptedAgentCall",
    "ScriptedMcpClient",
    "StepKind",
    "StepMode",
    "StepResultConflictError",
    "StepStatus",
    "StubToolEvaluator",
    "SubprocessCodexRunner",
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
    "ToolEvaluation",
    "ToolEvaluator",
    "ToolEvidence",
    "ToolExecutor",
    "ToolRefusal",
    "ToolResult",
    "ToolValidationError",
    "TrialCombinationIdentity",
    "TypedRef",
    "apply_reward_policy",
    "attempt_history_entries",
    "build_codex_command",
    "compute_identity_hash",
    "diff_check",
    "make_stub_evaluator",
    "rank_attempt_history",
    "require_full_hash",
    "step_request_reference",
    "step_result_reference",
    "strict_pareto_accepts",
    "tool_result_reference",
    "tool_result_to_mcp_content",
    "typed_ref_for_record",
]
