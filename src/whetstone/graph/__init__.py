"""Whetstone graph contracts over the released dr-graph break.

This package holds Whetstone's role-qualified use of the dr-graph Graph
Definition / Graph Config artifacts and ``graph_hash`` as the sole Rollout
Variant identity. Whetstone owns no parallel Graph Config identity payload,
document, schema, or hash: it consumes dr-graph's native ones. What
Whetstone owns here is role semantics:

* the closed, versioned Node Definitions ``whetstone.llm-call/v1`` and
  ``whetstone.eval/v1`` (``nodes``);
* the Eval identity partition rule that binds the Eval Node's statically
  assigned Evaluation Procedure Config to the composite dr-code Eval Config
  (``eval_config``);
* the Rollout Key / Rollout Execution Key / Evaluation Context measurement
  and execution identities (``rollout``);
* the Materialization Record, Proposal / Diff Check validation, fixed-axis
  expansion, Selection Policy, and dedup-by-``graph_hash`` execution
  planning (``materialization``);
* the Character Budget graph/runtime binding with no separate policy
  artifact (``character_budget``).
"""
