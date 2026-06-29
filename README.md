# dr-dspy

Graph-based HumanEval evaluation platform workbench. The current migration path
is toward graph-shaped generation specs, explicit LM/prompt boundaries, and
append-only terminal outcomes. The old direct and enc-dec DBOS workers remain in
the tree as legacy v0 data-generation surfaces until migration validation is
complete.

## Package layout

- `humaneval/` — task parsing, code extraction, scoring, compression metrics
- `lm/` — prompt/provider boundary helpers, DSPy compatibility, LM telemetry
- `graph/` — pure graph execution and graph-spec hashing
- `eval_failures/` — worker failure taxonomy, retry policy, recording/generation boundaries
- `serialization.py` — JSON-safe encoding for telemetry and DB payloads
- `harness/` — legacy v0 DBOS workflows, batch operations, repair, worker monitoring
- `experiments/` — legacy v0 HumanEval direct and enc-dec DBOS backends

## Design notes

- [Graph-based eval platform design](docs/append-only-eval-records-design.md)
  captures the planned migration toward graph-shaped generation specs,
  append-only outcomes, explicit prompt/LM boundaries, rescoring, metrics, and
  Unitbench-facing projections.

The legacy v0 direct and enc-dec workflows write mutable prediction rows that
mix requested specs, workflow status, generation artifacts, scores, and repair
state. Those rows remain source data for migration/backfill, but new
implementation work should not build domain contracts, graph workflows,
rescoring, or reporting on top of v0 repair/status/reporting flows.

## Failure handling (`eval_failures`)

Eval worker step failures are classified, summarized for DB/logs, and persisted
with structured metadata. This package is **not** a global exception registry.

### Module roles

| Module | Responsibility |
|--------|----------------|
| `serialization.py` | Typed `SerializationError` hierarchy for unencodable values |
| `eval_failures/recording.py` | `ensure_recordable` / `recordable_jsonb` bridge → `RecordingFailureError` |
| `eval_failures/generation.py` | `require_generation_text`, enc-dec/direct validators |
| `eval_failures/exceptions.py` | `EvalFailureError` hierarchy with `failure_class` |
| `eval_failures/policy.py` | Third-party heuristics, `summarize_exception`, `should_retry_step` |

### Recording boundary

All storable JSON/JSONB values pass through `ensure_recordable` or
`recordable_jsonb`. Unencodable LM telemetry or persistence payloads raise
`RecordingFailureError` (permanent, no step retry) instead of being silently
dropped or stored as empty objects. Call sites include LM logging, predictor
metadata, experiment DB writes, and harness batch operations.

### Generation boundary

Typed generation failures (`EmptyGenerationError`, `PredictionParseError`) are
raised from `eval_failures.generation.require_generation_text` and
`lm.runner.run_predictor`. Humaneval job builders call
`validate_encdec_generation` / `validate_direct_generation` before constructing
`GenerationResult`, so empty or unparseable LM output becomes
`generation_error` rather than a false `generated` row.

### Worker workflow pattern

Legacy v0 experiment DBOS workflows catch step exceptions, call
`summarize_exception`, and record errors via `failure_summary_payload`.
Retryable failures (`transient`, `rate_limited`) return recoverable status
strings; permanent failures do not step-retry.

Scoring test failures are domain semantics: a wrong answer records `score=0`
with `scoring_status='scored'`. That is not a worker failure.
