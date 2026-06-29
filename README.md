# dr-dspy

Durable HumanEval evaluation workers built on DBOS: submit predictions, generate
code with LLMs, score against HumanEval tests, and persist every inference
output. Experiment backends live under `src/dr_dspy/`.

## Package layout

- `humaneval/` — task parsing, code extraction, scoring, compression metrics
- `lm/` — DSPy signatures, OpenRouter LM adapter, predictor runner
- `harness/` — DBOS workflows, batch operations, repair, worker monitoring
- `experiments/` — HumanEval direct and enc-dec DBOS backends
- `eval_failures/` — worker failure taxonomy, retry policy, recording/generation boundaries
- `serialization.py` — JSON-safe encoding for telemetry and DB payloads

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

Experiment DBOS workflows catch step exceptions, call `summarize_exception`,
and record errors via `failure_summary_payload`. Retryable failures
(`transient`, `rate_limited`) return recoverable status strings; permanent
failures do not step-retry.

Scoring test failures are domain semantics: a wrong answer records `score=0`
with `scoring_status='scored'`. That is not a worker failure.
