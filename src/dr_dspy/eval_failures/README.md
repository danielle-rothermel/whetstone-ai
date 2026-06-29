# eval_failures

Eval worker step failure taxonomy, classification/retry policy, failure
summaries for DB/logs, and recording/generation boundaries.

This package is **not** a global exception registry. Encoding errors are
defined in `dr_dspy.serialization` and bridged via `eval_failures.recording`.
Generation output validation lives in `eval_failures.generation` with typed
errors in `eval_failures.exceptions`. Third-party exceptions are classified by
heuristics in `eval_failures.policy`.

## Generation boundary

Typed generation failures (`EmptyGenerationError`, `PredictionParseError`) are
raised from `eval_failures.generation.require_generation_text` and
`lm.runner.run_predictor`. Humaneval job builders call
`validate_encdec_generation` / `validate_direct_generation` before constructing
`GenerationResult`.

## Deferred swallow inventory (not recording-boundary failures)

| Location | Pattern | Why deferred |
|----------|---------|--------------|
| `harness/dbos.py` | enqueue race tolerance | Idempotency, not telemetry |
| `harness/batch.py` | workflow start race | Same |
| `harness/workers/monitor.py` | monitor loop catch | Operator visibility, not step success |
| `humaneval/code_extraction.py` | extraction fallbacks | Domain logic, not recording |
| `humaneval/task.py` | test-harness metadata | Local eval tool |
