# eval_failures

Eval worker step failure taxonomy, classification/retry policy, failure
summaries for DB/logs, and the recording boundary.

This package is **not** a global exception registry. Encoding errors are
defined in `dr_dspy.serialization` and bridged via `eval_failures.recording`.
Third-party exceptions are classified by heuristics in `eval_failures.policy`.

## Deferred swallow inventory (not recording-boundary failures)

| Location | Pattern | Why deferred |
|----------|---------|--------------|
| `dspy_runner.py` | predictor catch, fallback to buffer text | Intentional LM-failure recovery |
| `dbos_runtime.py` | enqueue race tolerance | Idempotency, not telemetry |
| `batch_operation.py` | workflow start race | Same |
| `worker_monitor.py` | monitor loop catch | Operator visibility, not step success |
| `code_extraction.py` | extraction fallbacks | Domain logic, not recording |
| `human_eval.py` | test-harness metadata | Local eval tool |
