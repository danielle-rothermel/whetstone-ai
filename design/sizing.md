# Sizing table

One place where every concurrency, worker-count, capacity, and pool number is
a named caller decision. This is a document, not a mechanism: it imports
nothing and computes nothing. When a number here changes in code, change it
here in the same commit.

## Live values today

| Decision | Value | Owner (named constant / field) | Consumers |
| --- | --- | --- | --- |
| Row-execution fanout concurrency | 5 | `DEFAULT_CONCURRENCY`, `src/whetstone/execution/fanout.py` | `run_call_pool` default; leaks into `EvaluationEngine(concurrency=...)` (`evaluation/engine.py`) and both code_comp drivers (`evaluation/drivers/code_comp/direct.py`, `.../encdec.py`) |
| Subprocess runtime rebuild concurrency | 5 | `CodeCompEvaluationRuntimeConfig.concurrency`, `src/whetstone/envs/code_comp/runtime_config.py` | duplicate of the fanout default carried on the persisted runtime config; the runtime home for the number once the fanout stack is deleted |
| ED1 baseline behavior-matrix row concurrency | 100 | `DEFAULT_CONCURRENCY`, `src/whetstone/envs/code_comp/behavior_matrix.py` | `run_code_comp_baseline_behavior_matrix`, `scripts/experiments/run_baseline_behavior_matrix.py` |
| Provider wire-call wall cap (seconds) | 600.0 | `DEFAULT_TIMEOUT_SECONDS`, `src/whetstone/runner/routes.py` | every canonical route builder in `runner/routes.py`; the dribble backstop — `DEFAULT_IDLE_SECONDS` is the real stall detector |

The two `DEFAULT_CONCURRENCY = 5` copies are the same decision recorded twice;
the fanout copy dies with the fanout stack in the gated wave, leaving the
runtime-config field as the single owner.

## Platform-cutover rows (decided at cutover)

These rows exist so that the numbers are decided together, in one place, when
whetstone moves onto dr-platform. Each is marked with the rule that decides
it; none has a value today.

| Decision | Value | Deciding rule |
| --- | --- | --- |
| DBOS queue concurrency (per queue) | decided at cutover | one queue per stage; queue concurrency bounds in-flight steps for that stage |
| Provider worker count | decided at cutover | bounded by provider rate limits and the wire-call wall cap above; a named number per provider lane, not a shared pool |
| Exec (code-scoring) worker count | decided at cutover | bounded by local CPU and the disposable-worker isolation cost; distinct from the provider worker count |
| Stage capacity (`batch_size`, `barrier_batch_size`) | decided at cutover | the dispatcher's batch grain; chosen per stage with the row grain forced by the platform decision |
| DB pool size | decided at cutover | dr-platform enforces `pool_size >= max(batch_size, barrier_batch_size)` (`required_checkpoint_workers`) at `register_scheduled_dispatcher` (`dr_platform/runtime/dispatcher.py`). The check fires at registration, not at config construction — size the pool before that call, from the stage capacities chosen above. |

Rules for maintaining this table:

- Every number is a named caller decision with one owner. No number is
  derived silently from another at runtime; when two rows must agree (DB pool
  vs. stage capacity), the deciding rule names the direction of the
  dependency and the numbers are set together.
- No rate targets, no throughput forecasts. The table records decisions, not
  predictions.
