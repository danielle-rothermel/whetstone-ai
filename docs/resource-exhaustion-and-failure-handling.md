# Worker Resource Exhaustion & Failure-Handling Design Notes

Status: analysis complete, fix implemented in the shared DBOS worker
resource/failure-handling path.
Scope: the DBOS HumanEval eval harness (`humaneval_encdec_dbos.py`,
`humaneval_direct_dbos.py`, and the shared `dbos_runtime` / `dspy_runner` /
`worker_monitor` / `eval_repair` modules).

## Summary

The first full enc-dec sweep (`encdec-budget-full-v0`, 55,104 jobs) failed
overnight: ~86% of generation jobs were marked `generation_error` and both
worker queues stalled. The stored error on every row was the opaque DBOS
wrapper `DBOSMaxStepRetriesExceeded`. Decoding the real exceptions that DBOS
recorded shows the true cause was **file-descriptor exhaustion**
(`OSError: [Errno 24] Too many open files`), which then surfaced as
`psycopg.OperationalError: connection is bad` against the local Postgres
socket. The root driver is a **per-job HTTP-client leak** hitting a
**self-imposed 8,192 open-file cap**. The incident also exposed several
design smells in how the harness handles failure — most importantly that it
has a single, undifferentiated failure path where it needs failure
classification with per-class policy.

This document records the problem, the evidence, the concerns, and each smell
in detail. The proposed fix is summarized at the end.

## The incident

- Run: enc-dec full sweep, experiment `encdec-budget-full-v0`.
- Jobs: 55,104 = 164 HumanEval+ tasks × 16 self-pair models × 7 budget ratios
  × 3 repetitions, encoder and decoder temperature 0.
- Worker: `--queue both`, default concurrency (generation 64, scoring 32),
  default `--open-file-limit 8192`, against a **local** Postgres.
- Observed: the submit/worker shell filled with repeating tracebacks; by
  morning both queues were stalled at a fixed state and no progress was being
  made. The visible traceback was a `psycopg` "connection is bad" failure
  inside the error-recording path, plus
  `cannot schedule new futures after shutdown`.

## State assessment (evidence)

Prediction table (`dr_dspy_encdec_eval_predictions`, experiment
`encdec-budget-full-v0`):

| generation_status | count | scoring_status | count |
| --- | ---: | --- | ---: |
| generation_error | 47,404 | pending | 48,437 |
| generated | 6,667 | scored | 6,309 |
| pending | 851 | queued | 354 |
| started | 182 | started | 4 |

- Fully complete (generated **and** scored): **6,309**. All 55,104 rows were
  inserted, so submission itself completed.
- Timeline: generation errors span **05:54 → 08:43**, sustained at
  ~3,300 per 10 minutes from 06:20 onward; successes span 05:54 → 07:51.
  This is a **continuous 2.5-hour failure**, not a single blip.
- Uniformity: every one of the 16 models has ~400 generated / ~2,960 errored;
  every one of the 7 budgets (including `none`, which uses the non-budgeted
  encoder) has ~950 generated / ~6,770 errored. The failure is **systemic and
  uniform**, not specific to a model or to the budget feature.
- Database health afterward: local Postgres (socket
  `/tmp/.s.PGSQL.5432`), `max_connections = 1000`, only 6 active — so this was
  **not** raw connection-count exhaustion.
- Real exceptions (decoded from `dbos.workflow_status.error`, which DBOS
  pickles; our own table stored only the wrapper):
  - `OSError: [Errno 24] Too many open files`
  - `psycopg.OperationalError: connection is bad: no error details available`
  - `psycopg.OperationalError: connection is bad: connection to server on
    socket "/tmp/.s.PGSQL.5432"`
- FD limits on this machine: OS `kern.maxfilesperproc = 245760`, hard limit
  `unlimited` — yet the worker raises its soft limit only to **8,192**
  (`DEFAULT_WORKER_OPEN_FILE_LIMIT`).

## Root cause

A compounding chain:

1. **Per-job HTTP-client leak.** `generate_code_for_job`
   (`humaneval_encdec_dbos.py`) builds a fresh `LoggingOpenRouterLM` — and
   therefore a new `OpenAI`/httpx client — for the encoder **and** the decoder
   on every job (`build_lm` → `dspy_runner.build_logged_lm` →
   `openrouter_lm.OpenRouterLM._get_client`, which calls `OpenAI(...)` at
   `openrouter_lm.py:83`). Nothing ever closes these clients — there is no
   `.close()`, `__del__`, or `__exit__` on `OpenRouterLM` /
   `LoggingOpenRouterLM` — and dspy does not reuse LMs across calls
   (`dspy_runner.run_predictor` opens a fresh `dspy.context` per call). Under
   sustained 64-way concurrency, client sockets accumulate faster than GC
   reclaims them.
2. **Self-imposed low ceiling.** The worker lowers/sets its soft open-file
   limit to 8,192 (`raise_open_file_limit`, `dbos_runtime.py:263`), far below
   what the OS allows. The leak reaches that ceiling in ~30 minutes (errors
   ramp 06:10 → 06:20).
3. **Cascade.** Once file descriptors are exhausted, the process can no longer
   open sockets: new Postgres connections fail (`connection is bad`), HTTP
   calls fail, and scoring subprocess pipes fail. The generate step fails, its
   retries (3 attempts; already exponential at 2s → 4s → 8s) all fail because
   the resource is still gone, and DBOS raises `DBOSMaxStepRetriesExceeded`.
   The workflow catches it and writes a **terminal** `generation_error`,
   converting a transient/systemic condition into 47,404 permanent failures;
   eventually the executor tears down (`cannot schedule new futures after
   shutdown`).

The same per-job client pattern exists in `humaneval_direct_dbos.py`
(`build_generation_lm`), so any large direct run is equally exposed.

## Concerns

- **Permanent damage from a transient cause.** A resource blip burned 47,404
  jobs into a terminal state that requires manual `repair --apply` to recover.
- **Silent multi-hour degradation.** The run failed at ~86% for 2.5 hours with
  no signal beyond "queues stalled." Nothing surfaced the failure rate or the
  error class, so the failure was invisible until morning.
- **Wasted spend.** A mostly-failed run still incurs API cost for the ~12% that
  succeeded plus retries, and partial encoder/decoder calls that failed late.
- **High diagnosis friction.** The true cause was only recoverable by decoding
  DBOS's pickled exceptions, because the harness stored only the
  `DBOSMaxStepRetriesExceeded` wrapper.
- **Shared blast radius.** The same leak and the same fixed FD cap apply to the
  direct experiment.

## Code smells

### Smell 1 — Per-job unbounded resource creation (the leak)

A new HTTP client is created for every encoder and every decoder call and
never closed, so a per-request resource grows without bound across a run. An
injection seam already exists and is unused: `build_logged_lm(client=…)`
(`dspy_runner.py:36`) accepts a client but no production caller passes one. The
correct shape is a single shared, connection-pool-bounded client, not one per
call.

### Smell 2 — A fixed, under-OS open-file ceiling that isn't derived from need

`DEFAULT_WORKER_OPEN_FILE_LIMIT = 8192` is a magic number set far below the
machine's capacity (245,760 per process here), and it is not computed from the
actual resource footprint. This is inconsistent with how the DB pool is
sized: `auto_db_pool_max_size` (`dbos_runtime.py:188`) derives the pool from
`generation_concurrency + scoring_concurrency + margin`, but the FD limit — the
budget that actually governs *all* sockets, pipes, and connections — is a
hard-coded constant. The resource that matters most is the one not
right-sized.

### Smell 3 — One undifferentiated failure path (the deep smell)

Every failure is handled identically at three layers, when different failure
types demand different responses:

- **3a. Uniform retry.** The generate step retries every exception the same
  way, and DBOS's per-step `should_retry(exception) -> bool` predicate
  (`_core.py:330`) is unused — so resource-exhaustion and permanent failures
  are retried just like transient ones, which is futile (resource) or wasteful
  (permanent). (Note: a related perception that retries were "fixed every 2
  seconds" is inaccurate — DBOS defaults `backoff_rate = 2.0`, so the existing
  retries are already exponential. The real smell is the lack of
  classification, not the backoff curve.)
- **3b. Terminal-on-transient.** The workflow's `except Exception`
  (`humaneval_encdec_dbos.py`, generate/score workflows) writes a permanent
  `generation_error` / `score_error` for *any* exception, so a transient or
  systemic condition becomes a permanent data state instead of a recoverable
  one.
- **3c. Repair is class-blind.** `reset_generation_errors_for_retry` and the
  `eval_repair` plan treat all `generation_error` rows identically, with no
  transient-vs-permanent distinction — so a genuinely permanent failure (e.g.
  a bad model id) is re-run indefinitely alongside recoverable ones.

The missing abstraction is a failure taxonomy plus a per-class policy; the
machinery to apply it (per-step `should_retry`, the workflow boundary, repair,
the monitor) already exists at every layer.

### Smell 4 — Error identity is discarded at the point of capture

The workflow stores `repr(error)` (`humaneval_encdec_dbos.py` generate/score
workflows), which is the `DBOSMaxStepRetriesExceeded` wrapper, not the real
exception type/message/cause. Diagnosing this incident required reading
`dbos.workflow_status` and un-pickling exceptions. The true cause should be
preserved at capture time.

### Smell 5 — No health observability and no safe stop

`worker_monitor.py` reports only absolute counts (queue depth, per-phase
generated/scored/errored), with no failure *rate*, no error *class*, and no
resource-headroom signal. The worker has no preflight safety check, no
degradation detection, and no clean self-halt: it runs at full concurrency
until manually killed. The result is exactly this incident's worst trait — it
degraded silently and kept grinding.

### Smell 6 — DBOS resilience features are under-used

The harness uses only `worker_concurrency` on the queues and a static step
retry. DBOS also exposes `should_retry`, `backoff_rate`, and
`max_recovery_attempts`, none of which are wired into a deliberate failure
policy.

## Proposed fix plan

(Direction chosen: prevent the root cause, then make the failure mode
observable and safe-stopping rather than self-healing. Bullets are intentional
one-liners; items marked **[scope]** need more design before implementation.)

- Share one bounded HTTP client per worker, injected through the existing
  `build_logged_lm(client=)` seam and closed on shutdown, so LM-call file
  descriptors are bounded by the client's connection pool rather than by job
  count.
- Replace the fixed 8,192 open-file cap with a soft limit derived from the
  actual bounded footprint (HTTP pool + DB pool(s) + scoring-subprocess pipes
  + margin), mirroring the existing `auto_db_pool_max_size` logic.
- Preserve the real failure at capture time: unwrap DBOS retry wrappers and
  store the actual exception type and message (plus a failure class) instead of
  `repr(wrapper)`.
- Introduce a single `classify(exception) -> FailureClass` function as the
  spine and wire it into step `should_retry`, the workflow terminal-vs-
  recoverable decision, repair, and the monitor. **[scope: the failure
  hierarchy itself — the exact classes (e.g. transient / rate-limited /
  resource-exhaustion / permanent), how each is identified from real
  exceptions and HTTP status codes, and the policy each maps to.]**
- Make only genuinely permanent failures terminal; treat transient,
  rate-limited, and resource-exhaustion failures as recoverable (re-raise so
  DBOS/`repair` re-run them) so a resource blip never burns the queue.
- Add a worker preflight check that refuses or loudly warns when the FD budget
  is too small for the chosen concurrency, so an unsafe configuration cannot be
  launched.
- Surface failure rate and dominant error class in the worker monitor so
  degradation is visible within seconds instead of hours.
- Add a clean threshold kill-switch that, on a sustained resource/transient
  failure spike, stops intake, drains in-flight work, and exits non-zero with
  the queue preserved. **[scope: thresholds, the measurement window, and
  whether to auto-resume or require a manual restart.]**
- Apply the same root fixes (shared client, FD budget) to
  `humaneval_direct_dbos.py`, which shares the per-job client pattern.
- **[scope: whether `repair` should skip permanent-class rows, and whether to
  persist a dedicated `error_class` column or derive it from the stored
  error.]**
