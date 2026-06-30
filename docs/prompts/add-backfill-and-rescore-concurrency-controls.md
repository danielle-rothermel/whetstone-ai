# Prompt: add backfill and rescore concurrency controls

You are working in `/Users/daniellerothermel/drotherm/repos/whetstone-ai` on
branch `today_exp`.

First read:

- `AGENTS.md`
- `docs/testing_logs.md`
- `src/dr_dspy/migration/v0_encdec_backfill.py`
- `src/dr_dspy/platform/worker.py`
- `src/dr_dspy/platform/rescoring.py`
- `src/dr_dspy/platform/scoring_workflow.py`
- `tests/test_v0_encdec_backfill.py`
- `tests/test_platform_worker_cli.py`
- `tests/test_platform_scoring.py`

The current push is about getting full enc-dec backfill/rescore completed fast
enough to choose HPMs. This task is only about operator-controlled throughput.
Do not change migration semantics, scoring semantics, prompt configs, analysis
scripts, projection movement, Unitbench/export, direct-mode behavior, or schema
migrations.

## Goal

Make both operational phases controllable from CLI:

1. `backfill-v0-encdec` should avoid one huge serial transaction and support
   faster bounded operation.
2. `rescore` should support an explicit maximum number of scoring workflows in
   flight at once.

Keep the implementation small and safe. Preserve idempotency and rerun safety.

## Current behavior to verify

Backfill currently:

- fetches all terminal rows from `dr_dspy_encdec_eval_predictions`
- opens one transaction for the full write path
- reshapes one row at a time
- inserts spec/run/node attempts row-by-row
- has `--dry-run`, `--limit`, `--target-experiment-name`
- has no `--chunk-size` or concurrency control

Rescore currently:

- selects scoreable generation runs in pages via `--chunk-size`
- schedules DBOS scoring workflows
- awaits scheduled handles only after scheduling completes
- has `--limit`, `--chunk-size`, `--dry-run`
- has no true max in-flight scoring workflow control

## Backfill intended change

Implement chunked backfill first. Add CLI options:

```bash
--chunk-size 1000
--reshape-workers 1
```

`--chunk-size` is required for this task. `--reshape-workers` can be accepted
and validated now even if it only supports `1`, but prefer implementing bounded
parallel reshape if it is straightforward and safe.

Required behavior:

- Process terminal v0 rows in deterministic chunks.
- Commit each chunk independently.
- Keep idempotent inserts.
- Print/report cumulative counts and chunk progress.
- `--limit` still limits the total selected terminal rows.
- `--dry-run` still writes nothing.
- Preserve default behavior when new flags are omitted.

Preferred implementation:

```text
for chunk in terminal row pages:
  reshape rows
  insert specs/runs/node attempts
  commit chunk
  update cumulative counts
```

Important details:

- Use deterministic keyset or offset paging over the existing order
  `generation_status ASC, prediction_id ASC`.
- If using offset paging, keep it simple and explain in a short comment why it
  is acceptable for a read-only frozen v0 table.
- Do not mutate v0 rows.
- Do not add a generalized migration framework.
- Do not parallelize database writes unless it remains clearly idempotent and
  easy to reason about.
- If implementing `--reshape-workers > 1`, parallelize reshape only, then write
  from one DB writer per chunk. Do not share SQLAlchemy connections across
  worker threads/processes.

Backfill command target shape:

```bash
uv run python -m dr_dspy.platform.worker backfill-v0-encdec \
  --chunk-size 1000 \
  --reshape-workers 4
```

If `--reshape-workers 4` is not implemented in this pass, the command should
reject values other than `1` with a clear error instead of silently ignoring it.

## Rescore intended change

Add a true in-flight scoring workflow limit:

```bash
--max-in-flight 30
```

Required behavior:

- `--max-in-flight` controls how many scheduled scoring workflow handles are
  allowed before awaiting.
- It is separate from `--chunk-size`.
- `--chunk-size` remains DB selector page size.
- `--limit` remains total candidate cap.
- `--dry-run` schedules nothing and should not need DBOS runtime.
- Default behavior should remain compatible; choose a conservative default such
  as current unbounded/end-await behavior or `1` only if needed for safety, and
  document the choice in code/tests.

Preferred scheduling shape:

```text
pending_handles = []
for candidate in candidates:
  schedule candidate
  if handle exists:
    pending_handles.append(handle)
  if len(pending_handles) >= max_in_flight:
    await pending_handles
    pending_handles = []
await pending_handles
```

This is acceptable even if it means `rescore_generation_runs` now does the
awaiting itself or exposes waves to the CLI, but keep the boundary simple.

Rescore command target shape:

```bash
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-budget-full-v0 \
  --generation-status success \
  --generation-status partial \
  --max-in-flight 30
```

## Reporting

Both commands should print enough progress to know they are alive during long
runs. Keep output concise and machine-readable where possible.

Backfill result should include:

- chunk size
- reshape workers
- selected/processed counts
- inserted/already-present counts
- reshape failures
- experiments touched

Rescore result should include existing counts plus enough information to infer
that max-in-flight was applied, for example:

- `max_in_flight`
- scheduled count
- recovered count
- in-flight/orphan/failed counts
- selected count

## Tests

Add focused tests. Do not add brittle timing tests.

Backfill tests:

- `--chunk-size` processes multiple chunks and accumulates counts.
- `--limit` caps total processed rows across chunks.
- dry-run writes nothing across chunks.
- rerun remains idempotent.
- `--reshape-workers` rejects unsupported values or verifies parallel reshape
  is used if implemented.

Rescore tests:

- CLI passes `--max-in-flight` into rescore logic.
- max-in-flight batching awaits handles after each wave.
- dry-run does not await/schedule handles.
- existing selector/idempotency behavior remains intact.

Run targeted tests:

```bash
uv run pytest tests/test_v0_encdec_backfill.py tests/test_platform_worker_cli.py tests/test_platform_scoring.py -q
```

If you add a dedicated test file, run it too.

## Testing log

Update `docs/testing_logs.md` with:

- command flags added
- tests run
- a small dry-run/backfill verification command
- a small rescore dry-run or limited rescore verification command
- exact recommended full-run commands using the new flags

Recommended final command examples:

```bash
uv run python -m dr_dspy.platform.worker backfill-v0-encdec \
  --chunk-size 1000 \
  --reshape-workers 4

uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-budget-full-v0 \
  --generation-status success \
  --generation-status partial \
  --max-in-flight 30
```

## Constraints

- Enc-dec v0 backfill only.
- Do not touch direct migration.
- Do not change score computation.
- Do not change stable IDs.
- Do not change DB schema.
- Do not add dependencies.
- Do not build a queue worker for backfill.
- Keep the patch small and operational.

## Handoff

When done, report:

- files changed
- new CLI flags
- whether `--reshape-workers > 1` is implemented or rejected
- tests run
- updated full-run commands
- any caveats before restarting/rerunning full backfill or full rescore
