# Prompt: execute enc-dec v0 backfill smoke and tiny rescore

You are working in `/Users/daniellerothermel/drotherm/repos/whetstone-ai` on
branch `today_exp`.

First read:

- `AGENTS.md`
- `docs/testing_logs.md`
- `docs/v0-migration-completion-checklist.md`
- `docs/remaining-implementation-intentions.md`
- `src/dr_dspy/migration/v0_reshape.py`
- `src/dr_dspy/platform/persistence.py`
- `src/dr_dspy/platform/worker.py`
- `src/dr_dspy/platform/rescoring.py`

The current push is about getting evaluation numbers quickly. Scope this task
to enc-dec v0 backfill preparation and tiny validation only. Do not implement
direct-mode backfill, projection movement, Unitbench/export, schema cleanup,
repo rename work, or post-migration deletion.

## Goal

Execute steps 1-5 of the enc-dec backfill plan:

1. Confirm v1 schema is at Alembic head.
2. Add the smallest enc-dec-only live backfill command around
   `reshape_v0_encdec_row`.
3. Run dry-run counts against live v0 data.
4. Run a tiny real enc-dec backfill sample.
5. Rescore the tiny migrated sample.

Stop after the tiny backfill/rescore validation. Do not run the full backfill
or full rescore in this task. Instead, update `docs/testing_logs.md` with the
results and end the new log entry with the exact commands needed for the full
enc-dec backfill and full enc-dec rescore.

## Current facts

- Tiny new v1 enc-dec jobs have already been smoke-tested successfully.
- `reshape_v0_encdec_row(row)` exists and returns:
  - `PredictionSpecRecord`
  - optional `GenerationRunRecord`
  - `NodeAttemptRecord` tuple
  - v0 source metadata
- Existing persistence helpers already provide first-write-wins/idempotent
  inserts for prediction specs, generation runs, node attempts, and score
  attempts.
- The live backfill job/CLI is still missing.
- Keep legacy v0 Postgres tables as read-only backup. Do not drop, mutate, or
  repair v0 rows.
- For today's decisions, only enc-dec migrated rows matter.

## Implementation requirements

Add a narrow enc-dec-only backfill command. Prefer a Typer command in the
existing platform CLI if it fits cleanly, for example:

```bash
uv run python -m dr_dspy.platform.worker backfill-v0-encdec ...
```

If a separate migration CLI is much cleaner, keep the command equally narrow
and document the exact command in `docs/testing_logs.md`.

The command must support:

- `--dry-run`
- `--limit N`
- `--database-url`
- `--env-file`
- a way to use a clean migrated experiment namespace, preferably
  `--target-experiment-name`

Use the target experiment name by copying/overriding the row mapping passed to
`reshape_v0_encdec_row`; do not mutate v0 tables. If preserving each legacy
experiment name is clearly safer than one target name, stop and explain the
tradeoff in the testing log before proceeding.

The command should:

1. Read from the legacy enc-dec table only.
2. Select terminal rows only, matching `V0_TERMINAL_GENERATION_STATUSES`:
   `generated` and `generation_error`.
3. Apply deterministic ordering so dry-runs and limited samples are repeatable.
4. For each selected row, call `reshape_v0_encdec_row`.
5. Insert the v1 records idempotently:
   - prediction spec
   - experiment row if the existing submit/build path requires it
   - generation run when present
   - node attempts when present
6. Report counts:
   - selected v0 rows
   - reshaped specs
   - inserted/already-present specs if easy to determine
   - inserted/already-present generation runs if easy to determine
   - inserted/already-present node attempts if easy to determine
   - skipped non-terminal rows
   - reshape failures

Keep the insert path simple. Do not add a general migration framework.

## Dry-run requirements

Before any write:

1. Confirm Alembic head:

```bash
uv run alembic current
```

2. Run a live dry-run of the new enc-dec command.

The dry-run should prove:

- it can connect using the normal `.env`/`DATABASE_URL` path
- v0 enc-dec table rows can be read
- terminal rows can be selected
- `reshape_v0_encdec_row` validates real rows
- expected insert counts are reported without writing

Record the dry-run command and summarized output in `docs/testing_logs.md`.

## Tiny write requirements

After dry-run succeeds, run a tiny real backfill sample. Use a clean target
experiment name with a date-stamped suffix, for example:

```text
v0_encdec_backfill_smoke_20260630
```

Use a small `--limit`, for example 5-20 rows. Choose the smallest sample that
includes at least one scoreable successful generation if available.

After the tiny write, validate with SQL or existing CLI helpers:

- v1 prediction specs exist for the target experiment
- v1 generation runs exist
- v1 node attempts exist
- migrated generation summaries include v0 source metadata
- status counts look sane

Record the validation queries and summarized results in `docs/testing_logs.md`.

## Tiny rescore requirements

Run batch rescore only for the tiny migrated experiment:

```bash
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name v0_encdec_backfill_smoke_20260630 \
  --generation-status success \
  --generation-status partial
```

If the command needs the actual target experiment name, substitute it. Do not
rescore all migrated/full v0 rows in this task.

After rescoring, validate:

- scheduled count
- failed count
- persisted score-attempt count
- score status counts
- a small pass-rate summary if any attempts succeeded

Record the command and summarized output in `docs/testing_logs.md`.

## Testing requirements

Add focused tests for the new backfill command or helper layer.

Minimum useful coverage:

- dry-run does not write
- enc-dec query/filter targets only the legacy enc-dec table and terminal
  statuses
- target experiment override affects the reshaped v1 spec without mutating the
  input row
- idempotent tiny import behavior is safe to rerun, at least at the statement or
  helper level

Run targeted tests, at least the v0 reshape and new backfill tests:

```bash
uv run pytest tests/test_v0_reshape.py tests/integration/test_v0_reshape_outcomes.py
```

Also run the specific new test file you add. If the backfill command is added
to `worker.py`, run the relevant worker CLI tests too:

```bash
uv run pytest tests/test_platform_worker_cli.py
```

Use `uv run python` and `uv run pytest`.

## Testing log update

Add a new entry at the top of `docs/testing_logs.md` with:

- date
- branch
- command/feature under test
- schema-head check result
- code changes under test
- tests run and result
- dry-run command and summarized output
- tiny backfill command and summarized output
- validation queries/results
- tiny rescore command and summarized output
- verdict
- blockers or caveats

The entry must end with this section title:

```markdown
### Full enc-dec backfill/rescore commands
```

Under that title, include one shell block with the exact commands needed to run
the full enc-dec backfill and full enc-dec rescore after review. These commands
must be runnable as pasted and must use the actual command names/options you
implemented.

The full commands should not include `--dry-run` or `--limit`.

## Constraints

- Enc-dec only.
- Do not touch direct rows.
- Do not mutate or drop v0 tables.
- Do not run the full backfill.
- Do not run the full rescore.
- Do not implement projection movement.
- Do not add generalized migration abstractions.
- Prefer Pydantic `BaseModel` for new structured data.
- Use Typer for new CLI surfaces.
- Keep changes as small as possible.

## Handoff

When done, report:

- files changed
- command implemented
- tests run
- dry-run result
- tiny backfill result
- tiny rescore result
- exact location of the full-run commands in `docs/testing_logs.md`
- whether the repo is ready for the operator to run full enc-dec backfill and
  full enc-dec rescore
