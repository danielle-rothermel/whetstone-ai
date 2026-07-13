# Cutover operator tooling

`whetstone-cutover` is dry-run by default and never dispatches work. Mutating
store commands require both `--execute` and an exact
`--confirm RUN_ID`.

## Fresh stores and bound commands

Configure base admin URLs only in `DATABASE_URL`, `MOTHERDUCK_DATABASE_URL`,
and `NEON_DATABASE_URL`. The descriptor contains names and ownership facts,
never URLs or credentials.

## Secret-backed commands

Store the hosted URLs with `mise run set-secret-env -- MOTHERDUCK_DATABASE_URL`
and `mise run set-secret-env -- NEON_DATABASE_URL`. `mise` does not export
those values into the parent shell, so verify only their presence from its
environment without printing them:

```sh
mise exec -- sh -c 'test -n "$MOTHERDUCK_DATABASE_URL" && \
  test -n "$NEON_DATABASE_URL" && echo "cutover database URLs are set"'
```

Run every command that needs either URL through `mise exec --`, including
`whetstone-cutover stores ...`, worker, publication, and live-sweep commands:

```sh
mise exec -- whetstone-cutover stores verify --descriptor /absolute/operator/stores.json
mise exec -- uv run python -m whetstone.platform.worker serve --worker-concurrency 4
mise exec -- uv run whetstone-publish publish --destination /absolute/operator/analysis.duckdb
mise exec -- uv run whetstone-live-sweep status /absolute/operator/campaign \
  --ledger /absolute/operator/live-sweep.sqlite3
```

Running these commands directly produces misleading missing-variable errors,
even when the secrets were stored successfully.

When Python passes a SQLAlchemy `URL` to another connection API, preserve the
`URL` object or call `render_as_string(hide_password=False)` only at that
connection boundary. Never rebuild a reusable DSN with `str(engine.url)` or
`str(url)`: SQLAlchemy replaces the password with the literal `***`. Do not log
or persist the unmasked rendered value.

```sh
mise exec -- whetstone-cutover stores prepare --run-id acceptance_171 \
  --descriptor /absolute/operator/stores.json
mise exec -- whetstone-cutover stores prepare --run-id acceptance_171 \
  --descriptor /absolute/operator/stores.json \
  --execute --confirm acceptance_171
mise exec -- whetstone-cutover stores verify --descriptor /absolute/operator/stores.json
mise exec -- whetstone-cutover stores run --descriptor /absolute/operator/stores.json -- \
  uv run whetstone-live-sweep submit-canary \
  /private/tmp/platform-v6-live-sweep-161 --execute \
  --ledger /absolute/operator/live-sweep.sqlite3
mise exec -- whetstone-cutover stores run --descriptor /absolute/operator/stores.json -- \
  uv run whetstone-live-sweep submit-remaining \
  /private/tmp/platform-v6-live-sweep-161 --execute \
  --ledger /absolute/operator/live-sweep.sqlite3 --page-size 100
```

`submit-remaining` records one bounded page at a time. It requires a stable
terminal lifecycle (`succeeded`, `typed_failure`, or `incomplete`) for every
cell in that page before recording the next. Observed provider cost and tokens
are telemetry: missing cost is reported explicitly and never blocks dispatch.
It never creates canary intent: a fresh ledger starts only locked non-canary
shards. If `submit-canary` was interrupted after journaling its immutable
canary intent, `submit-remaining` may replay that exact locked pending canary
operation before continuing; it never synthesizes canary intent itself.

Run `submit-scoring` only after Generation reconciliation. The command rejects
until every locked shard relationship and member is present, terminal, and
fully reconciled. Its first successful invocation journals one complete
campaign cut, including the exact serialized scoring targets, before dispatch.
An interrupted `submitting` intent replays those stored targets; once marked
`submitted`, later invocations are no-ops and cannot derive another cut.

`relock-generation-shards` writes a content-addressed artifact set under
`generation-locks/`, fsyncs every file and directory, and then atomically
switches `generation-lock.json`. The pointer is authoritative after it exists;
an interrupted relock can be rerun deterministically without repairing a mixed
set of top-level legacy files.

Cleanup is fail-closed and ownership-checked:

```sh
mise exec -- whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json
mise exec -- whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json \
  --execute --confirm acceptance_171
mise exec -- whetstone-cutover stores verify-cleanup \
  --descriptor /absolute/operator/stores.json
```

If preparation crashes before the descriptor is committed, the journal already
contains the complete recovery descriptor. Cleanup and verification can use it
directly:

```sh
mise exec -- whetstone-cutover stores cleanup \
  --journal /absolute/operator/stores.json.journal.json \
  --execute --confirm acceptance_171
mise exec -- whetstone-cutover stores verify-cleanup \
  --journal /absolute/operator/stores.json.journal.json
```

Each schema and the DBOS SQLite database carries an immutable run ID and
descriptor-digest marker. Binding and cleanup verify those markers; a missing or
replacement marker stops the operation before any resource is deleted.

Hosted parity is the release gate. Record the returned run ID; do not select a
run by recency when concurrent dispatch is possible:

```sh
gh workflow run release-parity.yml --repo drothermel/unitbench --ref main
gh run watch RUN_ID --repo drothermel/unitbench --exit-status
```

Before Vercel production promotion, record the project ID, current production
deployment ID/URL, new reviewed deployment ID/URL, and fingerprints of both
`ANALYSIS_DATABASE_URL` and `DATABASE_URL`. Promotion and rollback remain owner
operations: rollback must restore both old store values and the old deployment
together; mixed Analysis/Detail generations are forbidden.
