# Cutover operator tooling

`whetstone-cutover` is dry-run by default and never dispatches work. Mutating
store commands require both `--execute` and an exact
`--confirm RUN_ID`.

## Fresh stores and bound commands

Configure base admin URLs only in `DATABASE_URL`, `MOTHERDUCK_DATABASE_URL`,
and `NEON_DATABASE_URL`. The descriptor contains names and ownership facts,
never URLs or credentials.

```sh
whetstone-cutover stores prepare --run-id acceptance_171 \
  --descriptor /absolute/operator/stores.json
whetstone-cutover stores prepare --run-id acceptance_171 \
  --descriptor /absolute/operator/stores.json \
  --execute --confirm acceptance_171
whetstone-cutover stores verify --descriptor /absolute/operator/stores.json
whetstone-cutover stores run --descriptor /absolute/operator/stores.json -- \
  uv run whetstone-live-sweep submit-canary \
  /private/tmp/platform-v6-live-sweep-161 --execute \
  --ledger /absolute/operator/live-sweep.sqlite3
whetstone-cutover stores run --descriptor /absolute/operator/stores.json -- \
  uv run whetstone-live-sweep submit-remaining \
  /private/tmp/platform-v6-live-sweep-161 --execute \
  --ledger /absolute/operator/live-sweep.sqlite3 --page-size 100
```

`submit-remaining` records one bounded page at a time. It requires a stable
terminal lifecycle (`succeeded`, `typed_failure`, or `incomplete`) for every
cell in that page before recording the next. Observed provider cost and tokens
are telemetry: missing cost is reported explicitly and never blocks dispatch.

Cleanup is fail-closed and ownership-checked:

```sh
whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json
whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json \
  --execute --confirm acceptance_171
whetstone-cutover stores verify-cleanup \
  --descriptor /absolute/operator/stores.json
```

If preparation crashes before the descriptor is committed, the journal already
contains the complete recovery descriptor. Cleanup and verification can use it
directly:

```sh
whetstone-cutover stores cleanup \
  --journal /absolute/operator/stores.json.journal.json \
  --execute --confirm acceptance_171
whetstone-cutover stores verify-cleanup \
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
