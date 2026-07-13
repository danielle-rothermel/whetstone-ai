# Manual release recovery runbook

Automated progress-aware recovery (journals with per-boundary recovery states,
atomic checkpoints, ownership-proof deletion — closed PR #36) was descoped by
owner decision D1 in dr-platform
`docs/implementation/platform-whetstone-v6/descope-cleanup.md`. ADR 0022 pins
rollback to source control plus fresh schemas, so **manual recovery is the
supported path**. A failed release run is never repaired in place: remove its
run-owned resources, then rerun with a fresh store set. Everything here is
fail-closed — when ownership is uncertain, stop and investigate before
deleting anything.

## What a release run owns

`whetstone-cutover stores prepare --run-id <run_id>` creates, in order:

| Resource | Where | Name |
| --- | --- | --- |
| Source schema | `DATABASE_URL` | `whetstone_run_<run_id>` |
| Analysis schema | `MOTHERDUCK_DATABASE_URL` | `whetstone_analysis_<run_id>` |
| Detail schema | `NEON_DATABASE_URL` | `whetstone_detail_<run_id>` |
| DBOS store | local file | `<run_id>-dbos.sqlite3` next to the descriptor |
| Descriptor + journal | local files | `stores.json`, `stores.json.journal.json` |
| Publication bundles | local files | the `whetstone-publish publish --destination` output (and `--detail-destination` if used) |

Every schema and the DBOS store carries an immutable
`whetstone_cutover_ownership` marker recording the run ID and descriptor
digest. The journal is written **before** any remote mutation, so it exists
for every failure mode, including a crash before the descriptor was committed.

## Recovery procedure

Stop workers and any in-flight `stores run` commands first. All commands need
the store URLs, so run them through `mise exec --` (secrets are stored with
`mise run set-secret-env -- NEON_DATABASE_URL` and
`mise run set-secret-env -- MOTHERDUCK_DATABASE_URL`; direct invocation
produces misleading missing-variable errors).

### 1. Journaled cleanup (first resort — covers every boundary)

`stores cleanup` already handles a crash at any point of prepare/run: it
preflights ownership markers on every existing resource and refuses to drop
anything it cannot prove this run owns.

```sh
mise exec -- whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json
mise exec -- whetstone-cutover stores cleanup --descriptor /absolute/operator/stores.json \
  --execute --confirm <run_id>
mise exec -- whetstone-cutover stores verify-cleanup --descriptor /absolute/operator/stores.json
```

If prepare crashed before committing the descriptor, use the journal directly
(`--journal /absolute/operator/stores.json.journal.json` in place of
`--descriptor`).

### 2. Manual drops (only when cleanup fails closed)

Cleanup stops with "schema ownership marker disagrees" or "DBOS ownership
marker is unreadable" when a marker is missing or was replaced. That is the
signal a resource may not belong to this run. For each boundary, read the
marker yourself, compare it to `run_id` and `descriptor_sha256` in the
journal, and drop only on an exact match:

```sql
SELECT run_id, descriptor_sha256
FROM "whetstone_run_<run_id>".whetstone_cutover_ownership;
DROP SCHEMA "whetstone_run_<run_id>" CASCADE;
```

Boundary constraints (these are why the tooling is shaped the way it is):

- **Source (`DATABASE_URL`) and Neon (`NEON_DATABASE_URL`)**: the marker row
  is protected by an immutability trigger — drop the schema, never try to
  delete the marker row. On Neon use the **direct (unpooled) endpoint**: the
  tooling binds schemas via the `options` startup parameter
  (`-c search_path=...`), which Neon's pooled PgBouncer endpoint does not
  accept. Neon's schema names are `whetstone_detail_<run_id>`.
- **MotherDuck (`MOTHERDUCK_DATABASE_URL`)**: its Postgres endpoint rejects
  startup `search_path`, so always schema-qualify names; it reports **every**
  error as SQLSTATE `XXUUU`, so read the message text, not the code; and the
  marker table there has no immutability trigger — a matching marker is
  weaker evidence, so double-check the schema name before
  `DROP SCHEMA "whetstone_analysis_<run_id>" CASCADE`.
- **DBOS store**: check the marker
  (`sqlite3 <run_id>-dbos.sqlite3 'SELECT * FROM whetstone_cutover_ownership'`)
  then delete the file.

### 3. Publication bundle

If `whetstone-publish publish` failed mid-run, delete the files it wrote at
`--destination` (and `--detail-destination`). Bundles are integrity-signed, so
a partial bundle fails closed at consumers — deletion is hygiene, not a
safety requirement. Vercel promotion/rollback after publication remains an
owner operation; see `docs/cutover-operator-tooling.md`.

### 4. Rerun fresh

Prefer a **new run ID and a new descriptor path**: `stores prepare` refuses
schema collisions and existing descriptor/journal files, so reusing the old
identity requires complete verified cleanup plus removing (or archiving) the
old descriptor and journal.

```sh
mise exec -- whetstone-cutover stores prepare --run-id <new_run_id> \
  --descriptor /absolute/operator/stores-<new_run_id>.json \
  --execute --confirm <new_run_id>
mise exec -- whetstone-cutover stores verify --descriptor /absolute/operator/stores-<new_run_id>.json
mise exec -- whetstone-cutover stores run --descriptor /absolute/operator/stores-<new_run_id>.json -- <command>
```

Nothing from the failed run is carried forward. Rollback is source control
plus fresh schemas (ADR 0022) — not runtime repair.
