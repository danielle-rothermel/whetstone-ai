# v1 schema migrations (frozen)

**Current status:** Frozen schema reference. During the June 30 eval push, do
not change migration history or pursue schema cleanup unless it directly blocks
backfill, rescoring, model selection, the enc-dec budget sweep, or the minimal
COPRO-style experiment loop. Use [`../AGENTS.md`](../AGENTS.md) for active
priorities.

**Purpose:** Single source of truth for the v1 Alembic revision chain, fresh-database setup, and reset procedure for databases that applied draft schemas during hardening.

**Code anchor:** [`src/dr_dspy/db/migrations/head.py`](../src/dr_dspy/db/migrations/head.py) — `V1_MIGRATION_HEAD`, `V1_MIGRATION_BASE`, `V1_MIGRATION_REVISION_COUNT`.

**Schema authority:** [`src/dr_dspy/db/schema.py`](../src/dr_dspy/db/schema.py) (SQLAlchemy Core) plus Alembic revisions under [`src/dr_dspy/db/migrations/versions/`](../src/dr_dspy/db/migrations/versions/). Parity is enforced by [`tests/test_db_migrations.py`](../tests/test_db_migrations.py).

---

## Frozen head

| Constant | Value |
|---|---|
| `V1_MIGRATION_HEAD` | `20260630_0005` |
| `V1_MIGRATION_BASE` | `20260629_0001` |
| `V1_MIGRATION_REVISION_COUNT` | 9 |

The chain is **linear** (no branches). Future schema changes require **new forward revisions** only.

---

## Revision changelog

| Revision | Summary |
|---|---|
| `20260629_0001` | v1 domain schema — nine `dr_dspy_*` tables; batch items use `insert_status` + `enqueue_status` |
| `20260629_0002` | `dr_dspy_throttle_backoff` table |
| `20260629_0003` | `already_scheduled_count` on batch submit operations |
| `20260629_0004` | `enqueuing` operation status (alongside `prepared`) |
| `20260630_0001` | Append-only triggers on generation runs, node attempts, and score attempts |
| `20260630_0002` | Terminal enqueue accounting check constraints |
| `20260630_0003` | `claiming` enqueue status; heal stale pending metadata |
| `20260630_0004` | Remove `prepared` operation status (backfill rows to `enqueuing`) |
| `20260630_0005` | `dataset_name` / `dataset_split` on score attempts; profile uniqueness |

---

## Fresh database

On a database with no v1 platform tables applied yet:

```bash
uv run alembic upgrade head
uv run alembic current   # expect 20260630_0005
```

Connection config, driver normalization, and offline SQL rendering are documented in [README § Database migrations](../README.md#database-migrations).

---

## Reset path for draft v1 schemas

**When to reset:** A database applied an **unlisted draft** revision set during branch hardening — for example:

- Batch submit items with a single `status` column instead of `insert_status` + `enqueue_status`
- Batch operations missing `already_scheduled_count` or `enqueuing` status
- Outcome tables without append-only triggers
- Score attempts without `dataset_name` / `dataset_split`

There is **no supported upgrade path** from unlisted draft shapes. Drop v1 platform objects, reset Alembic, and replay from the frozen chain.

**Warning:** This is **destructive for all v1 platform data** in the listed tables. Legacy v0 Postgres tables (`dr_dspy_eval_predictions`, `dr_dspy_encdec_eval_predictions`, etc.) are separate and are not dropped by this procedure.

### 1. Drop v1 platform objects

Run against the target database (adjust `DATABASE_URL` as needed):

```sql
-- Drop v1 platform tables (CASCADE clears FKs and triggers on them)
DROP TABLE IF EXISTS dr_dspy_batch_submit_items CASCADE;
DROP TABLE IF EXISTS dr_dspy_batch_submit_operations CASCADE;
DROP TABLE IF EXISTS dr_dspy_throttle_backoff CASCADE;
DROP TABLE IF EXISTS dr_dspy_prediction_projection CASCADE;
DROP TABLE IF EXISTS dr_dspy_score_attempts CASCADE;
DROP TABLE IF EXISTS dr_dspy_node_attempts CASCADE;
DROP TABLE IF EXISTS dr_dspy_generation_runs CASCADE;
DROP TABLE IF EXISTS dr_dspy_prediction_specs CASCADE;
DROP TABLE IF EXISTS dr_dspy_experiments CASCADE;

-- Orphan append-only guard function (may survive table drops)
DROP FUNCTION IF EXISTS dr_dspy_reject_append_only_outcome_mutation() CASCADE;
```

These names match [`V1_TABLE_NAMES`](../src/dr_dspy/db/schema.py) and [`APPEND_ONLY_OUTCOME_REJECT_FUNCTION`](../src/dr_dspy/db/schema.py).

### 2. Reset Alembic version

Either stamp to base:

```bash
uv run alembic stamp base
```

Or delete the version row directly:

```sql
DELETE FROM alembic_version;
```

### 3. Replay migrations

```bash
uv run alembic upgrade head
uv run alembic current   # expect 20260630_0005
```

---

## Post-freeze policy

1. **Head revision is canonical** — `20260630_0005` until a new forward revision lands.
2. **Do not edit existing revision files** after freeze, except unavoidable typo fixes that would invalidate already-applied databases (avoid if possible).
3. **Future schema changes** — add new revisions only; never rewrite history.
4. **Single linear chain** — one head, one base; tests assert this in `test_alembic_v1_migration_chain_is_linear`.

---

## What is not promised

- Upgrade from unlisted draft schema shapes — **reset and replay** instead.
- Automatic migration of v0 legacy tables — out of scope; see [`v0-migration-completion-checklist.md`](v0-migration-completion-checklist.md).
- Unitbench / Neon projection publishing — deferred.
