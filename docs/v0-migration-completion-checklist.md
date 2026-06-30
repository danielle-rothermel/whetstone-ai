# v0 migration completion checklist

**Purpose:** Define what was removed when v0 runtime code was archived, what remains for backfill, and the exact cleanup required after live migration is validated.

**Related:** [`remaining-implementation-intentions.md`](remaining-implementation-intentions.md), [`TESTING.md`](../TESTING.md) Tier 3.5

---

## What the v0 archive PR removed

The following v0 **runtime/orchestration** code was deleted from the repository:

| Removed | Notes |
|---|---|
| `src/dr_dspy/experiments/` | Direct and enc-dec HumanEval DBOS backends, inline DDL, Typer CLIs |
| `src/dr_dspy/harness/` | Batch ops, repair, reporting, status, worker monitor, v0 queue naming |
| `src/dr_dspy/lm/runner.py`, `signatures.py`, `openrouter.py`, `logging.py` | DSPy `Predict` / ChatAdapter path |
| `src/dr_dspy/runtime.py` | Replaced by `platform/cli_env.py` |
| `scripts/humaneval_dspy_eval_only_dbos_v0.py` | Direct v0 CLI entrypoint |
| `scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py` | Enc-dec v0 CLI entrypoint |
| `scripts/reclassify_encdec_generation_http_errors.py` | One-off v0 row repair script |

v1 platform code now bootstraps DBOS through:

- [`src/dr_dspy/platform/dbos_bootstrap.py`](../src/dr_dspy/platform/dbos_bootstrap.py)
- [`src/dr_dspy/platform/dbos_compat.py`](../src/dr_dspy/platform/dbos_compat.py)
- [`src/dr_dspy/platform/cli_env.py`](../src/dr_dspy/platform/cli_env.py)

**Not removed (Postgres data):** legacy mutable prediction tables (`dr_dspy_eval_predictions`, `dr_dspy_encdec_eval_predictions`, `dr_dspy_batch_operations`, etc.) may still exist in databases as read-only backup until explicitly dropped.

---

## What remains for backfill (keep until migration validated)

| Path | Role |
|---|---|
| [`src/dr_dspy/migration/v0_reshape.py`](../src/dr_dspy/migration/v0_reshape.py) | Reshape legacy row dicts → v1 `PredictionSpecRecord`, generation runs, node attempts |
| [`tests/fixtures/v0_samples/*.json`](../tests/fixtures/v0_samples/) | Frozen legacy row contracts for tests |
| [`tests/test_v0_reshape.py`](../tests/test_v0_reshape.py) | Unit smoke |
| [`tests/integration/test_v0_reshape_*.py`](../tests/integration/) | Integration smoke |
| [`tests/integration/v0_sample_loader.py`](../tests/integration/v0_sample_loader.py) | Fixture loader |

These modules import **only v1 packages** (`graph/`, `records/`, `lm/boundary.py`, etc.) — no deleted v0 orchestration code.

**Still TODO (separate from this checklist):** a full backfill **job/CLI** that reads live v0 tables, calls `reshape_v0_*`, and inserts v1 append-only rows. Tier 3.5 tests prove reshape correctness on frozen samples only.

---

## Backfill prerequisites (before post-backfill cleanup)

Operational steps from the platform cutover plan:

1. **Freeze v0 writes** — policy/process; no new rows in mutable prediction tables.
2. **Run full backfill** — all terminal v0 direct and enc-dec rows reshaped into v1 append-only tables.
3. **Validate** — row counts, artifacts, costs, legacy inline scores vs migrated outcomes.
4. **Rescore** — migrated terminal artifacts as new `ScoreAttemptRecord` rows under `humaneval@v1`.
5. **Confirm v1-only experiments** — next COPRO / model-selection work uses `platform/` path only.

---

## Post-backfill deletion checklist

After steps 1–5 above pass review, open a **second cleanup PR** and delete:

### Code and tests

- [ ] `src/dr_dspy/migration/` (entire package)
- [ ] `tests/fixtures/v0_samples/` (entire directory)
- [ ] `tests/test_v0_reshape.py`
- [ ] `tests/integration/test_v0_reshape_outcomes.py`
- [ ] `tests/integration/test_v0_reshape_specs.py`
- [ ] `tests/integration/v0_sample_loader.py`

### Documentation

- [ ] Remove Tier 3.5 from [`TESTING.md`](../TESTING.md) and update CI test counts
- [ ] Remove backfill/migration sections from [`README.md`](../README.md) that reference retained reshape code
- [ ] Mark step 11 (migration and validation) **Done** in [`remaining-implementation-intentions.md`](remaining-implementation-intentions.md)
- [ ] Archive or delete this checklist (or mark it historical)

### Optional data housekeeping (not code)

- [ ] Drop or rename legacy v0 Postgres tables after backup export
- [x] Document final v1 migration revision as frozen/deployed — see [`v1-schema-migrations.md`](v1-schema-migrations.md) (separate from post-backfill package deletion)

---

## Definition of done: “v0 migration complete”

Migration is **complete** when all of the following are true:

1. No production or experimental writes target v0 mutable prediction tables.
2. All required v0 rows are represented in v1 append-only tables (via backfill + validation sign-off).
3. Rescored `ScoreAttemptRecord` batches are trusted for model ranking / pass-rate analysis.
4. The migration package and v0 sample fixtures listed above are deleted (second cleanup PR merged).
5. New experiment work runs exclusively through `dr_dspy.platform` commands and v1 schema.

**Explicitly still out of scope for “migration complete”** (separate backlog):

- Projection movement command
- Unitbench / generated TypeScript types
- First-class scoring/profile record tables
- Repo extraction / `dr_dspy` → `whetstone` rename
