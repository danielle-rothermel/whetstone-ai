# Remaining implementation intentions and open work

**Derived from:** `append-only-eval-records-design.md`, `platform-graph-workflow-implementation.md`, `repo-split-and-naming-plan.md`  
**Date:** 2026-06-30  
**Purpose:** Consolidated backlog of unfinished implementation, deferred decisions, follow-ups, and future work called out across platform docs.

---

## Implementation sequence — not yet complete

The stacked-PR sequence from the platform design doc. Items marked **partial** have first paths landed but gaps remain.

| Step | Scope | Status |
|---|---|---|
| 1 | Design doc as north star | **Done** |
| 2 | Core primitives (HumanEval, scoring, parsing, serialization, error classification) | **Partial** — persistable summaries exist; v1 score-attempt path landed |
| 3 | LM and prompt boundary (plain adapter, OpenRouter/OpenAI callers, tests) | **Partial** — see deferred LM items below |
| 4 | Pure graph execution core | **Done** (reusable runner, no DB/DBOS knowledge) |
| 5 | Archive v0 surfaces (old CLIs, manifests, repair, reporting, `experiments/`) | **Done** — runtime removed; see [`v0-migration-completion-checklist.md`](v0-migration-completion-checklist.md) |
| 6 | Domain contracts (graph specs, provider configs, outcomes, metrics, stable ids) | **Partial** — records exist; several contracts still string/metadata-based |
| 7 | Schema and migrations (SQLAlchemy Core + Alembic) | **Partial** — migration history frozen at `20260630_0005`; see [`v1-schema-migrations.md`](v1-schema-migrations.md) |
| 8 | Platform graph workflow (DBOS + append-only persistence) | **Done** (first path under `dr_dspy.platform`) |
| 9 | Batch submission, fairness, backoff | **Done** (chunked submit, fair-order enqueue, throttle table) |
| 10 | HumanEval scoring and metrics | **Partial** — scoring workflow + `humaneval@v1` landed; profile record tables deferred |
| 11 | Migration and validation (v0 backfill) | **Partial** — `migration/v0_reshape.py` + Tier 3.5 tests kept; backfill job and validation tooling **deferred** |
| 12 | Rescoring (workflow/CLI + projection movement) | **Partial** — `score-one` / `rescore` exist; projection movement **deferred** |
| 13 | Unitbench/export (Neon projections, generated TS types) | **Deferred** |

---

## Deferred platform phases (post–v1 core)

These belong to the same later phase as Unitbench: implement after the v1
generation/scoring path is code-complete and exercised on new experiments, not
before.

### v0 migration operations (step 11)

- **v0 backfill job/CLI** — read legacy prediction tables, call `reshape_v0_*`,
  bulk-insert v1 append-only rows.
- **Migration validation tooling** — row counts, artifacts, costs, legacy v0
  scores vs migrated/rescored outcomes.
- **Reshape hardening** (when backfill starts) — deterministic timestamps,
  enc-dec `PARTIAL` coverage, v0 score preservation for diffing, provider-axis
  fidelity.
- **Legacy import scope** — how much v0 inline score data to backfill versus
  validate only via rescoring.

### Analysis layer (step 12 projection + step 13 read path)

- **Projection movement command** — upsert selected generation/score attempts
  into `dr_dspy_prediction_projection` after validated score batches.
- **Projection storage shape** — physical table vs SQL view vs DuckDB/read-side
  projection (table exists today; contract not finalized).
- **Unitbench/export** — Neon publishing, stable projection views, generated
  TypeScript types, read-side query tooling.

---

## Open design questions

From the platform design doc — still unresolved or only partially resolved in code:

1. **Exact table names and primary keys** — implemented under `dr_dspy_*` names; migration history is frozen/deployed — see [`v1-schema-migrations.md`](v1-schema-migrations.md).
2. **Legacy import scope** — deferred with v0 backfill (see above).
3. **Projection storage shape** — deferred with analysis layer (see above).

**Resolved in implementation (no longer open):** node attempts grouped by `generation_run_id`; throttle coordination via `dr_dspy_throttle_backoff` when DBOS alone is insufficient.

---

## Platform workflow follow-ups

From `platform-graph-workflow-implementation.md`:

### Graph and provider contracts

- Replace prompt **metadata string keys** (`user_prompt_template`, `system_prompt`, `provider_config_id`) with **typed Pydantic fields** on graph/spec models (breaking contract change).
- **Extend persisted provider config** so experiments can vary runtime details from specs: `base_url`, `api_key_env`, temperature/reasoning capability flags, etc.
- Add a **supported spec-construction path** for v1 runs (CLI helper and/or standard integration-test fixture). Today specs must be created via tests, migration setup, ad-hoc insert, or `submit-jsonl`.

### Runtime and infrastructure

- Move **SQLAlchemy engine/pool ownership** into the platform worker runtime — stop creating short-lived engines inside each DBOS step.
- **DBOS queue-worker submit/resume E2E** — full path from enqueue through worker consumption; integration fixtures not yet standardized.

### Analysis and projections

- Projection movement and storage-shape decisions are **deferred** (see
  [Deferred platform phases](#deferred-platform-phases-postv1-core)).
- Terminal DBOS scoring failures that cannot replay into a persisted
  `ScoreAttemptRecord` may still need **manual DBOS admin** — v1 does not port
  v0 repair machinery.

### Execution fairness (optional enhancement)

- Stricter **multi-worker fairness policy** (queue or leasing design) if fair-prefix execution order matters more than throughput at concurrency > 1.

---

## Scoring and profile schema — later phases

- **First-class `ScoringProfileRecord` and `ParserProfileRecord` tables** — extraction rules, metric sets, and versioning semantics specified together (currently profile/parser stored as `(id, version)` references on score attempts only).
- **Graph layout closed enum** — remains a strict string until vocabulary is stable.
- **Compression metrics** as a typed metrics sub-shape inside HumanEval scoring profile (candidate, not finalized).
- **Profile/version ambiguity** — noted as deferred to scoring phase in review closure; may still need explicit resolution when profile tables land.

---

## LM and prompt boundary — deferred cleanup

From core-primitives / LM boundary implementation notes:

- **`LoggingOpenAILM` wrapper** — deferred until a caller needs direct OpenAI outside the graph-runner path.
- **Consolidate chat-style text extraction** between strict provider response parsing (typed failures) and `lm.utils.response_text()` (telemetry preview) if shapes drift.

---

## HumanEval and runner semantics — to decide

- **Stricter subprocess runner cardinality** — partial runner output is preserved today instead of whole-batch error; decide final behavior with per-test persistence and score-attempt semantics.
- **HumanEval domain module review** before schema freeze:
  - Split runtime AST from persistable summaries in `parsed_code.py`
  - Review `Any` fields and stable case ids in `parsed_tests.py`
  - Clarify naming/placement of `sampling.py` if useful
  - Fold `compression.py` into broader versioned metrics profile

---

## Metrics and analysis pipeline — future

- **External exploratory feature extraction** not in current scoring profiles: textdescriptives/spaCy, MinHash, tree-sitter, radon/lizard/complexipy, parquet, DuckDB — belongs to later analysis pipeline unless explicitly adopted into a future scoring profile.
- **Raw vs parsed artifacts** — out of scope for current platform phase; no decision to persist both shapes yet.

---

## v0 migration and cutover — deferred operations

v0 runtime code is **removed**. Reshape logic and frozen fixtures remain for when
backfill runs (see [`v0-migration-completion-checklist.md`](v0-migration-completion-checklist.md)).

Backfill, validation, rescoring on migrated rows, and migration-package deletion
are **deferred** with the analysis/Unitbench phase. Legacy v0 Postgres tables
may remain as backup until that work starts.

---

## Schema and deployment

- Unitbench/types/Neon/Drizzle items are **deferred** — see
  [Deferred platform phases](#deferred-platform-phases-postv1-core).
- v1 migration history is **frozen** — see [`v1-schema-migrations.md`](v1-schema-migrations.md).

---

## Testing gaps

| Gap | Notes |
|---|---|
| Full DBOS queue-worker submit/resume E2E | Follow-up after enqueue-to-worker fixtures standardized |
| Live Postgres/DBOS integration for scoring | Exists under `tests/integration/test_platform_scoring_*.py` (opt-in `@pytest.mark.integration`) |
| Projection movement | Deferred with analysis layer |

---

## Repository extraction and rename — not started

Blocked on landing **`graph-workflow` → `main`** first (one risky thing at a time).

After that, mechanical extraction runbook:

1. Land graph-workflow branch (CI, pinned `dspy==3.3.0b1`).
2. `git filter-repo --subdirectory-filter dr-dspy` on fresh clone → standalone repo with history preserved.
3. Remove `tool.uv.sources` workspace override and `scripts/ci/ensure_pypi_dspy.sh`; keep PyPI-pinned DSPy.
4. Create personal-org repo, push, wire **Depot + CI** at repo root (drop `working-directory: dr-dspy`).
5. **Rename** `dr_dspy` → `whetstone` in a separate dedicated commit.

Post-extraction housekeeping:

- README **future directions** for Cognee/KG, RL, agent-sandbox (not scaffolding).
- Reserve namespace on sibling packages (`kg/`, `rl/`, `agent/`) — do not build until scope expands.

---

## Defensive name reservations (optional)

Not blocking development; cheap insurance if the project grows:

1. **Cursor Origin** — waitlist; grab `whetstone-ai` if offered (GA fall 2026).
2. **Codeberg** — instant handle.
3. **Tangled** — instant handle.

Skip unless a service becomes actual home: GitLab, Bitbucket, Sourcehut, Radicle.

PyPI **`whetstone-ai`** and GitHub **`<personal-org>/whetstone-ai`** are the chosen identities; bare `whetstone` on PyPI is taken (stale placeholder).

---

## COPRO and next experiments

Explicit next use of the platform after cutover:

- **COPRO-oriented experiments** on the new graph/spec/outcome path with instruction mutation over encoder nodes.
- Optimizer reads from **projection** after score-attempt batches are validated and projection movement is implemented.

No implementation work for COPRO itself is documented in these three source files — only that the platform design targets it as the immediate post-migration consumer.

---

## v1 platform completion (before deferred phases)

Work required for a robust v1 path on **new** experiments (code complete; nothing
need be executed yet):

1. **Spec construction path** — CLI or config → `PredictionSpecRecord` generator
   (not only hand-built JSONL / test fixtures).
2. **PARTIAL run scoring policy** — decide and implement whether enc-dec/graph
   `PARTIAL` generation runs are scoreable; align `rescore` defaults and
   `validate_generation_run_for_scoring`.
3. ~~**Freeze v1 Alembic history**~~ — **Done** — see [`v1-schema-migrations.md`](v1-schema-migrations.md).
4. **Submit → worker E2E integration test** — enqueue through DBOS worker
   consumption.
5. **Small code cleanup** — remove dead `_provider_axis_from_row`; dedupe
   `failure_payload_from_exception`.
6. **`dspy` dependency review** — trim or dev-only now that v0 runtime is gone
   (serialization still references DSPy types).
7. **Stale docs** — align `completed-design-and-implementation-choices.md` and
   related notes with post–v0-removal layout (`dbos_bootstrap.py`, etc.).

---

## Summary priority sketch

**Now (v1 core):** items 1–7 in [v1 platform completion](#v1-platform-completion-before-deferred-phases).

**Later (deferred with Unitbench):** v0 backfill + validation + reshape
hardening; projection movement + storage shape; Unitbench/types/Neon; then delete
`migration/` after backfill sign-off.

**Follow-ups (explicitly deferred elsewhere in this doc):** provider config
extension, typed graph prompt fields, engine pooling, first-class profile tables,
optional multi-worker fairness, later analysis metrics, repo rename to whetstone.
