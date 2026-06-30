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
| 2 | Core primitives (HumanEval, scoring, parsing, serialization, error classification) | **Partial** — persistable summaries exist; v0 write paths still use legacy score columns |
| 3 | LM and prompt boundary (plain adapter, OpenRouter/OpenAI callers, tests) | **Partial** — see deferred LM items below |
| 4 | Pure graph execution core | **Done** (reusable runner, no DB/DBOS knowledge) |
| 5 | Archive v0 surfaces (old CLIs, manifests, repair, reporting, `experiments/`) | **Not done** |
| 6 | Domain contracts (graph specs, provider configs, outcomes, metrics, stable ids) | **Partial** — records exist; several contracts still string/metadata-based |
| 7 | Schema and migrations (SQLAlchemy Core + Alembic) | **Partial** — pre-deployment; draft migration history not frozen |
| 8 | Platform graph workflow (DBOS + append-only persistence) | **Done** (first path under `dr_dspy.platform`) |
| 9 | Batch submission, fairness, backoff | **Done** (chunked submit, fair-order enqueue, throttle table) |
| 10 | HumanEval scoring and metrics | **Partial** — scoring workflow + `humaneval@v1` landed; profile record tables and projection movement not |
| 11 | Migration and validation (v0 backfill) | **Not done** |
| 12 | Rescoring (workflow/CLI + projection movement) | **Partial** — `score-one` / `rescore` exist; projection movement command not |
| 13 | Unitbench/export (Neon projections, generated TS types) | **Not done** |

---

## Open design questions

From the platform design doc — still unresolved or only partially resolved in code:

1. **Exact table names and primary keys** — largely implemented under `dr_dspy_*` names, but migration history is not yet declared frozen/deployed.
2. **Legacy import scope** — how much v0/v1 data to backfill into append-only shape versus rescoring in place for short-term model selection.
3. **Projection storage shape** — physical table vs SQL view vs DuckDB/read-side projection first.

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
- Move **DBOS bootstrap** out of `dr_dspy.harness.dbos` into a shared runtime module (currently shared with v0 for coexistence).
- **DBOS queue-worker submit/resume E2E** — full path from enqueue through worker consumption; integration fixtures not yet standardized.

### Analysis and projections

- Add **explicit projection movement command** after live validation confirms score-attempt counts, failures, and model rankings/pass rates. Batch rescoring intentionally does not move projections today.
- Terminal DBOS scoring failures that cannot replay into a persisted `ScoreAttemptRecord` may still need **manual DBOS admin** — v1 does not port v0 repair machinery.

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
- **`lm.logging` import isolation** — recordability pulled at log time is intentional; pure boundary modules stay import-isolated, logging payload persistence is separate concern.
- v0 direct/enc-dec workflows still call **`dspy.Predict`** with ChatAdapter formatting — graph-runner stage should adopt caller-built messages via plain prompt path; **do not rewrite v0 experiments** in the same PR as LM boundary work.

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

## v0 migration and cutover — not started

Operational cutover steps still ahead:

1. **Freeze v0** for new experimental writes (policy/process, not just code).
2. **Backfill** existing v0 direct and enc-dec rows into append-only model (`migration/v0_reshape.py` exists for Tier 3.5 tests; full backfill job not done).
3. **Validate** migrated counts, artifacts, costs, legacy scores, and projections.
4. **Rescore** migrated terminal artifacts under new parser/scoring profiles.
5. **Run next COPRO experiments** on new path only.
6. **Retire v0 as active path** while keeping tables as backup until trusted.

Also: **archive v0 surfaces** (step 5 in implementation sequence) — move or clearly mark legacy CLIs, manifests, repair flow, old reporting, and `experiments/` implementation details.

Prompt for archiving work exists at `docs/prompts/archive-v0-surfaces.md` in the design sequence (file may need to be (re)created when that step starts).

---

## Schema and deployment

- **Declare v1 migration history frozen/deployed** and document upgrade path from any draft schemas applied locally or on Neon.
- **Reset guidance** for databases that applied earlier draft `20260629_0001` (batch item status shape change: single `status` → `insert_status` + `enqueue_status`).
- **Generated TypeScript types** for Unitbench from canonical Python contracts or DB introspection.
- **Unitbench Neon access** — stable projection tables/views for read-only server-side `SELECT`; no second schema owner.
- Consider **Drizzle** later as read-side query builder if Unitbench queries become painful (generated from canonical schema only).

---

## Testing gaps

| Gap | Notes |
|---|---|
| Full DBOS queue-worker submit/resume E2E | Follow-up after enqueue-to-worker fixtures standardized |
| Live Postgres/DBOS integration for scoring | Exists under `tests/integration/test_platform_scoring_*.py` (opt-in `@pytest.mark.integration`) |
| Projection movement | No command or integration coverage yet |

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

## Summary priority sketch

Suggested ordering implied across docs (not a new decision — synthesis of existing callouts):

1. Freeze v1 migrations + finish schema hardening
2. v0 backfill + validation + rescoring on migrated artifacts
3. Projection movement command + Unitbench/types
4. Archive v0 surfaces; adopt plain prompt path everywhere on v1
5. Provider config contract extension + typed graph prompt fields
6. Worker runtime pooling + DBOS bootstrap consolidation + E2E submit/worker tests
7. Land graph-workflow → extract repo → rename to whetstone
8. First-class scoring/profile tables; optional multi-worker fairness; later analysis pipeline metrics
