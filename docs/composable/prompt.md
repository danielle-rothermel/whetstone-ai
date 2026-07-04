# Migration Execution Brief

You are executing the composable extraction of whetstone-ai into five
libraries, in sequenced, individually-tested stages. This file is your
task definition. The design is already settled — read these before
writing any code, in this order:

1. `overall.md` — target shape, sequencing, house rules, resolved
   decisions
2. `serialize.md`, `code_parse_test.md`, `llm_provider.md`,
   `graph_runner.md`, `platform.md` — one per library
3. Whetstone's `README.md`, `TESTING.md`, and
   `docs/remaining-implementation-intentions.md` (rename section)

Do not re-litigate settled decisions. Where a doc marks something as an
open section, make the smallest conservative choice that preserves current
behavior, and record what you chose and why in that doc.

## Loop protocol (read first, every iteration)

You are running in a loop: each iteration starts with fresh context and
this same prompt. All state lives on disk and in git, not in your memory.

1. **Re-derive state.** Read `docs/composable/migration_log.md` (created
   in Stage 0), then verify it against reality: branch and `git log` of
   each repo, existence of the new repos. If the log and the repos
   disagree, trust the repos and correct the log first.
2. **Advance one bounded increment.** Complete the current stage, or one
   coherent sub-step of it — never more than one stage per iteration.
   Prefer finishing and verifying something small over starting something
   large.
3. **Leave clean state.** Commit and push all work, update the log's
   status table and append an entry, before ending the iteration. Never
   end with uncommitted changes; a half-done sub-step must be re-derivable
   from git by the next iteration.
4. **Thrash guard.** If the same acceptance check fails in two consecutive
   iterations with substantially the same error, mark the stage
   `blocked: <one-line diagnosis>` in the log and do not attempt it again;
   proceed only if a later stage is genuinely independent, otherwise halt.
   A human clears `blocked`.
5. **Termination.** When every stage is `done`, make your final message
   exactly `MIGRATION COMPLETE`. If blocked with no independent work
   remaining, end with `MIGRATION BLOCKED: <stage list>`. (These lines are
   the loop's stop signal.)

`migration_log.md` format — status table first, then append-only entries:

```markdown
# Migration Log
## Status
| stage | state | notes |
|-------|-------|-------|
| 0 baselines | done | ... |
| 1 rename | in_progress | ... |
...
## Environment
gh auth: yes/no · postgres: yes/no · keys: OPENROUTER/OPENAI/GEMINI y/n
## Entries
### <date> — stage N
what landed · what was verified (commands + results) · skips · choices
```

## Repos

| Repo | Location | Status |
|------|----------|--------|
| whetstone-ai | this repo | exists (currently `dr-dspy` package) |
| dr-serialize | `../dr-serialize` | **create** (private GitHub repo) |
| dr-graph | `../dr-graph` | **create** (private GitHub repo) |
| dr-providers | `../dr-providers` | exists |
| dr-code | `../dr-code` | exists |

## Ground rules

- **Branch discipline.** Never commit to `main` of any existing repo.
  Create a `composable-migration` branch from each repo's default branch
  (new repos: scaffold on `main`, push, then branch the same way). Push
  branches to origin; open a draft PR per repo once its first stage
  lands.
- **Frozen contracts are sacred.** Graph digests, `prediction_id` /
  `generation_run_id` / score-attempt ID axes, parser/scoring profile
  IDs, Alembic revision IDs, DBOS queue/workflow/step name strings, and
  DB table names must be byte-for-byte identical after every stage.
  Golden tests (below) enforce this; if a golden test fails and you
  cannot restore identity, stop that stage and write up the discrepancy
  instead of adjusting the golden values.
- **No shims.** Each cutover is a clean break: delete the extracted code
  from whetstone, swap imports, update callers/tests/docs in the same
  change. No re-export layers, no compatibility aliases.
- **No paid API calls** unless the needed key is present in the
  environment — and then only the minimal smoke calls listed in a stage.
  All required verification must pass without network access.
- **Dependency wiring during migration.** Consume the new packages from
  whetstone via `[tool.uv.sources]` path dependencies
  (`{ path = "../dr-serialize", editable = true }` etc.). Leave a note in
  each PR that these must become git/PyPI pins before merge.
- **Scaffolding conventions for new repos.** Match dr-providers: `uv`
  project, `src/` layout, Python ≥3.12, `ruff` + `ty` + `pytest` +
  `pre-commit`, README, `py.typed`. Use `gh repo create <name> --private`.
- **Every operation idempotent.** Any step may run twice across
  iterations: check-then-create for repos (`gh repo view || gh repo
  create`), branches (`git switch <b> || git switch -c <b>`), files, and
  scaffolding. Re-running a completed stage's acceptance checks must be
  harmless.
- **Per-stage definition of done.** (a) the library's own test suite
  passes; (b) whetstone's full unit suite passes after the cutover
  (`uv run pytest`); (c) integration-marked tests pass if Postgres is
  available (see `TESTING.md`), otherwise record that they were skipped;
  (d) golden/identity tests pass; (e) work committed and pushed with a
  stage-labeled message.
- **Record as you go.** Append a short entry per stage to
  `docs/composable/migration_log.md` (create it): what landed, what was
  verified and how, what was skipped, any conservative choices made on
  open sections.

## Stage 0 — Golden baselines (before touching anything)

In whetstone as it stands, generate and commit fixture files capturing
current identity outputs: canonical-JSON strings and digests for a
spread of values (`hashing.py`), graph digests for `direct_graph()`,
`encdec_graph()`, and `humaneval_encdec_graph()`, prediction/generation/
node/score-attempt IDs for fixed axis inputs (`records/hashing.py`), and
parser/scoring outputs for a sample of stored generations under the v1
profiles. Add a pytest module that asserts current code reproduces the
fixtures, so "goldens pass" is one command forever after. These fixtures
are the acceptance oracle for every later stage. Commit them on the
migration branch.

Also run the environment preflight once and record the matrix in the
log: `gh auth status`, Postgres reachability (`psql "$DATABASE_URL" -c
'select 1'` or equivalent), which API keys are present (never print
values), `uv --version`.

**Accept:** golden pytest module green (`uv run pytest -k golden`);
full suite green (`uv run pytest`); log initialized with status table +
environment matrix; committed and pushed.

## Stage 1 — Rename `dr_dspy` → `whetstone` (whetstone-ai only)

Python package and repo-internal naming only: package directory, all
imports, `pyproject.toml` name, ruff/ty config, README/docs references,
test imports. **Do not change** persisted string constants: DBOS
queue/workflow/step names (`dr-dspy-platform-generation-v1`,
`dr_dspy_platform_*` step names), profile IDs, Alembic revision IDs and
`alembic.ini`, table names, or the default database name in URLs.

**Accept:** `src/whetstone/` exists and `src/dr_dspy/` does not;
`grep -rn "dr_dspy" src tests` matches **only** the frozen string
constants (queue/workflow/step names, DB defaults) — and, positively,
`grep -rn "dr-dspy-platform-generation-v1" src` still matches; full
suite green; golden fixtures reproduce.

## Stage 2 — dr-serialize (per `serialize.md`)

Create the repo; extract `hashing.py` verbatim and the serialization
engine with the handler-registration API and `SerializationLimits`
config (Postgres preset). Port the relevant whetstone unit tests; add
golden tests from Stage 0 fixtures. Cutover: whetstone deletes both
modules, imports dr-serialize, registers its DSPy handlers at package
import, keeps `records/hashing.py` and `eval_failures/recording.py`
app-side. `SANITIZE_KEYS` stays in whetstone's `lm/` until Stage 4 moves
it to dr-providers.

**Accept:** dr-serialize suite green in its repo; whetstone's
`hashing.py`/`serialization.py` deleted with no remaining imports of
them; whetstone full suite + goldens green (digests now produced via
dr-serialize); both repos committed and pushed.

## Stage 3 — dr-code nucleus (per `code_parse_test.md`, steps 1–2 only)

On the dr-code migration branch: prune to the nucleus (delete
`pipeline/`, `generation/`; absorb the load-bearing code-eval core), then
port whetstone's parsing/scoring behavior **byte-identically under the
existing v1 profile IDs**, with golden tests from Stage 0 fixtures and
the code-eval corruption corpus as the regression suite. Cutover:
whetstone's `humaneval/` package is replaced by the dependency plus a
thin app-side module for anything whetstone-specific the doc keeps
(e.g. the `recordable_text` injection point). The doc's improvement
steps 3–4 (markdown container experiment, profile v2, fork sandbox) are
**out of scope** for this migration — do not do them.

**Accept:** dr-code suite + corruption-corpus regression suite green;
whetstone's `humaneval/` package deleted; parser/scoring golden fixtures
byte-equal under the existing v1 profile IDs; whetstone full suite
green.

## Stage 4 — dr-providers v0.2 (per `llm_provider.md`)

On the dr-providers migration branch, breaking release: config records
with endpoint kinds / reasoning shapes / token-limit params; OpenAI and
Gemini presets (Gemini via Google's OpenAI-compatible endpoint,
`GEMINI_API_KEY` — decided: AI Studio); raw-httpx transport with opt-in
retry; `FailureClass` + failure-record home; public `build_payload` and
payload-on-response; conformance module; `TokenUsage`/`CostInfo`;
warnings channel; continuation handle field; **`FixtureProvider`** as
public API; `SANITIZE_KEYS` moves in. Grow the audit corpus with
whetstone's real response fixtures; all parsers corpus-tested. Cutover:
whetstone's `lm/boundary.py` becomes the thin adapter, `eval_failures`
keeps only domain exceptions and the psycopg/DBOS heuristics, and
`node_attempt_id` is passed as the idempotency key. Optional (only if
keys are present): one live smoke call per configured provider.

**Accept:** dr-providers suite green including corpus-backed parser
tests; a whetstone test exercises a node execution end-to-end against
`FixtureProvider` with no network; whetstone's `eval_failures/policy.py`
no longer references `openai`/`httpx` module heuristics; whetstone full
suite + goldens green.

## Stage 5 — dr-graph (per `graph_runner.md`)

Create the repo; extract `graph/` with the v1 generalizations
(open-string ops/field types, parameterized reserved namespace with
`"task"` default, structural `ClassifiedFailure` protocol, neutral
builders, subgraph composition, `completed`-nodes hook). **Digest golden
tests are the gate**: Stage 0 graph digests must reproduce exactly.
Cutover: whetstone import swap; `spec_builder.py`, `prompts.py`, and the
DBOS `run_node` wrapping stay.

**Accept:** dr-graph suite green; Stage 0 graph digests byte-equal when
computed through dr-graph; whetstone's `graph/` package deleted;
whetstone full suite + goldens green.

## Stage 6 — Platform library (per `platform.md`) — gated

This doc has the most open sections. Before writing code: fill in the
open sections (package name, protocols, schema ownership,
projection/artifact API sketches, cutover plan) as a design commit, and
write the two consumer sketches (nl_latents loop, optimizer population
evaluation) against the proposed facade. **If any design choice feels
underdetermined by the docs, stop this stage and leave the written
proposal for review instead of guessing.** If the design completes
cleanly: create the repo, extract the generic layer behind the
`SubmittableItem` protocol, library-owned tables with their own Alembic
lineage, and cut whetstone over.

**Accept (design half):** `platform.md` open sections filled + both
consumer sketches committed. **Accept (extraction half, only if
un-gated):** platform suite green; whetstone integration tier green
against Postgres; whetstone full suite + goldens green.

## Final — End-to-end verification

In whetstone with all landed stages: full unit suite; integration tier
against Postgres if available (`TESTING.md`); then a zero-spend E2E
smoke — build specs from `configs/experiments/humaneval_encdec_smoke.json`,
route them through a provider config backed by dr-providers'
`FixtureProvider`, run `submit-jsonl` + worker + `rescore` against a
scratch database, and confirm append-only outcomes and scores land.
Record the E2E evidence (commands, counts, IDs) in the migration log.
If Postgres is unavailable, state exactly which verifications could not
run.

**Accept:** unit suite green; integration tier green or explicitly
recorded as unavailable; E2E smoke shows expected spec/outcome/score
counts in the log; all branches pushed with draft PRs open; every stage
row in the status table reads `done` (or `blocked` with diagnosis).

## Escalation / stop conditions

Stop the current stage (finish the log entry, push what passes) rather
than improvise when: a golden/identity test cannot be made to pass
without changing frozen values; a design question isn't answered by the
docs and the conservative choice isn't obvious; anything would require
force-pushing, deleting a repo, touching `main`, or spending money
beyond the listed smoke calls. Completed stages are independently
valuable — a clean stop after stage N is success, not failure.
