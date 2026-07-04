# Migration Log

## Status

| stage | state | notes |
|-------|-------|-------|
| 0 baselines | done | golden fixtures + tests committed; full suite 696 passed; integration 45 passed |
| 1 rename | done | src/whetstone; pyproject whetstone-ai; frozen strings intact; 696 unit + 45 integration + goldens green |
| 2 dr-serialize | done | repo created + cutover; dr-serialize 64 tests; whetstone 666 unit + 45 integration + goldens green |
| 3 dr-code nucleus | in_progress | 3a + 3b done (port green: 309 tests, goldens byte-equal, corpus baseline pinned); next: 3c whetstone cutover |
| 4 dr-providers v0.2 | pending | repo exists at ../dr-providers |
| 5 dr-graph | pending | repo not yet created |
| 6 platform | pending | gated on design completion |
| final e2e | pending | |

## Environment

gh auth: yes · postgres: yes · keys: OPENROUTER y / OPENAI y / GEMINI y

- uv 0.11.25; Python per `.python-version` (3.13 venv).
- Postgres: local socket, no `postgres` role. Dev DB from `.env`
  (`DATABASE_URL`, socket URL to db `dr_dspy`). Integration tier runs with
  `DATABASE_URL="postgresql:///dr_dspy_test" uv run pytest -m integration
  tests/integration/ -q` (created `dr_dspy_test` via `createdb`; the CI
  default `postgres:postgres@localhost` role does not exist locally).
- OPENROUTER key in shell env; OPENAI + GEMINI keys added to `.env` on
  2026-07-04 (user) — Stage 4's optional one-call-per-provider live smoke
  is possible. All required verification still runs without network.

## Entries

### 2026-07-04 — stage 0

- Landed: `composable-migration` branch; golden fixture generator
  (`scripts/golden/generate_golden_fixtures.py`, typer CLI) writing
  `tests/fixtures/golden/{hashing,graph_digests,record_ids,parser_scoring}.json`;
  golden pytest module `tests/test_golden_fixtures.py` (loads the generator
  via the repo's importlib script-loading pattern and compares payloads to
  committed fixtures).
- Fixture coverage: 13 canonical-JSON/digest value cases; canonical payload
  strings + digests for `direct_graph` (b00851facf9fe358), `encdec_graph`
  (ec4e636b819ecfbf), `humaneval_encdec_graph` (9a1f1b1b791a5057); record ID
  axes for `dimensions_digest`, `stable_prediction_id`, `fair_order_key`,
  `stable_generation_run_id`, `stable_node_attempt_id`,
  `stable_score_attempt_id` (default + explicit dataset); parser extraction
  for 8 samples × both v1 profiles (methods exercised: bare_python,
  fenced_code, field_marker, json_code_field, cleaned_candidate, plus
  failure cases) and scoring outputs for 5 samples under the default
  `humaneval@v1` scoring profile (outcomes: passed, tests_failed,
  extraction_failed, empty_generation).
- Verified: `uv run pytest -k golden` → 4 passed; `uv run pytest` → 696
  passed; integration tier → 45 passed against `dr_dspy_test`; `ruff` and
  `ty` clean on new files; regenerating fixtures reproduces byte-identical
  content.
- Fix folded in: `test_rescore_cli_dry_run_wires_options_without_launching_dbos`
  was failing on `main` (stale expectation — the rescore CLI now passes a
  `progress` kwarg, added by the sliding-window commit 36cbd36). Updated the
  test to pop `progress`, assert it is an `OperationProgress`, and compare
  the rest. Pre-existing failure, not caused by migration work; fixed to
  make the full-suite acceptance gate meaningful.
- Choices: golden test compares full recomputed payloads (not just digests)
  so mismatches show which serialized bytes moved; fixture regeneration is
  a script (never regenerate to paper over a migration mismatch — see the
  test module docstring). Added `.claude/ralph-loop.local.md` to
  `.gitignore` (loop state, not repo content).
- Skips: none.

### 2026-07-04 — stage 1

- Landed: `git mv src/dr_dspy src/whetstone`; all `dr_dspy.` module
  references rewritten to `whetstone.` across src/tests/scripts (sed on
  `dr_dspy\.` — safe because every frozen string uses `dr_dspy_` or
  `dr-dspy-` shapes, never a dot); pyproject `name = "whetstone-ai"`
  (bare `whetstone` is taken on PyPI per
  docs/remaining-implementation-intentions.md), isort first-party
  `whetstone`, new `[tool.hatch.build.targets.wheel] packages =
  ["src/whetstone"]` (name no longer matches package dir); README/TESTING
  current-code references updated; `uv.lock` resynced.
- Frozen and verified untouched: queue `dr-dspy-platform-generation-v1`,
  `dr_dspy_platform_*` step names, `DBOS_APP_NAME =
  "dr-dspy-platform-graph-v1"` (same persisted-string family as
  queue/workflow names — conservative choice), all `dr_dspy_*` table/
  constraint/index/trigger names, Alembic revision IDs (all version files
  byte-identical except pure import lines; verified by diffing against
  HEAD blobs), `sqlalchemy.url` default DB `dr_dspy`, test DB
  `dr_dspy_test`, CI workflow Postgres env.
- Conservative choice: `alembic.ini` is listed as frozen, but its
  `script_location` is a filesystem path that must track the package move
  or alembic breaks; updated only that line, everything else
  byte-identical. Historical changelog entries and rename-plan references
  in TESTING.md/AGENTS.md/README.md deliberately left as `dr_dspy`.
- Verified: `uv run alembic heads` → 20260630_0006 (head); full suite 696
  passed; integration tier 45 passed; goldens 4 passed; ruff + ty clean
  (22 isort fixes after first-party rename).
- Skips: none.

### 2026-07-04 — stage 2

- Landed (dr-serialize): new private repo
  `danielle-rothermel/dr-serialize` (`../dr-serialize`), scaffolded to
  match dr-providers (uv, src layout, py≥3.12, strict ruff set, ty,
  pytest, pre-commit, py.typed, MIT). `hashing.py` extracted verbatim;
  engine with `register_handler`/`convert_value` public API,
  `SerializationLimits` + Postgres preset (`postgres_jsonb_limits`),
  `ValueTransformError` base. 64 tests including golden hashing fixture
  copied from Stage 0. Pushed `main` + `composable-migration` (branch ==
  main; draft PR deferred until the branch diverges).
- API choices recorded in `serialize.md` open-sections block (module
  layout, required-limits keyword, handler slot, contextvar depth,
  frozen `postgres_max_bytes` diagnostics key).
- Landed (whetstone cutover): `hashing.py` + `serialization.py` deleted;
  dep via `[tool.uv.sources]` path (editable — must become git/PyPI pin
  before merge, noted in pyproject and PR); DSPy handlers re-homed to
  `whetstone/dspy_serialization.py` with `SignatureSummaryError`/
  `ExampleSerializationError` subclassing `ValueTransformError`,
  registered in `whetstone/__init__`; `SANITIZE_KEYS` +
  `sanitize_lm_kwargs` moved to `whetstone/lm/utils.py` (Stage 4 moves
  them to dr-providers); generic serialization tests moved out, DSPy
  handler tests stay (`tests/test_dspy_serialization.py`).
- Layering caught by tests: registering handlers in `__init__` initially
  imported `whetstone.lm` at package import, breaking
  `test_graph_imports`; fixed with a lazy `sanitize_lm_kwargs` import
  inside the BaseLM handler. `test_platform_boundaries` updated (deleted
  `serialization.py` removed from its scan list).
- Verified: dr-serialize `ruff`/`ty`/`pytest` all green (64 passed);
  whetstone full suite 666 passed (30 generic serialization tests moved
  to dr-serialize); goldens 4 passed (digests now produced through
  dr-serialize); integration tier 45 passed; `grep` shows no remaining
  `whetstone.hashing`/`whetstone.serialization` imports.
### 2026-07-04 — stage 3, sub-step 3a (dr-code prune + absorb)

- Landed (dr-code branch `composable-migration`, PR #9 draft): deleted
  `pipeline/` + `generation/` with dependent tests/scripts (16 files);
  absorbed code-eval as `src/code_eval` in-repo source and dropped the
  `code-eval==0.2.0` / `dr-providers` / `dr-queues` deps + uv.sources +
  ruff override; ported code-eval's tests and the 4,100-sample corruption
  corpus to `tests/code_eval/` (regression harness for 3b).
- Conservative choices: absorbed the FULL code_eval package verbatim
  (slimming to the ~530-line load-bearing core is profile-v2 territory,
  out of scope); only fixture paths re-homed
  (`SNAPSHOT_REL_PATH`, `DEFAULT_DATASET_PATH`, ladder/viewer script +
  template locations); `src/code_eval` excluded from ty (legacy lineage
  typing — 10 pre-existing diagnostics), ruff clean without exclusions.
- Kept dr-code's own `tests/corpus/humanevalplus_snapshot.json` (11 MB,
  different schema from code-eval's 127 KB snapshot — two loaders, two
  fixtures; briefly clobbered during the port and restored from git).
- Verified: dr-code suite 226 passed (nucleus 90 + absorbed 136); ruff +
  ty clean. Whetstone untouched this sub-step (its suite unchanged from
  stage 2: 666 unit + 45 integration + goldens).
- Remaining for stage 3: **3b** port whetstone humaneval parsing/scoring
  into dr-code byte-identically under v1 profile IDs (golden fixtures
  from Stage 0; corpus regression run); **3c** whetstone cutover
  (humaneval/ deleted, thin app-side module keeps `recordable_text`
  injection).

### 2026-07-04 — stage 3, sub-step 3b (v1 parsing/scoring port)

- Landed (dr-code commit e1a9dbd): `src/dr_code/humaneval/` = whetstone's
  humaneval package verbatim (imports rewritten only; the package had
  exactly one whetstone-internal import). v1 profile IDs unchanged.
- Injection point per code_parse_test.md: `score_humaneval_generation`
  now takes a required `recordable_text: Callable[[Any], str]` keyword;
  whetstone will pass `eval_failures.recording.recordable_text` at 3c
  (for the golden fixtures' all-string inputs, `str` is behaviorally
  identical — used in dr-code tests).
- Regression gates: whetstone Stage 0 `parser_scoring.json` copied to
  `tests/humaneval/fixtures/parser_scoring_golden.json`, 21 golden tests
  byte-equal; corpus baseline
  `tests/humaneval/fixtures/corpus_baseline_v1.json` pins per-recipe
  extraction outcomes over the 4,100-sample corpus (3707 extracted);
  regenerate only to intentionally re-baseline via
  `uv run python tests/humaneval/corpus_baseline.py`.
- Verified: dr-code full suite 309 passed (226 + 61 ported + 21 golden +
  1 corpus); ruff + ty clean. Whetstone untouched this sub-step.
- Remaining for stage 3: **3c** whetstone cutover — delete `humaneval/`,
  depend on dr-code via path source, swap imports (callers:
  records/models, analysis/*, platform/{worker,scoring,spec_builder}),
  pass `recordable_text` at the platform scoring call site, move
  humaneval tests out of whetstone, full suite + goldens + integration.

- Note: `gh` is authenticated as `drothermel`; the first
  `gh repo create dr-serialize` call landed an **empty stray repo
  `drothermel/dr-serialize`** before I created the correct
  `danielle-rothermel/dr-serialize`. Token lacks `delete_repo` scope and
  repo deletion is a human call — please delete the stray repo.
- Skips: none.
