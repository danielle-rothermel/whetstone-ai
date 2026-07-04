# Migration Log

## Status

| stage | state | notes |
|-------|-------|-------|
| 0 baselines | done | golden fixtures + tests committed; full suite 696 passed; integration 45 passed |
| 1 rename | pending | |
| 2 dr-serialize | pending | repo not yet created |
| 3 dr-code nucleus | pending | repo exists at ../dr-code |
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
