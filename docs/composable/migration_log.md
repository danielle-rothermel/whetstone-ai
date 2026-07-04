# Migration Log

## Status

| stage | state | notes |
|-------|-------|-------|
| 0 baselines | done | golden fixtures + tests committed; full suite 696 passed; integration 45 passed |
| 1 rename | done | src/whetstone; pyproject whetstone-ai; frozen strings intact; 696 unit + 45 integration + goldens green |
| 2 dr-serialize | done | repo created + cutover; dr-serialize 64 tests; whetstone 666 unit + 45 integration + goldens green |
| 3 dr-code nucleus | done | dr-code 309 tests + corpus baseline; whetstone humaneval/ deleted; 605 unit + 45 integration + goldens green |
| 4 dr-providers v0.2 | done | kernel+transport+conformance+corpus (83 tests); whetstone thin adapter, FixtureProvider e2e, live smokes 3/3; 576 unit + 45 integration + goldens green |
| 5 dr-graph | done | repo created + cutover; dr-graph 111 tests incl. golden digests; whetstone 502 unit + 45 integration + goldens green |
| 6 platform | in_progress | design + 6a + 6b-i done (76 tests incl. Postgres migration/backoff); 6b-ii submission/enqueue/observability/projections/artifacts, 6c cutover, 6d validation pending |
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

### 2026-07-04 — stage 3, sub-step 3c (whetstone cutover) — stage 3 done

- Landed (whetstone): `src/whetstone/humaneval/` deleted; all imports
  swapped to `dr_code.humaneval` (19 src + 11 test/support files +
  golden generator); dr-code consumed via `[tool.uv.sources]` path dep
  (pin before merge); `recordable_text` injected at the three
  `score_humaneval_generation` call sites (platform/scoring.py, golden
  generator, test_platform_scoring); whetstone's
  test_humaneval_primitives + test_import_inference deleted (they moved
  to dr-code in 3b).
- Conservative choices: `requires-python` narrowed to `>=3.13, <3.15`
  (dr-code requires ≥3.13; whetstone already runs 3.13 per
  `.python-version`); CI workflow python bumped 3.12 → 3.13 to match.
- Verified: whetstone full suite 605 passed (61 tests moved out);
  goldens 4 passed — parser/scoring fixtures now byte-equal through
  dr-code under the v1 profile IDs; integration tier 45 passed; ruff +
  ty clean. dr-code suite unchanged green (309, from 3b).

- Note: `gh` is authenticated as `drothermel`; the first
  `gh repo create dr-serialize` call landed an **empty stray repo
  `drothermel/dr-serialize`** before I created the correct
  `danielle-rothermel/dr-serialize`. Token lacks `delete_repo` scope and
  repo deletion is a human call — please delete the stray repo.
- Skips: none.

### 2026-07-04 — stage 4, sub-step 4a (dr-providers v0.2 kernel)

- Landed (dr-providers branch `composable-migration`, PR #3 draft):
  `dr_providers/kernel/` subpackage — failures (FailureClass +
  ProviderFailure record, carriers, status-code classification,
  SANITIZE_KEYS), config records with `supported_controls` + presets
  (OpenRouter, OpenAI chat/responses, Gemini via the OpenAI-compat
  endpoint / GEMINI_API_KEY), `LlmRequest` + public pure
  `build_payload` (loud UnsupportedControlError; explicit
  `allow_unsupported_control_drop` opt-in preserving whetstone's
  silent-drop use case for knob-rejecting models), `LlmResponse` as
  materialized parts (TokenUsage w/ reasoning extraction, CostInfo,
  warnings channel, continuation handle, payload-on-response) with
  chat/responses body parsers, and `FixtureProvider` (public, Provider
  protocol, scripted outcomes).
- Conservative choices: kernel lives as a new subpackage; the 0.1.x
  `query/` package stays untouched until the v0.2 cutover settles its
  fate; extra_body flattens inline into the wire payload (raw httpx —
  the SDK-era extra_body indirection is gone).
- Verified: dr-providers 60 tests green (27 existing + 33 kernel);
  ruff + ty clean.
- Remaining for stage 4: **4b** raw-httpx transport (opt-in retry),
  conformance module, audit corpus grown with whetstone's real response
  fixtures, corpus-tested parsers; **4c** whetstone cutover (boundary →
  thin adapter over the kernel, eval_failures/policy.py drops the
  openai/httpx table, node_attempt_id as idempotency key,
  FixtureProvider end-to-end node execution test, optional live smokes
  — OPENAI/GEMINI/OPENROUTER keys all present).

### 2026-07-04 — stage 4, sub-step 4b (transport + conformance + corpus)

- Landed (dr-providers 68920d1): `kernel/transport.py` — `HttpProvider`
  over raw httpx implementing the Provider protocol; explicit timeouts;
  all failure classification in one place; retry strictly opt-in
  (`TransportPolicy.max_retries` default 0, bounded exponential backoff
  with jitter, injectable sleep/rng); `Idempotency-Key` header from
  `request.idempotency_key`. `kernel/conformance.py` — evidence-based
  post-response checks (reasoning-not-observed, token-limit-exceeded,
  model-substitution) emitted as WARNING-severity records; severity
  default recorded as resolved in `llm_provider.md`.
- Corpus: `data/kernel-corpus/responses.jsonl` seeded with 8 response
  shapes grown from whetstone's boundary fixtures, each with
  ground-truth parses; `test_kernel_corpus.py` regression-checks the
  parsers against it (never edit expectations to match a parser change
  without a recorded decision).
- Verified: dr-providers 83 tests green; ruff + ty clean.
- Remaining for stage 4: **4c** whetstone cutover — `lm/boundary.py`
  becomes the thin adapter over the kernel; `eval_failures/policy.py`
  drops the openai/httpx heuristic table (keeps psycopg/DBOS);
  `node_attempt_id` passed as idempotency key; whetstone test exercises
  node execution end-to-end against `FixtureProvider` with no network;
  optional one live smoke per configured provider (all keys present).

### 2026-07-04 — stage 4, sub-step 4c-i (FailureClass re-home)

- Landed (whetstone 9deae23): `eval_failures/types.py` deleted;
  `FailureClass` + RECOVERABLE/RETRYABLE sets import from
  `dr_providers.kernel.failures` (canonical home; enum values unchanged
  so persisted `failure_class` strings are byte-identical);
  `RETRYABLE_STEP_FAILURE_CLASSES` → kernel's
  `RETRYABLE_FAILURE_CLASSES` at all call sites; policy chain-walker
  unwraps kernel `ProviderFailureError` carriers; dr-providers wired as
  path dep.
- Landed (dr-providers 4c3434e): package root and kernel `__init__` use
  PEP 562 lazy exports so importing the failure taxonomy never loads
  httpx — required by whetstone's import-hygiene tests
  (test_lm_imports, test_eval_failures_policy). Public surfaces
  unchanged.
- Deliberately kept: policy's openai/httpx heuristic table stays until
  4c-ii removes the OpenAI SDK from node_execution (dropping the table
  first would degrade runtime classification while the SDK is the
  transport).
- Verified: whetstone 605 unit + 45 integration + 4 golden green; ty +
  ruff clean; dr-providers 83 green.
- Remaining for stage 4 (**4c-ii**): lm/boundary.py → thin adapter over
  the kernel (enums imported from kernel everywhere; whetstone
  ProviderConfig replaced by kernel config; LlmResponse →
  ProviderResult converter); node_execution.py drops the OpenAI SDK for
  Provider injection (HttpProvider default, node_attempt_id as
  idempotency key); policy drops the openai/httpx table; FixtureProvider
  end-to-end node execution test; test updates (test_lm_boundary,
  node_execution/graph_workflow mocks); optional live smokes (keys
  present).

### 2026-07-04 — stage 4, sub-step 4c-ii (boundary/SDK cutover) — stage 4 done

- Landed (whetstone 6db16e1): `lm/boundary.py` → thin adapter
  (`llm_request_for_node`, `provider_result_from_response`,
  `translate_provider_failure`, `PlainPromptAdapter`); OpenAI SDK
  removed from `node_execution.py` in favor of kernel `Provider`
  injection (HttpProvider default); `node_attempt_id` threaded from
  `run_prediction_graph_workflow`'s node closure into
  `execute_lm_node_step` (new optional trailing arg — step NAME
  unchanged; DBOS replays recorded outputs so in-flight workflows are
  unaffected) and passed as the request idempotency key with the same
  axes persistence uses; Gemini branch added to
  `runtime_provider_config`; `eval_failures/policy.py` dropped the
  openai/httpx heuristic table (kept psycopg/DBOS + generic shapes);
  kernel enums imported directly everywhere (no re-export layer); copro
  proposal call rewired to the kernel.
- Tests: `test_lm_boundary` rewritten for the adapter (old wire tests
  live in dr-providers' kernel/corpus suites); provider fakes are now
  Provider objects; 11 dead openai/httpx heuristic tests deleted;
  acceptance e2e test `test_execute_lm_node_end_to_end_with_fixture_provider`
  runs the full node path against `FixtureProvider` with no network.
- Verified: whetstone 576 unit + 45 integration + 4 golden green
  (count drop from 605 = removed dead heuristic tests + slimmer adapter
  test file; coverage moved to dr-providers, 83 green); ruff + ty clean.
- Live smokes (one minimal call each, ~16-token caps):
  openai/gpt-4o-mini OK (15 tokens), gemini-2.5-flash-lite via
  OpenAI-compat endpoint OK (9 tokens), openrouter llama-3.2-3b OK
  (44 tokens). Gemini-via-AI-Studio decision validated end-to-end.
- Wire-shape note (recorded): extra_body/extra_kwargs flatten inline
  into the raw-httpx payload — byte-equivalent to what the SDK's
  extra_body indirection put on the wire.

### 2026-07-04 — stage 5 (dr-graph extraction + cutover)

- Landed (dr-graph): new private repo `danielle-rothermel/dr-graph`
  (`../dr-graph`), scaffolded to match dr-serialize conventions;
  scaffold on `main`, library on `composable-migration` (draft PR #1).
  models/execution/hashing ported from whetstone's `graph/` with the v1
  generalizations: open-string `op` (required, non-empty — neutrality
  choice recorded in graph_runner.md) and `type_name` (default "str");
  parameterized external namespace via validation context (default
  "task"; `BindingSource.TASK` → `EXTERNAL`, `validate_task_bindings` →
  `validate_external_bindings(allowed_fields=...)`; graphs reject mixed
  namespaces and node-id/namespace collisions); `ClassifiedFailure`
  runtime-checkable Protocol (partial conformance tolerated — the
  wrapped-step-failure shape has no `underlying` attr); neutral
  `node()`/`graph()`/`as_binding_ref` builders; `inline_subgraph`
  flattening composition (separator ":"); `completed` resume hook on
  `execute_graph` (upfront validation → `CompletedNodeError`, skipped
  nodes appear as SUCCESS outcomes and in execution_order). Depends on
  dr-serialize (path dep — pin before merge, noted in PR).
- Tests (dr-graph): 111 green — ported whetstone graph execution suite
  (test doubles localized; dict-shape source values "task"→"external"),
  golden digest tests reproducing Stage 0 `graph_digests.json`
  canonical payloads AND digests byte-for-byte, plus new namespace/
  builders/compose/completed/protocol/import-hygiene suites. ruff + ty
  clean.
- Landed (whetstone cutover): `src/whetstone/graph/` deleted; dr-graph
  path dep; imports swapped in 9 src + 15 test/support files + golden
  generator; new `whetstone/node_ops.py` (`LLM_CALL_OP = "llm_call"`,
  frozen persisted spec content) used by node_execution dispatch
  (`node.op != LLM_CALL_OP`), spec_builder, and v0_reshape; NodeSpec
  constructions pass `op` explicitly (dr-graph made it required);
  test_platform_boundaries drops the moved `graph/` scan path;
  moved tests deleted (test_graph_execution, test_graph_imports — the
  hygiene checks live in dr-graph now).
- Process note: an over-eager `ruff format` pass initially reformatted
  ~30 unrelated whetstone files (whetstone CI runs `ruff check` only,
  not format) — fully reverted via `git restore` and re-applied as
  scoped edits; final diff touches only cutover files.
- Verified: whetstone 502 unit (74 graph tests moved out, 1 message
  expectation updated for the external-bindings rename) + 45
  integration (`dr_dspy_test`) + 4 golden green — graph digests now
  byte-equal through dr-graph; ruff + ty clean; no remaining
  `whetstone.graph`/`NodeOp`/`validate_task_bindings` references.
- Choices recorded in graph_runner.md "Resolved at extraction" block;
  additive-canonicalization and routing/iteration shapes stay open (not
  needed for the migration).
- Skips: none.

### 2026-07-04 — stage 6, design half

- Landed: platform.md open sections filled (package name dr-platform,
  module map, SubmittableItem/ItemIdentity/EnqueueItem/seed-hook
  protocol definitions, prefix-parameterized schema with own Alembic
  lineage + stamped baseline for whetstone, projection API, artifact
  store API, cutover plan 6a-6d with escalation points); consumer
  sketches committed (docs/composable/sketches/nl_latents_loop.py,
  optimizer_population_eval.py) — both import only the proposed facade.
- Key design findings (from a symbol-level map of the generic layer):
  (1) frozen table names + "library-owned tables with own lineage"
  jointly force PlatformSchema(prefix=...) with whetstone stamping the
  library's baseline revision; (2) batch_submit_item_id hashes the
  literal JSON key "prediction_id", so the facade takes
  ItemIdentity(item_key_label=...) to keep persisted ID bytes
  identical; (3) both consumer sketches independently need an
  await-work-completion primitive -> await_operation over recorded
  deterministic workflow ids (replaces copro's
  wait_for_generation_runs and nl_latents' hand-rolled polling).
- Underdetermination check (the stage's stop condition): each open
  section resolved to either a forced choice (name, schema strategy)
  or an obvious conservative one (projection column mapping scalars+
  JSONB, artifact local-dir backend, dr_dspy_prediction_projection
  stays app-owned). No guessing required -> extraction proceeds, with
  escalation points recorded in the cutover plan (item-id byte
  mismatch, Alembic state touching frozen revisions, passthrough-
  wrapper smell).
- Verified: docs-only change; ruff/ty unaffected (sketches live under
  docs/, outside lint includes; they are illustrative, not executable).
- Next: 6a repo + pure kernel (progress, batch_status, fairness,
  jsonl, dbos_config, items) with ported tests; no whetstone change.

### 2026-07-04 — stage 6, sub-step 6a (dr-platform pure kernel)

- Landed (dr-platform, new private repo danielle-rothermel/dr-platform,
  ../dr-platform; scaffold on main, kernel on composable-migration,
  draft PR #1): items.py (SubmittableItem runtime protocol,
  ItemIdentity digest config, stable_item_id with frozen
  {namespace, axes} payload, batch_item_id + claim_token), fairness.py
  ((order_key, item_id) sort + windowing over structural Orderable),
  jsonl.py (byte-offset index/load, JsonlFieldNames parameterization,
  caller-owned parse callable), batch_status.py (operation/item status
  enums with frozen string values, pure counts + operation status
  derivation, is_terminal_enqueue_status), dbos_config.py (URL
  normalization/resolution, PlatformDbosConfig without app concurrency
  knobs, DBOSConfig builder, dbos._error race shim + workflow status
  vocabulary), progress.py (verbatim port of progress_log).
- Identity gates pinned in tests: batch_item_id/claim_token reproduce
  whetstone's persisted bytes under
  ItemIdentity(item_key_label="prediction_id") (compared against
  direct dr_serialize digests of the {"operation_key","prediction_id",
  ...} payloads); status string values asserted frozen; stable_item_id
  payload shape asserted frozen.
- De-domaining choices: fairness drops PredictionSpecRecord
  revalidation (validation is app-side; refs/items arrive typed);
  jsonl index error messages name the configured field
  ("duplicate prediction_id ..." under whetstone's field names);
  concurrency knobs (generation/scoring) stay app-side, dropped from
  the library DBOS config.
- Verified: dr-platform 48 tests green; ruff format/check + ty clean.
  Whetstone untouched this sub-step (502 unit + 45 integration + 4
  golden unchanged from stage 5).
- Remaining for stage 6: 6b schema (PlatformSchema prefix param +
  Alembic lineage 0001/0002) + stateful modules (backoff w/ holds+tags,
  submission claim/lease, enqueue, await_operation/observability,
  projections, artifacts); 6c whetstone cutover; 6d consumer
  validation.

### 2026-07-04 — stage 6, sub-step 6b-i (schema + lineage + backoff)

- Landed (dr-platform 634c78a): naming.py (PlatformNaming — prefix +
  item/order/group column-word labels; whetstone config dr_dspy/
  prediction_id/fair_order_key/experiment_name preserves all frozen
  physical names); records.py (BatchOperationRecord/BatchItemRecord
  with ported validators on neutral field names; EnqueueFailure ==
  FailureMetadataPayload JSONB shape; frozen enqueue_metadata keys);
  db/schema.py (PlatformSchema — batch ops/items, throttle w/ holds+
  tags head shape, projections registry; constraint/index names
  library-generated, only table/column names load-bearing for
  stamped adopters); db/alembic (0001 baseline + 0002 holds/tags/
  registry, version table {prefix}_platform_alembic_version) with
  programmatic upgrade/stamp; backoff.py (ported math + explicit
  failure fields instead of whetstone FailureSummary; set/clear hold
  with duration or absolute expiry, tags with containment filter,
  delay = max(blocked_until, hold_until)).
- Port notes: clear_throttle_backoff now explicitly preserves holds/
  tags (whetstone's full-row upsert predates those columns);
  record_throttle_failure re-reads state after write so returned
  state includes hold/tag fields. Behaviorally identical for existing
  whetstone callers.
- Verified: 76 tests green — including real-Postgres coverage
  (scratch DB dr_platform_test, created via createdb): fresh upgrade
  under default naming, fresh upgrade under whetstone naming
  (column-word parity asserted), stamp-then-0002 with a sentinel row
  proving 0001 never re-ran; backoff/hold/tag behavior end-to-end.
  ruff format/check + ty clean. dr-providers consumed via path dep
  (version constraint >=0.1.1 — note: dr-providers' pyproject still
  says 0.1.1 despite the v0.2 kernel; version bump is a pre-merge
  task for that repo). Whetstone untouched.
- Remaining for stage 6: 6b-ii submission claim/lease + dedup enqueue
  + observability/await_operation + projections + artifacts; 6c
  whetstone cutover; 6d consumer validation.
