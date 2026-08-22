# whetstone-ai

Generic toolkit for **evaluating and optimizing** LLM prompts and programs.

Whetstone sits above the **dr-*** libraries (graphs, providers, store, serialize,
exec, platform) and below domain-specific environments. It owns the reusable
experiment contract, batched evaluation engine, optimizer harness, and
evidence/analysis plumbing — not task datasets, domain scoring rules, or
application UI.

**In scope here:** evaluation at scale, a shared optimization harness, and
stepping through runs to inspect behavior. Optimizers are not co-equal:

| Optimizer | Harness adapter | Platform pipeline | Sandbox |
|-----------|-----------------|-------------------|---------|
| **COPRO** | Live; pass a COPRO adapter in the `build_runtime` registry | Wired (`submit_optim_run`, inline and PLATFORM deferral) | `whetstone-sandbox copro` |
| **GEPA** | Live harness adapter + step engine; pass via the `build_runtime` registry | Not registered | `whetstone-sandbox gepa` |
| **MIPROv2** | Adapter/control exist | Not on the pipeline | `whetstone-sandbox miprov2` (plan preview only) |

**Out of scope here:** particular benchmarks or envs (those live in separate
packages or repos), one-off experiment scripts, and product-facing runners.

## Core capabilities

1. **Evaluation** — batched, efficient sweeps over candidates and tasks;
   configurable splits, graph rollouts, concurrency, and durable evidence.
   Bundled reference drivers: `GraphRolloutEvalDriver`
   (`eval/drivers/graph_rollout.py`) — the default, parallel in-process graph
   rollouts with injected `EvalProcedureRunner` — and
   `SubprocessGraphRolloutEvalDriver`
   (`eval/drivers/subprocess_graph_rollout.py`), which runs the same rows on a
   dr-exec worker pool with per-row and per-batch wall-time budgets.
2. **Evaluation analysis** — bootstrap confidence intervals, power analysis, and
   anchor calibration over persisted evaluation evidence (`eval/analysis/`).
3. **Optimization** — shared harness and adapters that propose candidates and
   drive evaluation intents in a loop. COPRO is the platform-wired optimizer;
   GEPA and MIPROv2 exist as adapters (GEPA also has a step engine) but are
   not registered in the default runtime.
4. **Sandbox & interpretation** — dry-run previews and toy-graph helpers to step
   through optimizer behavior before spending full eval budget
   (`whetstone-sandbox`).
5. **Codex MCP eval** — `whetstone-mcp-eval` serves the Codex evaluate-candidate
   tool over stdio.

```text
Evaluation  →  Evaluation analysis
     ↓
Optimization  →  Sandbox / interpretation
```

## dr-* libraries

| Package | Role in whetstone |
|---------|-------------------|
| **dr-graph** | Rollout graphs: LLM-call → eval nodes, executed per task row |
| **dr-providers** | Provider call configs, transport, and invocation evidence |
| **dr-store** | Content-addressed persistence for candidates, evidence, and step records |
| **dr-serialize** | Strict JSON and canonical identity hashing |
| **dr-exec** | Budgeted subprocess execution: Codex optimizer steps, and the subprocess rollout driver's worker pool |
| **dr-platform** | Durable pipeline stages, deferral/fan-in, and run submission (`platform` extra) |

## Stable seams

- **Experiment** — generation graph, initial/ceiling candidates, eval configs, reward policy
- **EvaluationEngine** — validates and evaluates a candidate; returns typed evidence refs
- **OptimizerAdapter** — COPRO and GEPA plug into the shared harness when present in the `build_runtime` registry; MIPROv2 exists as an adapter but is not platform-wired. Adding or removing an adapter changes controller identity.
- **StepContractProvider** — each optimizer declares its first-step and continuation contracts and parses its own launch control, registered by adapter key; `StepRequestBuilder` and `HarnessRunController` dispatch through it
- **Step evidence** — a step reports evaluations it asked the harness to run in `resolved_intents`, and evaluations its own search drove in `search_evidence`, each bound to its run and step index and verified by the harness; a terminal step whose contract sets `terminal_proposal_count` and that accepted no improvement over the run's own initial candidate sets `seed_retained`
- **Graph rollouts** — `experiment/graph/` builds standard two-node graphs; drivers execute them per row

## Platform pipeline

The optim pipeline (`whetstone.optim.v1`) has stages `optim_step` → `eval_row`
→ `eval_fanin`, plus `run_completion`. `EvalDispatchMode.INLINE` evaluates
inside the step. `EvalDispatchMode.PLATFORM` persists eval intents, fans out
row jobs, fans results back in, then resumes the step. Submit a run with
`submit_optim_run`.

`whetstone-optim` (requires the `platform` extra) is the production entry
point. `run` resolves a bound launch from a SQLite store, assembles
`build_runtime` + `deploy_platform`, submits a members tuple, and prints the
receipt. Adapter-set membership is part of controller identity: adding an
adapter changes `runtime.controller.runtime_hash`. `status` reads the run
manifest and release state; `result` loads `OptimPlatformRunResult`.

```bash
uv sync --extra platform
uv run whetstone-optim run \
  --run-id <bound-run-id> \
  --store-path runtime.sqlite \
  --database-url "$WHETSTONE_DATABASE_URL" \
  --campaign-key campaign-1 \
  --run-key run-1 \
  --adapter copro \
  --application-version 0.1.3 \
  --executor-id local-1
uv run whetstone-optim status --run-key run-1 --store-path runtime.sqlite
uv run whetstone-optim result --run-key run-1 --store-path runtime.sqlite
```

## Sandbox

```bash
uv run whetstone-sandbox copro --task-prompt "Say hello"
uv run whetstone-sandbox graph --run
```

Requires Python 3.13+. Optional extras: `dbos`, `postgres`, `platform`.

## Platform integration tests

Tier 2 tests exercise the dr-platform harness against Postgres + DBOS:

```bash
uv sync --extra platform
createdb whetstone_platform_test   # once, if needed
uv run pytest -m integration tests/integration/
```

Set `WHETSTONE_TEST_DATABASE_URL` when not using the default
`postgresql+psycopg:///whetstone_platform_test`. Locally, tests skip when
Postgres is unavailable; in CI they fail hard. Default `uv run pytest` excludes
integration tests via the pytest marker.
