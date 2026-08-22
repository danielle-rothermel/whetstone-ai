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
| **GEPA** | Live harness adapter + step engine; pass via the `build_runtime` registry | Wired (`submit_optim_run`, inline and PLATFORM deferral) | `whetstone-sandbox gepa` |
| **MIPROv2** | Live via `register_toy_runtime(..., extra_adapters=...)` + `prepare_miprov2_run` | Not on the pipeline | `whetstone-sandbox miprov2` (plan preview only) |
| **Codex direct** | Live via the `build_runtime` registry + `prepare_codex_run`; the only tool-using optimizer, and macOS-only (its sandbox is `sandbox-exec`) | Not on the pipeline | No sandbox command |

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
   GEPA is platform-wired the same way as COPRO when present in the
   `build_runtime` registry. MIPROv2 plugs into the in-process harness via
   `register_toy_runtime(..., extra_adapters=...)` plus
   `prepare_toy_miprov2_run` and is not on the platform pipeline.
4. **Sandbox & interpretation** — dry-run previews and toy-graph helpers to step
   through optimizer behavior before spending full eval budget
   (`whetstone-sandbox`).
5. **Codex MCP eval** — whetstone hosts, outside the Codex sandbox, the one
   tool a Codex run is granted: evaluate a candidate on the run's internal
   split and read back the aggregate reward and per-task scores. The agent
   receives only an authenticated loopback endpoint, never the store. Every
   call is admitted through `ToolAdmissionAuthority` against a per-run
   capacity, leased, persisted, and recorded in the step's Issued Tool Call
   ledger. The Codex output artifact
   carries no candidate body -- it names the `call_id` it selected, and the
   adapter rebuilds that candidate from the call's recorded, content-addressed
   arguments, so a template that was never evaluated through the tool cannot be
   returned.

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
| **dr-exec** | Budgeted subprocess execution: the Codex optimizer's `ProcessExecutor`, and the subprocess rollout driver's worker pool |
| **dr-platform** | Durable pipeline stages, deferral/fan-in, and run submission (`platform` extra) |

## Stable seams

- **Experiment** — generation graph, initial/ceiling candidates, eval configs, reward policy
- **EvaluationEngine** — validates and evaluates a candidate; returns typed evidence refs
- **OptimizerAdapter** — COPRO and GEPA plug into the shared harness and platform pipeline when present in the `build_runtime` registry; MIPROv2 and Codex are harness-only. Codex is the only `TOOL_USING` adapter, so its run carries a `ToolConfig` and `build_runtime` needs a `tool_executor` and a durable `admission` authority. Adding or removing an adapter changes controller identity.
- **StepContractProvider** — each optimizer declares its first-step and continuation contracts and parses its own launch control, registered by adapter key; `StepRequestBuilder` and `HarnessRunController` dispatch through it
- **Step evidence** — a step reports evaluations it asked the harness to run in `resolved_intents` (COPRO, MIPROv2), and evaluations its own search drove in `search_evidence` (GEPA), each bound to its run and step index and verified by the harness. A `TOOL_USING` step (Codex) carries `tool_evidence` instead: intent/search evidence and tool evidence are mutually exclusive, and the Issued Tool Call ledger records one entry per admitted call. A terminal step whose contract sets `terminal_proposal_count` and that accepted no improvement over the run's own initial candidate sets `seed_retained`
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
  --proposer provider \
  --application-version 0.1.5 \
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

## Real-Codex ladder

The Codex-direct optimizer's CI suites drive a scripted fake CLI
(`whetstone.testing.fake_codex_cli`), which speaks real MCP to the real
evaluation server but never validates the output schema, consults an
approval policy, or guesses a tool argument. The ladder in
`tests/real_codex/` closes that gap by driving the **real** Codex CLI:

```bash
scripts/check-real-codex.sh              # every rung, stopping at the first break
scripts/check-real-codex.sh -k rung3     # one rung
```

It is a manual check. It spends real Codex agent turns on a logged-in
subscription session, so it is excluded from the default suite by the
`real_codex` marker and runs only when `WHETSTONE_REAL_CODEX=1` is set,
which the script does for you. The task model stays fake throughout —
evaluations use the reference transport — so the ladder costs Codex turns
and no eval-provider credit. Nothing reads credential material: the
runner's own auth staging copies the existing session into each run's
scratch `CODEX_HOME`.

Requirements: macOS (the sandbox is `sandbox-exec` only), the Codex CLI
(`/opt/homebrew/bin/codex`, or set `WHETSTONE_REAL_CODEX_BINARY`), and a
session from `codex login`. Transcripts and a per-rung result table are
written outside the repository, under
`~/drotherm/data/whetstone-ai/real-codex/<timestamp>/` by default
(override with `WHETSTONE_REAL_CODEX_OUTPUT_DIR`).

The rungs run cheapest-first, each presupposing the ones below it: config
the runner writes is accepted by the real binary (no session), the auth
preflight proves a session, one real Step through the hosted MCP server,
the capacity/wall-budget/no-tool-call edge paths, a real multi-evaluation
selection loop, reasoning-effort variants, output retention against a
truncated real transcript, the sandbox denying the store, and a foreign
bearer token being refused.

The `Real Codex ladder` GitHub Actions workflow runs the same script but
is **`workflow_dispatch`-only** and never triggers automatically. It needs
a self-hosted macOS runner with its own logged-in Codex session; a
GitHub-hosted runner cannot satisfy that.
