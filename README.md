# Whetstone

[![CI](https://github.com/danielle-rothermel/whetstone-ai/actions/workflows/whetstone_tests.yml/badge.svg)](https://github.com/danielle-rothermel/whetstone-ai/actions/workflows/whetstone_tests.yml)

| [Repo Definitions](https://danielle-rothermel.github.io/whetstone-ai/) | [dr-code v0.1.0](https://github.com/danielle-rothermel/dr-code) | [dr-graph v0.1.0](https://github.com/danielle-rothermel/dr-graph) | [dr-providers v0.2.1](https://github.com/danielle-rothermel/dr-providers) | [dr-serialize v0.1.1](https://github.com/danielle-rothermel/dr-serialize) | [dr-store v0.1.0](https://github.com/danielle-rothermel/dr-store) | [whetstone-envs v0.1.0](https://github.com/danielle-rothermel/whetstone-envs) |
| --- | --- | --- | --- | --- | --- | --- |

**Whetstone evaluates and optimizes prompt candidates through typed,
reproducible experiment contracts.** Its functionality is organized into
these areas:

- **Experiment modeling and identity** bind candidates, computation graphs,
  objectives, task plans, and execution settings into typed,
  content-addressed configurations.
- **Environments and sampling** assemble task pools, evaluation roles and
  splits, prompt transformations, rollout definitions, and reward policies for
  code-generation and encoder-decoder experiments.
- **Provider interaction** translates language-model inputs and results,
  classifies transport and semantic failures, applies bounded retry policies,
  and retains evidence for each attempt.
- **Execution and recovery** fan out work through managed subprocesses, reuse
  cached provider results, preserve partial progress, and resume previously
  completed work exactly.
- **Evaluation and scoring** execute graphs over planned tasks and repeats,
  capture component traces, and aggregate correctness, compression, reward,
  and statistical evidence.
- **Optimization** provides a shared candidate-evaluation harness and native
  COPRO, MIPROv2, and GEPA flows, including proposal generation, tool use,
  algorithm state, and result artifacts.
- **Authority and orchestration** coordinate durable proposal and evaluation
  effects, ownership claims, and terminal result binding across replay and
  recovery.

## Testing

The authoritative unit lane is serial:

```bash
uv run pytest tests/ -q \
  -m "not process_integration and not sqlite_time_integration and not sqlite_contention"
```

For a faster local iteration loop, the same selection can use a fixed four
workers with load balancing:

```bash
uv run pytest tests/ -q \
  -m "not process_integration and not sqlite_time_integration and not sqlite_contention" \
  -n 4 --dist=load
```

The parallel command is a local convenience, not the CI default. Keep the
worker count bounded; do not replace it with `-n auto`. Run the complete serial
suite with `uv run pytest -q`. Live PostgreSQL tests additionally require
`WHETSTONE_TEST_POSTGRES_DSN`. CI separately exercises process, SQLite timing,
SQLite contention, installed-wheel, and Python 3.14 compatibility contracts.

Process-integration cleanup is watchdog-bounded and best effort. POSIX process
and process-group identifiers can be reused, and macOS does not provide an
atomic pidfd-equivalent signaling handle, so abrupt-failure cleanup cannot
guarantee that a late signal still identifies the original process. Run these
tests in an isolated local or CI environment; their process-group assertions
exercise observed behavior, not a strict containment guarantee.
