# Whetstone

Whetstone is a typed experiment system for evaluating and optimizing prompt
candidates with reproducible configuration, execution, and evidence.

## At a Glance

- [Whetstone design reference](https://danielle-rothermel.github.io/whetstone-ai/)
- First-party dependencies:
  - [dr-code](https://github.com/danielle-rothermel/dr-code) — `0.1.0`, for
    code-task datasets and records
  - [dr-graph](https://github.com/danielle-rothermel/dr-graph) — `0.1.0`, for
    typed computation graphs and deterministic interpretation
  - [dr-providers](https://github.com/danielle-rothermel/dr-providers) —
    `0.2.1`, for provider requests, responses, and transport
  - [dr-serialize](https://github.com/danielle-rothermel/dr-serialize) —
    `0.1.1`, for canonical serialization and identity hashing
  - [dr-store](https://github.com/danielle-rothermel/dr-store) — `0.1.0`, for
    content-addressed and durable document storage
  - [whetstone-envs](https://github.com/danielle-rothermel/whetstone-envs) —
    `0.1.0`, for benchmark tasks and environment data

## High-Level Design

Whetstone connects experiment definitions to reproducible evaluation evidence
and candidate optimization. Its functionality is organized into these areas:

- **Experiment modeling and identity** bind candidates, computation graphs,
  objectives, task plans, and execution settings into typed,
  content-addressed configurations.
- **Environments and sampling** assemble task pools, evaluation roles and
  splits, prompt transformations, rollout definitions, and reward policies for
  code-generation and encoder-decoder experiments.
- **Provider interaction** translates language-model inputs and results,
  classifies transport and semantic failures, applies bounded retry policies,
  and retains evidence for each attempt.
- **Execution and recovery** fan out work through guarded processes, reuse
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

## Development

Whetstone supports Python 3.13 and 3.14. Install the locked environment and run
the local checks with:

```sh
uv sync --all-groups
./scripts/ci/lint.sh
./scripts/ci/unit.sh
```
