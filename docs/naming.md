# Whetstone naming vocabulary

Authoritative identifiers for eval/optim code, wire schemas, and persisted JSON fields.

## Data and stochasticity

| Token | Meaning | Pattern |
| ----- | ------- | ------- |
| `split` | Ordered task set + role | `EvalSplit`, `split_role`, `split_id` |
| `seed` | Eval stochasticity (= provider `rng_seed`) | field `rng_seed`; never `data_seed` for eval |
| `data_seed` | RNG for data split creation | split/manifest provenance only |
| `num_seeds` | Stochastic eval draws per task | replaces `num_samples` |
| `task_trial` | One draw: `(task, seed_index, rng_seed)` | `TaskTrial`, `TaskTrialKey` |
| `seed_index` | Draw index in `[0, num_seeds)` | replaces `sample_index` |

## Configuration and optimization

| Token | Meaning | Pattern |
| ----- | ------- | ------- |
| `candidate` | One optimized parameter instance | `Candidate`, `candidate_id` |
| `config` | Fixed while scoring candidates | `EvalConfig`, `config_hash` |
| `evaluation` | One `(config, candidate)` over `split × num_seeds` | `EvalEngine.evaluate()` → `EvalResult` |
| `sweep` | One config, many candidates | `*Sweep*` transcript types |

## Time and execution

| Token | Meaning | Pattern |
| ----- | ------- | ------- |
| `generation` | **LLM output text only** | `generation` / `output_text` on row payloads |
| `rollout` | One full graph run per `task_trial` | `RolloutDriver`, `run_rollout_row`, `rollout_graph` |
| `trace` | Full I/O sequence for a rollout | `EvalTraces`, `EvalTraceRow`, `traces_ref` |

## Structural abbreviations

| Token | Pattern |
| ----- | ------- |
| `eval` | `EvalEngine`, `whetstone.eval.*` package |
| `optim` | `OptimRun`, `whetstone.optim.*` package |
| `ref` | `CandidateRef`, `EvalConfigRef`, `traces_ref` |
| `hash` | `config_hash`, `identity_hash` |
| `id` | `run_id`, `candidate_id`, `split_id` |

## Package paths

| Old | New |
| --- | --- |
| `whetstone.evaluation` | `whetstone.eval` |
| `whetstone.optimization` | `whetstone.optim` |

## Reserved rules

- `generation` is LLM text only — not matrix index keys (`TaskTrialKey` instead).
- `seed` in identifiers → `rng_seed` (eval) or `data_seed` (split); bare `seed` only in `seed_index` / `num_seeds`.
- Retired matrix units: `sample`, `trial` (`num_trials` in MIPROv2 control is optimizer schedule, not eval matrix).
- Statistical bootstrap: `bootstrap_rng_seed` in `eval/analysis/statistics.py`.
