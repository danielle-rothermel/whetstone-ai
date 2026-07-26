# MIPROv2 Frozen Source Audit and Differential Oracle

Status: independent source contract for the PR #91 replacement.

This document is the behavioral oracle for the Whetstone MIPROv2 port. It was
derived directly from DSPy commit
`6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0`. It is deliberately more specific
than the implementation plan: tests should cite the source row they cover, and
an implementation that disagrees with this document must either be corrected
or declare an intentional, identity-versioned compatibility difference.

## 1. Frozen source set

| Concern | Frozen source |
| --- | --- |
| Top-level configuration and search | `dspy/teleprompt/mipro_optimizer_v2.py` |
| Minibatch, evaluation, demo-set, promotion helpers | `dspy/teleprompt/utils.py` |
| Bootstrap tracing and demo assembly | `dspy/teleprompt/bootstrap.py` |
| Labels-only demo assembly | `dspy/teleprompt/vanilla.py` |
| Grounded instruction proposal | `dspy/propose/grounded_proposer.py` |
| Dataset summary | `dspy/propose/dataset_summary_generator.py` |
| Prompt string helpers | `dspy/propose/utils.py` |
| Full evaluation score | `dspy/evaluate/evaluate.py` |
| Categorical optimizer | DSPy `uv.lock`: `optuna==4.8.0` |

DSPy's declared dependency is only `optuna>=3.4.0`; the lock, not that lower
bound, defines this oracle.

## 2. Intentional Whetstone differences

The following are permitted differences and must be bound into optimizer
identity:

1. DSPy signature field descriptions and Python source formatting are replaced
   by Whetstone prompt templates, components, placeholders, and rendering
   metadata.
2. Whetstone rejects invalid numeric controls eagerly. Frozen DSPy generally
   accepts them and fails later.
3. Whetstone rejects proposed instructions that violate required native prompt
   placeholders. DSPy performs no equivalent validation.
4. Ambient models, metrics, filesystem logs, and in-memory calls are replaced
   by explicit routes, Eval Configs, evidence, artifacts, and DBOS effects.
5. Natural-language output is not expected to be byte-identical across live
   endpoints. Scripted semantic outputs are the differential oracle.
6. Pure algorithm seams remain multi-component-generic, but the integrated
   runtime rejects more than one executable component before any effect.
   Whetstone's current rollout primitive exposes only one provider trace, so
   accepting such a layout would fabricate or omit DSPy's ordered predictor
   traces. The structured trace contract remains for the future multi-trace
   executor (#94).

No other difference is implied by this list.

## 3. Constructor and compile contract

### Constructor defaults

| Argument | Frozen default / behavior | Required test |
| --- | --- | --- |
| `metric` | required | constructor snapshot |
| `prompt_model` | `None`; truthy value or ambient LM | falsey-model fallback |
| `task_model` | `None`; truthy value or ambient LM | falsey-model fallback |
| `teacher_settings` | `None`; stored as `teacher_settings or {}` | falsey mapping |
| `max_bootstrapped_demos` | `4` | constructor snapshot |
| `max_labeled_demos` | `4` | constructor snapshot |
| `auto` | `"light"` | allowed-mode matrix |
| `num_candidates` | `None`; copied to instruction and few-shot counts | manual count propagation |
| `num_threads` | `None` | evaluator construction |
| `max_errors` | `None`; later ambient fallback | explicit/ambient resolution |
| `seed` | `9` | seed fallback |
| `init_temperature` | `1.0` | per-proposal route snapshot |
| `verbose` | `False` | decision-invariance test |
| `track_stats` | `True` | result-shape matrix |
| `log_dir` | `None` | no-filesystem compatibility projection |
| `metric_threshold` | `None` | threshold truthiness matrix |

`auto` accepts exactly `None`, `light`, `medium`, and `heavy`. The constructor
also rejects when either resolved model is falsey.

### Compile defaults

| Argument | Frozen default |
| --- | ---: |
| `teacher` | `None` |
| `valset` | `None` |
| `num_trials` | `None` |
| `max_bootstrapped_demos` | `None` |
| `max_labeled_demos` | `None` |
| `seed` | `None` |
| `minibatch` | `True` |
| `minibatch_size` | `35` |
| `minibatch_full_eval_steps` | `5` |
| `program_aware_proposer` | `True` |
| `data_aware_proposer` | `True` |
| `view_data_batch_size` | `10` |
| `tip_aware_proposer` | `True` |
| `fewshot_aware_proposer` | `True` |
| `requires_permission_to_run` | `None` |
| `provide_traceback` | `None` |

### Error precedence and reference validation

The order is observable and should be tested:

1. `requires_permission_to_run=False` warns; `True` raises.
2. Resolve error and demo overrides.
3. Compute zero-shot status using exact equality to zero.
4. In manual mode with candidates but no trials, raise the recommendation
   diagnostic.
5. In other incomplete manual configurations, raise the candidates-required
   diagnostic.
6. Auto mode rejects either explicit candidate count or explicit trial count.
7. Resolve `seed = compile_seed or construction_seed`; compile seed zero falls
   back.
8. Initialize the shared RNG.
9. Reject empty trainset.
10. With no valset, reject a trainset shorter than two; otherwise split.
11. Reject an explicit empty validation set.
12. Apply auto settings.
13. Reject only
    `minibatch and minibatch_size > len(resolved_valset)`.

DSPy has no eager positive/nonnegative/finite checks for the other numeric
controls. Whetstone's stricter checks are safety adaptations.

### Dataset and auto-mode branches

Without an explicit validation set:

```text
valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
cutoff = len(trainset) - valset_size
trainset = trainset[:cutoff]
valset = trainset_before_split[cutoff:]
```

Auto settings:

| Mode | N | Requested validation sample |
| --- | ---: | ---: |
| light | 6 | 100 |
| medium | 12 | 300 |
| heavy | 18 | 1000 |

Auto sampling executes:

```text
size = min(requested_size, len(valset))
indices = shared_rng.sample(range(len(valset)), size)
valset = [valset[i] for i in indices]
```

It always consumes RNG and reorders the set, including when `size` equals the
entire validation length. Auto minibatching is then
`len(sampled_valset) > 50`.

Candidate counts are:

```text
instruction_candidates = N if zero_shot else int(N * 0.5)
fewshot_candidates = N
num_vars = component_count * (1 if zero_shot else 2)
num_trials = int(max(2 * num_vars * log2(N), 1.5 * N))
```

## 4. Exact top-level call order

```text
resolve permission/error/demo controls
validate manual/auto relationship
initialize shared RNG
split datasets
apply auto sampling/settings
construct evaluator and deep-copy student
bootstrap demo candidates under task-model context
construct grounded proposer and propose outside task-model context
overwrite instruction candidate zero with original
discard demos when zero-shot
baseline full evaluation under task-model context
create Optuna sampler/study and add baseline
run exactly num_trials sampled objectives with interleaved promotions
attach optional statistics
return strict best full-evaluated program
```

## 5. RNG oracle

The shared algorithm RNG is one `random.Random(resolved_seed)`.

### Shared RNG operations, in order

1. Auto validation `sample`, when auto mode is active.
2. For each demo-set seed that reaches the generic bootstrap branch:
   - `shuffle(trainset_copy)`;
   - `randint(1, max_bootstrapped_demos)`.
3. Predictor-major and proposal-major:
   - `choice` from the insertion-ordered tip keys, when random tips are on;
   - `randint(0, 10**9)` for the rollout id.
4. For each true minibatch evaluation:
   - `sample(range(len(valset)), minibatch_size)`.

Full baseline and promotion evaluations do not consume this RNG.

### Independent RNGs

| Source | Seed and behavior |
| --- | --- |
| Labels-only `LabeledFewShot` | one `random.Random(0)`; sequential per-predictor samples from the complete trainset |
| Bootstrap unaccepted-example shuffle | `random.Random(0).shuffle` |
| Bootstrap raw-demo sampling | one `random.Random(0)`; later predictors sample from the prior predictor's sampled subset |
| Multiple trace choice | `random.Random(Hasher.hash(tuple(demos)))`; random threshold then choice if needed |
| Optuna | independent `TPESampler(seed=resolved_seed, multivariate=True)` |

### Required RNG tests

- auto sample smaller than validation;
- auto full-size sample and reorder;
- manual mode consumes no validation-sample draw;
- seed `-2` labels branch and shuffled fallthrough;
- zero-shot proposal-context bootstrap;
- one and two predictors;
- tips enabled and disabled;
- selected empty `none` tip;
- minibatch and full evaluation;
- crash/replay after every draw.

The oracle records operation name, arguments, result, and cursor ordinal rather
than relying only on a final RNG state.

## 6. Bootstrap oracle

`create_n_fewshot_demo_sets` subtracts three from the requested count and then
iterates:

```text
range(-3, requested_count - 3)
```

This produces exactly the originally requested number of sets.

| Seed | Frozen branch |
| ---: | --- |
| `-3` | reset/zero-shot candidate when non-bootstrapped sets are included |
| `-2` | labels-only only when `max_labeled_demos > 0`; otherwise generic shuffled bootstrap |
| `-1` | unshuffled bootstrap |
| other | generic shuffled bootstrap |

Do not force all named special sets when `N < 3`.

For a bootstrap candidate:

1. Student becomes `student.reset_copy()`.
2. Teacher is the explicit teacher's deep copy, or `student.deepcopy()`.
3. If labeled demos are enabled and teacher is not compiled, reset the teacher
   and add labeled demos first.
4. Require matching predictor count, names, and signatures.
5. Iterate training examples in order until the requested number of successful
   complete traces is reached.
6. Temporarily remove the current example from every teacher predictor's demos
   by equality.
7. Execute teacher and call `metric(example, prediction, trace)`.
8. If `metric_threshold` is truthy, accept on
   `metric_value >= metric_threshold`; otherwise use metric truthiness.
9. Convert trace entries for known predictors into augmented examples.
10. If one predictor appears multiple times in a successful trace, use the
    hash-seeded 50/50 earlier-versus-last selection.
11. Restore teacher demos.
12. Shuffle unbootstrapped examples with `random.Random(0)`.
13. For each predictor, prepend augmented demos and then sample raw demos.

The bootstrap error counter persists across examples/rounds within this one
candidate compilation. It resets for the next candidate set because a new
`BootstrapFewShot` is constructed. It raises when
`current_error_count >= effective_max_errors`.

The `raw_demos` narrowing across predictors is part of the reference:

```text
raw_demos = validation
for predictor:
    raw_demos = rng.sample(raw_demos, sample_size)
    predictor.demos = augmented + raw_demos
```

Required tests cover every seed branch, threshold `None`, `0.0`, positive and
negative values, failed rollouts, unknown predictor traces, multiple traces,
teacher demo exclusion/restoration, multi-predictor narrowing, and error reset
between candidate sets.

## 7. Grounded proposal oracle

### Dataset-summary topology

When enabled, dataset summary is generated once during
`GroundedProposer` construction:

1. One initial `DatasetDescriptor` request over the first batch.
2. Up to nine `DatasetDescriptorWithPriorObservations` requests.
3. One `ObservationSummarizer` request.

The loop increments `calls` and breaks when `calls >= 10` before issuing that
iteration's request. Thus there are at most ten descriptor requests total.

`COMPLETE` is recognized only when the first eight response characters,
uppercased, equal `"COMPLETE"`. Such responses are not appended. The skip
counter is cumulative and stops after five; it is not reset by other output.
Non-`COMPLETE` observations are concatenated without an inserted separator.
An exception during follow-ups ends the loop and proceeds to summary. An
initial or final-summary exception causes the parent proposer to disable data
awareness.

All dataset-summary calls use the prompt model at temperature `1.0`. They do
not use the per-proposal rollout id or `init_temperature`.

### Proposal iteration and effects

If demo candidates are absent, proposal count per predictor is `N` and task
demos are disabled. Otherwise the loop length is derived from predictor zero:

```text
num_demos = max(len(demo_candidates[0]), 1)
count = min(N, num_demos)
```

Iteration is predictor-major, then demo-set-major. Per proposal:

1. Optionally choose one of the insertion-ordered tips:
   `none`, `creative`, `simple`, `description`, `high_stakes`, `persona`.
2. Set `use_tip = bool(selected_tip)`. The `none` choice removes the tip field
   from that proposal's generated signature.
3. Draw rollout id `randint(0, 10**9)`.
4. Copy the prompt model once with that rollout id and
   `temperature=init_temperature`.
5. When program aware, call `DescribeProgram`.
6. When program aware, call `DescribeModule`.
7. Call final instruction generation.

Steps 5–7 share the same copied model configuration and rollout id and repeat
for every proposal. They are not precomputed summaries.

### Demo and response quirks

- Demo sets rotate current, later, then earlier.
- Only examples containing an `"augmented"` key are included; the value is not
  checked.
- At most three examples are gathered in reference field order.
- Demo-set index zero is forced to `"No task demos provided."` even if examples
  were gathered.
- MIPROv2 supplies an empty trial log and disables instruction history.
- Program-aware description failure substitutes unavailable/not-provided text
  for that proposal.
- Proposal response goes through `strip_prefix` twice.
- Candidate index zero is overwritten with the original instruction after all
  model calls; the displaced response remains charged.
- Duplicate instructions remain distinct categorical entries.

Required tests snapshot semantic prompt sections for each awareness toggle,
tip shape, zero/current/rotated demo context, one/two predictors, failures, the
displaced first response, and duplicate responses.

## 8. Evaluation and score oracle

Full candidate evaluation preserves validation order. True minibatch
evaluation samples ordered indices with the shared RNG.

`Evaluate` returns:

```text
round(100 * sum(per_example_metric_values) / number_of_examples, 2)
```

The score supplied to Optuna, promotion means, strict-best comparisons, logs,
and candidate statistics is this percentage value. Bootstrap acceptance uses
the raw unscaled per-example metric value.

If the entire evaluation helper raises, `eval_candidate_program` returns score
`0.0` and empty results. MIPROv2 still increments `total_eval_calls` by the
nominal requested batch size. Per-example failures handled within `Evaluate`
receive its failure score.

When `minibatch=True` and `minibatch_size == len(valset)`, the helper invokes
the full-evaluation route and `score_data.full_eval` is true, but the optimizer
still applies minibatch winner and promotion control flow.

Required tests cover scale/rounding, ordering, whole-evaluation failure,
per-row failure, nominal accounting, equality-sized minibatches, and
concurrent result reassembly.

## 9. Optuna and promotion oracle

After baseline full evaluation:

```text
sampler = TPESampler(seed=seed, multivariate=True)
study = create_study(direction="maximize", sampler=sampler)
study.add_trial(completed baseline trial)
study.optimize(objective, n_trials=num_trials)
```

Baseline is Optuna trial zero. Its categorical parameters are all zero. It
counts toward Optuna startup observations.

Suggestion order is predictor-major:

```text
0_predictor_instruction
0_predictor_demos  # only when demos active
1_predictor_instruction
1_predictor_demos
...
```

For each sampled objective:

1. Optuna opens the sampled trial.
2. Suggest all categorical parameters in the order above.
3. Assemble the candidate.
4. Sample minibatch, if active.
5. Evaluate and log the sampled score.
6. Append under a comma-joined human-readable key such as
   `"Predictor 0: Instruction 2,Predictor 0: Few-Shot Set 1"`.
7. If promotion is due, full-evaluate and `study.add_trial` the promotion while
   the sampled trial remains open.
8. Return sampled score.
9. `study.optimize` then completes/tells the sampled trial.

The promotion trial therefore receives an Optuna trial number before the
currently open sampled trial is completed. Replay must use `ask`, ordered
suggestions, optional `add_trial`, then `tell`.

Promotion cadence:

```text
extra_at_end = 1 if num_trials % full_eval_steps != 0 else 0
adjusted = num_trials + num_trials // full_eval_steps + 1 + extra_at_end
display_trial_num = optuna_trial.number + 1
promote if:
    display_trial_num % (full_eval_steps + 1) == 0
    or display_trial_num == adjusted - 1
```

Promotion:

1. Arithmetic mean all observations for each combination.
2. Stable-sort descending; insertion order resolves tied means.
3. Choose first combination not already fully promoted.
4. Use its first-observed candidate and raw parameters.
5. Full-evaluate, add a completed Optuna trial, then mark fully evaluated.
6. Update winner only on strict improvement.

If every observed combination is already promoted, the helper raises
`ValueError("No valid program found in param_score_dict")`.

The baseline and promotion trials affect when Optuna leaves its default startup
sampling phase. Reconstructing only sampled observations is not equivalent.

Required tests cover baseline insertion, exact suggestions for one/two
predictors, demos on/off, repeated combinations, tied means, promotion
insertion before tell, divisible/nondivisible schedules, final promotion,
promotion exhaustion, strict full-score ties, and transcript replay after
every event.

## 10. Winner and statistics oracle

Baseline initializes `best_score` and `best_program`. Every update is strict
`score > best_score`.

- Non-minibatch sampled trials are winner-eligible.
- Minibatch sampled trials are never directly winner-eligible.
- Baseline and promoted candidates are winner-eligible in minibatch mode.
- The implementation never uses `study.best_trial` to select the result.

`score_data` begins with baseline and then appends sampled and promoted
observations in execution order. Final sorting is stable descending by score,
so earlier observations win equal-score ordering.

Baseline trial log key is `1`; its filesystem save ordinal is `-1`. Sampled
log keys derive from `optuna_trial.number + 1`. Promotion logs use the current
display number plus one and do not include categorical parameter-index fields.

When `track_stats=True`, the best program receives:

- `trial_logs`;
- `score`;
- `prompt_model_total_calls == 0`;
- `total_calls == 0`;
- stable score-sorted `mb_candidate_programs`;
- stable score-sorted `candidate_programs`.

The two call counters are initialized but never incremented. Whetstone's real
effect accounting remains separate.

`score_data.full_eval` is a batch-size classification, not winner eligibility.
The minibatch-equals-validation quirk can therefore place sampled candidates
in `candidate_programs` even though they could not directly become the winner.

When `track_stats=False`, frozen MIPROv2 performs no statistics-attachment
step.

Required tests cover baseline ties, sampled ties, promoted ties, stable
ordering, log numbering around inserted promotions, nominal cumulative call
counts, equality-sized minibatches, and `track_stats` on/off.

## 11. Function/branch to test matrix

| Frozen function or branch | Required Whetstone test |
| --- | --- |
| `MIPROv2.__init__` allowed auto modes | constructor mode matrix and exact defaults |
| `MIPROv2.__init__` model truthiness | explicit/ambient and falsey route resolution |
| `compile` permission branch | warning, rejection, and no-op cases |
| `compile` manual recommendation branch | candidates without trials diagnostic and formula |
| `compile` incomplete manual branch | missing candidates/trials diagnostic |
| `compile` auto conflict branch | explicit candidates and/or trials rejection |
| `seed = seed or self.seed` | compile seed zero fallback |
| `_set_and_validate_datasets` | empty sets, tail split, order |
| `_set_hyperparams_from_run_mode` | all modes, full-size sampling/reorder, `> 50` boundary |
| `_set_num_trials_from_num_candidates` | zero-shot/few-shot and multi-component formulas |
| compile minibatch-size guard | greater-than rejection and equality acceptance |
| `create_n_fewshot_demo_sets` range | `N=1`, `N=2`, `N=3`, ordinary `N` |
| seed `-3` | reset candidate |
| seed `-2` labels branch | labels-only samples and local RNG |
| seed `-2` fallthrough | shared shuffle and size draw |
| seed `-1` | no shared shuffle/size draw |
| generic bootstrap seed | exact shared draw order |
| `_prepare_student_and_teacher` | explicit/default teacher and labeled preloading |
| bootstrap current-demo exclusion | equality exclusion and restoration |
| bootstrap threshold branch | falsey zero threshold and truthy thresholds |
| bootstrap trace mapping | unknown predictor skip and structured augmented demo |
| bootstrap repeated predictor trace | hash-seeded earlier/last choice |
| bootstrap error counter | cumulative within set, reset between sets |
| bootstrap `_train` | augmented-first and cross-predictor raw subset narrowing |
| zero-shot proposal context | three augmented, zero labels, discard after proposal |
| `create_dataset_summary` | initial + 0–9 follow-ups + final summary |
| dataset `COMPLETE` branch | prefix rule, cumulative five skips, no append |
| dataset follow-up exception | summarize accumulated observations |
| proposer source failure | program awareness disabled before proposals |
| proposal tips | exact key order, empty-tip signature shape |
| proposal loop | predictor-major/demo-major order and count |
| proposal demo rotation | current/later/earlier, first-set suppression |
| per-proposal program awareness | describe program, describe component, generate |
| rollout model copy | draw order, shared rollout id, temperature |
| instruction overwrite | original at zero, displaced response charged |
| `Evaluate.__call__` | percentage scaling and two-decimal rounding |
| `eval_candidate_program` | full/minibatch callback route and swallowed exception |
| `_select_and_insert...` | exact categorical names and suggestion order |
| `_get_param_distributions` | exact ranges and order |
| baseline `create_trial` | all-zero trial zero before sampled trials |
| objective evaluation | exactly `num_trials`, ordered minibatches |
| promotion condition | divisible/nondivisible and end-of-run cases |
| `get_program_with_highest_avg_score` | means, stable ties, first observation, exhaustion |
| `_perform_full_evaluation` | add promotion before sampled tell |
| best updates | strict improvement only and eligibility rules |
| result sorting | stable descending full/minibatch lists |
| `track_stats` branch | complete attachment versus no attachment |
| call-count fields | frozen zeros plus separate canonical accounting |

## 12. Acceptance trace

The differential fixture must emit and compare this normalized sequence:

```text
resolved controls and ordered datasets
RNG operation/result transcript
bootstrap seed/branch/train order
teacher rollout and metric acceptance
component trace and final demo contents
dataset-summary requests
tip and rollout-id choices
program/component/instruction proposal requests
instruction pool order and overwritten index zero
baseline evaluation and normalized score
Optuna create/add/ask/suggest/add/tell transcript
candidate parameter selection
minibatch indices and evaluation score
combination observation insertion
promotion mean/rank/selection/full score
strict-best updates
trial-log and candidate-stat ordering
terminal winner and canonical effect accounting
```

Raw DSPy and Whetstone prompt strings are not compared. Separate Whetstone
snapshots assert equivalent semantic sections without DSPy field-description
formatting.
