# MIPROv2 DSPy-Compatibility Replacement Plan

Status: implementation plan and recovery checkpoint for PR #91.

This plan replaces the existing MIPROv2 approximation. The present adapter is
useful only as evidence that Whetstone can persist proposal and evaluation
artifacts; its pool construction, demonstration synthesis, acquisition policy,
and promotion cadence are not MIPROv2 and must not survive as algorithmic
behavior.

The target is a near-exact behavioral port of DSPy's MIPROv2 onto Whetstone's
prompt, evaluation, identity, and DBOS durability primitives. "Compatible"
means the same public algorithm controls, resolved defaults, semantic prompt
stages, random-number consumption, candidate-space construction, Optuna
interaction, evaluation cadence, strict-improvement rules, accounting, and
result ordering. It does not mean byte-identical natural-language generations:
LLM endpoints are nondeterministic even when requests are identical.

## 1. Frozen reference and compatibility boundary

The acceptance oracle is frozen to:

- DSPy repository commit
  `6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0`.
- Primary implementation:
  `dspy/teleprompt/mipro_optimizer_v2.py`.
- Few-shot and minibatch helpers:
  `dspy/teleprompt/utils.py` and `dspy/teleprompt/bootstrap.py`.
- Grounded proposal behavior:
  `dspy/propose/grounded_proposer.py`,
  `dspy/propose/dataset_summary_generator.py`, and
  `dspy/propose/utils.py`.
- DSPy's resolved test lock uses Optuna `4.8.0`; Whetstone will pin
  `optuna==4.8.0` (including the resolved artifact hash in `uv.lock`) rather
  than rely on DSPy's declared lower bound of `optuna>=3.4.0`.
- The Whetstone optimizer identity must contain the DSPy commit, exact Optuna
  version, Whetstone algorithm version, and every versioned prompt/parser
  schema described below.

Before implementation, record a source-audit matrix that maps every branch in
the functions above to a Whetstone test. A reference change is a deliberate
upgrade requiring a new algorithm version and a new differential oracle; it is
not a dependency refresh.

### Intentional differences

Only these differences are allowed:

1. DSPy `Signature` input/output field descriptions and DSPy program source
   formatting are not reproduced. Whetstone owns complete prompt templates,
   placeholders, task records, component metadata, and rendering rules.
2. DSPy ambient models, metrics, settings, and filesystem logging become
   explicit Whetstone model routes, Eval Configs, reward policy, execution
   policy, content-addressed artifacts, and evidence.
3. DSPy in-memory execution becomes restart-safe Whetstone/DBOS steps.
4. Natural-language proposal text is compared through scripted semantic
   fixtures, not expected to match a live DSPy run byte-for-byte.
5. The port targets Whetstone prompt components, not arbitrary executable DSPy
   `Module` subclasses. The pure bootstrap, proposal, and study seams preserve
   multiple components as separate categorical variables. The integrated
   runtime currently rejects layouts with more than one executable component
   before any effect because Whetstone's rollout primitive exposes only one
   provider trace. The structured trace contract remains for the future
   multi-trace executor (#94).

Any other divergence needs an explicit compatibility note, an identity version
bump, and reviewer approval.

### Whetstone safety adaptations

The frozen DSPy implementation performs almost no eager numeric validation.
For example, it does not explicitly reject zero or negative candidate/trial
counts, demo maxima, minibatch sizes, full-evaluation intervals, or dataset
summary batch sizes, and it does not check that temperatures are finite. Those
values generally fail later, sometimes with unrelated exceptions. Whetstone
will reject structurally invalid values while resolving `Miprov2Control`.
These fail-fast checks are intentional Whetstone safety adaptations, not
claims about DSPy's validation behavior. They must be called out in the
compatibility manifest and bound into the algorithm-version identity.

Whetstone also validates that generated instructions preserve the native
prompt format's required placeholders. DSPy has no equivalent check. This is
an intentional prompt-safety adaptation required to keep candidates executable
on the Whetstone stack.

## 2. Public control contract

Introduce an immutable `Miprov2Control` and a `configure_miprov2(...)` entry
point, following `CoproControl`. No algorithm setting may remain an untyped
entry in runner `hyperparameters`. Conceptual `None` defaults are resolved at
construction and the exact resolved control is persisted before the first
observable effect.

### Construction controls

| DSPy argument | DSPy default | Whetstone representation |
| --- | ---: | --- |
| `metric` | required | exact full-evaluation `EvalConfigRef` plus reward-policy hash |
| `prompt_model` | ambient LM | exact `ProposerConfig`; explicit injected default only |
| `task_model` | ambient LM | bound by the three canonical Eval Configs and provider execution identity |
| `teacher_settings` | `{}` | typed bootstrap teacher route/settings; empty by default |
| `max_bootstrapped_demos` | `4` | `StrictInt`, nonnegative (Whetstone safety adaptation) |
| `max_labeled_demos` | `4` | `StrictInt`, nonnegative (Whetstone safety adaptation) |
| `auto` | `"light"` | `"light"`, `"medium"`, `"heavy"`, or `None` |
| `num_candidates` | `None` | optional positive integer; required in manual mode |
| `num_threads` | `None` | evaluation-service concurrency setting; does not change result order |
| `max_errors` | `None` | explicit resolved evaluation/bootstrap failure policy |
| `seed` | `9` | integer controlling the algorithm RNG and Optuna |
| `init_temperature` | `1.0` | finite proposal temperature, equal to proposer route temperature (Whetstone safety adaptation) |
| `verbose` | `False` | diagnostics only; excluded from decisions |
| `track_stats` | `True` | controls optional detailed result projections |
| `log_dir` | `None` | replaced by content-addressed artifacts; accepted only as a deprecated/no-op compatibility input if exposed |
| `metric_threshold` | `None` | optional finite bootstrap acceptance threshold |

### Compile/run controls

| DSPy argument | DSPy default | Required behavior |
| --- | ---: | --- |
| `trainset` | required | nonempty, ordered task-identity sequence |
| `teacher` | `None` | starts as a deep copy of the base candidate/route; when labeled demos are enabled and that copy is not compiled, the frozen bootstrap resets it and attaches labels before teacher rollouts |
| `valset` | `None` | explicit ordered validation sequence, or DSPy's split below |
| `num_trials` | `None` | manual-mode requirement; auto-derived otherwise |
| `max_bootstrapped_demos` | `None` | optional run override of construction value |
| `max_labeled_demos` | `None` | optional run override of construction value |
| `seed` | `None` | optional run override; preserve DSPy's `seed or self.seed` behavior, including `0` falling back to construction seed |
| `minibatch` | `True` | manual-mode evaluation choice; auto mode may override |
| `minibatch_size` | `35` | positive as a Whetstone safety adaptation; DSPy rejects only a value larger than the resolved validation set |
| `minibatch_full_eval_steps` | `5` | positive as a Whetstone safety adaptation; use the reference's exact trial-number condition |
| `program_aware_proposer` | `True` | controls component/program summary stages |
| `data_aware_proposer` | `True` | controls dataset summary stages |
| `view_data_batch_size` | `10` | positive dataset-summary batch size (Whetstone safety adaptation) |
| `tip_aware_proposer` | `True` | controls the reference tip choice and RNG draws |
| `fewshot_aware_proposer` | `True` | controls inclusion of bootstrapped examples in proposal context |
| `requires_permission_to_run` | `None` | deprecated: `True` rejects, `False` records a warning, `None` does nothing |
| `provide_traceback` | `None` | canonical evaluation failure-detail policy |

Validation must match the reference:

- `auto` accepts only `None`, `light`, `medium`, and `heavy`.
- Manual mode requires both `num_candidates` and `num_trials`. Supplying
  `num_candidates` without `num_trials` raises the reference-style diagnostic,
  including the computed recommendation.
- Auto mode rejects either explicit `num_candidates` or `num_trials`.
- An empty trainset rejects. With no valset, a trainset of fewer than two
  rejects.
- With no valset, validation size is
  `min(1000, max(1, int(len(trainset) * 0.80)))`, taken from the tail; training
  is the prefix.
- An explicit empty valset rejects.
- When minibatching, DSPy rejects only
  `minibatch_size > len(resolved_valset)`. Equality is accepted. Whetstone
  preserves that case even though the frozen evaluation helper routes the
  sampled evaluations through its full-evaluation path.

All additional positive-integer, nonnegative-integer, finite-number, and
placeholder-preservation checks are the explicitly documented Whetstone
safety adaptations above.

Auto settings are exact:

| Mode | N | sampled validation size |
| --- | ---: | ---: |
| `light` | 6 | 100 |
| `medium` | 12 | 300 |
| `heavy` | 18 | 1000 |

Auto mode samples validation with the shared algorithm RNG, sets minibatching
to `len(resolved_valset) > 50`, uses `N` instruction candidates for zero-shot
optimization and `int(N * 0.5)` otherwise, always uses `N` few-shot candidates,
and calculates:

```text
num_vars = component_count * (1 if zero-shot else 2)
num_trials = int(max(2 * num_vars * log2(N), 1.5 * N))
```

The auto validation call always executes `random.sample`, even when the
requested size is greater than or equal to the validation-set length. It
therefore consumes shared RNG and reorders the entire validation set in that
case.

The validation sample occurs before bootstrapping and therefore consumes RNG
state that affects later shuffles, tip choices, rollout IDs, minibatches, and
selection.

## 3. Replace the current data model

Delete the algorithmic use of `DemoPair(rendered_input, observed_output)`,
Cartesian combination materialization, and
`_materialize_demonstrations(...)`. Demonstrations are not instruction text.

Add content-addressed, typed artifacts:

- `Miprov2Example`: task identity, structured task inputs, structured expected
  outputs when labeled, generated component inputs and outputs when augmented,
  `augmented` flag, component identity, and exact source evidence/trace refs.
- `Miprov2DemoSet`: ordered examples for one prompt component.
- `Miprov2ProgramDemos`: an ordered component-id-to-demo-set mapping.
- `Miprov2InstructionPool`: an ordered instruction list per component.
- `Miprov2Combination`: ordered instruction index and, when enabled, demo-set
  index per component.
- `Miprov2TrialObservation`: Optuna trial number, exact parameters and
  distributions, combination identity, sample task identities, score,
  evaluation evidence refs, purpose, and full/minibatch classification.
- `Miprov2PromotionObservation`: chosen combination key, accumulated minibatch
  mean, first-observed parameter record, full score, and evidence refs.
- `Miprov2State`: resolved datasets/config, RNG transcript or reconstructible
  RNG cursor, bootstrap products, proposal products, ordered study event log,
  best full-evaluated candidate, score data, trial logs, and accounting.

The canonical `Candidate` must continue to carry an executable native
Whetstone prompt. Its payload references structured demos and component
instructions separately. The evaluation renderer, not the optimizer, combines
them according to the prompt-format adapter. Candidate identity includes every
component instruction, every ordered demo identity, the prompt-format adapter
identity, and the base route. Changing a demo changes candidate identity and
evidence without altering the instruction string.

For the present one-component Whetstone prompt, use component id
`user_prompt_template`. Keep all collections component-indexed so adding a
second optimizable prompt creates two instruction variables and, unless
zero-shot, two demo variables exactly as DSPy does.

## 4. Exact algorithm phases

Each row is a durable phase with typed input and output state. Split phases
further whenever one row would otherwise contain more than one external model
or evaluation effect.

### A. Resolve configuration and datasets

1. Validate and persist the fully resolved `Miprov2Control`.
2. Apply DSPy's seed fallback and initialize one `random.Random(seed)`.
3. Resolve the train/validation split.
4. Apply auto-mode validation sampling using the same RNG and reference
   `random.sample` ordering.
5. Resolve candidate counts, trial count, minibatch mode, demo maxima, error
   policy, and zero-shot status.
6. Persist task identities in exact order. Resume must reject changed task
   contents, ordering, component structure, routes, prompt adapter, metric, or
   reward policy.

### B. Bootstrap few-shot candidates

Port `create_n_fewshot_demo_sets` and the relevant `BootstrapFewShot` behavior,
not its name or a loose sample generator.

The candidate-set seed sequence is `range(-3, num_candidate_sets - 3)`:

- seed `-3`: reset/zero-shot program;
- seed `-2`: labels-only program only when labeled demos are enabled;
- seed `-1`: unshuffled bootstrap;
- seeds `>= 0`: shuffle a trainset copy with the shared RNG, then draw bootstrap
  size with `rng.randint(1, max_bootstrapped_demos)`.

If labeled demos are disabled, seed `-2` falls through to the generic shuffled
bootstrap branch and consumes the same shared-RNG `shuffle` and `randint`
operations as a nonnegative seed. This is especially important for zero-shot
optimization, whose proposal-context candidates use zero labeled examples.
Preserve this fallthrough and the reference's small-`N` behavior; do not force
all three nominal special sets when `N < 3`.

For each bootstrap compilation:

1. Start from the explicit teacher's deep copy, or the base candidate's deep
   copy. If labeled demos are enabled and that teacher is not compiled, reset
   it and attach labels before rollout. Exclude the current example from each
   teacher component's demos by equality during its rollout.
2. Evaluate training examples in order until the bootstrap maximum is reached.
3. Accept based on metric truthiness, or `score >= metric_threshold` when the
   threshold is truthy, preserving the reference's treatment of `0.0`.
4. Capture component-level generated inputs and outputs as augmented examples.
5. If a component has multiple traces for one example, preserve the reference
   hash-seeded 50/50 choice between an earlier trace and the last trace.
6. Shuffle unbootstrapped validation examples with `random.Random(0)`.
7. Per component, take augmented examples first and then sample labeled/raw
   examples with one `random.Random(0)`. Preserve the reference quirk that the
   `raw_demos` variable is replaced by the sampled subset after each component,
   so a later component samples from the prior component's subset rather than
   independently from the original pool. The labels-only `LabeledFewShot`
   branch instead samples every component from the complete trainset using a
   separate sequential `random.Random(0)`.
8. Enforce the reference's scoped `max_errors` semantics. Error count is
   cumulative across attempts within one `BootstrapFewShot` compilation, but
   resets for every candidate set because each seed constructs a new
   teleprompter. Ordinary candidate evaluation similarly constructs a new
   evaluation executor per invocation.

Zero-shot optimization still bootstraps proposal context with three augmented
examples and zero labeled examples. Those demo candidates participate in
instruction proposal, then are discarded before the optimization variables are
created.

Every teacher rollout, metric result, accepted/rejected trace, sampled labeled
example, and constructed demo set is persisted. The old behavior that derives
several demo sets by taking combinations of one evaluation's rendered strings
is removed.

### C. Grounded instruction proposal

Reproduce `GroundedProposer`'s semantic stages and RNG order while replacing
DSPy `Signature` formatting with a versioned Whetstone prompt-format adapter.

The adapter must support:

- Dataset observations over ordered batches of
  `view_data_batch_size`: one initial observation, at most nine follow-up
  observation calls (ten descriptor calls total), the reference cumulative
  `COMPLETE`/five-skip behavior, and a final two-to-three-sentence summary.
- Program summary and per-component summary when program awareness is enabled.
  Instead of Python/DSPy source and signature fields, supply the Whetstone
  prompt-component graph, complete current template, allowed placeholders,
  rendering rules, and an example execution.
- Up to three augmented task demonstrations, rotating from the current demo
  set through adjacent sets in reference order. Demo-set zero and a lack of
  augmented examples produce `No task demos provided.` Preserve the frozen
  check for the presence of an `augmented` key rather than its truth value.
- The exact six reference tips (`none`, `creative`, `simple`, `description`,
  `high_stakes`, and `persona`) and `rng.choice` timing. Choosing `none`
  removes the tip field from that proposal's request shape because the frozen
  proposer sets `use_tip = bool(selected_tip)`.
- The original/basic component instruction and the reference's empty
  instruction-history behavior.
- One generated instruction output, parsed as a complete replacement for that
  component. Required Whetstone placeholders must remain valid as the explicit
  Whetstone prompt-safety adaptation described above.

Dataset, program, component-summary, and instruction-generation requests get
separate prompt schema tags and parsers. Each rendered optimization prompt,
response, parser result, provider execution policy, and route is persisted as
proposal evidence.

For each component, generate `min(N, number_of_demo_sets)` proposals in
demo-set order (`N` calls when no demos are available). Preserve RNG draws for
tip choice and the `rng.randint(0, 10**9)` rollout id. After generation,
overwrite instruction candidate index zero with the original instruction,
exactly as DSPy does; the displaced first model response remains recorded and
charged even though it cannot enter the search space.

The exact per-proposal endpoint topology is also preserved. After the optional
tip choice and rollout-id draw, every proposal with program awareness enabled
performs a program-description request, a component-description request, and
the final instruction-generation request, in that order. All three requests
use the same copied proposer-model configuration, rollout id, and
`init_temperature`. These summaries are regenerated for each proposed
instruction rather than computed once and reused. Dataset-summary requests are
the exception: they are constructed once before proposal iteration, use
temperature `1.0`, and do not use the per-proposal rollout id.

Do not retain the current prompt that merely asks for a diverse replacement
while listing accepted strings. Do not use retry-until-distinct semantics:
DSPy has a fixed proposal-call topology, and duplicate proposed instructions
remain categorical entries if the reference retains them.

### D. Baseline and categorical Optuna study

1. Full-evaluate the untouched base candidate on the entire resolved
   validation set.
2. Initialize `optuna.samplers.TPESampler(seed=seed, multivariate=True)` and a
   maximize study.
3. Define categorical variables in component order:
   `<i>_predictor_instruction`, followed by `<i>_predictor_demos` when demos
   are active. Choices are `range(pool_size)`.
4. Add a completed baseline Optuna trial with every categorical parameter set
   to zero and the baseline full score.
5. Preserve the reference quirk that the separately evaluated base program may
   not be byte-equivalent to categorical combination zero when an input
   candidate arrived with existing demos.

The numeric score supplied to Optuna and all best/promotion comparisons must
match DSPy's evaluation score: the per-example metric values are averaged,
scaled by 100, and rounded to two decimal places. Bootstrap acceptance remains
based on the unscaled per-example metric value, with the trace supplied to the
metric. Whetstone may retain its canonical raw rewards separately, but the
compatibility projection must make this normalization explicit.

Remove `_seeded_tpe_choice` completely. Optuna must select the combination.
Whetstone must not eagerly materialize the Cartesian product.

For restart safety, persist an ordered **study API transcript**, not an opaque
pickled study. Reconstruct a fresh pinned Optuna study from the seed by
replaying the same API calls:

- add the baseline completed trial;
- for each sampled trial, call `ask`, call `suggest_categorical` in exact
  component/instruction/demo order, and assert the resulting parameters equal
  the persisted observation;
- when a promotion occurred while that sampled trial was running, add its
  completed trial before telling the sampled trial, matching upstream call
  order;
- call `tell` with the persisted minibatch/full score;
- only then ask for the next trial.

Fail closed if replay produces another trial number, parameter set,
distribution, or event order. This both consumes the sampler RNG identically
and avoids using an unstable pickle as durable truth.

### E. Trial evaluation and promotion

Run exactly `num_trials` Optuna objective trials.

For each trial:

1. Assemble a candidate by inserting each selected instruction and structured
   demo set.
2. If minibatching, draw task indices with the shared algorithm RNG using
   `random.sample(range(len(valset)), minibatch_size)` and preserve the sampled
   order. Otherwise use the complete validation set in its existing order.
3. Resolve the canonical evaluation and persist its exact Reward and
   EvaluationEvidence.
4. Record the score under a comma-joined reference combination key in insertion
   order. The program and raw parameters used for promotion are from the first
   observation of that key.
5. In non-minibatch mode, update the best only on strict `score > best_score`.

If the whole frozen evaluation helper raises, it substitutes score `0.0` with
no row results and still increments cumulative evaluation calls by the nominal
requested batch size. Preserve that compatibility projection while recording
the actual Whetstone failure evidence and endpoint effects separately.

When `minibatch=True` and `minibatch_size == len(valset)`, DSPy's evaluation
helper classifies the sampled evaluation as full because the requested batch
size is not smaller than the dataset. The optimizer nevertheless retains
minibatch control flow: this sampled candidate cannot replace the winner
directly and promotion still occurs on the minibatch schedule.

Minibatch promotion must copy the reference's actual numbering rather than an
intuitive "every N trials" rewrite:

```text
run_additional_full_eval_at_end =
    1 if num_trials % minibatch_full_eval_steps != 0 else 0
adjusted_num_trials =
    num_trials
    + num_trials // minibatch_full_eval_steps
    + 1
    + run_additional_full_eval_at_end

trial_num = optuna_trial.number + 1
promote when:
    trial_num % (minibatch_full_eval_steps + 1) == 0
    or trial_num == adjusted_num_trials - 1
```

At promotion:

1. Calculate each observed combination's arithmetic mean.
2. Stable-sort descending by mean, preserving dictionary insertion order on
   ties.
3. Select the first combination not already fully promoted.
4. Full-evaluate its first-observed candidate on the complete validation set.
5. Add that score to Optuna as a completed trial with the exact categorical
   distributions **while the current sampled trial is still open**, then tell
   the sampled trial after the objective returns.
6. Mark the combination fully evaluated and update best only on strict
   improvement.

If no observed combination remains unpromoted, the frozen helper raises
`ValueError("No valid program found in param_score_dict")`. Do not silently
skip the promotion or invent early stopping; either preserve this failure or
record a separately approved compatibility divergence.

Baseline and promoted full scores are the only candidates eligible for final
selection in minibatch mode. There is no arbitrary `returned_proposal_count`
ranking contract in DSPy MIPROv2: return the single best program. If
Whetstone's generic caller requests more than one output, reject the control
rather than invent extra winners.

### F. Final result and accounting

The terminal projection is pure and reconstructs the winner from persisted
full-evaluation evidence. It must not call a model, re-evaluate, truncate an
internal ranking, or accept a minibatch-only candidate.

When `track_stats=True`, persist the reference-equivalent:

- score and selected program;
- ordered trial logs, including exact parameter indices, minibatch/full scores,
  cumulative evaluation calls, and candidate refs instead of filesystem paths;
- all score observations, stable-sorted descending;
- minibatch candidate programs and full-evaluated candidate programs as
  separate collections;
- proposal-model calls and task/evaluation calls.

DSPy's `full_eval` classification in the candidate-statistics lists is based
on whether the requested batch size is at least the validation-set size, not
on winner eligibility. Consequently, the minibatch-equals-validation quirk can
place sampled programs in `candidate_programs` even though they were not
eligible to update `best_program` directly. Preserve this distinction in the
compatibility projection.

When `track_stats=False`, omit those optional projections rather than emitting
empty or partial lookalikes. DSPy's frozen implementation initializes
`prompt_model_total_calls` and `total_calls` to zero and does not increment
them; preserve that field behavior in the compatibility projection, while
Whetstone's separate canonical budget/evidence accounting records actual
calls. Document the distinction rather than silently "fixing" the reference.

Budget debits must be derived from durable external effects: dataset-summary
calls, program/component-summary calls, instruction proposal calls, bootstrap
task calls, minibatch evaluation rows, and full evaluation rows. A rejected or
discarded proposal still consumes its call. Replaying a completed effect
consumes nothing again.

## 5. DBOS and restart model

No step may contain an uncheckpointed loop of external effects.

- Resolve/configure, RNG-only choices, Optuna replay, candidate assembly,
  promotion choice, state folding, and finalization are pure steps.
- Each proposal-model request is its own identity-bound DBOS effect.
- Each canonical evaluation intent is an identity-bound effect whose output is
  the existing Reward/EvaluationEvidence record.
- Bootstrap trace capture uses canonical evaluation/rollout effects and stores
  component traces; it does not call a hidden DSPy runtime.
- Effect keys include run identity, algorithm version, phase, logical ordinal,
  component, candidate/combination identity, ordered task identities,
  Eval Config, route, execution policy, prompt adapter, and prompt schema.
- State transitions bind immutable snapshots. A repeated request returns the
  bound result; a different request at the same logical ordinal conflicts.

Crash-injection tests must stop and resume:

- after configuration and dataset sampling;
- after every bootstrap rollout and demo-set assembly;
- after every dataset/program/component summary call;
- after every instruction proposal, including the overwritten index-zero call;
- after baseline evaluation;
- after Optuna `ask`, candidate assembly, and evaluation;
- immediately before and after promotion insertion and sampled-trial `tell`;
- after the last trial and before terminal binding.

Every resumed run must produce the same effect-key sequence, sampled task
identities, Optuna parameters, promotions, best candidate, trial/stat records,
and result refs as an uninterrupted run, with zero duplicate endpoint calls.

## 6. Identity and provenance

`Miprov2Control.identity_payload()` must bind:

- Whetstone algorithm version and frozen DSPy commit;
- exact Optuna version;
- all resolved construction and run controls;
- base candidate and ordered prompt-component identities;
- ordered training and validation task refs;
- proposer route/config and prompt temperature;
- task/teacher routes through exact Eval Configs;
- reward-policy and provider-execution-policy hashes;
- prompt-format adapter and parser identities;
- dataset-summary, program-summary, component-summary, instruction-proposal,
  bootstrap, and candidate-rendering schema tags;
- error/failure policy;
- Whetstone state/result schema versions.

Every proposal and evaluation artifact links back to that control identity.
Every demo links to its source task, rollout, trace, output, score, and
acceptance decision. Every trial links to the exact pool entries, sampled
tasks, evaluation evidence, and study transcript event. Every promotion links
to all minibatch observations used to compute its mean.

Resume rejects changed source/version identity, dependency version, prompt
adapter, task ordering/content, component structure, model route, generation
config, metric, reward policy, Eval Config, seed, or HPM—even if the next
candidate would coincidentally be the same.

## 7. Verification and differential oracle

### Source-contract tests

- Snapshot constructor and compile arguments/defaults from the frozen source.
- Cover every auto/manual validation branch and exact auto-derived values.
- Cover train/validation splitting, seed-zero fallback, validation sampling
  including full-size sampling/reordering, `N < 3` demo-set behavior, seed
  `-2` fallthrough, zero-shot proposal-only demos, threshold truthiness,
  predictor-to-predictor raw-demo narrowing, and per-candidate-set
  maximum-error reset behavior.
- Assert no old approximation symbols or policies remain:
  `_seeded_tpe_choice`, `seeded_categorical_tpe/v1`,
  `_materialize_demonstrations`, combinatorial demo synthesis, retry-until-
  distinct instruction generation, or eager Cartesian candidate pools.

### Scripted differential oracle

Build a tiny frozen DSPy program and an equivalent one-component Whetstone
prompt graph. Exercise the two-component pure seams separately and assert the
integrated runtime rejects a two-component layout before any paid effect until
the multi-trace executor exists. Feed the executable adapters scripted outputs
keyed by semantic effect rather than raw formatted prompt. Capture a normalized
trace containing:

- configuration and resolved dataset identities;
- every RNG-consuming operation;
- bootstrap seed/mode, training order, trace acceptance, and per-component
  demo contents;
- dataset/program/component summary and instruction proposal topology;
- selected tip, rollout id, demo context order, and instruction pool order;
- Optuna API event order, trial number, categorical parameters, and values;
- minibatch task indices and scores;
- promotion combination, mean, full score, and study insertion position;
- strict best updates, final winner, accounting, and stats order.

The normalized traces must be exactly equal. Raw Whetstone optimization prompts
are separately snapshot-tested to verify equivalent semantic sections without
DSPy field-description formatting. Live LLM text is never an equality oracle.

Run a settings matrix covering:

- auto light/medium/heavy and manual mode;
- zero-shot and demo-enabled search;
- explicit and derived validation sets;
- minibatch on/off, divisible/nondivisible final promotion cadence;
- one executable component, plus two-component pure-seam and fail-fast
  integration coverage;
- duplicate instructions, repeated combinations, tied means and tied full
  scores, including promotion exhaustion after all observed combinations have
  already been promoted;
- small candidate counts and small datasets;
- minibatch size equal to validation size;
- `track_stats` on/off;
- bootstrap failure, proposal failure, evaluation failure, and exact budget
  exhaustion.

### Whetstone integration tests

- Structured demos alter rendered execution and candidate identity without
  altering instruction text.
- EvaluationEvidence, Reward, proposer evidence, and source traces all belong
  to the exact candidate/control.
- Concurrent evaluation completion is folded back into input order.
- Terminal selection cannot reference a minibatch-only candidate.
- Control/result/state validation rejects tampering, missing evidence,
  cross-run refs, reordered observations, changed Optuna parameters, and
  identity drift.
- Full repository Ruff, format, type, unit, and PostgreSQL/DBOS suites pass.

## 8. Adversarial review gates

An independent reviewer who did not implement the phase must return GO after
attempting at least:

1. Auto/manual conflicts and defaults that merely resemble the reference.
2. Zero-shot demo leakage into final combinations or loss from proposal
   context.
3. Rendered-string demos masquerading as structured demos.
4. Original instruction at the wrong index or failure to charge the displaced
   first proposal.
5. RNG drift caused by validation sampling, bootstrapping, tips, rollout ids,
   minibatches, concurrency, or restart.
6. Fresh-Optuna reconstruction that skips historical `ask` calls and therefore
   changes sampler RNG state.
7. Promotion off-by-one errors, promotion inserted after `tell`, repeated
   promotion, unstable tie handling, or selection from full scores instead of
   minibatch means.
8. Final selection of a minibatch-only candidate or non-strict tie
   replacement.
9. Duplicate external effects after crashes at every boundary.
10. Identity omissions that permit changed prompts, routes, datasets,
    policies, HPMs, dependency versions, or source versions on resume.
11. Statistics/accounting that hide discarded proposals, bootstrap calls, or
    promotions.
12. Single-component assumptions that silently conflate variables in a
    multi-component fixture.

All blocking and high-severity findings are fixed and the differential,
crash-recovery, focused, and full-stack suites are rerun before PR #91 is
declared ready.

## 9. Commit checkpoints, ownership, and restacking

Work proceeds in recoverable, pushed checkpoints. During implementation PR #91
may temporarily contain multiple commits; after adversarial review it is
rewritten to one coherent MIPROv2 commit, with this plan retained in that
commit.

1. **Plan checkpoint:** this file, frozen references, and no behavior change.
2. **Contract checkpoint:** pinned Optuna dependency, typed control, exact HPM
   validation/default tests, and source-audit matrix.
3. **Data checkpoint:** structured component/demo/combination/trial artifacts,
   identities, rendering seam, and migration of fixtures.
4. **Bootstrap checkpoint:** exact special sets, teacher traces, threshold and
   sampling behavior, plus crash tests.
5. **Proposal checkpoint:** dataset/program/component summaries, Whetstone
   prompt-format adaptation, tip/RNG topology, evidence, and snapshots.
6. **Study checkpoint:** baseline trial, exact Optuna transcript replay,
   categorical selection, minibatches, promotion cadence, and differential
   traces.
7. **Result checkpoint:** terminal projection, statistics, accounting,
   identity/provenance closure, and integration tests.
8. **Review-fix checkpoint:** adversarial findings resolved and all validation
   green.
9. **Final history checkpoint:** squash/reword PR #91 to the replacement commit,
   restack PRs #92, #93, and #94, force-push the authorized stack, and verify
   each PR's base/head, patch ownership, mergeability, and CI.

PR #91 owns the MIPROv2 algorithm, typed control, structured demo primitives,
Optuna pin, prompt schemas, and optimizer-level tests. PR #92 must be rebased
without absorbing MIPRO-specific changes. PR #93 remains Codex-only. Generic
runner/controller integration belongs to PR #94; once PR #91's final API is
known, PR #94 is updated to construct and persist `Miprov2Control`, drive its
typed phases, and expose its terminal result. Any reusable primitive required
by GEPA should be placed in the lowest correct owning commit, not copied into
PR #92.

At every pushed checkpoint, record the commit SHA and passing focused commands
in the PR body. Never leave descendants based on an obsolete parent after a
history rewrite.

## 10. Explicit non-goals

- Do not preserve backward compatibility with the fake MIPROv2 pool, state,
  candidate ids, or result artifacts.
- Do not rename an approximate sampler, demonstration concatenation scheme, or
  custom cadence "MIPROv2."
- Do not reimplement TPE.
- Do not vendor DSPy or execute arbitrary DSPy programs in production.
- Do not reproduce DSPy Signature field-description formatting.
- Do not claim live LLM output determinism.
- Do not improve reference quirks under the compatibility algorithm version.
- Do not add early stopping, retry-until-distinct behavior, alternate ranking,
  extra winners, or a new budget-based search policy.
- Do not let filesystem logs, wall-clock time, unordered concurrency,
  timestamps, UUIDs, or ambient settings influence decisions or identity.

Completion means the approximation has been removed from PR #91's history, the
frozen differential oracle and adversarial reviewer both return GO, crash
recovery duplicates no effects, all repository checks pass, and the entire PR
stack is correctly restacked and green.
