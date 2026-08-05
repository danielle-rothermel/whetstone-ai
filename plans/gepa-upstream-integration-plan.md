# GEPA upstream-engine integration plan

Status: implemented; frozen-source differential and DBOS recovery review complete

Owner: PR #92 (`rebuild/10-gepa-adapter`)

Stack order: PR #90 COPRO → PR #91 MIPROv2 → PR #92 GEPA → PR #93
Codex adapter → PR #94 canonical runner cutover

## Goal

Replace Whetstone's current GEPA-named approximation with the real, frozen
upstream GEPA algorithm. Run upstream selection, reflective mutation, Pareto
bookkeeping, merging, budget accounting, and final selection on Whetstone's
canonical evaluation and proposer primitives, with DBOS-backed durability and
Whetstone-owned prompt-format adaptation.

The preferred implementation calls the frozen `gepa.optimize(...)` engine
directly. We will reimplement algorithmic code only if a measured durability
constraint makes direct hosting impossible, and then only as a minimal,
auditable patch over the frozen source.

Success means:

- upstream GEPA, not a Whetstone approximation, makes every algorithmic
  decision;
- task and reflection model calls flow only through Whetstone services;
- evaluations, scores, traces, feedback, prompts, responses, and result
  projections retain canonical evidence and provenance;
- interruption and DBOS replay preserve stable logical effect identities;
  `provider_idempotent` proposal execution prevents duplicate provider effects,
  while `at_least_once` explicitly retains the irreducible
  provider-accept-before-checkpoint duplicate window;
- Whetstone can customize the optimization prompts for each native prompt
  format without importing DSPy Signature field formatting;
- the public algorithmic controls and defaults match the pinned DSPy wrapper;
- a scripted differential oracle produces the same logical trace and result
  through upstream-direct and Whetstone-durable executions.

## What is being replaced

`src/whetstone/optimization/gepa.py` currently implements
`whetstone_multi_objective/v1`: a Whetstone-specific correctness/compression
loop with hashed parent selection and `same_minibatch_strict_pareto/v1`
acceptance. That is not GEPA.

The current tool and storage seams may inform the replacement, but none of
these behaviors are compatibility constraints:

- correctness/compression as the hard-coded objective pair;
- same-minibatch two-objective Pareto acceptance;
- hashed parent selection;
- `max_reflection_attempts_per_step`;
- `max_reflection_lm_calls`;
- accepted-candidate-count termination;
- ranking accepted candidates by correctness, compression, and candidate id.

Delete the algorithm, its variant and acceptance-policy names, and tests that
assert those semantics. Do not preserve them behind a "legacy GEPA" mode.

## Frozen references

All source identities below must be persisted in `GepaControl`, asserted at
import/configuration time where practical, and included in result provenance.

| Reference | Frozen value |
| --- | --- |
| DSPy repository commit | `6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0` |
| DSPy wrapper | `dspy/teleprompt/gepa/gepa.py` at that commit |
| DSPy adapter | `dspy/teleprompt/gepa/gepa_utils.py` at that commit |
| GEPA package | `gepa==0.1.1` |
| GEPA repository tag | immutable release `v0.1.1` |
| GEPA repository commit | `b4dbb55b7601dac448cdb836d5a401ca7d9eb920` |
| PyPI wheel SHA-256 | `71ead7c591eafcc727b83509cdc4182f20264800a6ddf8520d61419daeb47466` |
| PyPI sdist SHA-256 | `643fda01c23de4c9f01306e01305dd69facc29bcb34ad59e4cd07e6621d34aa1` |

Before implementation:

1. Add the base `gepa==0.1.1` dependency without DSPy extras unless an audited
   upstream import requires one.
2. Lock the artifact hash in `uv.lock`.
3. Save a source manifest containing the tag, full commit, distribution
   hashes, license, public APIs used, internal APIs used, and hashes of every
   imported internal source file.
4. Verify the wheel/sdist source corresponds to the immutable release commit.
   Treat a mismatch as a stop condition, not as permission to choose one
   silently.
5. Record why every internal import is required. Prefer `gepa.optimize`,
   `GEPAAdapter`, `EvaluationBatch`, `ProposalFn`, and `GEPAResult`; use
   internals only when there is no public seam.
6. Add a CI guard that fails if the installed package version or recorded
   source hashes drift.

Upgrades are separate changes. A run bound to `0.1.1` must never resume under
another GEPA package, source manifest, or Whetstone adapter/prompt schema.

## Reference contract

DSPy delegates the actual search to `gepa.optimize`. Whetstone must preserve
the following topology:

1. Build the seed candidate as an ordered mapping of named text components.
2. Evaluate it on the validation set and initialize per-instance Pareto state.
3. Until the upstream stop condition fires:
   1. run a scheduled merge attempt when upstream schedules one;
   2. otherwise select a parent using the configured upstream selector;
   3. select component(s) using the configured upstream component selector;
   4. sample a reflection minibatch using the upstream batch sampler;
   5. evaluate the parent with trace capture;
   6. optionally skip an all-perfect minibatch;
   7. construct per-component reflective examples;
   8. propose replacement component text;
   9. evaluate the child on the same minibatch;
   10. accept only according to the upstream strict-improvement rule;
   11. full-evaluate accepted candidates according to the upstream validation
       policy;
   12. update candidates, parent lineage, per-instance subscores, Pareto sets,
       discovery counts, merge scheduling, and budgets.
4. Select `best_idx` with upstream aggregate-score ordering.
5. Return the complete upstream result shape when statistics are tracked.

No test may replace this contract with assertions derived solely from the
current Whetstone implementation.

## Exact controls and defaults

Create an immutable, identity-bearing `GepaControl`, following the
`CoproControl` pattern. Resolve all conceptual defaults before binding a run.
The algorithmic surface must match the pinned DSPy wrapper:

| Control | DSPy default/validation | Whetstone behavior |
| --- | --- | --- |
| `auto` | `None`; one of `auto`, `max_full_evals`, `max_metric_calls` is required | same |
| `max_full_evals` | `None` | same conversion to metric calls |
| `max_metric_calls` | `None` | same upstream stopper and accounting |
| `reflection_minibatch_size` | `3` | same |
| `candidate_selection_strategy` | `"pareto"` | same; initial support exactly `"pareto"` and `"current_best"` |
| `reflection_model` | no default; required unless a custom proposer exists | resolved `ProposerConfig`, never ambient |
| `skip_perfect_score` | `True` | same |
| `add_format_failure_as_feedback` | `False` | same |
| `instruction_proposer` | `None` | Whetstone prompt builder/proposer identity; see prompt section |
| `component_selector` | `"round_robin"` | same; also support `"all"` |
| `use_merge` | `True` in DSPy's wrapper | same, despite upstream API's direct default |
| `max_merge_invocations` | `5` | same; reject `None` before upstream arithmetic |
| `failure_score` | `0.0` | same |
| `perfect_score` | `1.0` | same |
| `track_stats` | `False` | same result exposure rule |
| `track_best_outputs` | `False`; requires `track_stats=True` | same |
| `warn_on_score_mismatch` | `True` | same semantic check, with warning captured as evidence |
| `seed` | `0` | same default; canonical runs reject `None` because it selects entropy rather than a replayable RNG seed |
| `valset` | omitted means trainset | same, with identity-bound ordered data ids |
| `teacher` | unsupported | reject |

Match DSPy's auto presets (`light=6`, `medium=12`, `heavy=18`) and copy its
`auto_budget` arithmetic exactly, including validation and dataset-size
dependence. Freeze that arithmetic in golden tests.

The frozen arithmetic is:

```text
num_trials = int(max(2 * (num_components * 2) * log2(num_candidates),
                     1.5 * num_candidates))
V = validation_size
N = num_trials
M = 35
m = 5
budget = V + num_candidates * 5 + N * M
if N > 0:
    periodic_fulls = (N + 1) // m + 1
    extra_final = 1 if N < m else 0
    budget += (periodic_fulls + extra_final) * V
```

`max_full_evals=n` resolves to
`n * (len(trainset) + (len(explicit_valset) if valset was supplied else 0))`.
The omitted-valset case therefore counts the shared train/validation data only
once. The constructor requires exactly one budget mode before compile.

Audit and classify `gepa_kwargs` from `gepa==0.1.1`:

- `batch_sampler="epoch_shuffled"`;
- `merge_val_overlap_floor=5`;
- `stop_callbacks`;
- `use_cloudpickle=False`;
- `val_evaluation_policy="full_eval"`;
- `frontier_type="instance"`;
- `cache_evaluation=False`;
- any candidate selector added by upstream `0.1.1`.

Expose only serializable, identity-bearing variants whose behavior can be
replayed. Initially reject opaque callbacks, samplers, selectors, evaluation
policies, and stoppers; a Python object's claimed identity is not sufficient
to prove that its behavior is replayable. Do not silently ignore a passthrough
argument. Preserve Python's duplicate-key behavior: a passthrough key already
supplied explicitly by the wrapper is an error, not an override. Reject
`reflection_prompt_template` because the Whetstone adapter owns
`propose_new_texts`, exactly as DSPy's adapter seam makes that upstream
template ineffective.

Operational DSPy settings are not optimization HPMs:

- `num_threads` maps, once supported, to Whetstone evaluation concurrency while
  preserving input-order reconstruction; initial canonical runs reject a
  non-`None` value instead of silently ignoring it;
- `log_dir` is replaced by canonical Whetstone storage and must not activate
  upstream checkpoint ownership;
- W&B, MLflow, progress bars, and upstream filesystem callbacks are disabled
  in canonical runs initially;
- custom stop callbacks are accepted only after a deterministic,
  identity-bearing protocol exists.

Document these stack-specific substitutions explicitly; do not assign them
fake matching behavior.

## Direct-engine-first architecture

Use the upstream package for:

- the optimization loop;
- candidate and component selection;
- epoch-shuffled minibatch sampling;
- reflective-mutation orchestration;
- same-minibatch comparison and acceptance;
- per-validation-instance Pareto state;
- full-validation policy;
- merge scheduling, parent selection, merge acceptance, and lineage;
- budget and discovery accounting;
- final best-candidate selection;
- `GEPAResult`.

Whetstone owns:

- run/control identity and configuration validation;
- train/validation dataset identities and ordered materialization;
- candidate conversion between Whetstone and `dict[str, str]`;
- task execution and scoring;
- trace capture and textual evaluation feedback;
- reflection and merge model calls;
- optimization-prompt rendering and response parsing;
- retries and provider execution policy;
- DBOS steps/workflows and effect idempotency;
- ObjectStore records and canonical evidence;
- result projection and terminal binding.

The intended object graph is:

```text
OptimizationRunControl
  -> GepaControl
  -> DBOS GEPA workflow
  -> gepa.optimize(...)
       -> WhetstoneGepaAdapter.evaluate(...)
            -> durable evaluation effect broker
                 -> canonical EvaluationService
       -> WhetstoneGepaAdapter.make_reflective_dataset(...)
            -> pure evidence-to-reflection projection
       -> WhetstoneGepaAdapter.propose_new_texts(...)
            -> durable proposal effect broker
                 -> Whetstone reflection prompt builder
                 -> ProposerTransport
       -> frozen merge proposer
            -> evaluation effect broker only
  -> GEPAResult
  -> canonical Whetstone result + full GEPA result artifact
```

Set `run_dir=None`, `display_progress_bar=False`, `use_wandb=False`, and
`use_mlflow=False`. Upstream files, pickle checkpoints, logging integrations,
wall-clock values, UUIDs, and environment state must not become sources of
truth.

## DBOS deterministic-replay feasibility spike

This is a mandatory go/no-go gate before production integration.

Build a tiny scripted adapter and run the real `gepa.optimize` under the
intended DBOS workflow with:

- two named components;
- distinct train and validation ids;
- `candidate_selection_strategy="pareto"`;
- `component_selector="round_robin"`, then `"all"`;
- merge disabled, then enabled;
- a fixed seed;
- scripted evaluations, trajectories, feedback, and proposer outputs;
- Whetstone effect records keyed as described below.

Inject termination after every external effect boundary and resume in a fresh
process. Assert:

- the engine replays to the identical sequence of effect identities;
- completed effects are read, not executed again;
- candidate indexes, selected parents, selected components, minibatches,
  accepted/rejected decisions, frontier membership, merge decisions, lineage,
  budgets, and final result are identical to uninterrupted execution;
- concurrent evaluation results are reconstructed in requested data-id order;
- no global RNG, hash randomization, iteration over sets, timestamp, UUID,
  logger, temporary path, or callback changes a decision;
- code/control identity drift refuses resume before any new effect.

Run the spike under at least two fresh `PYTHONHASHSEED` values. Natural-language
model output is scripted; this test proves control-flow replay, not endpoint
determinism.

### Go criterion

Use unmodified `gepa.optimize` when replay from the beginning plus durable
effect reuse reconstructs an identical engine trace and result at every crash
point.

### No-go criterion

Do not proceed with the direct call if:

- upstream mutates hidden state that cannot be reconstructed from fixed input,
  seed, and recorded effect results;
- effect order changes after process restart;
- a required side effect bypasses adapter/reflection boundaries;
- upstream requires its filesystem checkpoint to resume correctly;
- serialization or concurrency changes an algorithmic decision.

The spike result, test command, frozen versions, and observed effect trace must
be committed before the main replacement begins.

## Durable effect boundary

Implement a single run-scoped effect broker used by the adapter. The broker is
the only path to task or reflection endpoints. Frozen GEPA `0.1.1` merge uses
the adapter's evaluation boundary, but never the proposal boundary.

An effect identity includes:

- optimization run/control hash;
- frozen GEPA source-manifest hash;
- Whetstone adapter and prompt-schema versions;
- effect kind (`evaluate` or `propose`);
- replay-stable invocation ordinal;
- candidate mapping identity and upstream candidate index when available;
- ordered data ids and `capture_traces`;
- ordered component names;
- rendered prompt hash for proposal effects;
- evaluation config, reward policy, provider route, execution policy, prompt
  adapter, and response parser identities.

The ordinal is a guard, not the sole key. On replay, both ordinal and semantic
payload must match the prior record. A changed call at the same ordinal is a
hard conflict.

Each completed effect persists:

- request and response records;
- status and exhausted failure where applicable;
- ordered input ids;
- candidate/component identities;
- evaluation outputs and per-instance scores;
- optional objective scores;
- trace/trajectory references;
- textual feedback and format-failure evidence;
- provider attempt evidence;
- call-cost and metric-call accounting;
- the exact rendered optimization prompt and raw response for proposal calls.

Retry policy belongs to Whetstone. A provider retry does not become another
GEPA proposal or metric call. GEPA sees one logical result after Whetstone's
configured attempt policy completes.

Proposal durability supports two explicit, identity-bound modes. The default
`at_least_once` mode checkpoints every physical attempt but honestly retains
the irreducible crash window after a provider accepts a request and before
DBOS commits that attempt. `provider_idempotent` requires a transport with a
stable provider-side idempotency contract and is the only mode that claims
exactly-once physical proposal execution across that window. Both modes reuse
completed ObjectStore/DBOS evidence without another endpoint call.

## Upstream adapter

Implement `WhetstoneGepaAdapter` against upstream
`GEPAAdapter[DataInst, Trajectory, RolloutOutput]`.

### `evaluate(batch, candidate, capture_traces=False)`

1. Validate the complete candidate component map against the bound Whetstone
   prompt format.
2. Resolve each `DataInst` to its immutable task/data identity.
3. Submit canonical Whetstone evaluations through the durable effect broker.
4. Reassemble results in exactly the input batch order.
5. Return upstream `EvaluationBatch` with:
   - one output per input;
   - one numeric score per input;
   - optional objective scores when configured;
   - trajectories only when `capture_traces=True`.
6. Map exhausted failures to `failure_score` while retaining the full failure
   evidence.
7. Charge metric calls exactly as upstream charges returned batch entries.

The adapter must not aggregate scores before returning them, manufacture a
correctness/compression objective, or choose acceptance.

### `make_reflective_dataset(...)`

This is a pure projection over already recorded trajectories:

- retain upstream semantic keys `Inputs`, `Generated Outputs`, and `Feedback`;
- select component-specific trace material with the same choice order as the
  pinned DSPy adapter;
- use textual metric feedback when available;
- fall back to `This trajectory got a score of {score}.` as DSPy does;
- honor `add_format_failure_as_feedback`;
- preserve structured multimodal values rather than stringifying them;
- apply `warn_on_score_mismatch` once per run and record the mismatch;
- raise the same terminal condition when no valid reflective examples exist.

Any RNG choice made while selecting trace instances must use the same
seeded/replayed RNG stream as the reference and be covered by the differential
trace.

### Candidate mapping

For a single Whetstone prompt, use a stable component name such as
`user_prompt_template`; do not special-case the engine to a bare string. For
multi-component formats, bind an ordered name-to-component-schema map.

Converting an upstream candidate back to Whetstone must:

- preserve unmodified components byte-for-byte;
- validate required placeholders and rendering constraints;
- generate candidate ids from canonical content plus lineage, not wall-clock or
  random UUID state;
- retain the upstream candidate index separately from Whetstone identity.

## Whetstone-owned reflection prompts

Using upstream control flow does not surrender optimization-prompt control.
`WhetstoneGepaAdapter.propose_new_texts` is the intentional prompt seam.

For each component selected by upstream, in upstream order:

1. Receive the semantic inputs:
   `candidate`, `reflective_dataset`, and `components_to_update`.
2. Resolve the component's Whetstone prompt-format descriptor.
3. Render a Whetstone optimization prompt containing the same semantic
   sections as GEPA `0.1.1`:
   - current instruction/component text;
   - numbered task inputs;
   - generated outputs;
   - evaluation feedback;
   - instruction to infer the task and produce improved instructions.
4. Add format-specific constraints:
   - valid variables/placeholders and their semantics;
   - required rendering/escaping rules;
   - output contract for the replacement component;
   - structured multimodal parts when present.
5. Do not synthesize DSPy Signature input/output field descriptions or DSPy
   output-prefix formatting.
6. Invoke `ProposerTransport` through the durable broker.
7. Parse one replacement component with a versioned Whetstone parser.
8. Validate it against the prompt format before returning it upstream.

The frozen GEPA fence extractor can return an empty string. Native Whetstone
prompt components cannot execute an empty instruction, so empty output is a
terminal proposal-format failure before it reaches GEPA. Persist the raw
response and failure evidence, bind this native-format validity constraint in
the parser/control identity, and never substitute fallback text. For valid
outputs, the extraction and control flow match upstream.

Default proposal topology should remain as close as possible to GEPA's
`InstructionProposalSignature`: current instruction, reflective examples, then
the new-instruction request. Snapshot the rendered prompt for text-only,
format-failure, multimodal, and multi-component cases.

Support format-specific builders through a registry:

```text
prompt format identity
  -> GEPA reflection prompt builder identity
  -> response parser identity
```

The builder and parser identities are part of `GepaControl`. Changing either
refuses resume.

Frozen GEPA `0.1.1` merge has no prompt, parser, or language-model call. It
recombines component text already present in candidate ancestors and owns the
selection and acceptance decisions. Bind its frozen merge-policy/source
identity in the control and result provenance, but do not invent a Whetstone
merge prompt. Differential tests must prove that a successful multi-parent,
multi-component merge produces zero proposal-endpoint effects.

Do not compare endpoint text to DSPy output. Compare rendered prompt semantics,
effect topology, parsing rules, and downstream decisions under scripted
responses.

## Identity and provenance

`GepaControl.identity_payload()` must include:

- DSPy reference commit;
- GEPA package version, repository commit, artifact hashes, and source-manifest
  hash;
- Whetstone GEPA adapter/control schema versions;
- reflection prompt schema/parser tags and frozen merge-policy identity;
- response parser versions;
- prompt-format/component-schema identities;
- reflection `ProposerConfig`;
- distinct evaluation and proposer execution-policy identities;
- proposer prompt-adapter and physical-attempt durability identities;
- evaluation configs and reward-policy identity;
- ordered train/validation data identities;
- resolved budget mode and all HPMs;
- seed and supported upstream passthrough controls;
- failure/perfect-score policy;
- `track_stats` and `track_best_outputs`.

At bind and resume:

- compare the serialized control to the registered adapter/effect broker;
- verify the installed GEPA package/source manifest;
- verify every evaluation and proposer identity;
- reject extra generic hyperparameters or pools that duplicate `GepaControl`;
- reject a changed dataset order even when the set of ids is unchanged.

Every candidate/result must be traceable to the control, upstream candidate
index, parent indices, proposal/evaluation evidence, and discovery metric-call
count.

## Result fidelity

Persist a typed, lossless `GepaDetailedResult` projection of upstream
`GEPAResult` containing:

- all candidate component mappings in upstream order;
- `parents`, including multi-parent merge lineage;
- `val_aggregate_scores`;
- `val_subscores` keyed by validation data identity;
- `per_val_instance_best_candidates` with set membership normalized only for
  serialization;
- `discovery_eval_counts`;
- `best_outputs_valset` when enabled;
- `total_metric_calls`;
- `num_full_val_evals`;
- `seed`;
- `best_idx`;
- Whetstone candidate and evidence references for every upstream candidate;
- source/control identity.

Also preserve `val_aggregate_subscores`,
`per_objective_best_candidates`, and `objective_pareto_front` when upstream
returns them. Do not silently discard fields added within the pinned result
schema.

When `track_stats=False`, still persist the minimum internal result needed for
durable correctness and audit, but expose only the DSPy-compatible terminal
surface. When `track_stats=True`, expose the complete detailed result.

`output_count` must not change GEPA's best-candidate rule. If Whetstone supports
returning more than one candidate, define that as a separate projection over
the complete upstream result, with an explicit ordering contract and tests; it
must not alter the engine run.

## Differential and recovery test oracle

Create a standalone scripted oracle using the frozen package, not Whetstone
implementation details. The same scenario runs:

1. directly through `gepa.optimize` with a scripted upstream adapter; and
2. through Whetstone/DBOS with semantically identical scripted effects.

Capture an event trace with:

- seed evaluation;
- parent/component selection;
- sampled ordered data ids;
- `capture_traces` flag;
- reflection dataset;
- proposal request/response;
- same-minibatch parent/child scores;
- accept/reject/skip decision;
- full-validation evaluations;
- frontier changes;
- merge schedule, parents, decision, lineage, and proof that no proposal
  effect occurred in the merge subphase;
- cumulative metric calls and discovery counts;
- final result.

Require exact equality except for explicitly normalized Whetstone record ids
and intentionally different rendered prompt-format text. For optimization
prompts, compare a semantic normalized form and independently snapshot the
Whetstone rendering.

Required test matrix:

- all DSPy defaults and constructor validation;
- `auto=light|medium|heavy` budget goldens;
- explicit `max_full_evals` and `max_metric_calls`;
- trainset-as-valset and distinct valset;
- parent selectors `pareto` and `current_best`;
- component selectors `round_robin` and `all`;
- one and multiple components;
- perfect-score skipping on/off;
- failure score and no-valid-trajectory behavior;
- format-failure feedback on/off;
- score/feedback mismatch warning behavior;
- duplicate and unchanged proposals;
- strict ties and stable best-index tie handling;
- merge disabled/enabled, accepted/rejected, overlap floor, invocation limit,
  and multi-parent lineage;
- exact exhaustion before/after evaluation and proposal boundaries;
- track statistics and best outputs;
- multimodal reflective examples;
- every supported evaluation policy, sampler, frontier type, and cache mode;
- interruption after every evaluation, proposal, state/result persistence, and
  merge boundary;
- resume with a fresh process and no live adapter objects;
- conflicts from changed source, prompt schema, parser, model route, evaluation
  config, dataset order, or HPM.

Assertions for every recovery case:

- identical terminal result and event trace;
- no duplicate completed logical evaluation or proposal effects; physical task
  endpoint guarantees remain those of the bound evaluation execution policy;
- no duplicate provider calls in `provider_idempotent` mode;
- in `at_least_once` mode, an injected accept-before-checkpoint crash exposes
  the documented duplicate-call window rather than claiming exact-once
  behavior;
- no lost evidence;
- identical budget/call count;
- identical candidate/lineage order;
- no use of an upstream checkpoint directory.

## Minimal-fork fallback policy

If the feasibility spike is no-go, stop and document the exact blocker before
changing upstream code.

A fallback fork may:

- expose serialization of upstream engine/state at a stable boundary;
- inject an effect interface;
- inject Whetstone state load/save hooks;
- remove nondeterministic logging/filesystem coupling;
- make ordering explicit without changing the chosen order.

It may not:

- rewrite candidate selection, Pareto logic, reflective mutation, merge logic,
  acceptance, budgets, or final selection;
- copy fragments into unrelated Whetstone control code;
- introduce a Whetstone-specific objective or stopping rule;
- track upstream `main`.

Vendor the minimum files from commit
`b4dbb55b7601dac448cdb836d5a401ca7d9eb920`, preserve copyright/license
headers, and keep a patch series plus source/diff hashes. Add an upstream-direct
versus patched-engine differential suite. The fallback is acceptable only when
the algorithmic event trace remains exact.

## Adversarial review gates

Use a reviewer who did not implement the integration. They receive the frozen
source, DSPy wrapper, this plan, and the differential fixtures.

The reviewer must actively try to prove:

- Whetstone, rather than upstream, is making an algorithmic decision;
- an LLM/evaluation call bypasses Whetstone;
- replay can unexpectedly duplicate a call outside the bound durability
  policy, or reuse a semantically different call;
- RNG or unordered iteration drifts after restart;
- merge behavior or lineage is incomplete;
- aggregate-best logic was confused with per-instance Pareto membership;
- metric-call accounting differs at a boundary;
- a prompt-format or parser change can resume under an old identity;
- DSPy field-description formatting leaked into Whetstone prompts;
- a result field, rejected proposal, failure, or evidence reference was lost;
- `track_stats=False` changes algorithm behavior;
- a package/source upgrade can occur without an identity change.

Blocking severity:

- any algorithm/control-flow divergence;
- duplicate external effect outside the explicitly bound durability policy, or
  any untracked external effect;
- incorrect budget/result/frontier/lineage;
- unsafe resume;
- missing source or prompt identity;
- tests that only validate the Whetstone implementation against itself.

Merge requires a written `GO` with no unresolved blocking or high-severity
finding.

## Implementation and commit checkpoints

Each checkpoint must leave its branch buildable, include focused tests, and be
pushed to PR #92 before the next begins.

1. **Plan checkpoint**
   - commit this document on PR #92;
   - restack PRs #93 and #94;
   - verify CI before algorithm work.
2. **Frozen-source checkpoint**
   - add `gepa==0.1.1`, lock and source manifest;
   - add installed-version/source-hash guard;
   - add a direct upstream smoke test.
3. **Replay-spike checkpoint**
   - add scripted upstream adapter, event-trace oracle, DBOS crash injection;
   - record explicit direct-engine `GO` or `NO-GO`.
4. **Control checkpoint**
   - add `GepaControl`, exact defaults/validation, data/component identity;
   - add auto-budget and configuration golden tests.
5. **Evaluation-adapter checkpoint**
   - add upstream adapter types, candidate mapping, evaluation and trajectory
     projection through Whetstone services;
   - prove ordered results, failures, feedback, and metric-call accounting.
6. **Prompt checkpoint**
   - add versioned format-specific reflection builders/parsers;
   - route mutation proposals through Whetstone;
   - add prompt snapshots and identity-conflict tests.
7. **Engine checkpoint**
   - call upstream `gepa.optimize` with side effects disabled except through the
     broker;
   - enable selectors, perfect-score behavior, validation policy, and budgets;
   - delete the fake GEPA loop.
8. **Merge/result checkpoint**
   - enable upstream merging;
   - add lossless result projection, lineage, frontier, stats, and best outputs.
9. **Durability checkpoint**
   - complete crash matrix and fresh-process resume;
   - prove exact result equality and policy-accurate call behavior:
     exact-once provider calls for `provider_idempotent`, and the explicit
     accept-before-checkpoint window for `at_least_once`.
10. **Adversarial-fix checkpoint**
    - land reviewer fixtures and fixes;
    - obtain final `GO`.
11. **Stack checkpoint**
    - rewrite PR #92 to the intended cohesive commit history;
    - restack PRs #93 and #94;
    - push all affected branches;
    - require lint, type checking, unit tests, DBOS/Postgres tests, focused
      differential tests, and stack topology checks to pass.

During active work, push recovery commits after each checkpoint. Before final
handoff, consolidate only when doing so does not discard the recovery history
the user requested; if the PR stack requires one commit per PR, preserve
checkpoint tags/refs or a published recovery branch until the final rewritten
stack is verified remotely.

## PR ownership and stack discipline

- PR #91 owns MIPROv2 primitives that are genuinely shared and introduced
  there.
- PR #92 owns the GEPA dependency, source manifest, control, adapter, prompt
  integration, engine hosting, GEPA results, and GEPA-specific runner changes.
- PR #93 remains the Codex adapter and receives only mechanical restacking.
- PR #94 owns generic canonical-runner integration only; move GEPA-specific
  behavior down to PR #92.

After every PR #92 rewrite:

- restack #93 onto #92 and #94 onto #93;
- preserve each downstream PR's own tree delta;
- confirm base/head relationships and intended commit count remotely;
- run CI on every rewritten head;
- do not report completion until remote checks are green.

## Non-goals

- Reimplementing GEPA from the paper or from a prose description.
- Preserving the current Whetstone GEPA variant or its tests.
- Importing DSPy or reproducing DSPy Signature field formatting.
- Matching nondeterministic natural-language model outputs.
- Letting upstream call task/reflection endpoints directly.
- Letting upstream filesystem checkpoints, W&B, or MLflow become durable
  authority.
- Tracking a floating GEPA release or repository branch.
- Supporting arbitrary unidentifiable Python callables in canonical runs.
- Adding new Whetstone objectives, selectors, acceptance rules, or stopping
  rules under the GEPA name.
- Optimizing only one flattened string if the Whetstone prompt format has
  multiple independently addressable components.

## Definition of done

The replacement is done only when:

- the fake algorithm is gone;
- the frozen upstream engine or approved minimal patch makes all GEPA
  decisions;
- the direct differential suite and full crash-recovery matrix pass;
- Whetstone-specific reflection prompts are customizable, identity-bound, and
  free of DSPy field formatting;
- controls/defaults, budgets, result fields, Pareto state, merges, and lineage
  match the frozen reference;
- successful merge iterations make zero proposal-endpoint calls;
- all task and proposer calls have canonical Whetstone evidence;
- adversarial review returns `GO`;
- PRs #92–#94 are correctly restacked and pushed;
- all remote CI checks are green.
