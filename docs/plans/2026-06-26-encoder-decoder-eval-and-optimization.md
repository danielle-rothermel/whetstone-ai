# Encoder-Decoder HumanEval Eval and Optimization Plan

## Purpose

The next experiment should evaluate a two-stage DSPy flow over HumanEval Plus:

```text
ground truth code
  + encoder prompt template
  -> encoder model
  -> encoded code description
  + decoder prompt template
  -> decoder model
  -> decoded code
```

The experiment has two related goals:

1. Build a durable eval-only DBOS script that can run this encoder-decoder
   setup at scale and report both decoded-code correctness and description
   compression.
2. Shape the encoder-decoder flow so it can become the DSPy "program" optimized
   by existing optimizers, using COPRO as the first concrete example.

The implementation should follow the package split described in the README:
experiment-defining choices stay visible in `scripts/`, while reusable mechanics
move to `src/dr_dspy/` only when they are expected to remain stable across
multiple experiments.

## Goal 1: Durable Encoder-Decoder Evaluation

The first implementation should copy the current eval-only DBOS script and keep
its operational shell: Postgres tables, DBOS runtime setup, experiment-specific
queues, workers, status reporting, repair, and analysis commands. The copied
script should replace the current single-model generation unit with a
two-model encoder-decoder unit.

The current flow is:

```text
HumanEval prompt -> model -> raw generation -> extracted code -> tests -> score
```

The new flow should be:

```text
ground truth code
  -> encoder model
  -> encoded code description
  -> decoder model
  -> decoded generation
  -> extracted decoded code
  -> HumanEval tests
  -> code performance score
```

The scoring phase should also evaluate compression:

```text
encoded code description
  -> compression methods
  -> compression percentages compared with ground truth code length
```

The new prediction row should persist enough state to debug each boundary:

- HumanEval identifiers, prompt, test, entry point, and ground truth code.
- Encoder model, temperature, reasoning config, raw encoded description,
  response metadata, usage metadata, provider cost, and error state.
- Decoder model, temperature, reasoning config, raw decoded generation,
  response metadata, usage metadata, provider cost, and error state.
- Extracted decoded code plus compile, extraction, stdout, stderr, timeout, and
  test-result diagnostics.
- Compression metrics for the encoded description against the ground truth code.

The conservative first queue shape is one DBOS generation workflow that runs
encoder and decoder sequentially, then enqueues the scoring workflow. This keeps
the dependency obvious: decoding cannot happen until the encoded description
exists. A later version can split encoder and decoder into separate queues if
there is real pressure to cache encodings, retry decoders independently, or
evaluate many decoders against the same encoded description.

The existing decoded-code scoring logic can stay close to the current
`score_generated_code` behavior: clean/extract candidates, validate Python
source, select the first compilable candidate, run the HumanEval test in the
subprocess sandbox, and persist structured diagnostics. Compression scoring
should be a separate focused helper that accepts ground truth code and encoded
description, then returns byte counts and ratios for a small fixed set of
lossless compression methods. If that helper proves stable, it belongs in
`src/dr_dspy/`; the experiment-specific choice of which methods to report can
remain in the script.

## Goal 2: Optimization Evaluation Shape

The encoder-decoder flow should also be expressible as a normal DSPy module, so
optimizers can treat it as the program under optimization.

A first module shape would be:

```text
EncodeCode(ground_truth_code -> encoded_description)
DecodeCode(encoded_description -> decoded_code)

EncodeDecodeProgram.forward(ground_truth_code)
  -> encoded_description
  -> decoded_code
```

Training and dev examples should include `ground_truth_code`, `test`,
`entry_point`, and `task_id`, with `ground_truth_code` as the DSPy input. The
metric should evaluate the program prediction by:

1. extracting, validating, and testing `decoded_code`;
2. computing compression metrics for `encoded_description`;
3. combining those results into the scalar score expected by the optimizer.

For COPRO, the simplest metric should be correctness-gated compression:

```text
if decoded code fails tests:
    score = 0.0
else:
    score = compression_quality(encoded_description, ground_truth_code)
```

This avoids rewarding short descriptions that lose information. The exact
compression score can evolve, but the first version should be easy to inspect,
for example a clipped percentage improvement based on compressed encoded length
relative to ground truth code length.

The non-DBOS optimization shape is still the standard DSPy pattern: build
`dspy.Example` train/dev sets, create a metric, evaluate baseline, run an
optimizer, then evaluate the compiled module. COPRO differs in that it optimizes
predictor instructions and output prefixes across the predictors in a module.
The encoder and decoder predictors should therefore remain visible as separate
named predictors inside the DSPy module.

## Longer-Term Direction: COPRO as a DBOS Workflow

The intended final shape is not just "run COPRO locally, then evaluate with
DBOS." The optimization flow itself should eventually become durable DBOS work.

At a high level, that means representing the full optimization routine as a
recoverable workflow:

```text
initialize optimization run
  -> evaluate baseline program
  -> generate candidate encoder/decoder prompt variants
  -> evaluate candidates on train/dev examples
  -> select/update best candidate state
  -> repeat for optimizer depth/breadth
  -> persist compiled program artifact
  -> run final evaluation
```

For COPRO specifically, DBOS needs to preserve optimizer progress across prompt
candidate generation, candidate evaluation, score aggregation, and best-prompt
selection. The durable state should include optimizer configuration, candidate
instructions/prefixes, scores, failures, selected best prompts, and enough
program metadata to resume without changing the meaning of the run.

This is a later design step. The immediate eval-only script should not overfit
to the future DBOS COPRO implementation, but it should avoid choices that make
that direction harder. In particular:

- Keep the encoder-decoder program shape explicit and compatible with DSPy
  modules.
- Keep metric logic factored from DBOS persistence and queue mechanics.
- Persist stable identifiers for examples, model configs, prompt templates, and
  optimizer candidates.
- Keep scoring deterministic enough that optimization comparisons are
  meaningful.

## Script and Library Boundary

The copied eval script should own experiment-defining decisions:

- HumanEval Plus dataset slice and seed.
- Encoder and decoder signatures.
- Default encoder and decoder prompt templates.
- Model sweep dimensions.
- Queue topology and worker CLI.
- Analysis output fields.
- The scalar metric used for optimization experiments.

Reusable library candidates are narrower:

- Compression metric calculation.
- Shared decoded-code extraction and subprocess scoring helpers.
- DSPy `EncodeDecodeProgram` only if more than one script uses exactly the same
  module shape.
- Event/log serialization helpers if the optimization DBOS workflow needs the
  same observability across scripts.

The first pass should prefer a readable copied script over premature
generalization. Move code into `src/dr_dspy/` only after reuse is concrete.

## Open Design Questions

- Should the first DBOS eval persist encoder and decoder outputs in one
  prediction table, or split them into related stage tables once independent
  retries/caching matter?
- Which lossless compression methods are in scope for the first comparison?
- Should the optimization metric use only compression after a full correctness
  pass, or should partial decoded-code diagnostics contribute nonzero signal?
- Should encoder and decoder use independent LMs during optimization, or should
  COPRO initially optimize prompts while using one configured LM context?
- What artifact format should represent a durable optimized encoder-decoder
  program before the DBOS COPRO workflow exists?
