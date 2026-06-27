# Encoder-Decoder HumanEval Optimization Plan

## Purpose

The durable eval-only encoder-decoder HumanEval DBOS workflow has been
implemented. This plan now tracks only the remaining future work: making the
encoder-decoder flow optimizable as a DSPy program, then eventually making that
optimization process durable with DBOS.

The future optimization work should keep following the package split described
in the README: experiment-defining choices stay visible in the experiment flow
or entrypoint, while reusable mechanics move to `src/dr_dspy/` only when they
are expected to remain stable across multiple experiments.

## Future Goal 1: Non-DBOS DSPy Optimization

The encoder-decoder flow should be expressible as a normal DSPy module, so
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

The non-DBOS optimization shape is the standard DSPy pattern: build
`dspy.Example` train/dev sets, create a metric, evaluate baseline, run an
optimizer, then evaluate the compiled module. COPRO differs in that it
optimizes predictor instructions and output prefixes across the predictors in a
module. The encoder and decoder predictors should therefore remain visible as
separate named predictors inside the DSPy module.

## Future Goal 2: Durable COPRO Workflow

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

## Boundaries for Future Work

Future optimization work should reuse the existing eval-only foundations:

- `compression.py` for encoded-description compression metrics.
- `scoring.py` and `human_eval.py` for decoded-code correctness evaluation.
- `dspy_runner.py` for logged DSPy predictor execution where applicable.
- `serialization.py` for DSPy-aware JSON-safe serialization.
- `humaneval_dbos_flow.py` for shared HumanEval DBOS lifecycle mechanics if the
  optimized program is evaluated through the existing DBOS/Postgres path.

Optimization-defining choices should remain visible in the optimization flow or
entrypoint:

- HumanEval Plus dataset slice and seed.
- Encoder and decoder signatures.
- Default encoder and decoder prompt templates.
- Optimizer configuration.
- Model choices for candidate generation and candidate evaluation.
- Scalar optimization metric.
- Compiled program artifact path and format.

Avoid re-planning the completed eval-only DBOS workflow. Add new library
surface only when a future optimization script and an existing evaluation flow
would otherwise share the same code unchanged.

## Open Future Questions

- Should the optimization metric use only compression after a full correctness
  pass, or should partial decoded-code diagnostics contribute nonzero signal?
- Should encoder and decoder use independent LMs during optimization, or should
  COPRO initially optimize prompts while using one configured LM context?
- What artifact format should represent a durable optimized encoder-decoder
  program before the DBOS COPRO workflow exists?
- Should the durable optimization workflow split encoder and decoder stages for
  caching and retry, or keep the sequential stage shape until reuse pressure is
  concrete?
