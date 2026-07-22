# Quick-Test Candidates — Domain: Structured Extraction & Canonicalization to Schemas

Breadth-focused literature scan for the whetstone-ai quick-test task. The task must satisfy the
`quick-test-rubric.html`: single LLM call + exact deterministic programmatic scoring (no sandbox),
output bounded to multiple choice / a few words, default prompts scoring well below a KNOWN ceiling,
gap that closes INCREMENTALLY, difficulty decomposing into independent latent rules, a
synthetic/generatable pool with exact ground truth (hundreds of instances, contamination-resistant),
diagnostic failures, few-shot demos that help, and prompt differences resolvable on 10-20 evals at
temperature 0.

**How this domain maps to the rubric.** "House-style" canonicalizers (canonical JSON, slugs, phone/date
normalizers, checksum validators, CSV/fixed-width serializers) are an unusually good fit: the ground truth
is produced by a deterministic reference function the designer writes, so exact-string scoring is free and
the ceiling is provably 100%. The transformation is governed by a **ladder of independent normalization
rules** (sort keys, strip whitespace, lowercase, transliterate, escape, zero-pad, trim hyphens, compute a
checksum...), which is exactly the "difficulty decomposes into independent latent rules" mechanism behind
incremental gap-closing. Contamination is dodged by inventing a house convention that exists nowhere in
pretraining. The one caveat this literature repeatedly surfaces: if you want per-field partial credit you
must design the scoring metric carefully, but for the quick test a **single exact-match on the whole
canonical string** (or on a bounded field) sidesteps that entirely.

Dates below are as reported by sources; several arxiv IDs surfaced by search carried implausible
future-year prefixes (2602..2607) in this environment and are flagged as **[date unverified]** — treat those
as leads to confirm, not settled citations. The well-established works (IFEval, FollowBench, FoFo,
JSONSchemaBench, StructEval, EDC, RFC 8785) are solidly dated.

---

## A. Direct-use / closest-analog benchmarks

### 1. RFC 8785 JSON Canonicalization Scheme (JCS) — canonical-JSON *as* the task
- **What it is.** An IETF spec (informational, 2020) defining a deterministic byte-exact JSON
  serialization: I-JSON subset, lexicographic sorting of object keys by UTF-16 code units, ECMAScript
  number formatting, minimal string escaping, no insignificant whitespace. Independent implementations
  produce byte-identical output, which is the whole point (hash/sign stability).
- **Citation.** RFC 8785, *JSON Canonicalization Scheme (JCS)*, IETF, 2020.
  https://www.rfc-editor.org/info/rfc8785/ · https://datatracker.ietf.org/doc/html/rfc8785
- **Adaptation for the rubric.** Not a benchmark — it's a **reference spec you weaponize into a generator**.
  Emit random small JSON objects (bounded depth/width, mixed types), ask the model to return the JCS
  canonical form; score exact-string against a reference canonicalizer. This is the near-exact analog of
  the project's own `dr-serialize` work. To dodge contamination and keep the ceiling designer-owned,
  replace the RFC's rules with an **invented house convention** (e.g. sort keys by length-then-reverse-alpha,
  a bespoke escape table, a custom number format) so no pretrained JCS knowledge shortcuts the task.
- **Rubric fit.** Excellent. Deterministic exact scoring; provable 100% ceiling; rules are independent and
  stackable (sorting, whitespace, escaping, number format, unicode handling) → clean incremental ladder and
  inspectable "which rule did the prompt discover"; few-shot demos plainly teach individual rules;
  synthetic and infinite. **Weakness:** genuine JCS is a plausible pretraining hit, so you *must* invent the
  convention; and full canonical JSON output can grow long — bound object size to keep tokens small and
  keep outputs to "a few tokens"-scale, or score only one canonicalized field.

### 2. StructEval — LLM structured-output generation across 18 formats
- **What it is.** Benchmark evaluating generation of renderable (HTML, React, SVG, Mermaid) and
  non-renderable (JSON, YAML, CSV, TOML) formats over 44 task types, with Syntax / Keyword-Matching / VQA
  scores. Reports meaningful headroom: even o1-mini ≈ 75.58 average; open models ~10 pts lower.
- **Citation.** *StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs*, arXiv:2505.20139
  (May 2025). https://arxiv.org/abs/2505.20139
- **Adaptation.** Use only the **non-renderable, programmatically-checkable slice** (JSON/YAML/CSV/TOML) and
  swap its LLM-judge/VQA components for exact-string equality against a synthetically-generated target. Pick
  one format, invent a house style, and generate instances. The value here is the format taxonomy and the
  empirical evidence of a below-ceiling floor, not the harness itself.
- **Rubric fit.** Good as a source of formats and difficulty evidence. **Weakness:** as published it uses
  Keyword/VQA/LLM-judge scoring (not exact-match), some formats are renderable (need a sandbox — disqualified
  by criterion 2), and it's a published set (contamination) so instances must be regenerated.

### 3. JSONSchemaBench — 10K real-world JSON schemas for constrained decoding
- **What it is.** Rigorous benchmark of ~10K JSON schemas spanning constraint types of varying complexity,
  used to compare six constrained-decoding stacks (Guidance, Outlines, llama.cpp, XGrammar, OpenAI, Gemini)
  on efficiency, coverage, and output quality. Frameworks fail notably on GitHub-Hard / JSON Schema Store.
- **Citation.** Geng et al., *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language
  Models*, arXiv:2501.10868 (Jan 2025). https://arxiv.org/abs/2501.10868 ·
  https://github.com/guidance-ai/jsonschemabench
- **Adaptation.** Not a plug-in task (it measures decoder compliance, not a canonicalization gap), but a
  **rich, graded schema pool**: sample a simple schema, generate a valid instance, and make the task
  "return the canonical/normalized instance conforming to schema S" scored by exact-match. Also a useful
  reference for calibrating schema-complexity as a difficulty knob and for the partial-credit caveat.
- **Rubric fit.** Moderate. Great difficulty-graded schema corpus and a clear demonstration that JSON
  structure is hard below the ceiling. **Weakness:** the benchmark's own objective is constrained decoding,
  not prompt-optimizable canonicalization; you'd be borrowing the schemas, not the task; contamination on
  real schemas → regenerate.

### 4. EDC (Extract, Define, Canonicalize) — schema canonicalization pipeline
- **What it is.** Three-phase LLM KG-construction framework: open extraction → schema definition → **post-hoc
  canonicalization** of triples to a (possibly auto-built) schema, with a trained retriever for relevant
  schema elements. The "canonicalize" phase — mapping free surface forms onto a fixed vocabulary — is the
  conceptual heart of this domain.
- **Citation.** Zhang & Soh, *Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph
  Construction*, EMNLP 2024. https://aclanthology.org/2024.emnlp-main.548/ · arXiv:2404.03868 ·
  https://github.com/clear-nus/edc
- **Adaptation.** Distill the canonicalize step into a bounded task: give a surface entity/relation string
  and a small fixed target vocabulary, ask for the exact canonical label. Latent rules = the mapping
  conventions (casing, synonym collapse, unit normalization). Generate synthetically with an invented
  vocabulary → exact-match on the chosen label (multiple-choice-like, bounded output).
- **Rubric fit.** Good conceptual template; bounded output is natural (one label). **Weakness:** real KG
  canonicalization is semantic/fuzzy (many "correct" mappings) which fights determinism — you must constrain
  to a designer-owned invented vocabulary with a single correct answer per instance.

---

## B. Format-following & instruction-constraint benchmarks (mechanism donors)

### 5. IFEval — verifiable format/lexical instruction following
- **What it is.** ~500 prompts with 25 types of *programmatically verifiable* instructions (word/keyword
  counts, casing, JSON-only output, bullet counts, "wrap in quotes", etc.), scored by strict and loose
  accuracy — no LLM judge. The canonical example of "rule-based, exact, reproducible" format eval.
- **Citation.** Zhou et al., *Instruction-Following Evaluation for Large Language Models (IFEval)*,
  arXiv:2311.07911 (Nov 2023). https://arxiv.org/abs/2311.07911 ·
  https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Adaptation.** Borrow the **verifier library** (each instruction type ships a Python checker) but compose
  *many independent constraints per instance* to build a latent-rule ladder, and generate synthetic
  input+constraint combos so instances are fresh. Bound output to a few words / one line so scoring stays a
  simple exact/verifiable check.
- **Rubric fit.** Strong on determinism and diagnostic failures (checker names which rule failed).
  **Weakness:** single-constraint instances are near one-shot guessable by a competent prompt engineer
  (violates criterion 4); you fix this only by *stacking* independent constraints. Published prompts →
  regenerate for contamination.

### 6. FollowBench — multi-level incremental-constraint following
- **What it is.** Benchmark that starts from a base instruction and **incrementally adds constraints (levels
  1→5)** across Content/Situation/Style/Format/Example categories. The level mechanism is essentially the
  rubric's "incremental gap that closes in steps" made explicit.
- **Citation.** Jiang et al., *FollowBench: A Multi-level Fine-grained Constraints Following Benchmark*,
  ACL 2024. https://aclanthology.org/2024.acl-long.257/ · arXiv:2310.20410
- **Adaptation.** Adopt the **level-ladder design pattern** for a synthetic canonicalization task: level N =
  N house rules active. Restrict to the Format constraints that are exactly checkable; replace FollowBench's
  strong-LLM judge with programmatic exact-match; generate fresh instances.
- **Rubric fit.** Excellent conceptual donor for the incremental-rule structure and for measuring "how many
  rules did the prompt recover". **Weakness:** as published it leans on LLM-as-judge for open-ended
  constraints — not directly usable; you keep the *design*, not the scoring.

### 7. FoFo — domain-specific format-following
- **What it is.** 494 instructions requiring adherence to real-world domain formats (medical reports, LaTeX,
  etc.), built via AI-human collaboration. Demonstrates format-following is hard and below ceiling.
- **Citation.** Xia et al., *FoFo: A Benchmark to Evaluate LLMs' Format-Following Capability*, arXiv:2402.18667
  (Feb 2024). https://arxiv.org/abs/2402.18667
- **Adaptation.** Evidence source for "default prompts score below ceiling on format tasks" and a catalog of
  real formats to imitate. Not directly usable: scoring is **LLM-as-judge**, formats are open-ended, and it's
  published. Would need full synthetic regeneration + programmatic checker.
- **Rubric fit.** Weak as a direct task (judge-based, not exact), useful as a floor-evidence and format-idea
  donor.

---

## C. Synthetic-generation methods & extraction benchmarks (pool builders)

### 8. StructText — synthetic table→text benchmark generator (reverse into text→schema)
- **What it is.** End-to-end framework that starts from **tabular ground truth** and generates natural-language
  passages via a plan-then-execute pipeline, yielding key-value extraction benchmarks at scale with exact cell
  ground truth; scored on factuality/hallucination/coherence/numeric-temporal accuracy.
- **Citation.** Shirai et al., *StructText: A Synthetic Table-to-Text Approach for Benchmark Generation with
  Multi-Dimensional Evaluation*, arXiv:2507.21340; VLDB 2025 TaDA workshop.
  https://arxiv.org/abs/2507.21340 · https://github.com/IBM/struct-text
- **Adaptation.** Run it **backwards** as your generator: since the table cells are known ground truth, the
  quick-test task = "extract field X and return it in canonical form", scored exact-match on that one bounded
  field. The plan-then-execute text synthesis gives contamination-free inputs with guaranteed labels.
- **Rubric fit.** Good pool-generation recipe with exact cell truth. **Weakness:** its own metrics are
  multi-dimensional/soft; you use the *generator*, not the metrics. Free-text fields need bounding to a few
  tokens for clean exact scoring.

### 9. DTBench — synthetic document→table extraction
- **What it is.** Synthetic benchmark for document-to-table extraction with programmatically known ground
  truth. **[date unverified — arxiv:2602.13812]**
- **Citation.** *DTBench: A Synthetic Benchmark for Document-to-Table Extraction*, arXiv:2602.13812.
  https://arxiv.org/pdf/2602.13812 (confirm date/venue).
- **Adaptation.** Similar to StructText: use the synthetic generator to produce input docs with exact
  ground-truth cells; reduce the task to a single bounded canonical-field extraction for exact scoring.
- **Rubric fit.** Moderate; promising synthetic-generation lead but must be verified (future-dated ID). Table
  outputs are wide → bound to one field to keep tokens small and scoring exact.

### 10. ExtractBench — complex structured extraction with exact & tolerance metrics
- **What it is.** Benchmark + methodology for complex structured extraction; notably specifies **exact-match
  and tolerance-based numeric comparison** plus semantic array alignment as metrics. **[date unverified —
  arxiv:2602.12247]**
- **Citation.** *ExtractBench: A Benchmark and Evaluation Methodology for [complex structured extraction]*,
  arXiv:2602.12247. https://arxiv.org/pdf/2602.12247 (confirm date/venue).
- **Adaptation.** Mine its **metric design** (exact vs tolerance vs alignment) to justify the quick-test's
  choice of strict whole-string exact-match, and borrow task shapes for a synthetic single-field extractor.
- **Rubric fit.** Moderate; primarily a metric-design reference. Verify provenance before citing.

### 11. LLMStructBench — structured data extraction across prompting strategies
- **What it is.** Benchmarks many LLMs on structured extraction across parsing scenarios and ~five prompting
  strategies; reports that **prompting strategy matters more than model size** and that semantic errors
  persist even when structural validity holds. **[date unverified — arxiv:2602.14743]**
- **Citation.** *LLMStructBench: Benchmarking Large Language Model Structured Data Extraction*,
  arXiv:2602.14743. https://arxiv.org/html/2602.14743v1 (confirm date/venue).
- **Adaptation.** The "prompting strategy > model size" finding is **direct evidence** that prompt
  optimization has real headroom on structured tasks — useful to justify the whole quick-test premise. Borrow
  task templates; regenerate synthetically; bound output.
- **Rubric fit.** Moderate as a task source, strong as motivation evidence. Verify provenance.

---

## D. Deterministic normalizer families (pure synthetic, designer-owned ceiling)

### 12. House-style normalizer family — slugify / phone / date / checksum (invented conventions)
- **What it is.** A **synthetic family you build**, not a published benchmark, drawing on well-defined
  deterministic normalization pipelines: slugification (Unicode NFKD → transliterate → lowercase → strip to
  `[a-z0-9-]` → collapse/trim hyphens → dedupe suffix), ISO-8601 date normalization, E.164 phone formatting,
  and checksum validators/formatters (Luhn, ISBN-13, IBAN mod-97). Each pipeline is an ordered stack of small,
  independent, exactly-checkable rules.
- **Citations / references (rule sources).**
  - Slugify pipeline & Unicode NFKD/transliteration rules — python-slugify conventions, Unicode Normalization
    Form KD; see practitioner write-ups (e.g. peasydesign slug best-practices).
    https://peasydesign.com/guides/slug-url-safe-string-generation/
  - Luhn / ISBN-13 / IBAN mod-97 checksum algorithms — standard definitions (ISO 7064 for IBAN; ISBN-13 EAN
    check digit; Luhn for card numbers).
  - PolyNorm — *Few-Shot LLM-Based Text Normalization for TTS*, arXiv:2511.03080 (2025), as evidence that
    date/phone/number normalization is a live LLM few-shot task. https://arxiv.org/html/2511.03080v1
- **Adaptation.** Pick one family (slugify is the strongest single choice) and **invent a house convention**
  that diverges from any public library (e.g. custom transliteration table, non-standard hyphen rules, a
  bespoke check digit) so pretraining can't shortcut it. Generate hundreds of random inputs; the reference
  function is the exact ground truth; score whole-output exact-match. Latent rules = each pipeline stage;
  few-shot demos teach individual stages; failures are diagnostic ("expected `cafe-2`, got `café_2`" reveals
  the transliteration+separator rule was missed).
- **Rubric fit.** **Best all-around fit in this domain.** Provable 100% ceiling (you own the function);
  fully synthetic + infinite + contamination-proof (invented convention); output is one short slug/number
  (bounded); deterministic at temp 0; long independent rule ladder → clean incremental gap and inspectable
  rule discovery; few-shot measurably helps; checksum/format violations give natural, injectable failure
  cases (criterion 13); a length-budget knob (e.g. max slug length) provides the vestigial second metric the
  rubric suggests. **Weakness:** rules must be tuned so the naive-prompt floor is genuinely low but the ladder
  is long enough to avoid one-shot guessing — i.e. don't make it a single obvious "lowercase and hyphenate"
  step; stack 5-8 non-obvious rules. Per-field partial credit (if you ever want it) needs care, but a single
  whole-string exact-match avoids that.

---

## Cross-cutting notes

- **Strongest direct candidates for the quick test:** #12 (house-style slugify/normalizer family) and #1
  (invented-convention canonical JSON) — both give a designer-owned 100% ceiling, exact scoring, an
  independent-rule ladder, and trivial contamination-proofing. #6 (FollowBench level-ladder) and #5 (IFEval
  verifier library) are the best *design/mechanism* donors to structure the incremental rule ladder and to
  reuse ready-made exact checkers.
- **Recurring caveat across the literature:** most published structured-output/format benchmarks (StructEval,
  FoFo, FollowBench, ExtractBench) lean on LLM-judge or soft/multi-dimensional scoring and on published
  prompts. For this rubric they contribute *task shapes, difficulty evidence, and rule inventories*, but the
  actual quick-test should be **synthetically regenerated with a programmatic exact-match checker**.
- **Partial-credit caveat (from the extraction literature):** per-field scoring invites tolerance/alignment
  metrics (ExtractBench) that break determinism-friendly exact-match. For the quick test, prefer a single
  bounded output scored by exact equality; use rule-count only as *post-hoc* analysis of which latent rules a
  prompt recovered, not as the training signal.
- **Provenance flags:** #9 DTBench, #10 ExtractBench, #11 LLMStructBench surfaced with future-year arxiv IDs
  in this environment; confirm exact dates/venues before citing. The lettered/established works (RFC 8785,
  IFEval, FollowBench, FoFo, EDC, JSONSchemaBench, StructEval, StructText, PolyNorm) are reliably dated.

## Sources
- RFC 8785 (JCS): https://www.rfc-editor.org/info/rfc8785/ · https://datatracker.ietf.org/doc/html/rfc8785
- StructEval: https://arxiv.org/abs/2505.20139
- JSONSchemaBench: https://arxiv.org/abs/2501.10868 · https://github.com/guidance-ai/jsonschemabench
- EDC: https://aclanthology.org/2024.emnlp-main.548/ · https://github.com/clear-nus/edc
- IFEval: https://arxiv.org/abs/2311.07911
- FollowBench: https://aclanthology.org/2024.acl-long.257/ · https://arxiv.org/abs/2310.20410
- FoFo: https://arxiv.org/abs/2402.18667
- StructText: https://arxiv.org/abs/2507.21340 · https://github.com/IBM/struct-text
- DTBench (verify): https://arxiv.org/pdf/2602.13812
- ExtractBench (verify): https://arxiv.org/pdf/2602.12247
- LLMStructBench (verify): https://arxiv.org/html/2602.14743v1
- PolyNorm: https://arxiv.org/html/2511.03080v1
- Slugify best practices: https://peasydesign.com/guides/slug-url-safe-string-generation/
