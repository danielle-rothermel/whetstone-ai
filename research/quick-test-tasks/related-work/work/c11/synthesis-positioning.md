# c11 · House-Convention Canonicalizer — Positioning Synthesis

*Scope: positions c11-as-designed against the 7 related-work papers in this folder, using the per-paper dossiers (each verified 2026-07-22, confidence high) and the c11 candidate page. Citation counts are OpenAlex, fetched 2026-07-22; see §4 for caveats. Throughout, **[E]** marks evidence drawn from the dossiers/manifest/candidate page and **[J]** marks my judgment/inference.*

The 7 papers, for reference:

| key | short name | venue (per dossier) |
|---|---|---|
| 2404.03868 | EDC (Extract-Define-Canonicalize) | EMNLP 2024 |
| 2501.10868 | JSONSchemaBench | arXiv 2025 (not noted peer-reviewed) |
| 2505.20139 | StructEval | arXiv 2025-2026 (no venue listed) |
| 2507.02833 | IFBench | NeurIPS 2025 D&B |
| 2507.21340 | StructText | VLDB 2025 Workshop (TaDA) |
| 2511.03080 | PolyNorm | EMNLP 2025 Industry (per metadata) |
| 2602.14743 | LLMStructBench | arXiv 2026 (not actually cited on c11's page) |

---

## 1. Positioning map

I organize the family on three protocol/design axes that the dossiers repeatedly use to separate these papers. **[J]** These are the axes that actually discriminate the set; other candidate axes (single-shot vs multi-turn) turned out to be nearly degenerate here (see below).

**Axis A — Verifier granularity / output contract.** From *exact whole-string 0/1 match against one reference string* (strictest) → *per-field or per-constraint programmatic boolean* → *graded/fuzzy partial credit* → *soft edit-distance / LLM-as-judge* (most tolerant).

**Axis B — Convention provenance.** From *invented, adversarially standard-deviating rules* (must be learned/inferred, cannot be recalled) → *public standard conformance* (JCS/JSON-Schema/real locale conventions, recallable from pretraining) → *emergent / model-chosen conventions* (no fixed rule at all).

**Axis C — Output shape / task direction.** *Single canonical string (normalize-in-normalize-out)* vs *full structured object (extract or generate)* vs *free natural-language text*.

### Placement (evidence from dossiers)

| Paper / task | A. Verifier granularity | B. Convention provenance | C. Output shape |
|---|---|---|---|
| **c11 (as designed)** | Exact whole-string 0/1, no partial credit **[E, candidate p.17]** | **Invented**, adversarially deviating from RFC 8785 JCS / ISO / slugify **[E, candidate p.17]** | Single canonical string **[E]** |
| PolyNorm | Soft: WER/BLEU/CER, partial credit **[E, dossier]** | Real locale conventions (recallable) **[E]** | Single normalized string **[E]** |
| IFBench | Per-constraint programmatic boolean (strict/loose) **[E]** | Invented-but-*instructed* constraints, held-out from train (not standard-deviating) **[E]** | Free NL text satisfying constraints **[E]** |
| JSONSchemaBench | Schema validator; broad valid-class **[E]** | Public JSON-Schema standard **[E]** | Full JSON instance (any valid one) **[E]** |
| StructEval (T-slice) | Graded: 0.2·syntax + 0.8·keyword/dot-path (+VQA for V) **[E]** | Public formats (JSON/YAML/CSV/TOML) **[E]** | Full structured document **[E]** |
| LLMStructBench | Graded fuzzy: Levenshtein/coercion F1_micro + DOC_micro **[E]** | Public JSON schema (no house rules) **[E]** | Full nested JSON object (extraction) **[E]** |
| StructText | Fuzzy: LLM-judge rubric + BERTScore/Levenshtein, 0.1% numeric tol. **[E]** | Emergent, model-chosen groupings **[E]** | Free NL prose (gen) / JSON dict (extract) **[E]** |
| EDC | Mixed: token-triplet Partial/Strict/Exact F1 + human eval **[E]** | Semantic-equivalence classes, discovered/retrieved schema **[E]** | Set of (subj,rel,obj) triplets **[E]** |

### Crowded and empty regions

**[E]** On **Axis A**, six of seven papers cluster in the *graded / fuzzy / soft* band. Explicitly: StructEval, LLMStructBench, StructText, EDC all use partial-credit metrics; PolyNorm uses soft edit-distance; JSONSchemaBench uses broad-class validation; IFBench is the only one with a hard per-item boolean, and even it offers strict-vs-loose leniency. **[E]** Every dossier's `what_c11_would_add` field independently states c11's exact-match, zero-partial-credit contract is the thing the paper lacks — this is verbatim in all 7.

**[E]** On **Axis B**, the field clusters on *public-standard conformance* (JSONSchemaBench, StructEval, LLMStructBench) or *real/recallable conventions* (PolyNorm, StructText, EDC). IFBench is *invented-but-instructed* (the rule is stated in-prompt each time). **No paper occupies c11's cell: invented conventions that adversarially deviate from a named public standard and are held stable across instances rather than restated per prompt.** **[E]** This is asserted in the EDC, JSONSchemaBench, StructEval, PolyNorm, and IFBench dossiers each.

**[E]** On **Axis C**, the field splits between *full structured object* (JSONSchemaBench, StructEval, LLMStructBench, EDC) and *free text* (IFBench, StructText-gen), with only PolyNorm sharing c11's *single-string* shape — and PolyNorm differs on both other axes.

**[J] Crowded region:** graded-verifier + public-or-real-convention + full-object/free-text. That is where structured-output benchmarking lives in 2025-2026.
**[J] Empty region c11 targets:** exact-match verifier **×** invented-standard-deviating convention **×** single-string output. c11 is the only point in the union of these three extremes. That emptiness is simultaneously its differentiation and its publishability risk (§3): an empty cell can be empty because it is unclaimed, or because it is a narrow instrument the field does not need as a benchmark in its own right.

**[E] Note on the single-shot/multi-turn axis (why I dropped it):** the dossiers show it is nearly degenerate for this family. EDC and IFBench are tagged "both," but in each the "multi-turn" is *not* an oracle-in-the-loop correction protocol — EDC's refinement is self-bootstrapped with diminishing returns, and IFBench's "multi-turn" is constraint-isolation across a fixed 3-turn context scored on one final generation. JSONSchemaBench, StructEval, StructText, LLMStructBench, PolyNorm, and c11 are all single-shot at evaluation time. The axis does not separate the set, so it is not a useful positioning dimension here **[J]**.

---

## 2. Per-paper: what c11-as-designed adds (honest about overlap)

**EDC — Extract, Define, Canonicalize (2404.03868).** **[E]** Real overlap is at the *framing* level only: both use the word "canonicalization" and both motivate it by redundant surface forms needing unification. But EDC canonicalizes toward *semantic-equivalence classes* (mapping "profession"/"job"/"occupation" to one meaning-label, verified by LLM judgment and embedding retrieval over a discovered/target schema), scored with partial-credit triplet F1 plus human annotation, inside a compound multi-call pipeline that also does open extraction and RAG-style retrieval. **[J]** c11 adds a clean isolation of *syntactic rule-following precision*: a single forward pass, whole-string exact match against a fixed ~40-line reference function, with no retrieval, no schema discovery, and no semantic judgment. EDC's "canonicalization" number is entangled with extraction quality and reference-set noise (the paper's own headline failure is scorer/dataset noise, not model error **[E]**); c11's number is not. The overlap is honest but shallow — EDC is, per its own dossier, "primarily c12's parent."

**JSONSchemaBench (2501.10868).** **[E]** Genuine overlap: both are structured-JSON-output evaluations that care about graded, real-world schema complexity, and JSONSchemaBench is c11's actual schema-pool donor (via json-schema-faker, which forks the same ecosystem). But JSONSchemaBench asks for *any schema-valid instance* (an under-constrained generative task scored by a validity checker that accepts a broad equivalence class), studies *constrained-decoding engines* as the primary subject with the LM held fixed, and evaluates conformance to the *public* JSON-Schema standard. **[J]** c11 adds: a single-reference-string exact-match target (vs broad-class acceptance); adversarial deviation from the public standard so the answer cannot be recalled; per-rule diff attribution; and direct measurement of the LM's own rule-following unmediated by a grammar engine. **[E] Correction carried from the dossier:** the corpus claim that JSONSchemaBench is "contamination-proofed by regeneration" is *unsupported* — it is a static public release (itself a contamination risk); c11 must supply its own regeneration layer.

**StructEval (2505.20139).** **[E]** Strong overlap on the format family: StructEval-T (JSON/XML/YAML/Markdown/CSV/TOML generation and conversion) is exactly the substrate c11 draws its canonical-JSON instances from, and both surface format-specific difficulty cliffs and the fluent-vs-format-conforming gap. But StructEval scores with graded keyword/dot-path matching + syntax validity (a document can be 70-90% "right"), evaluates conformance to *real, publicly documented* formats, and dials difficulty by *format breadth* (18 formats × 44 tasks). **[J]** c11 adds zero-partial-credit exact match, invented deviations that cannot be produced from format familiarity alone, per-rule (not per-document) attribution, and *rule-breadth within one format* as the difficulty dial instead of format-breadth. **[E]** StructEval's below-ceiling floor evidence (open-source TOML gen at ~8.6%, some subtasks near single digits) is directly reusable as a floor prior for c11's harder strata.

**IFBench (2507.02833).** **[J] This is c11's closest and most dangerous neighbor.** **[E]** Overlap is deep and mechanical: both are programmatically-verifiable constraint-following with reference-function-as-ceiling, both use synthetic/curated contamination-proof pools, both stratify by rule/constraint category with per-rule failure attribution, and IFBench's "custom" group even contains CSV-quoting/delimiter and date-reformat (`date_format_list`, YYYY-MM-DD) constraints that structurally overlap c11's date/CSV instances. **[E]** IFBench is also NeurIPS 2025 D&B, and it precedents c11's exact "freeze one trivial normalization" plan via its strict-vs-loose accuracy. **[E]** The decisive differences: (1) IFBench verifies a *property* of an otherwise open-ended free-text response (many surface strings pass; there is generally no single correct string), whereas c11's ceiling *is* one specific reference string with whole-match 0/1; (2) IFBench constraints are *stated in natural language per instance* ("use at least N conjunctions"), whereas c11's rules are a fixed latent convention held across held-out instances, not restated in full; (3) IFBench's novelty is not adversarial deviation from a named public standard, and (4) IFBench carries an RLVR-training contribution entirely absent from c11. **[J]** c11 adds a stronger, more brittle single-string signal that catches subtle formatting slips (whitespace/quote/key-order) that IFBench's coarser boolean checks miss, plus the standard-deviation contamination-resistance angle. But because IFBench already owns "verifiable, decomposable, stratified, contamination-proof constraint-following," c11's differentiation from IFBench must be argued precisely, not assumed — it rests almost entirely on Axes A and B, not on the family paradigm.

**StructText (2507.21340).** **[E]** Overlap is narrow and structural: both start from tabular/record data and build a synthetic, regenerable, contamination-resistant pool with fully-specified-by-construction ground truth. But StructText's core direction is table→text (structured→prose), scored with LLM-judge rubrics + fuzzy Levenshtein/BERTScore/0.1%-numeric-tolerance, and its "conventions" are the LLM's own emergent narrative choices, not imposed rules. **[J]** c11 adds the reverse-and-constrained direction (normalize a given value to one string), a deterministic zero-partial-credit verifier, and pre-specified adversarial standard-deviating rules with rule-level diagnostics. **[E]** StructText's practical contribution to c11 is a design lesson, not a task overlap: its structured→prose→structured "information-accessibility gap" and its τ-threshold quality-filtering sweep are templates for c11's generator-quality control.

**PolyNorm (2511.03080).** **[E]** Overlap is real at the task-category level: both are single-forward-pass text→canonical-string transforms, both include date/phone/currency/cardinal/ordinal normalization as explicit categories, and both rely on a small set of curated in-context examples to convey a convention. PolyNorm is the primary evidence that "date/phone/number normalization is a live LLM few-shot task" **[E]**. But PolyNorm scores with *soft* WER/BLEU/CER (partial credit for near-misses), and — critically — its targets are *real locale conventions* that a model may have partially seen in pretraining, and its ICL set is always fully inclusive/in-distribution. **[J]** c11 adds a hard exact-match verifier (isolating whether the whole rule chain executes without any slip, not just "close"), invented standard-deviating conventions that close the prior-exposure gap PolyNorm cannot, and an explicit demo/table-coverage dial to probe copy-from-context vs genuine rule-induction — an axis PolyNorm's always-inclusive curation does not test. **[E]** PolyNorm's iteration effect (Japanese WER 12.32→7.88 via better ICL curation) is direct evidence that demo-set coverage strongly swings accuracy — which is exactly c11's central leakage worry.

**LLMStructBench (2602.14743).** **[E] Provenance flag first:** this paper is *not actually cited on c11's page* — it is a c12 motivation-only placeholder, and the manifest confirms it is a Feb-2026 arXiv preprint with 0 citations. Its identity is genuinely resolved (Tenckhoff et al., HTW Berlin). **[E]** Overlap: both are JSON-centric structured-output tasks scored against a schema/reference, both frame format-compliance vs content-correctness as the live tension, and both are positioned as complementary to JSONSchemaBench. Its most useful transferable finding for c11: structural/schema compliance (its DOC_micro, MK errors) saturates quickly with scale and prompting, while fine-grained value correctness (its WV errors) is the persistent bottleneck at *every* scale including GPT-4o — directly supporting c11's premise that per-rule value-exactness is where discriminative signal lives. **[J]** But LLMStructBench is an *extraction* task (free-text → schema-conformant JSON) with inherently fuzzy ground truth (Levenshtein/coercion credit), varying prompting-strategy × model-scale over a *fixed standard* schema. c11 adds strict exact-match that isolates rule-following from extraction ambiguity, per-rule attribution finer than the MK/MV/WV taxonomy, invented (not merely under-documented) conventions, and the demo-coverage/optimization-leakage axis LLMStructBench has no analogue for. It is more relevant to c12 than c11.

---

## 3. Publishability analysis

**Framing.** **[E]** The project runs an *un-optimized baseline first*: plain RFC 8785 (JCS) canonical-JSON with **no invented house-rule deviations yet** — inputs sampled from pinned schemas via json-schema-faker with fresh seeds, oracle = the trailofbits rfc8785-py reference implementation, output = the canonical JSON string, exact match. Then *potentially* prompt-optimize with COPRO / MIPROv2 / GEPA. This section addresses the conditional: **IF optimization produces large *verified* gains on this task.**

**[J] Critical caveat that colors everything below.** The baseline as specified is *plain JCS with no invented deviations*. Plain JCS canonicalization is a *public, documented, recallable* standard — it sits in the crowded Axis-B "public-standard-conformance" cell, **not** in c11's differentiating "invented-adversarial" cell. So a large optimizer gain on the *plain-JCS baseline* is a result about *prompt-optimizing a model to emit a known public canonical form*, which is materially weaker and more contestable than a gain on the *invented-house-rule* task the candidate page actually designs. The publishability of a headline gain depends heavily on **which** task produced it. I treat both cases.

### (a) The workshop-paper claim, stated precisely

**[J]** If the large verified gain is on the **plain-JCS baseline only** (no invented rules yet), the honest claim is narrow:

> "On a deterministic, single-reference, whole-string exact-match canonical-JSON serialization task (RFC 8785 JCS, instances regenerated per-seed from pinned schemas, oracle = the rfc8785-py reference implementation), prompt optimization with {COPRO/MIPROv2/GEPA} raises small-model exact-match from X% (un-optimized baseline) to Y%, a Δ of Z points, verified against a held-out fresh-seed split with no demo-set/table leakage. The gain is driven by [mechanism identified via prompt inspection]."

**[J]** If the gain is on the **full invented-house-rule task** (post-diet, ≤6 single-pass rules, table-width > minibatch/demo width), the stronger and more novel claim is:

> "On a canonicalization task whose rules are *invented to deviate adversarially* from every public standard (so the answer cannot be recalled from pretraining), with whole-string exact-match scoring and per-rule stratified diagnostics, prompt optimizer {O} closes the gap from floor F% to Y%, and the per-rule breakdown shows the gain concentrates in [rule types], while [computation-heavy rule types] remain at ceiling C%. The optimizer's advantage over {baselines} is D points; ablating table-width-vs-minibatch coverage confirms the gain is genuine rule-induction, not copy-from-context."

**[E]** The candidate page independently flags that a required guardrail is: run the "infer every rule from the examples" one-shot with the largest permitted demo set and treat >50% as a rule-mix bug. That guardrail-result must be reported as part of any claim — otherwise reviewers will assume leakage.

### (b) Which publication line it belongs to

**[J]** Two candidate lines, per the trends doc:
- **This family's own evaluation line** (Areas 1-2: verifiable instruction-following / structured-output compliance — IFBench, JSONSchemaBench, StructEval). c11 as a *new benchmark* slots here.
- **Prompt-optimizer benchmarking** (Area 9: GEPA is an ICLR 2026 *oral*; MAS-PromptBench 2026 exists explicitly to compare optimizers **[E, trends §9]**).

**[J] Verdict:** a *large-verified-gain* result belongs primarily in **Area 9 (prompt-optimizer benchmarking)**, not the family's own evaluation line. Reason: the novel object is not "here is a new capability benchmark" (the family already has IFBench/JSONSchemaBench covering the verifiable-structured-output space densely — see §1's crowded region) but "here is a clean, exact-match, contamination-proof, per-rule-decomposable task on which optimizer O beats optimizer O' by Δ." **[E]** The trends doc explicitly says (§9, and the tie-breaker note) that publishing the quick-test as a small optimizer benchmark "slots straight into this active 2025-2026 line" and is "mutually reinforcing" with a task drawn from areas 1-6 — which c11 is. The family's own evaluation line would only be the home if the headline were about *model* capability (a gap that persists across models/scales), not about *optimizer* deltas.

### (c) Baselines, ablations, comparisons reviewers would demand

**[J]** For an Area-9 (optimizer-benchmark) submission:
1. **Optimizer sweep, not a single optimizer.** COPRO *and* MIPROv2 *and* GEPA, plus a no-optimization few-shot baseline and a hand-written-prompt baseline. **[E]** GEPA's own headline is beating MIPROv2 by ~13%; reviewers will expect c11 to reproduce or contextualize that ordering.
2. **Rollout-budget / cost normalization.** **[E]** GEPA's selling point is 35× fewer rollouts; any Δ claim must be reported *at matched rollout budget*, or the comparison is meaningless.
3. **Multiple models / at least 2-3 sizes.** **[E]** LLMStructBench (22 models) and StructEval (12) are the field norm; a single-model result will be flagged as under-powered. The candidate's small-model framing must be shown across ≥2 small models to argue generality.
4. **Leakage ablation (the load-bearing one).** **[E]** Table-width > minibatch/demo-width must be varied and shown to matter; the "infer all rules from demos" one-shot control must be reported; without these, reviewers will assume the optimizer just transcribed the rule table from a batch (PolyNorm's ICL-curation effect and IFBench's train/test constraint split are the precedents reviewers will cite).
5. **Per-rule / per-stratum breakdown**, showing where the gain lands and that computation-heavy strata behave as predicted (near a known ceiling after the diet).
6. **Ceiling certification.** **[E]** The candidate's own pre-freeze step (reference prompt ≥95% exact-match with three agreeing temp-0 repeats) must be shown, and the temp-0 provider-nondeterminism residual quantified — otherwise a "gain to Y%" is not distinguishable from noise.
7. **Contrast with an existing benchmark's protocol.** **[E]** Reviewers will ask why not just use IFBench's date/CSV custom constraints; the answer (exact-match single-string + adversarial standard-deviation) must be demonstrated to *change the optimizer ranking*, not just restated.

**[J]** For a family-evaluation-line submission (if reframed as a capability benchmark), reviewers additionally demand: many models across scales, saturation analysis, and evidence the task is *not* saturated by frontier models (the JSONSchemaBench/StructEval/IFBench floor-evidence pattern).

### (d) Realistic venues / workshops and why

**[J]**
- **Most realistic: a workshop, not a main track.** A single-task optimizer-delta result is workshop-scale. Natural homes: an ICLR/NeurIPS/EMNLP **workshop on LLM evaluation, prompt optimization, or efficient/agentic methods**, or a **DSPy/prompt-optimization-adjacent workshop** riding the GEPA-ICLR-2026-oral momentum. **[E]** MAS-PromptBench (2026) demonstrates the optimizer-benchmarking sub-field is receptive to exactly this shape of contribution.
- **Datasets & Benchmarks tracks** (NeurIPS D&B — IFBench's home **[E]**) are plausible *only* if c11 is framed as a reusable benchmark with the full model×optimizer matrix, contamination-proofing story, and released generator — i.e., substantially more than a single verified gain.
- **[J] Why not a main conference track:** the empty cell c11 fills is narrow (§1); a main-track structured-output-benchmark paper needs breadth (StructEval's 18 formats, LLMStructBench's 22 models) that a single canonicalization family does not have. The differentiation from IFBench is real but is an *axis-level refinement*, not a new paradigm — that reads as workshop-strong, main-track-thin.

### (e) What would NOT be publishable (recognize early)

**[J]**
1. **A large gain on the plain-JCS baseline with no invented-rule task ever run.** This is "prompt-optimizing a model to emit a known public canonical form" — recallable, contamination-exposed, and not novel against the crowded Axis-B region. Not publishable as a research result; at best an engineering note.
2. **A gain that survives only because of demo/table leakage.** **[E]** If the "infer all rules from demos" control exceeds ~50% or the gain vanishes when table-width > minibatch is enforced, the result is an artifact. Recognize early via the guardrail controls in (c.4).
3. **A gain indistinguishable from temp-0 provider nondeterminism.** **[E]** The candidate's measurement-skeptic lens warns the true ceiling for computation-heavy rules is an unknowable 70-90% band with residual slips landing exactly where temp-0 tokens flip. If Δ is inside that noise band, it is not a result.
4. **A degenerate second-objective result.** **[E]** Output-bytes is *provably degenerate* here (correct canonical strings have fixed length), so any "two-objective optimization win" on this task validates only plumbing, not a real multi-objective gain — publishing it as a two-objective result would be misleading.
5. **A single-model, single-optimizer Δ with no rollout-budget normalization.** **[J]** Reviewers in Area 9 will reject on GEPA-comparison grounds alone.
6. **A gain that does not change the optimizer ranking or the model ranking vs an existing benchmark.** **[J]** If c11 produces the same COPRO<MIPROv2<GEPA ordering as everything else at the same cost, it adds no information — the exact-match/adversarial angle has to *change* something to be worth a paper.

---

## 4. Research-currency verdict

**[E] Citation data (OpenAlex, fetched 2026-07-22, `match: high` for all):**

| paper | cited_by | since_2025 | by_year |
|---|---|---|---|
| EDC (2404.03868, EMNLP 2024) | 10 | 8 | 2026:4, 2025:3, 2024:3 |
| JSONSchemaBench (2501.10868) | 3 | 3 | 2026:2, 2025:1 |
| StructEval (2505.20139) | 1 | 1 | 2026:1 |
| IFBench (2507.02833, NeurIPS 2025 D&B) | 1 | 1 | 2026:1 |
| StructText (2507.21340) | 0 | 0 | — |
| PolyNorm (2511.03080) | 0 | 0 | — |
| LLMStructBench (2602.14743) | 0 | 0 | — |
| *(context) IFEval 2311.07911* | 30 | 22 | 2026:7, 2025:15, 2024:8 |

**Stated caveats (required).** **[E]** These are **OpenAlex** counts, which **undercount** relative to Google Scholar; they are **near-zero-by-construction for 2026-dated papers** (a paper published in 2026 has had almost no time to accrue indexed citations, and OpenAlex indexing lags). So the 0-count papers (StructText, PolyNorm, LLMStructBench) and the low-count 2025 papers (StructEval, IFBench each at 1) are **not** evidence of low impact — their counts are dominated by recency and indexing lag, not reception.

**[J] Verdict: the task family is research-current, and the evidence is consistent with the trends doc rather than contradicting it.**
- The one paper old enough for counts to be meaningful, EDC (EMNLP 2024), has 10 cites with **8 since 2025 and 4 already in 2026** — i.e., its citation stream is *accelerating*, not decaying. That is the strongest single currency signal in the set. **[E]**
- The IFEval anchor (2023) at 30 cites (22 since 2025) confirms the *verifiable-instruction-following* line c11 rides is high-volume and still growing. **[E]**
- The 2025-2026 papers' near-zero counts are uninformative by the stated caveats and should **not** be read as coolness. **[J]** The correct read is: the family is populated by *very recent* work (four of seven papers are 2025-2026), which is itself a currency signal — an active, fast-moving area — corroborated by the trends doc placing verifiable-IF (Area 1) and structured-output compliance (Area 2) at "very high / high" activity and prompt-optimizer benchmarking (Area 9) at "high, self-referential" **[E]**.
- **[J] Honest limitation:** citation counts cannot *distinguish* c11's specific differentiating cell (exact-match × invented-adversarial × single-string) from the crowded neighbors, because no paper occupies that cell to have counts. Currency here is inferred at the *family* level (well-supported) and cannot be established at the *c11-specific* level from citation data (an inherent limit, not a data gap). The candidate page's own research-currency "against" note — that no dedicated currency re-audit ran past the tie-breaker line — remains the honest caveat.

---

## Executive summary (10 lines)

1. **Positioning uses three axes** — verifier granularity, convention provenance, output shape — because single-shot-vs-multi-turn is degenerate for this family (dossier evidence).
2. **The field is crowded** in the graded-verifier × public/real-convention × full-object/free-text region; all seven papers cluster there.
3. **c11 occupies an empty cell**: exact whole-string match × invented adversarially-standard-deviating rules × single canonical string — no paper sits there, per every dossier's own `what_c11_would_add`.
4. **IFBench is the closest, most dangerous neighbor** (NeurIPS 2025 D&B): same verifiable-stratified-contamination-proof paradigm, even CSV/date constraints; c11's differentiation rests narrowly on exact-match + standard-deviation, which must be argued precisely, not assumed.
5. **The plain-JCS baseline sits in the crowded cell, not c11's cell** — a large gain there is a weaker, recallable-standard result; a gain on the invented-rule task is the novel one.
6. **A large verified gain belongs in the prompt-optimizer-benchmarking line (Area 9)**, riding GEPA's ICLR-2026-oral / MAS-PromptBench momentum — not the family's saturated own-evaluation line.
7. **Reviewers will demand**: a full COPRO/MIPROv2/GEPA sweep at matched rollout budget, ≥2 models, the table-width-vs-leakage ablation, per-rule breakdowns, and ceiling certification.
8. **Realistic venue is a workshop** (LLM-eval / prompt-optimization); D&B track only with a full model×optimizer matrix; main track is out of reach for one narrow family.
9. **NOT publishable**: gains from leakage, gains inside the temp-0 noise band, the provably-degenerate output-bytes objective, plain-JCS-only gains, or single-model/single-optimizer deltas.
10. **Currency verdict: family is current** — EDC's citations are accelerating (10, 4 in 2026) and IFEval anchors a growing line; 2025-2026 zeros are recency/indexing artifacts (OpenAlex, undercounts vs Scholar, near-zero-by-construction for 2026), so they signal recency, not low impact; c11-specific currency cannot be established from citation data because its cell is unclaimed.
