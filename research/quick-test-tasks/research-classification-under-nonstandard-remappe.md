# Literature scan: Classification under nonstandard / remapped label semantics

**Domain owner note.** Breadth-focused scan for the whetstone-ai quick-test task. Goal: find
existing tasks / benchmarks / dataset-generation methods that could be **used or adapted** as the
single cheap quick-test task that validates the prompt-optimizers (Eval, COPRO, MIPROv2, GEPA,
Codex agent) end to end. Up to 12 candidates, **not ranked or narrowed**. Rubric essentials
(from `design/quick-test-rubric.html`): single LLM call + exact deterministic scoring (no sandbox);
output bounded to MC or a few words; default prompts score well below a KNOWN ceiling; gap closes
INCREMENTALLY through prompt improvements (not one-shot guessable); difficulty decomposes into
independent latent rules; synthetic/generatable pool with exact ground truth, hundreds of
instances, contamination-resistant; diagnostic failures; few-shot demos measurably help;
prompt-quality differences resolvable on 10-20 evals at temperature 0.

This domain is the seed domain ("classification under nonstandard label semantics") and maps
directly onto the brainstorm's **Theme F** (`design/brainstorm.md`): contrarian sentiment
codebooks, cipher-category tagging, inverted MC, priority/override policies. The research base has
two useful pillars: (1) a mature ICL literature quantifying how much models follow *stated label
mappings* vs *pretrained priors* — this is the mechanism that makes default prompts score near
chance with a known 100% ceiling; (2) rule-based / decision-list classification and cipher /
codebook generation methods that give the incremental, decomposable ladder and the synthetic
generator with exact ground truth.

---

## How to read the rubric-fit notes

The strongest fit is almost never an off-the-shelf benchmark (those are contaminated, use natural
labels, or lack decomposable latent rules). The realistic path is: **take a generation method or
task template from this literature and regenerate it synthetically** with (a) an opaque per-instance
codebook that defeats the prior, (b) 3-5 independent latent rules with precedence/override so the
score ladder is incremental, and (c) a bounded MC or short-token output for exact scoring. The
foundational ICL papers below are mostly *evidence that the mechanism works* rather than drop-in
tasks; the rule/cipher/tabular generators are the drop-in adaptation targets.

---

## Candidate 1 — Flipped-Label ICL (Wei et al., "Larger language models do ICL differently")

**What it is.** The canonical flipped-label / semantically-unrelated-label (SUL-ICL) setup. Take a
standard classification task, systematically flip labels in the in-context demonstrations (positive
→ negative), or replace labels with semantically-unrelated symbols (foo/bar). Measures whether the
model follows the *in-context mapping* or its *pretrained prior*. Key result: overriding semantic
priors is emergent with scale; large models drop to well-below-chance (e.g. code-davinci-002
90% → 22.5%) when labels are flipped, small models stay flat.

**Citations.**
- Wei et al., "Larger language models do in-context learning differently," 2023. arXiv:2303.03846 — https://arxiv.org/abs/2303.03846

**Adaptation for the rubric.** This is a *methodology*, not a fixed dataset — good. Regenerate over
synthetic instances (not SST/TREC) to dodge contamination. Convert to a single-call task: give the
codebook + rules in the *instruction* rather than only via demos, so the default (no-mapping) prompt
scores at the prior (near chance under a flipped map) and the ceiling prompt that states the full
mapping scores ~100%. Bound output to a single label token. Add independent latent rules
(base map + inversion trigger + carve-out + precedence) to get the incremental ladder — a bare flip
is one-shot guessable.

**Rubric-fit notes.** Excellent evidence base for "default prompt scores below a known ceiling" and
"few-shot demos measurably help." Weakness: a pure label flip is a *single* rule, so on its own it
fails the "decompose into independent latent rules / not one-shot guessable" criterion — must be
layered. Also the emergence-with-scale finding means results depend on the quick-test model being
large enough to be *able* to override priors; verify the chosen small/cheap model can climb the
ladder at all (see Candidates 3 and 12 for the small-model caution).

---

## Candidate 2 — Symbol Tuning / SUL-ICL benchmark suite (Wei et al., 2023)

**What it is.** Symbol tuning replaces natural-language labels with arbitrary symbols (foo/bar) so
the model must learn input→label mappings rather than lean on label semantics. Ships an evaluation
suite of SUL-ICL tasks (arbitrary-symbol classification) and flipped-label robustness evals across
NLU tasks. Directly operationalizes "the label carries no usable prior."

**Citations.**
- Wei et al., "Symbol tuning improves in-context learning in language models," EMNLP 2023.
  arXiv:2305.08298 — https://arxiv.org/abs/2305.08298 · ACL: https://aclanthology.org/2023.emnlp-main.61/

**Adaptation for the rubric.** Borrow the arbitrary-symbol label convention (ALPHA/BETA/GAMMA or
foo/bar) as the label space for the quick-test — this is exactly the brainstorm's "opaque codebook."
Generate synthetic inputs with a programmatic ground-truth rule set. Bound output to one symbol.
Add precedence/override rules for the ladder.

**Rubric-fit notes.** Strong conceptual fit for the label-space design (opaque symbols defeat the
prior). The suite itself uses real NLU datasets (contaminated) and is oriented toward *fine-tuning*,
not prompt optimization, so use it as a design pattern rather than a dataset. Symbol labels alone
are trivially learnable from a few demos, so on their own they satisfy "few-shot helps" but risk
being one-shot guessable — again, needs layered rules.

---

## Candidate 3 — Semantic Anchors: "Why Small LLMs Cannot Flip Their Labels" (2025)

**What it is.** Treats LLMs as prompt-induced classifiers; compares natural vs inverted
demonstrations across 8 classification tasks and 8 open models (1–12B). Finding: with inverted
demonstrations, small models produce **zero** coherent anti-semantic classifiers — "semantic
override rate remains exactly zero" in the 1–12B few-shot regime; ICL mostly re-projects inputs
onto stable pretrained semantic directions rather than remapping labels.

**Citations.**
- A. P. Krishna Kumar, "Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their
  Labels," 2025. arXiv:2511.21038 — https://arxiv.org/abs/2511.21038

**Adaptation for the rubric.** Primarily a **cautionary calibration source** rather than a task.
Use it to set the floor/ceiling expectations and to choose the quick-test model: if the cheap model
is ≤~12B, a *pure* label inversion may be uniformly unlearnable (floor stays at floor regardless of
prompt) — which would break criterion 4's incremental-gap requirement. Mitigate by making the rules
*compositional and instruction-stated* (not inversion-via-demos-only), which the study suggests is
more tractable than demo-only remapping.

**Rubric-fit notes.** Very relevant, very recent, directly quantifies the risk that a nonstandard-
label task is *too hard* for a cheap model. Weakness: it argues the naive version of this whole
domain may not produce an incremental ladder on small models — take it seriously when picking model
size and when deciding how much to lean on instruction vs demonstration to convey the mapping.

---

## Candidate 4 — In-Context Fixation (Liu, 2026)

**What it is.** Shows that the *label slot content* in demonstrations acts as an exhaustive answer
vocabulary. Homogeneous labels collapse accuracy to ≤12% across 6 models (0.8–8B) and 4 tasks;
with nonsense tokens {foo,bar,vex,nit,orb} the model puts 42-67% mass on the demonstrated set and
<0.2% on the true class. Generalizes to 4-way classification and multitoken verbalizers ("very
positive"), decomposing into format-level and content-level components. Includes mechanistic
patching localizing the effect.

**Citations.**
- M. Liu (Amazon), "In-Context Fixation: When Demonstrated Labels Override Semantics in Few-Shot
  Classification," 2026. arXiv:2605.08295 — https://arxiv.org/abs/2605.08295

**Adaptation for the rubric.** Use the nonsense-token label vocabulary design and the 4-way variant
as the label space and difficulty knob. The "format-level vs content-level" decomposition maps
neatly onto independent latent rules (one rule about the label *format*, another about the *content
mapping*) — useful for the decomposability + diagnostic-failure criteria. Regenerate synthetically.

**Rubric-fit notes.** Recent, directly on-domain, and its decomposition into independent components
is a gift for criteria 10-11. Weakness: it studies small models (≤8B) and documents *pathological
collapse*, so like Candidate 3 it warns that the ladder may be non-monotonic on cheap models. Also
it's about demo-induced behavior; the quick-test wants prompt-instruction-driven behavior, so adapt
the framing.

---

## Candidate 5 — Rectifying Demonstration Shortcut / label bias (Kang et al., 2024)

**What it is.** Characterizes the "demonstration shortcut": models rely on superficial label-bias
patterns in demonstrations instead of the underlying input→label logic, hurting generalization when
labels/formats change. Provides an evaluation across demonstration formats, example sets, and
alternative label assignments.

**Citations.**
- Kang et al., "Rectifying Demonstration Shortcut in In-Context Learning," NAACL 2024.
  arXiv:2403.09488 — https://arxiv.org/abs/2403.09488

**Adaptation for the rubric.** The alternative-label-assignment evaluation harness is a template for
generating instances where the "obvious" (prior-driven) answer is wrong and only the stated mapping
gives the right label. Regenerate synthetically with an opaque codebook; bound output; add rules.

**Rubric-fit notes.** Good on "default prompt scores low because it takes the shortcut" and diagnostic
failures (you can see the shortcut answer vs the correct one). Weakness: built on standard NLU
datasets (contamination), and the "shortcut" is a single failure mode — needs layered rules for the
ladder. Metadata was partly obscured in the PDF; confirm exact author list/venue before citing.

---

## Candidate 6 — LLMTabBench: synthetic tabular classification with LLM-distilled decision rules (2026)

**What it is.** Binary tabular classification, zero-to-few-shot. The LLM-synthetic suite contains 24
datasets, each generated to mirror a real-world counterpart, **together with the threshold-based
decision rules the generator produced**. The generator is asked for three measurable, threshold-based
rules mapping features → a binary label; a row is positive if ≥1 rule fires. Evaluates combinations
of few-shot examples, task descriptions, and LLM-distilled decision rules.

**Citations.**
- "LLMTabBench: Evaluating LLMs on Binary Tabular Classification From Zero to Few Shots," 2026.
  arXiv:2605.24417 — https://arxiv.org/abs/2605.24417

**Adaptation for the rubric.** This is the closest **drop-in generator**: a program emits both the
rows and the ground-truth rule set, giving exact labels and hundreds of instances for free. Adapt by
(a) making the rules *latent* (not given in the default prompt) so the default scores at base rate
and the ceiling prompt that states all rules scores ~100%; (b) expanding from OR-of-3-thresholds to
a **decision list with precedence/override** so partial rule knowledge earns partial score
(criterion 10); (c) relabeling classes with opaque symbols to defeat the prior; (d) bounding output
to the class token / a 4-way MC.

**Rubric-fit notes.** Strong on synthetic generation, exact ground truth, hundreds of instances,
decomposable rules, few-shot help, and diagnostic failures ("which rule did the model miss"). This
is arguably the best-fit *method* in the scan for the incremental-ladder + decomposability criteria.
Weaknesses: tabular inputs are longer than MC stems (watch token cost, criterion 3), and the base OR
structure is order-independent — you must add precedence to get true rule interaction rather than a
bag of independent thresholds. Very recent; verify the released artifact/license.

---

## Candidate 7 — Decision lists / optimal rule lists as a synthetic label oracle

**What it is.** Classical ordered rule lists (decision lists): a sequence of if-then rules; an
instance is labeled by the *first* rule that fires. Rich literature on constructing sparse/optimal
rule lists (MaxSAT / MIP formulations). Not an LLM benchmark — a **generator of latent, ordered,
precedence-bearing labeling functions** with exact ground truth.

**Citations.**
- Angelino et al., "Learning Certifiably Optimal Rule Lists," KDD 2017 / JMLR 2018 —
  https://arxiv.org/abs/1704.01701
- Yu et al., "Optimal Rule Sets and Lists for Binary Classification (MaxSAT)," 2020 —
  https://arxiv.org/abs/2104.10751 (representative of the line)

**Adaptation for the rubric.** Programmatically sample a short ordered rule list over synthetic
categorical features; the first-match semantics *is* the precedence/override structure the rubric
wants. Each rule the prompt captures adds score; the "first-match" ordering makes rule interactions
non-trivial and not one-shot guessable. Present features as a short structured stem; output = the
label token or a 4-way MC. Contamination-proof by construction (nothing published).

**Rubric-fit notes.** Excellent on criteria 7-11 (synthetic, known ceiling, decomposable independent
rules, first-match precedence, diagnostic "which rule fired"). This is the archetypal
"independent-rules with override" task the brainstorm's Theme F "Priority Override Router" already
sketches. Weakness: it's a build-it-yourself generator, not a ready benchmark, so more engineering.
Also, over-simple feature encodings can be one-shot solved by a competent engineer — control
difficulty via rule count, feature arity, and depth of the override chain.

---

## Candidate 8 — Instruction Hierarchy / priority-override policy classification (Wallace et al., 2024; Control Illusion 2025)

**What it is.** Instruction-hierarchy work trains/evaluates models to resolve *conflicting*
instructions by source priority (System > Developer > User > Tool), using synthetic conflict data.
"Control Illusion" (2025) shows current models often fail to honor stated precedence when
instructions conflict — i.e., precedence resolution is a genuine, unsolved, incremental skill.

**Citations.**
- Wallace et al., "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions,"
  2024. arXiv:2404.13208 — https://arxiv.org/abs/2404.13208
- Geng et al., "Control Illusion: The Failure of Instruction Hierarchies in Large Language Models,"
  2025. arXiv:2502.15851 — https://arxiv.org/abs/2502.15851

**Adaptation for the rubric.** Recast as a *classification* task: an instance carries several flags,
and a set of override rules with a stated precedence order determines the final label (e.g. incident
→ P0/P1/P2/P3/SUPPRESS, where "security forces P0," "region downgrades," "paying customer never
suppress," with precedence). This is exactly the brainstorm's "Priority Override Router." Output =
5-way label. Generate synthetically; make precedence latent so the default prompt mishandles
conflicts.

**Rubric-fit notes.** Very strong on the override/precedence ladder (criterion 4 + 10) and on 2025
research interest (tie-breaker). Diagnostic failures are clean ("model applied region-downgrade
before security-override"). Weakness: the source literature is about *safety/prompt-injection*, not
classification accuracy, so you're adapting the mechanism, not reusing a dataset. Need to ensure the
5-way label space + flag encoding stays short (criterion 3) and that precedence conflicts are
frequent enough that a no-precedence prompt visibly loses points.

---

## Candidate 9 — CipherBank / CipherBench / dynamic Caesar-cipher benchmarks (2024-2025)

**What it is.** Benchmarks for decoding ciphers — CipherBank (ACL Findings 2025) and CipherBench
(3 tiers: common like Base64/ROT13, uncommon, novel user-designed) — plus the "Towards Contamination
Resistant Benchmarks" argument that Caesar ciphers are *dynamically generatable* (infinite instances,
different shifts = different tasks), which makes them contamination-resistant. Directly the
brainstorm's Theme F "cipher-category tagger" and Theme C "cipher chains."

**Citations.**
- "CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography," ACL
  Findings 2025. arXiv:2504.19093 — https://arxiv.org/abs/2504.19093
- "When 'Competency' in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel
  Ciphers," ICLR 2025. arXiv:2402.10601 — https://arxiv.org/abs/2402.10601
- "Towards Contamination Resistant Benchmarks," 2025. arXiv:2505.08389 —
  https://arxiv.org/abs/2505.08389

**Adaptation for the rubric.** Use the *dynamic per-instance cipher key* as the mechanism that
defeats memorization and the prior. But for the quick-test, invert the emphasis: instead of full
decode (unbounded output, char-arithmetic noise), make it a **cipher-category classifier** — decode
just enough to emit a single MC/short-token label under a per-instance symbol→category map. Layer
rules (base polarity map + co-occurrence → mixed-tag + suffix rule + cancellation) for the ladder.

**Rubric-fit notes.** Strong on contamination-resistance and active 2025 research (tie-breaker).
Weakness the brainstorm already flags: **char-arithmetic ciphers add per-token noise that threatens
temp-0 resolvability on 10-20 evals (criterion 5)** — keep the transform to table lookups + a
classification output rather than long free-form decode. A full-decode cipher task also risks
unbounded output (criterion 3) and one-shot vulnerability for simple shifts.

---

## Candidate 10 — Synthetic few-shot rule-induction benchmarks: MIR-Bench, WILT, HERO'S JOURNEY (2025-2026)

**What it is.** A cluster of synthetic inductive-reasoning benchmarks where the model must infer a
*hidden rule* from examples: MIR-Bench (many-shot input→output function induction), WILT (Wason
2-4-6-style multi-turn rule discovery), HERO'S JOURNEY (complex rule induction in text games), and
ARISE (iterative rule induction + synthetic data generation for text classification).

**Citations.**
- "MIR-Bench: Many-shot In-context Reasoning benchmark," 2025 (see survey refs) —
  https://arxiv.org/abs/2502.09933
- "WILT: A Multi-turn, Memorization-Robust Inductive Logic Benchmark," 2025 —
  https://arxiv.org/abs/2410.10998
- "HERO'S JOURNEY: Testing Complex Rule Induction with Text Games," 2026. arXiv:2606.02556 —
  https://arxiv.org/abs/2606.02556
- "ARISE: Iterative Rule Induction and Synthetic Data Generation for Text Classification," 2025.
  arXiv:2502.05923 — https://arxiv.org/abs/2502.05923

**Adaptation for the rubric.** These supply *hidden-rule + synthetic-generation* templates. For the
quick-test, the optimizer's job is to *write a prompt that states the rules* (rules are given to the
designer, latent to the model), so take the generators but flip the framing from "model induces rule"
to "prompt supplies rule, model applies it." Constrain to a classification output. WILT's
memorization-robust design and MIR-Bench's function library are the reusable pieces.

**Rubric-fit notes.** Good on synthetic generation, decomposable rules, few-shot help, and
diagnostic failures. Recent, active research interest. Weaknesses: WILT/HERO'S JOURNEY are
*multi-turn* (violates single-call criterion 1 unless simplified); several emit longer outputs than
MC (criterion 3); and pure rule-*induction* difficulty can be high-variance across instances,
threatening temp-0 resolvability on 10-20 tasks. Use the generators, not the interaction protocols.

---

## Candidate 11 — Option-label / format-remapping sensitivity in MC (label-format bias line, 2024)

**What it is.** Work showing MC accuracy shifts drastically when the *label symbols* change
(alphabetic vs numeric vs Roman) even with explicit instructions — e.g. reported ~30-point MMLU drop
for Roman vs numeric labels. Demonstrates models carry priors over label *tokens*, not just
semantics, and that instructions to use a nonstandard label convention are imperfectly followed.

**Citations.**
- Representative: "Deconstructing Instruction-Following" and related MC-format-bias analyses, 2024
  — https://arxiv.org/abs/2601.18554 (and the broader "LLMs are not robust to option ID" line,
  e.g. Zheng et al. 2024, arXiv:2309.03882 — https://arxiv.org/abs/2309.03882)

**Adaptation for the rubric.** Use a **nonstandard option-label convention** as one latent rule:
e.g. "answer with the Greek letter two positions after the correct option," or "map A/B/C/D →
a per-instance permuted symbol set." This is the brainstorm's "Inverted Multiple-Choice." Compose it
with content rules so it's more than a relabeling. Output = single MC token.

**Rubric-fit notes.** Cleanly bounded output, exact scoring, and it directly defeats the label prior
(criterion 4/8). Weakness the brainstorm flags: a *bare* inverted-MC convention is **HIGH one-shot
risk** — a competent engineer states "shift the answer by +2" in one line and hits the ceiling. The
author's own note: use it as a **floor/plumbing check**, not the main incremental task, unless
layered with several interacting conventions. The cited effect sizes also depend on model; verify on
the cheap model.

---

## Candidate 12 — Contrarian sentiment codebook (synthetic-family, brainstorm-native)

**What it is.** Not from a single paper — a **synthetic family** synthesizing the ICL-flipped-label
findings (Candidates 1-5) with the decision-list/precedence structure (Candidates 7-8). Inputs are
short texts; labels are opaque codes (ALPHA..DELTA) under: a base sentiment→code map, a
sarcasm-inversion trigger, a grudging-praise trigger, a neutral-fact carve-out, and a precedence
order among triggers. Output = single code token. This is the brainstorm's "Contrarian Sentiment
Codebook."

**Citations (grounding).**
- Flipped/SUL-ICL basis: Wei et al. 2023 (arXiv:2303.03846, 2305.08298) — as above.
- Sentiment flip / steering evidence (2025): "Belief Dynamics Reveal the Dual Nature of In-Context
  Learning and Activation Steering," 2025. arXiv:2511.00617 — https://arxiv.org/abs/2511.00617
  (shows models can be driven between standard and flipped sentiment label spaces predictably).

**Adaptation for the rubric.** Build the generator: sample base sentiment with a template, inject
sarcasm / grudging / neutral-fact features probabilistically, apply the precedence rules
deterministically for the ground-truth code. Default prompt ("classify sentiment") scores near the
prior/base-rate because it ignores the codebook and the inversion triggers; each rule the prompt
captures adds score; full-rule prompt ≈100%. Opaque codes defeat the prior (criterion 8); output is
one token (criterion 3); demos of the triggers measurably help (criterion 12).

**Rubric-fit notes.** Best alignment with *all* core criteria simultaneously — this is why the
brainstorm nominates it. It inherits the flipped-label evidence base (default scores low, ceiling
known) and adds genuine independent-rule decomposition + precedence for the incremental ladder and
diagnostic failures. Brainstorm-flagged risk: **one-shot risk MEDIUM-HIGH** — a small fixed codebook
can be enumerated in one paragraph by a diligent engineer; mitigate with 4+ interacting rules, a
per-instance key, and/or compositional labels so no single paragraph captures the full mapping.
Also inherits Candidate 3/4's caution: confirm the cheap quick-test model is large enough to climb
the ladder rather than sitting at the floor regardless of prompt.

---

## Cross-cutting observations

- **Two reusable ingredients dominate.** (1) *Opaque/flipped label space* (Candidates 1-5, 11) makes
  the default prompt score low with a known 100% ceiling. (2) *Ordered rule list / precedence*
  (Candidates 6-8) gives the incremental, decomposable, diagnostic ladder. The best quick-test task
  combines both — which is exactly what Candidate 12 (and the brainstorm's Theme F) does.

- **The realistic deliverable is a generator, not a benchmark.** Every published benchmark here is
  either contaminated (real NLU datasets), unbounded-output (full cipher decode), multi-turn
  (WILT/HERO'S JOURNEY), or single-rule (bare flip / inverted-MC). LLMTabBench (Candidate 6) and the
  decision-list literature (Candidate 7) are the most directly reusable *methods* for building the
  synthetic pool with exact ground truth.

- **Biggest domain-wide risk: floor immovability on cheap models.** Candidates 3 and 4 (2025-2026)
  show that ≤~12B models may produce a *zero* override rate / pathological collapse under label
  inversion. If the quick-test model is small, a nonstandard-label task can sit at the floor for
  *every* prompt, killing the incremental gap (criterion 4). Mitigations: convey rules via
  instruction (not demo-only inversion), keep per-rule difficulty low, and pilot the floor/ceiling
  on the actual quick-test model before committing.

- **Second risk: one-shot guessability.** Bare relabeling / bare flip / bare inverted-MC are all
  one-paragraph-solvable (criterion 4 fail). Every candidate needs 3-5 interacting rules with
  precedence and ideally a per-instance key to stay off the ceiling for a competent prompt engineer.

- **Tie-breaker (research interest).** The flipped-label / semantic-prior-override question is very
  active in 2025-2026 (Candidates 3, 4, and the belief-dynamics/steering work), as is instruction-
  hierarchy precedence (Candidate 8) and contamination-resistant dynamic generation (Candidate 9).
  A quick-test built on a synthetic opaque-codebook-with-precedence generator sits on live research
  ground and could grow into its own "do models follow stated mappings or priors, and can prompt
  optimization recover the mapping" study.

## Sources
- https://arxiv.org/abs/2303.03846 — Larger LMs do ICL differently (Wei et al., 2023)
- https://arxiv.org/abs/2305.08298 · https://aclanthology.org/2023.emnlp-main.61/ — Symbol tuning (Wei et al., 2023)
- https://arxiv.org/abs/2511.21038 — Semantic Anchors: why small LLMs cannot flip labels (2025)
- https://arxiv.org/abs/2605.08295 — In-Context Fixation (Liu, 2026)
- https://arxiv.org/abs/2403.09488 — Rectifying Demonstration Shortcut (2024)
- https://arxiv.org/abs/2605.24417 — LLMTabBench (2026)
- https://arxiv.org/abs/1704.01701 · https://arxiv.org/abs/2104.10751 — decision/rule lists
- https://arxiv.org/abs/2404.13208 · https://arxiv.org/abs/2502.15851 — instruction hierarchy / Control Illusion
- https://arxiv.org/abs/2504.19093 · https://arxiv.org/abs/2402.10601 · https://arxiv.org/abs/2505.08389 — cipher benchmarks / contamination-resistant generation
- https://arxiv.org/abs/2502.09933 · https://arxiv.org/abs/2410.10998 · https://arxiv.org/abs/2606.02556 · https://arxiv.org/abs/2502.05923 — rule-induction benchmarks
- https://arxiv.org/abs/2601.18554 · https://arxiv.org/abs/2309.03882 — MC option-label format bias
- https://arxiv.org/abs/2511.00617 — belief dynamics / sentiment-flip steering (2025)
