# Quick-Test Candidates — Constructed-Language Translation & Layered Ciphers/Encodings

Breadth-focused literature scan for the whetstone-ai quick-test task (validate COPRO / MIPROv2 /
GEPA / Codex-agent optimizers before the HumanEval+ code-compression run). Domain seed:
**constructed-language translation and layered ciphers/encodings** (brainstorm Themes B & C).

Rubric anchors used to score fit (see `design/quick-test-rubric.html`):

1. Single LLM-call node + one eval node (no encoder→decoder chain).
2. Exact deterministic programmatic scoring, no sandbox.
3. Bounded output — multiple choice or a few words.
4. Default prompt scores well below a KNOWN ceiling; gap closes **incrementally** (not one-shot guessable).
5. Prompt-quality differences resolvable at temp-0 on 10–20 evals (effect ≥10 pts > residual noise).
7. Synthetic/parameterized pool, hundreds of instances, exact ground truth.
8. Contamination-resistant (not a memorizable published set).
9. Known ceiling/floor/reference prompt.
10. Difficulty decomposes into independent latent rules.
11. Diagnostic failures ("expected X, got Y" reveals which rule broke).
12. Few-shot demos measurably help.

**Recurring caveat from the brainstorm:** character-arithmetic ciphers (Caesar/ROT/shift schedules)
risk *criterion-5 failure* — LLMs do char-level bijections stochastically, so temp-0 outputs are
noisy and prompt effects get swamped. The cipher literature below is unanimous that even reasoning
models sit at ~40-45% exact-match on char-level ciphers. **Mitigation across all cipher candidates:
score against a closed vocabulary with nearest-match snapping, or convert decode → multiple-choice,
so the residual char noise doesn't leak into the score.** Constructed-language/morphology candidates
generally fit the rubric better than raw ciphers because the "rules" are lexical/structural, not
per-character arithmetic.

BREADTH over depth — 12 candidates below, unranked. Each is tagged `existing-benchmark`,
`adaptation`, or `synthetic-family`.

---

## 1. MTOB — Machine Translation from One Book (Kalamang) · `adaptation`

**What it is.** ICLR 2024 benchmark (Tanzer, Suzgun, Visser, Jurafsky, Melas-Kyriazi). A model is
given ~several hundred pages of field-linguistics reference material (grammar textbook + bilingual
word list + 375 parallel sentences) for **Kalamang** (<200 speakers, ~zero web presence) and must
translate EN↔Kalamang. Frames the task as L2 learning from a *single book* rather than corpus
mining. LLM baselines: 44.7 chrF (Kalamang→EN), 45.8 (EN→Kalamang) vs. 51.6/57.0 for a human who
learned from the same book. Strong contamination resistance by construction.

**Adaptation the rubric needs.** MTOB itself violates criteria 1-3 hard: outputs are full
sentences, scoring is chrF (not exact match), inputs are hundreds of pages (huge token cost). To
use it as the quick test you would **not use Kalamang data directly** — instead borrow only the
*paradigm* ("learn language from a compact in-context grammar, then translate a short phrase") and
regenerate a tiny synthetic micro-grammar (see #7 below). If used near-verbatim: shrink to a
one-page grammar snippet, restrict test items to fixed-template 3-6 word phrases, and replace chrF
with exact-match on a canonicalized gloss.

**Rubric-fit notes.** Excellent conceptual template and directly on-domain; strong 2024-2026
research interest (MTOB is a magnet paper). But as-shipped it fails 1/2/3/5 (long inputs, soft
scoring, sentence-length outputs, chrF noise defeats 10-20-task resolution) and is a fixed corpus
(fails 7/8 at the instance level — only one language, can't generate hundreds of independent
instances). Best treated as *inspiration + a source of realism*, not the deployed task.

Sources:
- A Benchmark for Learning to Translate a New Language from One Grammar Book (2023/2024) — https://arxiv.org/abs/2309.16575 · project: https://lukemelas.github.io/mtob/ · code: https://github.com/lukemelas/mtob

---

## 2. LingOly — Olympiad-Level Linguistic Reasoning Puzzles · `existing-benchmark`

**What it is.** NeurIPS 2024 benchmark (Bean et al.). 1,133 problems over 90+ low-resource/extinct
languages, six formats (incl. **Rosetta** = translate from paired examples, **Pattern**,
**Match-Up**), five difficulty levels, six topics (phonology, morphology, syntax, semantics, number
systems, compound). Rosetta-format items are essentially "here are N glossed pairs, translate this
one" — exactly the in-context grammar-induction shape the domain wants. Hard: top model ~35% on the
hardest tier; models are well below ceiling → satisfies criterion 4's headroom naturally.

**Adaptation the rubric needs.** LingOly is a *published fixed set* → contamination risk (fails 8)
and the authors themselves report a no-context baseline to catch memorization. For the quick test:
**do not use LingOly items directly**; instead use it as a *format spec* and generate fresh
synthetic Rosetta/Match-Up puzzles from a parameterized grammar (nearest deployable form is #8).
If adapting existing items: constrain to Match-Up (which is naturally multiple-choice → exact match,
satisfies 2/3) and drop the free-form translation formats.

**Rubric-fit notes.** Format library is a goldmine and difficulty decomposes into independent
latent rules (criterion 10) cleanly. Weaknesses: fixed/published (8), many items are free-text
(2/3), difficulty is uneven and some items are one-shot-guessable while others are near-impossible
(threatens the *incremental* clause of 4 and the 10-20-task resolution of 5). Very active
2024-2026 research line (LingOly-TOO, LingBench++, LINGOLY follow-ups).

Sources:
- LINGOLY: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles… (NeurIPS 2024) — https://arxiv.org/abs/2406.06196 · https://arxiv.org/html/2406.06196v2
- Can LLMs Solve and Generate Linguistic Olympiad Puzzles? (EMNLP 2025) — https://arxiv.org/abs/2509.21820 · https://aclanthology.org/2025.emnlp-main.969.pdf
- LingBench++ (2025) — https://arxiv.org/pdf/2507.16809

---

## 3. From Rosetta to Match-Up — Paired Linguistic-Puzzle Corpus w/ generation · `adaptation`

**What it is.** 2026 paper presenting a paired corpus of linguistic puzzles with human + LLM
benchmarks, explicitly pairing **Rosetta-style** translation puzzles with **Match-Up** (which
pairs of words translate each other) versions of the same underlying data. The Match-Up framing is
inherently closed-set / multiple-choice, which is exactly what makes exact deterministic scoring
and temp-0 resolvability tractable.

**Adaptation the rubric needs.** Use the Match-Up transformation as the *scoring trick*: any
generated micro-translation puzzle can be presented as "which of A/B/C/D is the correct pairing,"
collapsing free-text translation into MC (satisfies 2/3/5). Regenerate instances synthetically for
contamination safety (8). Add latent rules (case marking, affix order, topic-fronting) to build the
incremental gap (4/10).

**Rubric-fit notes.** The Rosetta↔Match-Up duality is the single most useful idea in this scan for
reconciling "translation task" with "exact MC scoring." Weakness: the corpus itself is fixed/
published (contamination) and is recent enough that provenance/quality is less battle-tested than
LingOly. Best used as a design pattern, not a data source.

Sources:
- From Rosetta to Match-Up: A Paired Corpus of Linguistic Puzzles with Human and LLM Benchmarks (2026) — https://arxiv.org/pdf/2605.13408 · https://arxiv.org/html/2605.13408

---

## 4. ConLangs to Probe Metalinguistic Grammatical Knowledge · `synthetic-family`

**What it is.** 2025 paper (Taguchi & Sproat). *Systematically generates* constructed languages,
varying morphosyntactic features under precise control, specifically to avoid memorization — the
whole point is contamination resistance. Dataset released on HuggingFace. Scoring in the paper uses
METEOR (translation metric), and the task is generate/analyze structures in the invented language.

**Adaptation the rubric needs.** This is the closest thing to a ready-made *generator* for
Theme B. Reuse the grammar-generation code/spec; **replace METEOR with exact-match** by bounding
outputs to a fixed-template short gloss or a meaning-MC. Turn the varied morphosyntactic features
(word order, case, agreement, tense marking) into the independent latent rules (10) so partial rule
knowledge earns partial score (4/11). Sample hundreds of instances per grammar (7).

**Rubric-fit notes.** Directly synthetic + contamination-proof (7/8 strong), latent-rule
decomposition is native (10), and it's on-domain and 2025-current. Weakness: default eval is soft
(METEOR) and generation-oriented → must re-bound the output surface to satisfy 2/3/5. Requires
integrating their generator rather than an off-the-shelf dataset.

Sources:
- Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs (2025) — https://arxiv.org/pdf/2510.07591

---

## 5. CipherBank — Cryptographic Decryption Benchmark · `existing-benchmark` / caution

**What it is.** 2025 (Li et al., Shanghai AI Lab). 2,358 tasks, 9 algorithms in 3 categories:
Substitution (ROT13, Atbash, Polybius, Vigenère), Transposition (Reverse, SwapPairs), Custom
(DualAvgCode, ParityShift, WordShift). Scoring = **exact character match** (+ Levenshtein as
partial-credit report). 3-shot prompts. Findings: best model Claude-3.5 = 45.14%, o1 = 40.59%;
much better on ROT13 than custom ciphers; **char-level ciphers "solvable in principle but hard"** —
models can't reliably apply systematic per-character transforms.

**Adaptation the rubric needs.** Exactly the char-arithmetic temp-0 hazard flagged in the
brainstorm. To use: **do not score raw decrypt**; convert to closed-vocabulary decode (nearest-word
snap) or to MC over candidate plaintexts (2/3/5). Regenerate keys/plaintexts synthetically per
instance (already easy — ciphers are parameterized) for contamination (8). Use the 3-category
split as natural independent latent rules (10) and compose 2-3 layers for the incremental gap (4).

**Rubric-fit notes.** Native exact-match scoring (2), trivially generatable with exact ground truth
(7/8), clean per-layer diagnostics (11), and demos help (12, they use 3-shot). BUT the headline
risk is **criterion 5**: raw char-level accuracy is noisy at temp-0, so a 10-point prompt gain may
not resolve on 10-20 items *unless* you snap to a closed vocab. Also very-hard ciphers (Vigenère)
may sit below the floor with no incremental path. Prefer substitution/transposition layers over
polyalphabetic. Hot 2025-2026 area.

Sources:
- CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges (2025) — https://arxiv.org/abs/2504.19093 · https://arxiv.org/html/2504.19093v1

---

## 6. CipherBench (v2) — Common / Uncommon / Novel Cipher Decoding · `existing-benchmark`

**What it is.** Benchmark evaluating LLM decode over ten cipher techniques grouped as **common,
uncommon, and novel (user-created)** ciphers. Explicit thesis: as reasoning improves, models can
decode *novel* ciphers not in pretraining — directly relevant to the "invented-rule decoder"
brainstorm entry (Theme C) and to whether a fresh synthetic cipher is genuinely un-memorized.

**Adaptation the rubric needs.** Use only the **novel-cipher** methodology (invent a fresh
per-instance rule the model must apply from an in-context spec) — this is the contamination-proof
core (8) and gives real headroom (4). Bound output to closed vocab / MC (2/3/5). Layer 2-3 novel
ops for the incremental gap (4/10). Ground truth is exact by construction (7).

**Rubric-fit notes.** The common/uncommon/novel axis is a ready-made difficulty ladder mapping onto
criterion 4's incremental path and criterion 10's independent rules. Same char-noise caveat as #5.
The novel-cipher subset is the most rubric-aligned because it isn't memorizable. Weakness: as a
published benchmark the specific ciphers may leak; regenerate.

Sources:
- CipherBench v2 — https://cipherbench.github.io/
- (context) When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers (2024) — https://arxiv.org/abs/2402.10601

---

## 7. Synthetic Case-Marked Micro-Language Translation · `synthetic-family` (brainstorm Theme B)

**What it is.** Program-generated agglutinative micro-language: suffixes mark role/number/tense; a
**case suffix determines subject/object regardless of linear position**; a topic-fronting particle
reorders the output. Model gets a small in-context lexicon + rule hints (or few-shot pairs) and must
produce a **fixed-template 3-6 word English phrase**. This is MTOB's paradigm shrunk to quick-test
scale and is the most direct instantiation of the domain seed.

**Adaptation the rubric needs.** This IS the adaptation — build the generator. Key design choices:
(a) exact-match on a canonicalized template (choose a rigid slot order so "expected X got Y" is
clean → 2/11); (b) keep the affix inventory large/rotating enough that the mapping is *not
one-shot-guessable* (the brainstorm flags one-shot risk MODERATE-HIGH — mitigate by making case
assignment override word order, which naive English-SVO prompts get wrong → 4); (c) each affix rule
is an independent latent rule (10) so partial prompts earn partial score; (d) few-shot pairs teach
the affix→gloss map (12).

**Rubric-fit notes.** Best all-round fit in Theme B: cheap, exact-scored, generatable in the
hundreds, contamination-proof, latent-rule-decomposed, few-shot-friendly. Main risk is
**one-shot guessability** — a strong prompt engineer might infer "suffix -k = object" from a couple
examples in one shot, collapsing the incremental gap (4/5). Counter by stacking ≥3 interacting rules
(case + topic-fronting + number/tense agreement) and by making the *interaction* (precedence between
case and topic particle) the hard latent rule. Low direct research novelty on its own, but inherits
MTOB/ConLang relevance.

Sources:
- (paradigm) MTOB — https://arxiv.org/abs/2309.16575 ; (generation methodology) ConLang probing — https://arxiv.org/pdf/2510.07591 ; SIGMORPHON reinflection for affix-paradigm design — https://arxiv.org/abs/1910.11493

---

## 8. Synthetic Rosetta/Match-Up Grammar Puzzle Generator · `synthetic-family`

**What it is.** A parameterized generator that emits LingOly-style **Rosetta** items (N glossed
example pairs + one held-out item) but presents the query as **Match-Up multiple choice** (which
A/B/C/D pairing is correct), fusing candidates #2 and #3 into a deployable synthetic task. The
latent grammar (order, agreement, a number system, a compounding rule) is drawn per instance from a
seeded config.

**Adaptation the rubric needs.** Build the generator; expose knobs for #independent rules and #demo
pairs. MC framing gives exact deterministic scoring and temp-0 resolvability for free (2/3/5).
Seeded regeneration = contamination-proof (7/8). Choose distractors so that each latent rule the
prompt captures rules out one distractor → incremental gap + diagnostic failure (4/10/11).

**Rubric-fit notes.** Combines LingOly's *proven difficulty structure* with MC's *clean scoring* —
arguably the strongest rubric fit among translation-flavored options, and rides an active research
wave (linguistics-olympiad-for-LLMs, 2024-2026). Weakness: designing distractors that isolate one
rule each takes care; poorly chosen distractors make items either trivial or unresolvable.

Sources:
- LINGOLY (NeurIPS 2024) — https://arxiv.org/abs/2406.06196
- From Rosetta to Match-Up (2026) — https://arxiv.org/pdf/2605.13408
- Can LLMs Solve and Generate Linguistic Olympiad Puzzles? (EMNLP 2025) — https://arxiv.org/abs/2509.21820

---

## 9. Composable Cipher-Chain Decode → Closed Vocabulary · `synthetic-family` (brainstorm Theme C)

**What it is.** An ordered stack of named reversible ops (e.g. keyed block transposition, invented
digraph swap, positional skip) with a per-instance key; decoding applies inverses **in reverse
order** to recover a short **closed-vocabulary** plaintext (few nonsense-but-listed words). The
closed vocab + nearest-match snapping is the deliberate fix for the char-arithmetic temp-0 hazard.

**Adaptation the rubric needs.** Avoid Caesar/shift-arithmetic layers (the noisiest, per #5/#6);
prefer *structural* reversible ops (transposition, block reversal, digraph substitution) whose
correctness is deterministic and whose failures are legible ("applied layers in wrong order").
Score by snapping the output to the nearest closed-vocab token, or go full MC (2/3/5). Each layer =
one independent latent rule; the *reverse-order* rule is the hard meta-rule (4/10). Few-shot chains
teach the inversion pattern (12).

**Rubric-fit notes.** Cleanest per-layer diagnostics of any candidate (11) and a natural incremental
ladder (add layers) for criterion 4. Native exact scoring + trivially generatable (2/7/8). The
brainstorm rates one-shot risk MEDIUM-HIGH and char-noise as the main threat to 5 — both mitigated
by (a) structural (not arithmetic) layers and (b) closed-vocab/MC output. Watch that a very long
chain doesn't drop below the floor with no incremental path. Rides the 2025-2026 cipher-reasoning
wave.

Sources:
- CipherBank (2025) — https://arxiv.org/abs/2504.19093
- Benchmarking LLMs for Cryptanalysis (Findings-EMNLP 2025) — https://aclanthology.org/2025.findings-emnlp.1082.pdf · https://arxiv.org/pdf/2505.24621

---

## 10. Invented Tonal/Diacritic Orthography Disambiguation (MC) · `synthetic-family` (brainstorm Theme B)

**What it is.** Base glyph + stacked diacritics; a tone/diacritic marker resolves homographs, a
context particle can override the tone, and there's a precedence order among markers. Output is a
**meaning A/B/C/D** — natively multiple choice, so exact scoring and temp-0 resolution are easy.
Shares homograph/precedence structure with the Theme F enum/codebook tasks.

**Adaptation the rubric needs.** Generator emits glyph strings + a per-instance diacritic legend;
question asks the meaning of a target token. Independent latent rules: (a) which diacritic selects
which sense, (b) context-particle override, (c) marker precedence (4/10). MC output already
satisfies 2/3/5. Randomize glyph↔meaning maps per instance for contamination (8).

**Rubric-fit notes.** Because it's MC over meanings, it sidesteps *both* the translation-scoring
softness (unlike #1) *and* the cipher char-noise (unlike #5/#9) — a very safe criterion-5 profile.
Precedence/override interaction gives genuine incremental headroom and diagnostic failures ("picked
sense B, ignored the override particle" → 4/11). Weakness: with only ~4 diacritics the rule set can
become one-shot-guessable; keep the precedence/override interaction as the load-bearing hard rule.
Modest standalone research novelty (rides the "rare-script / unfamiliar-writing-system" line where
LLMs reliably underperform humans).

Sources:
- (LLMs weak on unfamiliar writing systems) LINGOLY (2024) — https://arxiv.org/abs/2406.06196 ; Reasoning Over the Glyphs: Decipherment of Rare Scripts (2025) — https://arxiv.org/pdf/2501.17785

---

## 11. Synthetic Morphological (Re)Inflection — SIGMORPHON-style, bounded · `adaptation`

**What it is.** SIGMORPHON/CoNLL shared tasks (2017-2024): given a lemma + a morphosyntactic feature
bundle (UniMorph tags), produce the inflected form; or fill a paradigm from partial exposure. High/
medium/low-resource conditions built in. Output is a **single short word** with exact-match
accuracy scoring — already criterion-2/3 compliant.

**Adaptation the rubric needs.** Real SIGMORPHON languages are memorizable (fails 8). Instead
**generate a synthetic morphology** (invented stems + invented affix rules + phonological
alternations) and ask for the inflected form of a novel lemma given an in-context paradigm. Exact
match on the produced word (2). Independent latent rules = each affix/alternation (10); few-shot =
partial paradigm (native to the task → 12).

**Rubric-fit notes.** Exact-match single-word output is an ideal scoring surface (2/3/5) and demos
provably help (the task is *built around* learning from a partial paradigm → 12 is guaranteed).
Latent-rule decomposition (stem change vs. suffix vs. phonological rule) is clean (10/11). Risk:
a single regular affix rule is one-shot-guessable → stack an irregular/allomorphy rule and a
phonological alternation to preserve the incremental gap (4). Char-level output means *some*
temp-0 spelling noise, milder than ciphers. Active area (SIGMORPHON through 2024; morphological
analogy work 2023-2025).

Sources:
- SIGMORPHON 2019 Shared Task (cross-lingual inflection) — https://arxiv.org/abs/1910.11493
- CoNLL-SIGMORPHON 2018 (103 languages) — https://arxiv.org/pdf/1810.07125
- SIGMORPHON shared-tasks index — https://sigmorphon.github.io/sharedtasks/

---

## 12. Char-Level String-Transform Tasks (StringLLM / CUTE / counting) · `existing-benchmark` / diagnostic

**What it is.** A cluster of 2024-2025 benchmarks probing LLM *character-level* competence, useful
mainly as **evidence for the criterion-5 caveat** rather than as the task itself:
- **StringLLM** (2024) — systematic string-processing capability suite.
- **CUTE** (2024) — measures whether LLMs understand their own tokens (spelling, char ops).
- **Counting Ability & Tokenization** (2024) and "Why LLMs Struggle to Count Letters" (2024) —
  quantify how BPE tokenization breaks letter-level operations.
- **Divide-and-Conquer char manipulation** (2025) — a *prompting* fix that measurably improves
  char-level accuracy → demonstrates a genuine incremental prompt-improvement gap.

**Adaptation the rubric needs.** If used as a task at all: pick a bounded char-op (e.g. apply an
invented positional substitution + count/report a single letter → short-word or numeric answer,
exact match). Generate instances synthetically (8). But note these are *tokenization-bound* — many
items are noisy at temp-0, which is precisely the hazard the rubric's criterion 5 guards against.

**Rubric-fit notes.** Most valuable as a **red-flag reference**: the Divide-and-Conquer result
shows char-level prompting *does* have an incremental improvement curve (supports 4/12 for a
carefully bounded design), while the counting/tokenization papers show raw char tasks are noisy
(threatens 5). Use these to justify closed-vocab/MC output framing across all cipher candidates
(#5/#6/#9), rather than as a standalone quick-test task. Strong 2024-2026 research interest.

Sources:
- StringLLM (2024) — https://arxiv.org/pdf/2410.01208
- CUTE: Measuring LLMs' Understanding of Their Tokens (2024) — https://arxiv.org/pdf/2409.15452
- Counting Ability of LLMs and Impact of Tokenization (2024) — https://arxiv.org/html/2410.19730v2
- Why Do LLMs Struggle to Count Letters? (2024) — https://arxiv.org/html/2412.18626v1
- Enhancing LLM Character-Level Manipulation via Divide and Conquer (2025) — https://arxiv.org/pdf/2502.08180

---

## Cross-cutting synthesis

- **Best rubric fits (translation flavor):** #8 (synthetic Rosetta/Match-Up MC), #7 (case-marked
  micro-language), #4 (ConLang generator) — all synthetic, exact-scorable, latent-rule-decomposed,
  contamination-proof. #8's Match-Up MC framing is the single cleanest way to reconcile
  "translation task" with "exact deterministic 10-20-item resolution."
- **Best rubric fits (cipher flavor):** #9 (composable structural cipher chain → closed vocab) and
  #6 (novel-cipher subset), **provided** outputs are snapped to a closed vocabulary or MC. Avoid raw
  char-arithmetic ciphers and polyalphabetic (Vigenère) layers — the literature (#5, #12) shows
  temp-0 char noise sits at ~40-45% and would defeat criterion 5.
- **Inspiration/realism, not deployable as-is:** #1 (MTOB), #2 (LingOly), #3 (Rosetta↔Match-Up),
  #11 (SIGMORPHON) — all fixed/published (contamination) and/or soft-scored, but each contributes a
  reusable ingredient (paradigm, format library, MC-duality, exact-match single-word surface).
- **Universal caveat (criterion 5):** every cipher/char candidate must move noise out of the score
  via closed-vocab snapping or MC. Every translation candidate must guard **one-shot guessability**
  (criterion 4) by stacking ≥3 interacting latent rules where the *interaction/precedence* is the
  load-bearing hard rule, not any single affix/cipher layer.
- **Research-tie-breaker signal (2024-2026):** MTOB, LingOly (+TOO/++ follow-ups), cipher-reasoning
  (CipherBank/CipherBench), and char-level/tokenization limits are all live 2024-2026 lines, so a
  synthetic quick-test grown from #7/#8/#9 could plausibly seed its own research direction.
