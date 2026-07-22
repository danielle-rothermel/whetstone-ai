# Quick-Test Candidates — Domain: Deterministic Simulation, State Machines, and Protocol/Game Execution

Breadth-focused literature scan for whetstone-ai's quick-test task selection.
Domain rationale: bounded single-token/few-word answers, rich decomposition into
independent latent rules, deliberate per-rule violation injection (criterion 13),
and a mature "execution/state-tracking" literature that quantifies where models
break — which directly informs floor/ceiling calibration (criteria 4 & 9).

Date: 2026-07-21. Author: literature subagent (breadth, not ranked).

## How to read the rubric-fit notes

The quick-test rubric's load-bearing criteria for this domain:
- **1-3 (cheap):** single LLM call, exact deterministic scoring, output bounded to MC / a few words.
- **4 (incremental gap):** default prompts well below a KNOWN ceiling, not one-shot guessable.
- **5 (small-sample resolvable):** ≥10-point effect resolvable on 10-20 tasks at temp 0.
- **7-9 (synthetic, contamination-resistant, known ceiling/floor/reference).**
- **10-11 (independent latent rules; diagnostic failures).**
- **12 (few-shot helps).** **13 (inject failures).**

A recurring tension in THIS domain: most execution/state-tracking benchmarks ask
for a **multi-step trace** (per-step queue/state), which violates criterion 3's
"bounded output." The adaptation is almost always the same: **ask only for the
final state / a single derived token**, and generate depth so that the final token
is not guessable. I flag this per candidate.

---

## Candidate 1 — TMBench (Turing Machine / m-Tag simulation)

**What it is.** LLMs simulate a Universal-Turing-complete *m-tag system*: given
production rules and an initial symbol queue, execute deterministic steps (read
head symbol, append per the matching rule, delete m symbols) until halt or a step
limit. Fully synthetic and "knowledge-agnostic." 100 sampled systems; difficulty
controlled by alphabet size, max queue length, and rule length. Models <8B score
near-zero; Gemini-2.5-Pro reaches ~94% at 30 steps but fails deep (step ~683).
Exact-match scoring per step.

**Citation.** Wu et al., *Turing Machine Evaluation for Large Language Model*, 2025. arXiv:2504.20771. https://arxiv.org/abs/2504.20771

**Adaptation for the rubric.** (a) Regenerate the m-tag system pool procedurally
with a seeded generator to control the split sizes (hundreds of instances) — the
generator is trivial to reimplement. (b) **Bound the output to a single token**:
ask for the head symbol after exactly N steps (or the halting length parity), not
the full trace — this satisfies criterion 3 and keeps exact-match diagnostic. (c)
Latent rules already decompose cleanly: "delete exactly m," "append rule for head
symbol," "halt condition," "read head not tail" — inject per-rule violations for
criterion 13.

**Rubric-fit notes.** Very strong fit for criteria 7-11: procedural, exact ground
truth, independent rules, floor/ceiling documented in the paper. Few-shot demos
should help teach the step mechanic (criterion 12). *Weakness:* native format is
a multi-step trace; you must truncate to a single derived answer or you blow the
token budget and reintroduce partial-credit ambiguity. *Weakness:* if N is small,
a competent prompt engineer may one-shot the mechanic — tune N and rule length so
default prompts land mid-range, not at ceiling.

---

## Candidate 2 — Long-Horizon Execution (in-context dictionary retrieve-and-sum)

**What it is.** A deliberately minimal synthetic execution task: a fixed in-context
dictionary maps vocab tokens to integers; the model is given a plan (a list of keys)
and must retrieve values and maintain a running sum across steps. Isolates *execution*
from *planning/knowledge*. Introduces H₀.₅ (max length at 50% success), the
"self-conditioning effect" (errors in context beget errors), and shows near-perfect
single-step accuracy collapsing over turns. Procedurally generated, contamination-free.

**Citation.** Sinha, Arun, Goel, Staab, Geiping, *The Illusion of Diminishing Returns:
Measuring Long Horizon Execution in LLMs*, 2025/2026. arXiv:2509.09677. https://arxiv.org/abs/2509.09677

**Adaptation for the rubric.** Almost drop-in. (a) Fix horizon length to a small K
so the answer is a **single integer** (final sum) — perfect for exact-match and
bounded output. (b) The dictionary + plan generator is a few lines; regenerate per
seed for splits. (c) Latent rules to layer for incremental gap: "sum vs last-value,"
"skip sentinel keys," "modular arithmetic wrap," "ignore distractor keys."

**Rubric-fit notes.** Excellent on cheapness (criteria 1-3) and determinism
(criterion 5): a single integer answer, exact match. Excellent on floor/ceiling —
the paper *is* a floor/ceiling study, so you inherit calibration curves. *Weakness:*
the base task is so clean that with a good instruction it may be one-shot solvable
(criterion 4 risk) — you must add latent rules (distractors, conditional skips,
modular wrap) to create an incremental path. *Weakness:* few-shot may help less
than instruction here, which is actually useful signal for MIPROv2 vs COPRO.

---

## Candidate 3 — CRUXEval (code output/input prediction)

**What it is.** 800 deterministic, side-effect-free Python functions each with one
input/output pair. CRUXEval-O = predict output given input (forward execution);
CRUXEval-I = predict an input that yields a given output (inverse). GPT-4 ~67%/63%;
Code-Llama-34B ~47%/44%. CRUXEval-X extends to 19 languages / 19k tasks.

**Citation.** Gu et al., *CRUXEval: A Benchmark for Code Reasoning, Understanding and
Execution*, 2024. https://crux-eval.github.io/ ; arXiv:2401.03065. CRUXEval-X: Xu et al.,
2024, arXiv:2408.13001. https://arxiv.org/abs/2408.13001

**Adaptation for the rubric.** (a) **Contamination is a real risk** — the 800
functions are public. You must *generate fresh functions* with a small DSL/templater
(the CRUXEval generation pipeline used a code-LLM + filters; reimplement with your
own operator set) so ground truth is exact and unseen. (b) Restrict operators to a
few families (string slicing, list ops, dict updates, arithmetic) so each family is
a latent rule and output is a short literal (bound to a few tokens). (c) Output is
already short (a Python literal) — enforce a canonical serialization for exact match.

**Rubric-fit notes.** Strong on decomposition (operator families = independent rules)
and diagnostic failures ("expected [3,1], got [1,3]" reveals a sort/reverse rule
miss). Well-studied floor/ceiling. *Weakness:* contamination of the published set is
severe — must regenerate. *Weakness:* output can be an arbitrary structure; you must
constrain the return type to keep exact-match clean and outputs bounded. *Weakness:*
"a few words" is stretched if functions return long structures — cap length.

---

## Candidate 4 — GridRoute / grid navigation execution

**What it is.** ASCII grid worlds; the model emits {UP,DOWN,LEFT,RIGHT} to route a
robot to a goal, or predicts the resulting cell after a fixed move sequence. GridRoute
benchmarks classical-algorithm-guided prompting (A*, Dijkstra, DFS embedded in the
prompt). Related: GRASP (commonsense spatial reasoning on grids), MiniGrid.

**Citation.** GridRoute, 2025, arXiv:2505.24306. https://arxiv.org/abs/2505.24306 ;
GRASP, 2024, arXiv:2407.01892. https://arxiv.org/abs/2407.01892

**Adaptation for the rubric.** (a) Flip from *plan generation* to *state prediction*:
give a start cell + a fixed move string + walls, ask for the **final coordinate or
cell contents** (single token). This makes scoring exact and output bounded (criteria
2-3) and dodges the "many valid paths" partial-credit problem. (b) Procedurally
generate grids with seeds (native to MiniGrid-style generators). (c) Latent rules:
"walls block (stay in place)," "grid wrap vs clamp at edges," "move order," "one-way
tiles" — each adds score incrementally.

**Rubric-fit notes.** Strong on synthetic generation and rule decomposition; failures
are highly diagnostic (off-by-one on a wall reveals the wall rule). Few-shot demos of
worked traces measurably help (shown in GridRoute's Algo-Reasoning). *Weakness:* if you
ask for a *path* it becomes non-deterministic-answer and multi-token — you must ask for
a *derived single state*. *Weakness:* spatial reasoning has known modality confounds;
keep grids small and ASCII to isolate execution.

---

## Candidate 5 — LLM-BabyBench (BabyAI text, Predict split)

**What it is.** Textual adaptation of the procedurally generated BabyAI grid world.
Three splits; the **Predict** split asks the model to predict the *final environment
state* after executing a low-level action sequence. Ground truth extracted from an
expert agent in the simulator. Procedural generation is inherited from BabyAI.

**Citation.** *LLM-BabyBench*, 2025, arXiv:2505.12135. https://arxiv.org/abs/2505.12135 ;
dataset: https://huggingface.co/datasets/salem-mbzuai/LLM-BabyBench

**Adaptation for the rubric.** (a) Use only the **Predict** split and reduce the
queried state to a **single field** (agent facing direction, or "carrying? y/n," or
final cell) so output is bounded and exact-match. (b) Regenerate levels with new seeds
to avoid the released set (contamination). (c) Latent rules: object-interaction
semantics (pickup/drop/toggle), direction turning, door open/closed — each is an
independent rule to be discovered by a prompt.

**Rubric-fit notes.** Good synthetic generator (BabyAI is battle-tested and free),
clean per-rule decomposition, diagnostic failures. *Weakness:* the released dataset is
public — regenerate. *Weakness:* full-state prediction is verbose; you must project to
one field. *Weakness:* action semantics are somewhat rich, so the naive-prompt floor
may be *too* low (near zero) — pick a subset of actions to keep the gap incremental.

---

## Candidate 6 — MLRegTest / FLaRe (regular-language membership)

**What it is.** MLRegTest: hundreds of regular languages from 11 subclasses of the
subregular/Piecewise-Local-Testable hierarchy, organized by formal complexity — a
membership (accept/reject) task. FLaRe (Formal Language Recognition) releases languages
across the Chomsky hierarchy with code. Classic result: transformers fail on non-star-free
/ periodic languages; LSTMs solve all Tomita grammars.

**Citation.** MLRegTest, van der Poel et al., JMLR 2024. https://www.jmlr.org/papers/volume25/23-0518/23-0518.pdf ;
*Training Neural Networks as Recognizers of Formal Languages*, 2024, arXiv:2411.07107. https://arxiv.org/abs/2411.07107

**Adaptation for the rubric.** (a) The task is **already a single binary token**
(accept/reject) — ideal for criteria 2-3 and 5. (b) Instead of training recognizers,
prompt the LLM to decide membership given the string; procedurally sample strings from
a chosen grammar with a seeded generator (exact ground truth by construction). (c)
Latent rules = the grammar's constraints (e.g., "even number of a's," "no 'ab'
substring," "ends in c"); compose several so partial rule-knowledge earns partial score.

**Rubric-fit notes.** Best-in-domain on output boundedness (one token) and on
independent-rule decomposition (each subregular constraint is a rule). Deep formal-language
literature gives principled floor/ceiling per grammar class. *Weakness:* binary output
means chance = 50%, so you must ensure the default-prompt floor is clearly below ceiling
and effect sizes ≥10 points resolve on 10-20 tasks — use class-balanced sampling and a
few composed constraints so guessing is penalized. *Weakness:* pure accept/reject gives
thin diagnostics (criterion 11) unless you also ask *which* rule was violated — consider
a "reject-because" variant with a small label set.

---

## Candidate 7 — bAbI entity/world-state tracking (regenerated)

**What it is.** The classic 20-task synthetic QA suite; several tasks are pure entity
movement / world-state tracking ("John went to the kitchen... Where is the apple?").
Synthetically generated over a tiny lexicon with exact answers. Even GPT-3-scale models
historically miss basic entity tracking; recent work shows LMs aggregate state at the
query token rather than tracking incrementally.

**Citation.** Weston et al., *Towards AI-Complete QA: bAbI tasks*, 2015, arXiv:1502.05698;
EntNet (solves bAbI), Henaff et al., ICLR 2017, arXiv:1612.03969. https://arxiv.org/abs/1612.03969 ;
entity-tracking-in-LMs: Kim & Schuster, ACL 2023; *Do Language Models Track Entities
Across State Changes?*, 2026, arXiv:2605.30233. https://arxiv.org/abs/2605.30233

**Adaptation for the rubric.** (a) **Regenerate** with your own grammar/generator (the
original generator is open) — the published bAbI set is thoroughly contaminated. (b)
Answers are already single words (a location/object) — perfect for exact-match. (c) Add
latent rules to lift the ceiling gap: negation ("no longer in"), transfer chains,
containment vs co-location, distractor sentences.

**Rubric-fit notes.** Excellent boundedness (single-word answer), excellent synthetic
generation, diagnostic failures ("expected kitchen, got garden" pinpoints the missed
move). *Weakness:* modern frontier models solve simple bAbI tracking near-ceiling — the
FLOOR is too high unless you deliberately add depth (long chains, negation, multi-hop) to
push default prompts down. This is the primary thing to engineer. *Weakness:* contamination
of the original — must regenerate.

---

## Candidate 8 — SATBench (Boolean satisfiability puzzles)

**What it is.** Logical puzzles auto-generated from Boolean SAT formulas; 2,100 puzzles.
Evaluates satisfiability prediction (SAT/UNSAT — binary) plus reasoning-trace validity.
Difficulty scales with clause count; o4-mini drops to ~65% on hard UNSAT (near the 50%
random baseline). Fully automated generation from SAT instances.

**Citation.** Wei et al., *SATBench: Benchmarking LLMs' Logical Reasoning via Automated
Puzzle Generation from SAT Formulas*, 2025, arXiv:2505.14615. https://arxiv.org/abs/2505.14615

**Adaptation for the rubric.** (a) **Output is one token** (SAT/UNSAT) — clean exact
match. (b) Generator is fully synthetic and parameterized (clauses, variables) — control
split sizes and difficulty directly. (c) For a richer, more diagnostic variant, ask for a
satisfying assignment as a bounded bit-string (still short) — exact-check by substitution.
Latent rules: unit propagation, pure-literal elimination, clause counting.

**Rubric-fit notes.** Strong on cheapness, synthetic generation, tunable floor/ceiling
(clause count is a difficulty dial). *Weakness:* binary SAT/UNSAT = 50% chance floor and
thin diagnostics (criterion 11) — mitigate by requiring an assignment or a "which clause
is violated" label. *Weakness:* the *reasoning* is search-heavy and may not decompose into
neat independent latent *prompt* rules the way an execution task does — the gap may be
driven more by raw reasoning than by prompt-discoverable rules, weakening criterion 4/10.

---

## Candidate 9 — Circuit / logic-gate DAG evaluation

**What it is.** Evaluate a Boolean circuit (AND/OR/NOT/XOR gates wired as a DAG) on a
given input assignment and predict the single output bit — a canonical deterministic
"execute the state machine" task. Related published work: CIRCUIT (analog-circuit QA,
510 pairs) and propositional-logic circuit-analysis studies, though these are adjacent
rather than exact matches to a synthetic gate-evaluation generator.

**Citation.** CIRCUIT, 2025, arXiv:2502.07980. https://arxiv.org/pdf/2502.07980 ;
*A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning*, 2024,
arXiv:2411.04105. (No off-the-shelf synthetic gate-eval benchmark found — this is a
synthetic-family candidate you'd build.)

**Adaptation for the rubric.** Build a small generator: random DAG of gates over k inputs;
sample an input assignment; ground-truth output computed by evaluating the DAG. (a) Output
= **one bit** (or a k-bit bounded string for multi-output). (b) Latent rules: gate
semantics per type, evaluation order (topological), NOT/negation handling, fan-out reuse.
(c) Difficulty dials: gate count, depth, gate-type mix.

**Rubric-fit notes.** Ideal boundedness and exact scoring; independent-rule decomposition
is clean (each gate type is a rule; get XOR wrong and a diagnostic pattern appears).
Contamination-proof (fully generated). Few-shot worked evaluations should help. *Weakness:*
no ready-made benchmark to borrow — this is **synthetic-family** work, more build effort.
*Weakness:* single-bit output = 50% chance; use several outputs or deeper circuits so
guessing is penalized and the effect size stays resolvable on 10-20 tasks.

---

## Candidate 10 — Dyck-n / bounded-depth balanced brackets

**What it is.** Membership or next-token/closing-bracket prediction for Dyck-n languages
(k bracket types, max depth m): a pushdown-stack tracking task. Heavily studied in the
transformer-expressivity literature; transformers struggle for n>1 and at higher depth.

**Citation.** Yao et al., *Self-Attention Networks Can Process Bounded-Hierarchy Languages*,
ACL 2021; *bounded Dyck grammars* case study, NSF-PAR:10489627. https://par.nsf.gov/servlets/purl/10489627 ;
FLaRe / *Training NNs as Recognizers*, 2024, arXiv:2411.07107.

**Adaptation for the rubric.** (a) Two clean bounded outputs: **accept/reject** (one
token) OR **predict the single next legal closing bracket** given a prefix (one token) —
the latter is more diagnostic and less chance-driven. (b) Generate strings from a seeded
sampler at controlled depth/length (exact ground truth). (c) Latent rules: matching by
bracket type, LIFO order, depth bound, ignore-non-bracket tokens.

**Rubric-fit notes.** Excellent boundedness and determinism; depth is a clean floor/ceiling
dial; failures diagnostic (wrong bracket type reveals the type-matching rule). *Weakness:*
if you use accept/reject you inherit the 50%-chance / thin-diagnostic problem — prefer the
next-closing-bracket variant. *Weakness:* very well-known task; a strong prompt engineer may
one-shot shallow instances — push depth so default prompts sit below ceiling (criterion 4).

---

## Candidate 11 — Invented card/resource game state execution (TextArena-style)

**What it is.** Simulate one deterministic turn (or N turns) of a small *invented* card /
resource game: given rules (in-prompt) + current state + a move, output the resulting
state field (score, whose turn, top-of-pile card). TextArena (50+ text games) and
GameBench/Game-Reasoning-Arena provide the surrounding infrastructure and rule-encoding
patterns; ZeroSumEval/GVGAI-LLM show synthetic/infinite game generation.

**Citation.** TextArena, 2025, arXiv:2504.11442; Game Reasoning Arena, 2025,
arXiv:2508.03368. https://arxiv.org/pdf/2508.03368 ; ZeroSumEval, 2025, arXiv:2504.12562;
GVGAI-LLM, 2025, arXiv:2508.08501.

**Adaptation for the rubric.** This is a **synthetic-family** candidate: invent a tiny
deterministic game with a seeded generator (deck/state/move), and ask for **one derived
field** after applying the move(s). (a) Bounded single-token output (e.g., resulting score,
or legal/illegal move). (b) *Invented* rules make it contamination-proof by construction
(criterion 8) — no published answers exist. (c) Latent rules: precedence of card effects,
resource caps, turn-passing, illegal-move rejection — each independently discoverable.

**Rubric-fit notes.** Best-in-domain on contamination-resistance (rules are novel) and on
deliberate rule-violation injection (criterion 13 — feed illegal moves and require a
"reject" token). Few-shot demos of worked turns should strongly help (criterion 12).
*Weakness:* highest design effort — you must author and validate the rules and a reference
100%-scoring prompt (criterion 9). *Weakness:* easy to accidentally make rules ambiguous;
keep the state transition a pure deterministic function to preserve exact-match.

---

## Candidate 12 — Scheduler / protocol-validator execution (synthetic family)

**What it is.** Execute a small deterministic scheduler or protocol state machine: given a
transition table (states × events → next-state/action) and an event trace, output the final
state or whether the trace is *accepted* by the protocol. This is the FSM-simulation and
protocol-acceptance theme; supported by the mechanistic state-tracking literature (transformers
learn "associative"/"parity-associative" algorithms; CoT helps) and TMBench's precedent.

**Citation.** *Finite State Automata Inside Transformers with CoT: A Mechanistic Study on
State Tracking*, 2025, arXiv:2502.20129. https://arxiv.org/abs/2502.20129 ;
*(How) Do Language Models Track State?*, 2025, arXiv:2503.02854. https://arxiv.org/html/2503.02854v1 ;
*Exploring State Tracking Capabilities of LLMs*, 2025, arXiv:2511.10457.

**Adaptation for the rubric.** Synthetic-family: seeded generator emits a random FSM
(transition table) + an event sequence; ground truth = final state or accept/reject computed
by running the FSM. (a) Output = single state label (from a small set) or one accept/reject
token. (b) Latent rules: the transition table itself (each event's effect), reset/error
states, self-loops, "unknown event → error." (c) Contamination-proof by construction.

**Rubric-fit notes.** Excellent on all cheap criteria and on independent-rule decomposition
(each transition is a rule; the reflection LM can read "in state S on event E, expected T,
got U" — maximally diagnostic, criterion 11). The mechanistic literature gives you *why*
CoT/few-shot help, aiding floor/ceiling design. Active 2025-2026 research interest (tie-breaker).
*Weakness:* if the state set is small, single-state output has a nontrivial chance baseline —
use enough states/steps so guessing is penalized and effects resolve on 10-20 tasks.
*Weakness:* build effort (no drop-in dataset), but the generator is genuinely small.

---

## Cross-cutting observations (not a ranking)

- **The dominant adaptation across the whole domain is output projection.** Almost every
  execution benchmark natively wants a multi-step trace (violating criterion 3). The fix is
  identical everywhere: **ask only for the final state or one derived token**, generated
  deep enough that it is not guessable. Candidates that are natively single-token (MLRegTest
  accept/reject #6, Dyck next-bracket #10, SATBench #8, FSM final-state #12, long-horizon
  final-sum #2) need the least surgery.
- **Binary-output candidates (#6, #8, #10-accept, #9-single-bit)** carry a 50%-chance floor
  and thin diagnostics. Prefer variants with a small-but->2 label set (next-bracket, final
  FSM state, "which rule was violated") to keep criterion 5 effect sizes resolvable and
  criterion 11 diagnostics rich.
- **Contamination** is a real hazard for the *published* sets (#3 CRUXEval, #5 LLM-BabyBench,
  #7 bAbI). All three have open generators, so regeneration is cheap. The **synthetic-family**
  candidates (#9 circuits, #11 invented games, #12 FSM/protocol) are contamination-proof by
  construction but cost more to author + validate a reference prompt (criterion 9).
- **Floor calibration is the opposite risk for #7 bAbI** (frontier models near ceiling on
  simple tracking) — must add depth. It is the *right* risk for #1 TMBench, #2 long-horizon,
  #12 FSM, where depth is a clean, monotone difficulty dial.
- **Tie-breaker (active 2025-2026 research):** state-tracking/execution is a hot area —
  TMBench (#1), long-horizon execution (#2), FSM-in-transformers mechanistic work (#12),
  and cellular-automata-as-simulation (LOGOS-CA/LifeGPT, adjacent) all appeared in 2025-2026,
  so a quick test built here could grow into its own research direction.

### Adjacent items noted but not promoted to candidates
- **Cellular-automata simulation** (LOGOS-CA arXiv:2602.00036; LifeGPT arXiv:2409.12182):
  deterministic local-rule update, single-cell-next-state is a bounded output; promising as a
  synthetic family but currently framed as generative/simulation rather than a scored benchmark.
- **CoRe / CRUXEval-X / "Reason About Complex Execution Paths"** (arXiv:2507.05269,
  arXiv:2408.13001, arXiv:2511.18288): code-execution variants; same contamination + output-bounding
  caveats as #3.
