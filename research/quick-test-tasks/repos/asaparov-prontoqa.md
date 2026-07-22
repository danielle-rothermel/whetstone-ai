# asaparov/prontoqa — skeptical reuse review

- Repo: https://github.com/asaparov/prontoqa
- Review date: 2026-07-21
- License: Apache-2.0 (LICENSE file present, full Apache 2.0 text)
- Serving candidate ids: c18 (Depth-Controlled Synthetic Deduction, T/F), c21 (Relational Micro-World with Invented Vocabulary)
- One-line verdict: Reusable deterministic seeded generator with a genuine constructive proof oracle and built-in self-verification; our diff is config + scoring glue. One real bug on the Composed/postorder path (trivially avoided). Score 3.

## Active path (what we hit for `--model-name json`)

Entry: `run_experiment.py:1692` (`__main__`, argparse) -> `run_experiment.py:1810` dispatch for `json` -> `run_experiment.py:1432` `run_experiment()` -> loop calling `generate_question()` (`run_experiment.py:327`) -> per deduction-rule theory construction -> `proof.generate_membership_question()` (`proof.py:47`) / `generate_de_morgans_question` / `generate_proof_by_cases_question` / `generate_compositional_question` -> serialize to `examples` dict -> `json.dump` at `run_experiment.py:1688`.

Supporting modules on the path:
- `theory.py:163` `generate_theory` -> `theory.py:68` `generate_ontology` (builds the OntologyNode DAG; the "world").
- `theory.py:183` `get_subsumption_formula` (turns graph edges into FOL rules; the latent rules).
- `syntax.py` `inflect` / `formula_to_clause` / `parse_sentence` (NL surface + round-trip parser).
- `fol.py` FOL term types, `substitute`, `unify`, `predicates`, `contains`.

### Active-path LOC (approx)
~1250 LOC of the code I traced as load-bearing for json generation:
- `run_experiment.py`: `generate_question` 327-664 (~337) + `run_experiment`/`__main__` 1432-1812 (~380). The other ~1100 lines of run_experiment.py (`is_provable`, `evaluate_response`, `parse_log`, `analyze_*`, prompting variants) are the MODEL-EVALUATION path and are NOT hit in json mode = dead code for our purposes.
- `proof.py`: 47-404 membership/DFS + de-morgans/proof-by-cases/compositional 406-960 (~360 core, more if OOD rules used).
- `theory.py`: 50-227 (~180).
- `syntax.py` (1004) + `fol.py` (628): used via the calls above; I read them through their call sites and verified behavior by execution + the round-trip self-check, not line-by-line.

## Checklist findings

### Seed plumbing — THREADED, clean
- CLI `--seed` default 62471893 (`run_experiment.py:1722`).
- Threaded into `run_experiment(args, ...)` and applied once at `run_experiment.py:1446-1447`: `seed(args.seed)` (stdlib `random`) and `np.random.seed(args.seed)`.
- Both RNGs are used: stdlib `random` (`choice/choices/randrange/shuffle/sample` imported `run_experiment.py:5`, `theory.py:2`, `proof.py:5`) and `np.random.choice` on the property-branch selection (`proof.py:207`). Both seeded, so fully reproducible.
- VERIFIED empirically: two runs with `--seed 12345` produced byte-identical JSON; `--seed 999` produced different answers (`['True','False']`). Determinism confirmed.
- No import-time `seed()` calls. No unseeded RNG on the active path found.
- Set/dict-iteration nondeterminism: `set(...)` uses on the active path are length comparisons or `sorted(list(set(...)))` (`run_experiment.py:1544`), not order-dependent iteration. No red flag.

### Oracle independence — constructive, with strong self-check (mild caveat)
- The proof/chain-of-thought is built constructively by walking the ontology DAG (`proof.py:99-258`), producing typed `ProofStep`s (AXIOM, UNIVERSAL_INSTANTIATION, etc.). It is derived from the same graph that generates the text, so it is NOT a fully independent re-derivation — it is generation-coupled. This is the expected design for a seeded-generator.
- MITIGATION 1 (strong): every generated sentence is re-parsed with an independent parser and asserted equal to its source formula — `run_experiment.py:532-536` and `:611-613`, `:649-653`. A generation/surface mismatch raises. This is a real independent check on the NL<->FOL mapping.
- MITIGATION 2: in DFS mode a search actually finds the proof path and raises `"DFS failed to find proof"` if it cannot (`proof.py:399-400`) — an independent validity check of the constructed proof.
- The T/F label: `expected_answer=True`, then negated with 50% prob at `run_experiment.py:617-623`. The label is thus definitional (comes from whether the query was negated), not re-derived — standard and correct for 0/1 exact match, but note it is not an independent theorem-prover verdict.

### Tests — NONE as a suite (runnable self-checks instead)
- No `test_*.py`, no `unittest`/`pytest`, no CI. The only `assert`s are internal invariants (`proof.py:407`, `:442`).
- The de-facto test is the parse round-trip + DFS check above, which fire on every generated example. I ran the generator myself (5 configs) as the substitute for a test suite; see Red flags for the one failure.

### Global state / hidden coupling
- `config` is a MODULE-LEVEL MUTABLE global (`run_experiment.py:325`). `generate_question` does `current_config = config` (reference, not copy) at `:455` then mutates its fields (`.stop_probability`, `.require_properties`, `.proof_width`, `.generate_distractor_parents`, ...). Every call mutates the shared object. Deterministic because generation is sequential and single-threaded, but it is hidden coupling and NOT thread-safe / not safe for parallel generation. Named red flag (low severity for our sequential use).
- `morphology` global (`run_experiment.py:51+`) and `available_entity_names` (`:321`) are read-only after init — fine.
- `bad_patterns` loaded at import from `bad_patterns.txt` (`run_experiment.py:674`) — used only by the eval path, not json generation.
- `OntologyNode.subsumption_formulas` is memoized on the node (`theory.py:184-227`); nodes are per-example so no cross-example leakage.

### Dead code on the active path
- Most of `run_experiment.py` (`is_provable`, `evaluate_response`, `do_chain_of_thought`/prompting variants, `parse_log`) is the evaluation harness, unreachable in json mode. Not a correctness risk but it dwarfs the generator and obscures the active path.

## Adaptation-diff sketch (config + scoring glue OUTSIDE the repo)

We do NOT edit the repo for the common case. Our glue:
1. Generation: call `run_experiment.py --model-name json` as a subprocess per stratum, e.g. loop `--min-hops/--max-hops/--hops-skip` for depth strata (c18), `--seed` per split, `--ontology fictional` (or `--disjoint-concept-names` for invented-vocab strata / c21), `--distractors {none,relevant,irrelevant}` as difficulty knobs. ~40 lines of Python driver + arg table. No repo edit.
2. Parse the emitted JSON (`example{i}.test_example.{question,query,answer,chain_of_thought}`) into our instance schema. ~30 lines. `answer` is our 0/1 exact-match gold; `chain_of_thought` optional gold trace.
3. Single LLM call: concatenate `question` + `query`, ask for True/False, exact-match against `answer`. Scoring glue ~20 lines, oracle-independent of our model.
4. For c21 compositional/relational strata: use `--OOD --deduction-rule Composed --proofs-only --ordering random` (random ordering avoids the postorder bug — see Red flags). ~5 extra lines in the arg table.

Estimated total new glue: ~100 lines, all outside the repo. ONE optional in-repo one-line fix if we want default (postorder) Composed generation (see below). Latent-rule strata map naturally onto `--deduction-rule` and hop count; invented/nonce vocab is native (`wumpus/yumpus/...` and the disjoint name pools at `run_experiment.py:1471-1481`).

## Red flags

1. BUG (reachable): `--deduction-rule Composed` with postorder/preorder ordering crashes. `run_experiment.py:521` sets `formulas = reversed(theory)` (an iterator), then `:642` `for i in range(len(formulas))` raises `TypeError: object of type 'list_reverseiterator' has no len()`. Reproduced; produces a 0-byte output file. WORKAROUND: `--ordering random` (takes the `theory[:]` copy branch at `:522-524`) — verified working. A one-line fix (`list(reversed(theory))`) would also do it. Since c21 is the Composed/relational path, note this explicitly in our driver.
2. Mutable module-global `config` mutated by reference each `generate_question` call (`run_experiment.py:455`) — not parallel-safe; fine for sequential seeded generation.
3. No test suite / no CI; reliability rests on the (genuinely good) parse round-trip and DFS self-checks that fire per example.
4. Retry-by-rejection generation: when a sampled theory is too small the code returns all-None and the caller loops (`run_experiment.py:1584-1588`, `:1597-1602`), emitting many `WARNING: Could not extend ontology...` lines. Harmless and still deterministic, but small concept pools + high hop counts can loop a lot; size the vocab pools to the requested depth.
5. Oracle label is definitional (negation flag), not an independent prover verdict — acceptable for 0/1 exact match but worth stating.
6. scipy is imported at module top (`betaincinv`, `logsumexp`) though only the eval path uses it — it is a hard import dependency even for json-only generation. Install scipy/numpy; nltk NOT required for the generation path.

## Verdict rationale (provisional I1 = 3)
Maintained (Oct 2024 regen w/ bug fixes); Apache-2.0 permissive; actively used in 2025-2026 reasoning-eval papers; our diff is config + scoring glue (~100 lines, no required repo edits for the ModusPonens/fictional/disjoint paths); active path read and executed, deterministic and self-verifying. The single Composed-postorder bug is real but trivially avoided with `--ordering random`, so it does not block a 3.

## Run verification (2026-07-21)

Verified by actually running the generator (not just code review). Confirms determinism, re-seeding, and ground-truth correctness.

### Environment
- Host: darwin (macOS), Apple silicon.
- `uv venv --python 3.12 .venv` inside `/tmp/qt-repo-review/asaparov-prontoqa`.
- `uv pip install --python .venv/bin/python numpy scipy` -> `numpy==2.5.1`, `scipy==1.18.0`. That is the ENTIRE dependency set for the json-generation path. torch / transformers / matplotlib / nltk NOT needed (they are eval-path only) and were NOT installed. Confirms Red-flag #6: scipy is a hard top-level import even for json mode, but numpy+scipy alone suffice. Total install ~5s, no dependency fight.

### Gotcha (not previously noted): must run from repo root
`run_experiment.py:674` opens `bad_patterns.txt` with a RELATIVE path at import time, so running from any other cwd dies with `FileNotFoundError: bad_patterns.txt` and a 0-byte / no output. The json output file is also written to cwd with a name derived only from config (e.g. `2hop_seed12345.json`; seed only appears in the name when != default 62471893). Our driver must `cd` into the repo (or copy `bad_patterns.txt` alongside) and collect the output file from cwd.

### Commands
Run from `/tmp/qt-repo-review/asaparov-prontoqa` (repo root); `.venv/bin/python` is the venv above.
```
# Seed A (12345), run 1:
.venv/bin/python run_experiment.py --model-name json --model-size 1 --num-trials 5 \
    --min-hops 2 --max-hops 2 --ontology fictional --seed 12345   # -> 2hop_seed12345.json ; saved as A1.json
# Seed A (12345), run 2 (rm output first, rerun):
.venv/bin/python run_experiment.py --model-name json --model-size 1 --num-trials 5 \
    --min-hops 2 --max-hops 2 --ontology fictional --seed 12345   # -> saved as A2.json
# Seed B (999):
.venv/bin/python run_experiment.py --model-name json --model-size 1 --num-trials 5 \
    --min-hops 2 --max-hops 2 --ontology fictional --seed 999     # -> 2hop_seed999.json ; saved as B.json
```

### Results
- Generator RAN: exit 0, valid JSON, 5 test examples each (`example1..5`, each with 8 in-context examples + `test_example{question,query,chain_of_thought,answer}`).
- Same seed deterministic (A1 vs A2): BYTE-IDENTICAL. `cmp A1.json A2.json` -> identical; md5 both `c5e93ed7602573e667c49cb77ad22ba7` (32481 bytes). Confirms Seed-plumbing finding.
- Different seed varies (A vs B): DIFFER. `cmp` differs at char 61 / line 4; md5 B `38aac2ff2b102c837dc0f22f8322f47a` (32029 bytes). Content genuinely different: A answers `[True,False,True,True,True]`, B answers `[False,False,True,False,False]`; different entities/queries (A ex1 "Wren is not loud"/True vs B ex1 "Stella is small"/False).

### Ground-truth hand-check (3 instances from A1.json, traced against the stated rules)
1. example1 — query "Wren is not loud." Chain: "Wren is an impus" + "Impuses are vumpuses" -> Wren is a vumpus; + "Vumpuses are not loud" -> Wren is not loud. Label True. CORRECT.
2. example2 — query "Stella is not windy." Chain: "Stella is a zumpus" + "Zumpuses are brimpuses" -> Stella is a brimpus; + "Brimpuses are windy" -> Stella is windy, so "not windy" is False. Label False. CORRECT.
3. example3 — query "Alex is not earthy." Chain: "Alex is a yumpus" + "Each yumpus is a dumpus" -> Alex is a dumpus; + "Dumpuses are not earthy" -> Alex is not earthy. Label True. CORRECT.
All 3 labels AND chains-of-thought are derivable purely from rules present in each question's premise text; no external/hidden knowledge needed. Oracle is sound on this sample.

### Conclusion
Running confirms the code-review verdict: the generator works with a minimal (numpy+scipy) environment, is byte-for-byte reproducible under a fixed seed, re-seeds to genuinely different problems, and emits correct ground-truth labels + valid proof chains. Only new caveat is the run-from-repo-root requirement for `bad_patterns.txt`.
