# CipherBank — Repo Review

- **URL:** https://github.com/Leey21/CipherBank
- **Review date:** undefined
- **License:** No LICENSE file in repo. README badge claims MIT (README.md:4) but no LICENSE text is present. HF dataset + ACL 2025 Findings paper attached → treat as paper-assumable / reimplementable. Note the gap; do not disqualify.
- **Serving candidate ids:** c09 (Composable Cipher-Chain Decode to Closed Vocab) — **partial fit only**
- **One-line verdict:** Clean, readable single-layer deterministic cipher encoders with a genuinely independent oracle, but it is a frozen 261-item dataset with NO cipher chaining, NO seeding/randomness, and NO closed-vocab/MC format — reusing it for c09 means rewriting the generator (composition + seeding + nonce vocab) ourselves, so it is scoring-glue-plus-generator work, not config glue.

## Active path

Two distinct paths exist. The *generator* path is what matters for c09; the *eval* path is separate.

**Generator / oracle path (what we'd reuse):**
`cipher/encryption.py::main` (line 358) → instantiates 9 fixed cipher objects (lines 360-368) → `CipherProcessor.process_jsonl(mode="cipher")` (line 298) reads `data/plaintext.jsonl`, calls `cipher.encrypt(data["plaintext"])` per line (line 330), writes each cipher's output as a new key into the same JSON record (line 331), serializes to `data/test.jsonl` (line 342).

**Cipher implementations** (all in `cipher/encryption.py`): CaesarCipher (20), AtbashCipher (46), PolybiusCipher (71), VigenereCipher (122), ReverseCipher (170), SwapPairsCipher (181), ParityShiftCipher (209), DualAvgCodeCipher (228), WordShiftCipher (265).

**Eval path (separate, not needed for our oracle):** `test.py::main` (93) → `read_jsonl` → picks a model client (95-104) → `test()` (7) → `prompt.get_prompt` (29) builds few-shot markdown prompt → single model call (41-46) → `extract_result_sentences` (utils/tools.py:30) → `compare_strings` (utils/tools.py:261) for 0/1 correctness + `levenshtein_similarity` (255).

**Active-path LOC:** generator+ciphers ≈ 300 LOC (`cipher/encryption.py`); prompt builder ≈ 95 LOC (`prompt.py`); scoring/model glue ≈ 275 LOC (`utils/tools.py`); `test.py` ≈ 145 LOC. Total ≈ 815 LOC, but the *reusable* core (encoders + compare_strings) is ≈ 320 LOC.

## Checklist findings

### Seed plumbing — ABSENT (by design; frozen dataset)
- There is **no randomness anywhere** in generation. Every cipher param is hardcoded: `CaesarCipher(shift=13)` (encryption.py:360), `VigenereCipher(key="ACL")` (362), `WordShiftCipher(shift=3)` (368). Polybius matrix is fixed (73-80).
- No `random`/`np.random` import; no seed argument threaded. Instances are a pure function of the fixed 261 plaintexts in `data/plaintext.jsonl`. So it is deterministic, but **not re-seedable** — you cannot ask it for "another 500 instances with seed 7". For c09's RE-SEEDABLE requirement this is a hard miss: seeding must be added by us.
- `seed=42` appears only in the OpenAI client call (utils/tools.py:96,117,149) — that is the LLM sampling seed, irrelevant to instance generation.

### Oracle independence — STRONG (independent, not tautological)
- Ground truth is the original `plaintext` field carried straight through from `data/plaintext.jsonl` (encryption.py:329-331 stores ciphertext alongside the untouched plaintext). Scoring compares model output to `item["plaintext"]` (test.py:49, 61), which was never produced by the decrypt logic. So the oracle is the human-authored source string, fully independent of both encrypt and decrypt code.
- Bonus: `process_jsonl(mode="decrypt")` (encryption.py:347-355) round-trips encrypt→decrypt and asserts equality against plaintext — an actual reversibility self-check, evidence the encoders are correct. This is the closest thing to a test (see below).

### Tests — NONE (only an inline round-trip check)
- No `pytest`/unittest files anywhere (`find` shows only source). The only correctness check is the `decrypt` mode in `encryption.py:347-355`, run via `python cipher/encryption.py --mode decrypt`, which prints failures to stdout. It is runnable but manual, not an assertion suite. `test.py` is an LLM-eval driver, not a unit test, and its imports (`from turtle import mode`, test.py:1) are junk left in.

### Global state / hidden coupling / dead code
- **Import-time side effect:** `prompt.py:52` runs `data = read_jsonl("data/shot_case.jsonl")` at module import into a module-global `data`, used inside `get_prompt_user_markdown` (77-79). Importing `prompt` from any CWD other than repo root crashes. Hidden coupling between few-shot examples and the prompt builder.
- **Dead / broken code:** `from turtle import mode` (test.py:1) shadows nothing useful and is dead. `run.sh` passes lowercase cipher names (`rot13`, `swap_pairs`, `lsb`, `openai`, `word_shift`) that do NOT match the capitalized keys in the data or the `Introduce()` branches (prompt.py:100-123) — so `run.sh` as shipped will KeyError/ValueError. `test.py` model wiring is half-broken: `OpenAILLM` is commented out (test.py:104 `model = None`), `Agent()` is constructed with no args (test.py:102) but `Agent.__init__` requires `Skey` (utils/tools.py:52). The eval harness is not runnable out of the box.
- `PolybiusCipher.decrypt` and several `decrypt` methods have edge cases but are off the generation path.
- `rail_fence` has a prompt-hint branch (prompt.py:112) but no implementing cipher class — dead vocabulary.

### Fit for c09 specifically — PARTIAL, structurally short
- **No chaining/composition.** Each of the 9 ciphers is applied as a single flat transform (encryption.py:370-380). c09 wants *layered* cipher-CHAIN decode. Composition would be new code we write.
- **Mix of shift vs structural.** c09 wants non-shift structural transforms. Shift-based here: Caesar/Rot13, Vigenere, ParityShift, WordShift. Usable structural/non-shift primitives: Atbash (46), Polybius (71), Reverse (170), SwapPairs (181), DualAvgCode (228). Only ~5 of 9 primitives qualify, and none compose.
- **No closed vocabulary / MC.** Answers are free-form plaintext strings scored by exact-match-after-space-strip (`compare_strings`, utils/tools.py:261-268) and Levenshtein. There is no closed answer set, no distractor generation, no invented/nonce vocab. c09's "closed vocab / MC answer" + "invented/nonce vocab" would be built by us.
- The exact-match scorer (`compare_strings`) is a clean, reusable 0/1 oracle and maps directly onto c09's "0/1 exact match" once we feed it a closed-vocab target.

## Adaptation-diff sketch

Reusing this for c09 is **generator rewrite + scoring glue**, not config-only. Concretely:

1. **New file `chain_generator.py` (~120-180 new LOC, outside repo):** import the 5 structural cipher classes from `cipher/encryption.py` unchanged (Atbash, Reverse, SwapPairs, Polybius, DualAvgCode). Add a seeded RNG (`rng = random.Random(seed)`), sample a chain of K transforms per instance, apply them in sequence (`text = c.encrypt(text)` folded over the chain), and record the chain as the latent-rule stratum. This is the composition + seeding the repo lacks.
2. **Nonce/closed-vocab source (~40-60 LOC):** replace `data/plaintext.jsonl` with a seeded generator that emits invented tokens from a fixed closed vocabulary, so the answer is one of a known set (MC). The repo's fixed 261 English sentences are not a closed vocab.
3. **Scoring glue (~20-30 LOC):** reuse `compare_strings` (utils/tools.py:261) as-is for 0/1; drop Levenshtein; add MC-option matching. Single LLM call + our own prompt (the repo's `prompt.py` is coupled to a fixed 3-shot file and would be replaced).
4. **Do NOT reuse:** `test.py`, `run.sh`, all model clients in `utils/tools.py` (broken/half-wired), `prompt.py` import-time global.

Estimated total new glue: ~200-270 LOC we write, importing ~5 cipher classes verbatim. Because the seeding, composition, closed-vocab, and MC layers are all absent and must be authored, our diff exceeds "config + scoring glue."

## Red flags (named)
- **Import-time file read into module global** (`prompt.py:52`) — CWD-dependent crash, hidden coupling.
- **Frozen dataset, zero seeding** — not re-seedable; instance count fixed at 261. (encryption.py:358-388)
- **No cipher composition** — flat single-layer only; c09 needs chains. (encryption.py:370-380)
- **No test suite** — only a manual stdout round-trip check. (encryption.py:347-355)
- **Shipped eval harness is broken** — `run.sh` cipher-name mismatch, `Agent()` missing required arg (utils/tools.py:52 vs test.py:102), `OpenAILLM` disabled (test.py:104), stray `from turtle import mode` (test.py:1).
- **No LICENSE file** despite MIT badge — paper-assumable but unconfirmed.
- **No closed-vocab / MC / nonce-vocab** anywhere — core c09 features must be built.

## Provisional score: 1
Reusable, well-written deterministic encoders with an independent exact-match oracle, and the paper is actively cited. But it is a frozen dataset with a partial generator (no seeding, no chaining), no closed-vocab/MC/nonce support, and a broken eval harness — the c09 diff is major surgery (seeding + composition + vocab + MC), not config glue. Matches the "major surgery or frozen-dataset with partial generator code" anchor.
