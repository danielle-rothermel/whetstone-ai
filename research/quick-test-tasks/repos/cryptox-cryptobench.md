# CryptoX / CryptoBench — Repo Review

- **URL**: https://github.com/multimodal-art-projection/CryptoX
- **Review date**: 2026-07-21
- **License**: NONE in repo (no LICENSE file, no package metadata). arXiv 2502.07813 attached, actively published. Treat as assumable/reimplementable — the encoding logic is trivial and re-implementable from scratch; do not copy files wholesale without attribution.
- **Serving candidate ids**: c09 (Composable Cipher-Chain Decode to Closed Vocab)
- **One-line verdict**: Runs end-to-end and the encoder is dead simple, but the "cipher chain" is a single-layer per-character substitution (Morse/emoji), the seed is hardcoded `random.seed(42)` inside the sampler (NOT re-seedable), and the closed-vocab MC oracle is the frozen source-benchmark answer — so this is a partial generator we'd largely re-implement, not reuse.

---

## Active Path (what we'd hit for c09)

Entry: `python -m data_construct.data_encode` (`data_construct/data_encode.py:106 main`)

1. `parse_init` (`data_construct/utils.py:205`) — argparse; parses `-r rule`, `-p "start n end"` into a linspace of percentages (`check_percentages` utils.py:69), `-d domain`, `-t type`, `--encode_type`.
2. `get_rule` (`utils.py:133`) — loads `rules.json` via pandas, returns a rule dict `{prompt_rule, rule}` where `rule` is a char→cipher map.
3. `files_encode` (`data_encode.py:80`) — iterates `ori_data/*.jsonl`, per percentage calls `one_file_encode`.
4. `one_file_encode` (`data_encode.py:21`) — per row: `sample_idx` picks word indices, `morse_code` substitutes each char, builds prompt via `get_prompt`, writes serialized record.
5. `sample_idx` (`data_construct/encode_lib.py:18`) + `morse_code` (`encode_lib.py:45`) — the only randomness + the actual cipher.
6. Prompt builders `data_construct/prompt_create/prompt_mmlu.py` (and `_bbh/_math/_mbpp`) — assemble few-shot + rule dictionary + question into the final prompt string.
7. Serialization: `df.to_json(..., lines=True)` (`data_encode.py:77`) → `<file>_percentage_<p>.jsonl`.

Oracle side (separate process): `eval/eval_lib.py:76 metric` dispatches to `eval/metric_lib/metric_mmlu.py:28 judge_correctness_mmlu` etc.

**Active-path LOC (approx)**: generator ~230 LOC (`data_encode.py` 133 + `encode_lib.py` 55 + relevant `utils.py` ~90); prompt assembly for MMLU ~200 LOC (`prompt_mmlu.py`, mostly hardcoded few-shot strings we would discard); oracle extractor ~45 LOC (`metric_mmlu.py`). Core cipher logic we actually care about is ~40 LOC.

**Confirmed runnable**: created a venv, `pip install pandas numpy tqdm jsonargparse tiktoken`, ran `python -m data_construct.data_encode -r morse_base -p "0 3 1" -d mmlu -t simple -o /tmp/qt-out` → produced 9 jsonl files. Records contain `question` (encoded), `ori_question`, `answer`, `options`, `crypto_word`/`sample_word`, `prompt`. Verified encoded example: `Find` → `..-.|..|-.|-..`.

---

## Checklist Findings

### Seed plumbing — RED FLAG (hardcoded, unthreaded)
- `data_construct/encode_lib.py:23` — `random.seed(42)` is called **inside `sample_idx`, on every invocation**, before `random.sample`. There is no seed parameter anywhere in the call chain (`main`→`files_encode`→`one_file_encode`→`sample_idx`). CLI has no `--seed`.
- Consequence: the set of encoded words is a deterministic function of `(question text, percentage)` only. It is **reproducible but not re-seedable** — you cannot draw N independent instances of the same question at the same percentage; every draw is identical. Empirically confirmed: two `sample_idx` calls returned identical `[12,1,0,4,11]`.
- For a re-seedable strata generator this is the wrong shape: we would need to remove the hardcoded seed and thread a per-instance seed. Also note `set(rule_dict.keys())` (`encode_lib.py:10`) + `random.sample` over a list built by iteration — order is list-based (deterministic given seed), so no dict-iteration nondeterminism in the sample itself, but `check_good` uses a set only for membership (fine).

### Oracle independence — MIXED
- **Answer-the-MC mode (the c09-relevant mode)**: ground truth is `item["answer"]`, the pre-existing answer from the source benchmark (MMLU/BBH/MATH), carried through untouched — `one_file_encode` never modifies `answer`. Confirmed record: `answer: 'B'` inherited from `ori_data/mmlu_dev_285_hop_1.jsonl`. The oracle (`metric_mmlu.py:28`) is an **independent regex extractor** (`Answer:\s*(.*)`, last match, normalized) compared to that stored answer. So the oracle is independent of the generator — GOOD — but the closed vocabulary is ABCD (fixed 4-way MC), and correctness depends on the *source dataset* answer being right, which is a frozen-dataset dependency, not something the generator computes or can regenerate.
- **hop-2 / hop-3 "answer conversion" modes**: `change_answer` (`prompt_mmlu.py:184`) applies a deterministic ABCD→shift or ABCD→digit rule to derive the reference — that logic lives in the prompt-example builder, and the stored `answer` field is still the raw source answer (the hop transform is described in the prompt text, expected to be done by the model, and re-applied in scoring). This is the closest thing to a "latent rule" stratum, but it is a 4-symbol cyclic map, not a composable multi-layer cipher.
- **Decode mode (`-t decode`)**: ground truth is the decoded question string, which is exactly what the generator produced (`ori_question`) — **tautological** (generator both encodes and holds the answer). Reproducible, but not an independent oracle.

### "Cipher chain" reality — RED FLAG vs c09 intent
- c09 wants *layered structural (non-shift) transforms*. What this repo implements is a **single layer** of per-character substitution: each letter → a Morse token / emoji token, joined by `|`, only on a sampled subset of words (`morse_code` encode_lib.py:45). There is no composition/chaining of multiple transforms; `morse_base`, `emoji_morse`, `emoji_shuffle` are three *alternative* single substitution alphabets, not stackable layers. Rebuilding "composable chains" is net-new work, not config.

### Tests — NONE for generator/oracle
- No `test_*.py` / pytest anywhere (`find` for tests returns only `eval/test_file/test.jsonl`, a data fixture, not a test).
- The only executable self-checks are `if __name__ == "__main__"` demo blocks in `prompt_mmlu.py:425` and `metric_mmlu.py:55` that just `print(...)`. `metric_mmlu.py:61` even calls `judge_correctness_mmlu(real_answer, test)` with the **wrong arity** (2 args for a 3-param function) — dead/broken demo code, evidence the module is untested.

### Global state / coupling / dead code — RED FLAGS
- **Import-time coupling**: `data_encode.py:14` imports `needle_data_encode`, which imports `jsonargparse` + `tiktoken` (`needle_data_encode.py:13-14`). So generating MMLU/BBH MC data forces the entire long-context-needle dependency stack even though needle is irrelevant to c09. Confirmed: the run failed until `jsonargparse` and `tiktoken` were installed.
- **Mutating instance state**: `needle_data_encode.run_sample` mutates `self.retrieval_question`/`self.needles` then restores them (`needle_data_encode.py:277,309`) — stateful, not on the c09 path but signals the coding style.
- **eval_lib** hard-imports `torch` and calls `torch.cuda.empty_cache()` (`eval_lib.py:11,70`) unconditionally, and `openai` client — the eval path is heavyweight and CUDA-coupled; we would not reuse it (we bring our own single-LLM-call + 0/1 scorer).
- `guide_nocode`/`guide_code` are defined twice (`prompt_mmlu.py:7-9` then overwritten `13-15`) — dead first definitions.

---

## Adaptation-Diff Sketch (if we reused it)

The reusable nucleus is ~40 LOC: `sample_idx` + `morse_code` + `rules.json`. Everything else (prompt few-shot scaffolding, eval harness, needle) we discard.

Realistically we would **re-implement rather than import**, because (a) no license, (b) import-time needle/tiktoken/torch coupling, (c) hardcoded seed needs surgery anyway. Sketch of the minimal-edit-in-repo path if forced:

- **`data_construct/encode_lib.py`** (~5 lines): change `sample_idx(question, rule_dict, percent, encode_type)` to accept a `seed` param and replace `random.seed(42)` with a local `rng = random.Random(seed)`; use `rng.sample`. Thread `seed` through `one_file_encode` and `files_encode` (~6 more lines in `data_encode.py`), add `--seed` to `parse_init` (~2 lines in `utils.py`).
- **New glue outside repo** (~120-180 LOC, this is where the real work is):
  - config layer: pick rule(s), percentage strata, and (for a real cipher *chain*) a wrapper that applies 2+ substitution layers in sequence — net-new, repo has no chaining.
  - invented/nonce-vocab: swap `rules.json` alphabets for nonce tokens and (for closed-vocab MC) supply our own question bank + gold answer, since the repo's answers are frozen source-benchmark answers we may not want.
  - single-LLM-call runner + 0/1 exact-match scorer: reuse only the regex idea from `metric_mmlu.py` (~15 LOC), NOT `eval_lib.py` (torch/openai/async).
  - latent-rule strata: implement per-stratum rule selection; repo has no strata concept.

Files touched in-repo: 3 (`encode_lib.py`, `data_encode.py`, `utils.py`), ~15 lines. New glue written outside: ~150 LOC. Because the cipher-chain composition, strata, nonce vocab, oracle, and runner are all net-new, **our diff is NOT "config + scoring glue" — it is a partial re-build on top of a 40-LOC substitution helper.**

---

## Red Flags (summary)
1. `random.seed(42)` hardcoded inside `sample_idx` (`encode_lib.py:23`) — not re-seedable; identical draws every call.
2. No seed parameter threaded anywhere; no `--seed` CLI.
3. Single-layer substitution only — no composable cipher chain (c09 core requirement absent).
4. Closed-vocab MC oracle = frozen source-benchmark `answer` (partial-generator / frozen-dataset dependency); decode-mode oracle is tautological.
5. Zero tests; broken demo call at `metric_mmlu.py:61` (wrong arity).
6. Import-time coupling drags needle/tiktoken/torch into the MC path; eval path is CUDA + OpenAI coupled.
7. No license.
8. Chinese-only docstrings/comments throughout (maintenance friction, not correctness).
