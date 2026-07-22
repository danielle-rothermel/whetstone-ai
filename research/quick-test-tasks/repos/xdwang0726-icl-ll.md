# Repo Review: xdwang0726/ICL_LL

- **URL:** https://github.com/xdwang0726/ICL_LL
- **Review date:** undefined
- **License:** MIT (Copyright 2022 Vector Institute) — permissive, reuse OK
- **Serving candidate ids:** c15 (Opaque-Codebook Remapped-Label Classification)
- **Provisional I1:** 1
- **One-line verdict:** A two-stage HF-dataset-derived label-corruption pipeline with a real ground-truth oracle, but it is a frozen-dataset generator (needs preprocessed real datasets), has no single-LLM-call scoring path (GPT2/GPT-J per-option loss only), messy multi-site seeding, and only an English-word remap — major surgery to serve c15.

## Active path

Generator is **two stages**, not one entry point:

1. **Instance sampling** — `preprocess/_build_gym.py` (`build_gym`, L80-144) shells out (`subprocess.run`, L71) to per-task scripts e.g. `preprocess/rotten_tomatoes.py` (L38-42), which call `FewshotGymClassificationDataset.generate_k_shot_data` in `preprocess/fewshot_gym_dataset.py` (L158-512). That method loads a **real HF dataset** (`load_dataset('rotten_tomatoes')`, rotten_tomatoes.py L34-35), shuffles, and writes k-shot train/dev/test jsonl.
2. **Label remap/corruption** — `create_data.py` `main` (L11-199) reads those jsonl files back and produces the label-effect variants (noisy `{75,50,25,0}_correct`, `random`, `random_english_words`, `no_labels`, `ood_inputs`, `random_labels_only`).

Serialization: `FewshotGymDataset.save`/`write` (fewshot_gym_dataset.py L78-153) and `create_data.py` L189-197 (`json.dumps` per line).

Eval/oracle-consuming path (would be **discarded**): `test.py` L82-233 → `utils/data.py::load_data` (L15-41) → `icl/model.py::do_inference`/`do_predict` (L246-272, per-option **loss** scoring, argmin) → `icl/data.py::evaluate` (L88-114).

**Active-path LOC (approx):** ~1400 read end-to-end (create_data 218, fewshot_gym_dataset 559 [~half is dead commented code], utils 281, one task 46, _build_gym 201, templates 100), plus ~500 in the eval path (test.py 277, icl/data.py evaluate, icl/model.py, utils/data.py).

## Checklist findings

### Seed plumbing — RED
Seed is threaded as a value but re-seeded at many independent sites, and one call site is unseeded:
- `create_data.py` L75: `np.random.seed(int(seed))` per seed loop.
- `fewshot_gym_dataset.py` L183 / L544: `np.random.seed(seed)` inside `generate_k_shot_data` (int seed, OK).
- `fewshot_gym_dataset.py` L21-34 `randomList` uses `from random import randint` (L16) — bare `random`, **never seeded**. It is only reachable via commented-out code (L491), so currently dead, but it is on the module surface.
- `test.py` L93, L127: separate `np.random.seed(int(seed))` for the `random_english_words` remap — the remap mapping is regenerated at eval time, so the codebook is NOT persisted with the instances (nondeterministic coupling between generation and scoring).
- Named red flags present: multiple `np.random.seed` calls (not a single threaded RNG object); **dict-iteration-order dependence** in `label_list` construction (fewshot_gym_dataset.py L189-196, L281-287) feeding `sorted_keys`/split assignment; hardcoded `split_list` tables keyed on label count (L201-278).

### Oracle independence — GREEN (but trivial)
Ground truth is the **real dataset label** carried from HF (`rotten_tomatoes.py` L21-31 `self.label` map; `preprocess/utils.py::preprocess` L171-262 emits `output`). It is NOT computed by the corruption generator — `create_data.py` corrupts a *copy* and the test split keeps the gold label (L82-83 comment, L93-100). So the oracle is independent of the perturbation logic. However for a remapped-codebook task the "oracle" is just string equality on the (possibly remapped) label — `icl/data.py::evaluate` L94-96 is exact/`in` match after `.strip()`. There is no latent-rule computation; the rule IS the fixed codebook.

### Tests — NONE
`test.py` and `test_imbalance.py` are **evaluation drivers**, not unit tests (no `pytest`/`unittest`/`def test_`; grep confirmed). `requirements.txt` lists pytest but there is no test suite covering generator or oracle. Nothing runnable to validate determinism.

### Global state / hidden coupling / dead code — RED
- `preprocess/fewshot_gym_dataset.py` parses `argparse` and builds `config_dict`/prompts **at import time** (L36-58), so importing the module requires CLI args and a `../config/tasks` cwd — heavy import side effects.
- `preprocess/utils.py::load_configs` (L26-33) hardcodes relative path `"../config/tasks"` — cwd-dependent; must run from inside `preprocess/`.
- `generate_k_shot_data` is ~350 lines of which ~250 are **commented-out dead alternatives** (L301-508). Real logic is small.
- Bugs on non-active branches: `L121` `os.path.joi` (typo), `L73` `NotImplementedError(o)` (undefined `o`) — both in the `use_instruct`/copa branches we would not hit, but signal low code hygiene.
- The `random_english_words` codebook is generated in TWO places (create_data.py L116-117 vs test.py L124-136) with different logic — coupling hazard.

### Frozen-dataset dependency — RED (disqualifying for "re-seedable generator")
The generator does not synthesize inputs; it samples from **downloaded real HF datasets** (`datasets.load_dataset`). Producing instances requires network + the exact HF dataset versions, and reproducibility depends on `md5sum` verification (`_build_gym.py` L157-190, `preprocess/_md5sum.py`). The `--build` step even hardcodes `all_tasks = ['rotten_tomatoes.py','trec.py']` with `assert all_tasks == ALL_TASKS` (L113-115), so only 2 tasks build out-of-the-box. This is a frozen-dataset generator, not a self-contained synthetic one.

## Adaptation-diff sketch

The repo's remap concept lives entirely in ~5 lines: `create_data.py` L116-119 (`np.random.choice(english_words_set, ...)` → `new_mapping` dict → rewrite `output`/`options`). To serve c15 we would essentially **reimplement the generator** rather than adapt:

- **Reuse (copy, ~40 lines):** the codebook-remap idiom from `create_data.py` L114-139 and the option/config schema (`config/tasks/*.json`, e.g. `ag_news.json`).
- **Replace (write new, ~150-250 lines outside repo):** a self-contained input source (drop `preprocess/` + HF `load_dataset` entirely, or pin a small local corpus), a single threaded `np.random.Generator(seed)` for shuffle + codebook draw + strata, nonce/invented vocab (repo only has real English words via the `english_words` package), and latent-rule strata labels.
- **Replace (write new, ~80 lines):** scoring glue — a single-LLM-call harness with 0/1 exact match. The existing `icl/model.py` (GPT2/GPT-J per-option loss argmin) and `icl/data.py::evaluate` (Macro-F1) do NOT map to "single LLM call, 0/1 exact match" and would be dropped.
- **Net:** our diff is NOT "config + scoring glue"; it is a rewrite of sampling + oracle-serialization + scoring, reusing only the remap idea and config schema. This is why it cannot score >1.

## Red flags (named)

- Frozen-dataset generator: instances come from downloaded HF datasets, not synthesized → not cleanly re-seedable.
- Seed set at 4+ independent sites; codebook regenerated at eval time (test.py L124-136) not persisted with instances → generation/scoring nondeterministic coupling.
- Unseeded bare `random.randint` in `randomList` (fewshot_gym_dataset.py L16,L32) — dead on active path but present.
- Dict-iteration-order + hardcoded `split_list` tables drive imbalance sampling.
- Import-time argparse + cwd-relative config paths (fewshot_gym_dataset.py L36-58; utils.py L28) → heavy hidden global state.
- ~250 lines of commented dead code in the core method; typos/undefined-name bugs on adjacent branches (`os.path.joi` L121, `NotImplementedError(o)` L73).
- No unit tests for generator or oracle.
- No confirmed 2025-2026 usage; last push 2024-03-29; 0 stars; unmaintained.
- Eval oracle produces Macro-F1 over per-option LM loss, not single-call exact match — wrong scoring shape for our need.
