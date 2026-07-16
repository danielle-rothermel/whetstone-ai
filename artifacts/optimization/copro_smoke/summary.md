# COPRO minimal enc-dec run

## Run config

- run_id: `b1f8a52d7f7f`
- experiment_name: `copro_minimal_b1f8a52d7f7f`
- model_config: `configs/models/gpt54-nano-openai.json`
- split: `configs/splits/tiny.json`
- compression_targets: [0.5]
- breadth: 2
- depth: 1
- repeats: [0]
- proposal_mode: manual
- execution_mode: sync
- dry_run: True

## Candidate table

| candidate_id | depth | proposal_source | pass_rate | scoreable | pass | gen_err | score_err |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| d0_c0 | 0 | carry_forward |  | 0 | 0 | 0 | 0 |
| d0_c1 | 0 | manual |  | 0 | 0 | 0 | 0 |

## Best candidate

- candidate_id: `d0_c0`
- depth: 0
- pass_rate: n/a
- instructions_start: 'Provide a concise description of the following code.'
- instructions_end: ''

## Caveats

- depth 0: sparse data; selected d0_c0 without score-success rows

## Command

```bash
uv run whetstone-copro --model-config configs/models/gpt54-nano-openai.json --split configs/splits/tiny.json --compression-target 0.5 --breadth 2 --depth 1 --repeats 1 --output-dir artifacts/optimization/copro_smoke --dry-run
```
