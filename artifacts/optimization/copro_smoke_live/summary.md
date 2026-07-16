# COPRO minimal enc-dec run

## Run config

- run_id: `e9ac044686a4`
- experiment_name: `copro_minimal_e9ac044686a4`
- model_config: `configs/models/gpt54-nano-openai.json`
- split: `configs/splits/tiny.json`
- compression_targets: [0.5]
- breadth: 2
- depth: 1
- repeats: [0]
- proposal_mode: manual
- execution_mode: sync
- dry_run: False

## Candidate table

| candidate_id | depth | proposal_source | pass_rate | scoreable | pass | gen_err | score_err |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| d0_c0 | 0 | carry_forward | 1.000 | 1 | 1 | 0 | 0 |
| d0_c1 | 0 | manual | 1.000 | 1 | 1 | 0 | 0 |

## Best candidate

- candidate_id: `d0_c0`
- depth: 0
- pass_rate: 1.000
- instructions_start: 'Provide a concise description of the following code.'
- instructions_end: ''

## Command

```bash
uv run whetstone-copro --model-config configs/models/gpt54-nano-openai.json --split configs/splits/tiny.json --compression-target 0.5 --breadth 2 --depth 1 --repeats 1 --output-dir artifacts/optimization/copro_smoke_live
```
