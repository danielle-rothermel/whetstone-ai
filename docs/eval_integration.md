# Eval integration for external environment repos

Whetstone owns the evaluation loop; your repo owns domain tasks, scoring, and transport.

See also [naming.md](naming.md) for canonical vocabulary (`num_seeds`, `task_trial`, `rollout`, etc.).

## Contract

| Your repo | Whetstone |
|-----------|-----------|
| `Experiment` (rollout graph, eval configs, reward policy) | `RuntimeEvalEngine`, drivers |
| `EvalTaskView` + domain fields (`gold`, `prompt_inputs`, …) | sampling / evidence persistence |
| `EvalProcedureRunner` (eval-node scoring) | graph orchestration, aggregation |
| Transport factory (API keys, routing) | `ProviderExecutionPolicy`, cache/partial log |

## Minimal in-process integration

```python
from whetstone.eval.drivers.graph_rollout import GraphRolloutEvalDriver
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import EvalRequest, eval_is_success
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy

experiment = build_experiment()  # your repo
policy = ProviderExecutionPolicy(
    transport_policy=default_transport_policy(api_key_env="MY_ENV_API_KEY")
)

driver = GraphRolloutEvalDriver(
    eval_runner=MyEvalProcedureRunner(),
    mutation_field="user_prompt_template",
    render_contract=experiment.render_contract,
    transport_factory=my_transport_factory,
)

engine = RuntimeEvalEngine(
    store=store,
    experiment=experiment,
    sampling=experiment.eval_configs.internal,
    execution_policy=policy,
    driver=driver,
    concurrency=8,
)

result = engine.evaluate(
    EvalRequest(
        request_id="run-1",
        candidate=candidate,
        metadata=metadata_with_purpose("official"),
    )
)
evidence = eval_is_success(result).evidence
```

## Reference runtime (toy / CLI)

`ReferenceEvalRuntimeConfig` builds a toy experiment with fakes:

| Field | Purpose |
|-------|---------|
| `driver_mode` | `"in_process"` (default) or `"subprocess"` |
| `partial_log_path` | Crash-durable partial rows (use resolved absolute paths) |
| `prompt_cache_path` | Prompt result cache directory |
| `row_job_entrypoint` | Subprocess worker (default `whetstone.eval.drivers.graph_worker:run_row`) |
| `split_role` | `"internal_eval"` or `"official"` |

```python
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

runtime = ReferenceEvalRuntimeConfig.model_validate({
    "driver_mode": "in_process",
    "partial_log_path": "/tmp/my-run/partials",
    "prompt_cache_path": "/tmp/my-run/cache",
})
engine = runtime.build_engine(store)
```

## Subprocess scaling

Use `driver_mode="subprocess"` or construct `SubprocessGraphRolloutEvalDriver` directly when you need process isolation. The default worker uses toy fakes; production env repos should provide a custom `row_job_entrypoint` that imports their `EvalProcedureRunner` in the worker module.

Row payloads use `GraphRowRequest` (`seed_index`, pre-rendered prompt, serialized graph config).

## Result handling

- `evaluate()` returns `EvalResult` = `EvalRejected | EvalEvidenceWithRef`
- Use `eval_is_success(result)` before accessing `.evidence`
- Key evidence fields: `aggregate_value`, `row_accounting.present`, `num_seeds`, `traces_ref`, `cache`

## Imports

Stable public surface:

- `whetstone.eval.protocol` — `EvalRequest`, `EvalEngine`, `EvalResult`, helpers
- `whetstone.eval.runtime_engine` — `RuntimeEvalEngine`
- `whetstone.eval.drivers.graph_rollout` — `GraphRolloutEvalDriver`, `run_rollout_row`
- `whetstone.eval.drivers.subprocess_graph_rollout` — `SubprocessGraphRolloutEvalDriver`
- `whetstone.eval.reference_runtime` — `ReferenceEvalRuntimeConfig`
- `whetstone.experiment.*` — `Experiment`, `Candidate`, `EvalConfigs`
