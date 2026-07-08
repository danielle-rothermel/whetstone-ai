# Limit Enforcement Points

Reference map of payload-size and related limits in the codebase. Domain caps
are aligned with the serialization ceiling and are catastrophe guards, not
experiment policy.

## Three Tiers

| Tier | Purpose | Where defined | Typical role |
|------|---------|---------------|--------------|
| 1 - Domain payload caps | Pydantic model construction guards | `src/whetstone/records/limits.py` | Fail fast before Postgres insert |
| 2 - Postgres / serialization ceiling | Last-resort JSONB guard | `src/whetstone/dspy_serialization.py` and DB row builders | Catch pathological blobs before persistence |
| 3 - Truncation / preview | Error messages and debug output only | Serialization and LM boundary modules | Keep diagnostics bounded |

## Domain Payload Caps

| Constant | Applies to |
|----------|------------|
| `DOMAIN_PAYLOAD_MAX_BYTES` | Alias for all domain byte caps |
| `TASK_INPUTS_MAX_BYTES` | `TaskInputsPayload.values` |
| `GRAPH_SNAPSHOT_MAX_BYTES` | `GraphSnapshotPayload` |
| `BATCH_SUBMIT_SPEC_MAX_BYTES` | `BatchSubmitOperationRecord.spec` |
| `PROVIDER_TELEMETRY_MAX_BYTES` | `UsageCostPayload.usage_metadata`, `ResponseMetadataPayload.response_metadata` |
| `NODE_OUTPUT_MAX_BYTES` | `NodeOutputPayload` |
| `METRICS_MAX_BYTES` | `ScoreAttemptRecord.metrics` |
| `PER_TEST_RESULTS_MAX_BYTES` | `ScoreAttemptRecord.per_test_results` |
| `METRICS_STAGES_MAX_COUNT` | `ScoreAttemptRecord.metrics.stages` |

`validate_payload_size(value, *, max_bytes, label)` in
`src/whetstone/records/limits.py` performs the domain-level byte check.

## Persistence Boundary

Every JSONB insert path validates recordability before writing:

- experiment config metadata
- prediction spec task snapshot, graph snapshot, dimensions, and provider config
- generation run summary
- node attempt provider config, output, usage/cost, response metadata, and
  failure payload
- score attempt extracted code, metrics, per-test results, and failure payload
- batch submit operation spec and metadata
- batch submit item enqueue metadata and failure payload

Domain validators run first when building Pydantic records. Row builders run a
second guard at the persistence boundary.

## Preview Limits

Preview and debug truncation limits keep error messages and logs bounded. They
do not block record persistence.

| Kind | Use |
|------|-----|
| message preview | short error/debug previews |
| debug detail limit | bounded structured debug detail |
| encoded preview slice | head/tail snippets in encoding failures |
| response preview | provider response logging |
