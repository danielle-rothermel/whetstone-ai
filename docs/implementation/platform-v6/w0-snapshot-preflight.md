# W0 snapshot content-identity preflight

This preflight fixes the dataset snapshot boundary before the Whetstone
runtime cutover. Snapshot bytes and validated header fields define canonical
identity. Filesystem paths remain locators and never enter identity. Execution
that follows registration must provide the registered identity. Callers cannot
inject detached rows and identity; they must provide a content-verified
`HumanEvalSnapshot` produced from one byte sequence.

## Scenario matrix

| Scenario | Registered state | Execution input | Required result | Evidence |
| --- | --- | --- | --- | --- |
| Same bytes, different paths | One content digest and header | Either locator | Equal canonical identity; neither path is persisted in identity | `test_same_snapshot_bytes_have_one_identity_across_paths` |
| Different bytes, same dataset name and split | Distinct content digests under the same logical axes | Both snapshots load successfully | Distinct canonical identities | `test_different_snapshot_bytes_have_distinct_identity_for_same_axis` |
| Corrupt bytes or mismatched header | No valid registration | Corrupt JSON or a different dataset header | Fail closed before rows are admitted | `test_corrupt_snapshot_and_header_mismatch_fail_closed` |
| Mutation after registration | Identity captured from the original bytes | Same locator now contains different bytes | Reject execution because observed identity differs from registration | `test_mutation_after_registration_is_rejected` |
| Injected content | Caller supplies content outside locator loading | Detached rows, verified snapshot, or wrong-dataset snapshot | Reject the detached API, propagate the verified snapshot identity, and reject a wrong-dataset snapshot | `test_injected_snapshot_carries_verified_identity`, `test_injected_snapshot_rejects_another_dataset` |
| Same dataset, detached content | Identity from snapshot A and bytes or rows from snapshot B | Construct a coupled snapshot | Reject construction even though both headers name the same dataset | `test_same_dataset_different_content_cannot_be_detached` |
| Post-construction mutation | A previously validated coupled snapshot | Nested row mutation or wholesale row-field replacement before consumption | Revalidate coupling at the consumption boundary and reject the mutated snapshot | `test_nested_row_mutation_is_rejected_at_consumption`, `test_row_field_replacement_is_rejected_at_consumption` |
| Canonical identity serialization | Content digest and header | Any absolute locator | Serialized identity contains no locator or absolute path | `test_same_snapshot_bytes_have_one_identity_across_paths` |
| Scoring resolution | Registered identity from the Prediction Spec | One durable snapshot-resolution step | Read once, verify identity, and return the task plus observed identity together; no path cache or detached identity read | `test_load_humaneval_scoring_input_step_couples_task_and_identity`, `test_scoring_workflow_uses_dbos_step_boundaries` |

## State transitions

| From | Event | To | Guard |
| --- | --- | --- | --- |
| Unread | Parse snapshot bytes | Validated | JSON shape and snapshot header validate |
| Validated | Capture identity | Registered | SHA-256 is computed from the exact parsed bytes |
| Registered | Resolve for execution | Ready | Fresh observed identity equals registered identity |
| Registered | Resolve mutated locator | Rejected | Fresh observed identity differs from registered identity |
| External snapshot | Build specs | Ready | Bytes, rows, and identity are revalidated at consumption and the identity names the configured dataset |
| Detached rows or mismatched snapshot | Build specs | Rejected | The legacy detached API and any content mismatch fail closed |
| Registered scoring input | Resolve task | Ready | One durable step reads bytes once and verifies the observed identity before returning both task and identity |

The loader reads each execution snapshot once, and the coupled model validates
that its rows and digest come from that exact byte sequence. Scoring performs
this resolution in one durable step checked against the identity registered on
the Prediction Spec. Locators can be persisted separately without changing
this identity contract.
