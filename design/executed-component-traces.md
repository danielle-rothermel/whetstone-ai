# Executed component traces

Each environment row records its state as `success`, `failed`, or `missing`
and carries the ordered component steps that actually completed during that
row. A successful row carries every environment-declared step. A missing row
carries none. A failed row carries the completed prefix, so a failure after a
provider accepted a generation does not erase that observation.

Each step contains:

- a zero-based `trace_index`, contiguous within the row;
- the stable Graph Definition node id as `component_id`;
- ordered, unique, non-overlapping input and output field names; and
- deeply immutable strict-JSON input and output objects whose key sets exactly
  match those names and whose serialized order is rebuilt from the name tuples.

The field-name tuples are authoritative even when a source object arrived in a
different insertion order. Construction defensively freezes every nested value;
serialization returns fresh ordinary JSON objects and arrays, so neither caller
mutation nor mutation of a prior dump can change an admitted trace.

Trace order is authoritative and component ids may repeat. Step count, field
count, JSON nesting, and serialized strict-JSON bytes are bounded before a
trace crosses the worker or partial-log boundary. Compact JSON bytes are
counted incrementally once per immutable step; aggregate admission sums those
cached sizes and aborts as soon as the trace array would exceed its fixed
bound, without materializing the full trace serialization. Trace values contain
only the semantic component inputs and accepted outputs; provider configuration,
model routing, credentials, and provider diagnostics are not trace fields.

Internal and D1 rows record the `generate` node with its exact rendered
`prompt` and accepted `generation`. ED1-family rows record `encode` with its
exact rendered encoder `prompt` and accepted encoder `generation`, followed by
`decode` only when its exact fixed-frame `prompt` and accepted decoder
`generation` exist. The ED1 optimizable component is `encode`.

Fresh execution, prompt-cache execution, worker transport, and exact partial
resume preserve the same trace values. Partial records store the explicit row
state and trace in an environment-owned strict observation payload; they are
never reconstructed from a row's final or display output.
