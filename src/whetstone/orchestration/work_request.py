"""The immutable Rollout Work Request and its opaque reference transport.

A :class:`RolloutWorkRequest` is the immutable object one Rollout Work Item
carries. Per ``design/vocab_and_defs.html`` (*Rollout Work Request*) it
contains exactly:

* one Rollout Execution Key,
* Graph Config and Evaluation Context **references** (not bodies),
* task inputs (dataset-task values that enter as Graph External Inputs),
* repeat data (the Repeat ID / repeat index),
* expected schema identities (the record schemas the terminal Result and its
  parts must validate against), and

**no Materialization Record reference.** Materialization lineage is
deliberately excluded from the Work Request, the keys, and the Results; the
schema has no field that could carry one.

Transport is opaque. dr-platform's :class:`WorkInput.input_reference` and the
stage ``output_reference`` are validated only as non-empty strings and are
never parsed, resolved, or decomposed by the platform. Whetstone therefore
encodes a Work Request as one **typed object-reference string**
(:func:`encode_work_request_ref`): a scheme-tagged URI carrying the record
schema and the Content Hash of the canonical Work Request record. The platform
carries that string byte-for-byte; only Whetstone resolves it back to the
stored Work Request. The same encoding shape is what the executor emits as the
stage ``output_reference`` for the terminal Rollout Result.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from dr_store import ObjectReference
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_validator,
)

from whetstone.graph.rollout import RolloutExecutionKey

__all__ = [
    "ROLLOUT_WORK_REQUEST_SCHEMA",
    "ExpectedSchemaIdentities",
    "RepeatData",
    "RolloutWorkRequest",
    "decode_object_reference",
    "encode_object_reference",
    "encode_work_request_ref",
    "work_request_reference",
]

#: dr-store record schema for the immutable Rollout Work Request Object.
ROLLOUT_WORK_REQUEST_SCHEMA = "whetstone.rollout_work_request"

#: The scheme of the opaque typed object-reference strings Whetstone puts in
#: ``WorkInput.input_reference`` / the stage ``output_reference``.
#: dr-platform never parses it; it is Whetstone-owned.
_OBJREF_SCHEME = "objref"

_HEX = frozenset("0123456789abcdef")


class RepeatData(BaseModel):
    """The repeat coordinate of one Rollout Work Item.

    ``repeat_id`` is the semantic Repeat ID (also present in the Rollout Key);
    ``repeat_index`` is the 0-based ordinal within the Repeat Plan. Both are
    execution inputs, not Rollout Variant identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repeat_id: StrictStr
    repeat_index: int

    @model_validator(mode="after")
    def _validate(self) -> RepeatData:
        if not self.repeat_id:
            raise ValueError("repeat_id must be non-empty")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")
        return self


class ExpectedSchemaIdentities(BaseModel):
    """The record schema identities the terminal Result must validate against.

    These are *expected schema* strings (identities the executor asserts its
    persisted parts carry), never Materialization Record references and never
    provider bodies. They travel with the Work Request so a worker can reject
    a schema-drifted result before binding it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rollout_result_schema: StrictStr
    graph_run_result_schema: StrictStr | None = None
    provider_call_attempt_schema: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> ExpectedSchemaIdentities:
        if not self.rollout_result_schema:
            raise ValueError("rollout_result_schema must be non-empty")
        return self


class RolloutWorkRequest(BaseModel):
    """The immutable object one Rollout Work Item carries.

    Contains the Rollout Execution Key, Graph Config and Evaluation Context
    references, task inputs, repeat data, and expected schema identities.
    Carries **no** Materialization Record reference — there is deliberately no
    field that could hold one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The terminal execution identity (Work Item + Result Store unique key).
    rollout_execution_key: RolloutExecutionKey

    # Graph Config and Evaluation Context references (identities, not bodies).
    graph_config_ref: StrictStr
    evaluation_context_ref: StrictStr

    # Dataset-task values that feed the run as Graph External Inputs / eval
    # inputs. Identities/values only; never Rollout Variant identity.
    task_inputs: dict[str, Any] = Field(default_factory=dict)

    # The repeat coordinate.
    repeat_data: RepeatData

    # Expected schema identities for the terminal Result and its parts.
    expected_schema_identities: ExpectedSchemaIdentities

    @model_validator(mode="after")
    def _validate(self) -> RolloutWorkRequest:
        if not self.graph_config_ref:
            raise ValueError("graph_config_ref must be non-empty")
        if not self.evaluation_context_ref:
            raise ValueError("evaluation_context_ref must be non-empty")
        # The Work Request's repeat_id must name the same cell as its key.
        key_repeat = self.rollout_execution_key.rollout_key.repeat_id
        if self.repeat_data.repeat_id != key_repeat:
            raise ValueError(
                "repeat_data.repeat_id must match the Rollout Key's repeat_id"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        """The complete canonical persisted content (for the Content Hash)."""
        return self.model_dump(mode="json")


def work_request_reference(request: RolloutWorkRequest) -> ObjectReference:
    """The typed Object Reference a Rollout Work Request resolves under."""
    return ObjectReference.for_record(
        ROLLOUT_WORK_REQUEST_SCHEMA, request.record_content()
    )


def encode_object_reference(reference: ObjectReference) -> str:
    """Encode any typed Object Reference as one opaque transport string.

    The result is a scheme-tagged URI ``objref://<schema>/?content_hash=<hex>``
    that dr-platform carries byte-for-byte as an opaque non-empty string. The
    ``schema`` is percent-encoded so a schema containing ``/`` or ``?`` cannot
    corrupt the shape; the encoding is injective over ``(schema,
    content_hash)``.
    """
    return urlunsplit(
        (
            _OBJREF_SCHEME,
            quote(reference.schema, safe=""),
            "",
            f"content_hash={reference.content_hash}",
            "",
        )
    )


def decode_object_reference(encoded: str) -> ObjectReference:
    """Resolve an opaque transport string back to its typed Object Reference.

    The inverse of :func:`encode_object_reference`. Whetstone alone performs
    this resolution; dr-platform never does. Raises ``ValueError`` for a
    string that is not a Whetstone object-reference URI.
    """
    parts = urlsplit(encoded)
    if parts.scheme != _OBJREF_SCHEME:
        raise ValueError(
            f"not a Whetstone object reference (scheme {parts.scheme!r})"
        )
    schema = unquote(parts.netloc)
    query = dict(
        pair.split("=", 1) for pair in parts.query.split("&") if "=" in pair
    )
    content_hash = query.get("content_hash", "")
    is_hash = len(content_hash) == 64 and all(
        char in _HEX for char in content_hash
    )
    if not is_hash:
        raise ValueError(
            "object reference is missing a valid 64-char content_hash"
        )
    if not schema:
        raise ValueError("object reference is missing a schema")
    return ObjectReference(schema=schema, content_hash=content_hash)


def encode_work_request_ref(request: RolloutWorkRequest) -> str:
    """Encode a Work Request as its opaque ``WorkInput.input_reference``.

    Convenience over :func:`work_request_reference` +
    :func:`encode_object_reference`: the caller persists the Work Request
    through dr-store and puts this string on the Work Item; the executor
    decodes it and resolves the stored request.
    """
    return encode_object_reference(work_request_reference(request))
