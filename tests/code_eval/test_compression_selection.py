"""Eval Experiment Compression Reference Selection.

Proves the experiment rule selects exact UTF-8 bytes of
``task.gt_code_wo_comments`` onto a generic dr-code Compression Reference Key,
and that the generic key/artifact layer stays dataset-ignorant (it never
learns the field name).
"""

from __future__ import annotations

import inspect

import dr_code.eval.compression_reference as generic_module
from dr_code.eval import (
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
)
from pydantic import BaseModel

from whetstone.code_eval import (
    COMPRESSION_REFERENCE_NAMESPACE,
    SELECTED_FIELD,
    build_resolver,
    compression_reference_binding,
    compression_reference_key,
    select_compression_reference,
)


class _Task(BaseModel):
    """A minimal experiment task view carrying the reference field."""

    gt_code_wo_comments: str


def test_selects_exact_utf8_bytes() -> None:
    task = _Task(gt_code_wo_comments="def f():\n    return 'π'\n")
    artifact = select_compression_reference(task)
    assert type(artifact) is CompressionReferenceArtifact
    assert artifact.content == task.gt_code_wo_comments.encode("utf-8")
    assert artifact.byte_length == len(
        task.gt_code_wo_comments.encode("utf-8")
    )


def test_key_is_generic_dr_code_key() -> None:
    key = compression_reference_key(task_identity="a" * 64)
    assert type(key) is CompressionReferenceKey
    assert key.namespace == COMPRESSION_REFERENCE_NAMESPACE
    assert key.name == "a" * 64


def test_binding_pairs_key_and_artifact() -> None:
    task = _Task(gt_code_wo_comments="x = 1\n")
    key, artifact = compression_reference_binding("t1", task)
    assert key == compression_reference_key("t1")
    assert artifact.content == b"x = 1\n"


def test_resolver_resolves_selected_bytes() -> None:
    tasks = {
        "t1": _Task(gt_code_wo_comments="def a(): return 1\n"),
        "t2": _Task(gt_code_wo_comments="def b(): return 2\n"),
    }
    resolver = build_resolver(tasks)
    assert isinstance(resolver, CompressionReferenceResolver)
    resolved = resolver.resolve(compression_reference_key("t1"))
    assert resolved.content == tasks["t1"].gt_code_wo_comments.encode("utf-8")


def test_generic_layer_is_dataset_ignorant() -> None:
    # The generic dr-code compression-reference module must not mention the
    # experiment field name anywhere: it stays dataset-ignorant. The only
    # mention is a docstring example, which is not executable knowledge; we
    # assert the *code* (identifiers/strings) carries no such field access.
    source = inspect.getsource(generic_module)
    # No attribute access or literal use of the field in code paths: the
    # generic key/artifact carry only namespace/name and bytes.
    key = CompressionReferenceKey(namespace="ns", name="n")
    assert not hasattr(key, SELECTED_FIELD)
    artifact = CompressionReferenceArtifact(content=b"x")
    assert not hasattr(artifact, SELECTED_FIELD)
    # The field name appears only inside a docstring (documentation), never as
    # a code token. Confirm it does not appear outside the module docstring.
    module_doc = generic_module.__doc__ or ""
    occurrences_in_code = source.replace(module_doc, "").count(SELECTED_FIELD)
    assert occurrences_in_code == 0


def test_selection_field_named_only_in_whetstone() -> None:
    # Whetstone is where the dataset field is named.
    assert SELECTED_FIELD == "gt_code_wo_comments"
