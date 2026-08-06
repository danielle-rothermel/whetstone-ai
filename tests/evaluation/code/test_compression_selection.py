from __future__ import annotations

import inspect

from pydantic import BaseModel

import whetstone.evaluation.compression as generic_module
from whetstone.evaluation import (
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
)
from whetstone.evaluation.code import (
    COMPRESSION_REFERENCE_NAMESPACE,
    SELECTED_FIELD,
    build_resolver,
    compression_reference_binding,
    compression_reference_key,
    select_compression_reference,
)


class _Task(BaseModel):
    gt_code_wo_comments: str


def test_selects_exact_utf8_bytes() -> None:
    task = _Task(gt_code_wo_comments="def f():\n    return 'π'\n")
    artifact = select_compression_reference(task)
    assert type(artifact) is CompressionReferenceArtifact
    assert artifact.content == task.gt_code_wo_comments.encode("utf-8")
    assert artifact.byte_length == len(
        task.gt_code_wo_comments.encode("utf-8")
    )


def test_key_is_generic_whetstone_key() -> None:
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
    source = inspect.getsource(generic_module)
    key = CompressionReferenceKey(namespace="ns", name="n")
    assert not hasattr(key, SELECTED_FIELD)
    artifact = CompressionReferenceArtifact(content=b"x")
    assert not hasattr(artifact, SELECTED_FIELD)
    module_doc = generic_module.__doc__ or ""
    occurrences_in_code = source.replace(module_doc, "").count(SELECTED_FIELD)
    assert occurrences_in_code == 0


def test_selection_field_named_only_in_whetstone() -> None:
    assert SELECTED_FIELD == "gt_code_wo_comments"
