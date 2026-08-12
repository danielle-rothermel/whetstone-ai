from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from whetstone.evaluation import (
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
)

#: The generic-key namespace for this experiment's compression references. It
#: is an opaque string to the generic resolver; the mapping to
#: ``task.gt_code_wo_comments``
#: is a Whetstone-only fact recorded here, never in the generic key.
COMPRESSION_REFERENCE_NAMESPACE = "whetstone.eval_experiment.compression"

#: The dataset field this experiment selects. Named here (Whetstone) and
#: nowhere in the generic reference layer.
SELECTED_FIELD = "gt_code_wo_comments"


@runtime_checkable
class ExperimentTaskView(Protocol):
    """Structural contract for a task carrying the experiment reference field.

    Any object exposing ``gt_code_wo_comments`` satisfies it. Kept a Protocol
    (not a subclass of ``HumanEvalTask``) so the generic task contract is not
    widened with an experiment field.
    """

    @property
    def gt_code_wo_comments(self) -> str: ...


def compression_reference_key(task_hash: str) -> CompressionReferenceKey:
    """The generic Compression Reference Key naming one task's reference.

    ``task_hash`` is the task identity. The returned key is a plain
    namespaced key with no dataset-field knowledge.
    """

    return CompressionReferenceKey(
        namespace=COMPRESSION_REFERENCE_NAMESPACE,
        name=task_hash,
    )


def select_compression_reference(
    task: ExperimentTaskView,
) -> CompressionReferenceArtifact:
    """Select the exact UTF-8 bytes of ``task.gt_code_wo_comments``.

    The artifact content is byte-for-byte the field's ``encode('utf-8')``
    (no normalization). The generic artifact carries only bytes; it does not
    know they came from this dataset field.
    """

    content = task.gt_code_wo_comments.encode("utf-8")
    return CompressionReferenceArtifact(content=content)


def compression_reference_binding(
    task_hash: str,
    task: ExperimentTaskView,
) -> tuple[CompressionReferenceKey, CompressionReferenceArtifact]:
    """The ``(key, artifact)`` binding for one task's compression reference."""

    return (
        compression_reference_key(task_hash),
        select_compression_reference(task),
    )


def build_resolver(
    bindings: Mapping[str, ExperimentTaskView],
) -> CompressionReferenceResolver:
    """Build a generic resolver over ``{task_hash: task}``.

    Each task's exact ``gt_code_wo_comments`` bytes become the resolved
    artifact for its generic key. The resulting resolver is an ordinary
    resolver with no dataset-field knowledge.
    """

    mapping = {
        compression_reference_key(task_hash): select_compression_reference(
            task
        )
        for task_hash, task in bindings.items()
    }
    return CompressionReferenceResolver.from_mapping(mapping)


__all__ = [
    "COMPRESSION_REFERENCE_NAMESPACE",
    "SELECTED_FIELD",
    "ExperimentTaskView",
    "build_resolver",
    "compression_reference_binding",
    "compression_reference_key",
    "select_compression_reference",
]
