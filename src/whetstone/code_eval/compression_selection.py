"""Eval Experiment Compression Reference Selection.

The Whetstone experiment-specific rule that selects the exact UTF-8 bytes of
``task.gt_code_wo_comments`` and binds them to a **generic** dr-code
:class:`~dr_code.eval.CompressionReferenceKey`. The dataset-field knowledge
lives **only** here: the generic dr-code key and artifact never learn the
field name, so the generic layer stays dataset-ignorant.

The seam:

* :class:`ExperimentTaskView` is the structural contract this experiment rule
  requires — any object exposing ``gt_code_wo_comments: str``. dr-code's
  ``HumanEvalTask`` does not carry this field (it is an experiment concern), so
  the experiment supplies the value; the generic kernel is untouched.
* :func:`compression_reference_key` produces the *generic* namespaced key. The
  namespace/name are opaque strings to dr-code; only Whetstone knows they
  correspond to ``task.gt_code_wo_comments``.
* :func:`select_compression_reference` resolves the exact UTF-8 bytes into a
  generic :class:`~dr_code.eval.CompressionReferenceArtifact`.
* :func:`compression_reference_binding` / :func:`build_resolver` bind the key
  to the artifact for a dr-code
  :class:`~dr_code.eval.CompressionReferenceResolver`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from dr_code.eval import (
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
)

#: The generic-key namespace for this experiment's compression references. It
#: is an opaque string to dr-code; the mapping to ``task.gt_code_wo_comments``
#: is a Whetstone-only fact recorded here, never in the generic key.
COMPRESSION_REFERENCE_NAMESPACE = "whetstone.eval_experiment.compression"

#: The dataset field this experiment selects. Named here (Whetstone) and
#: nowhere in the generic dr-code layer.
SELECTED_FIELD = "gt_code_wo_comments"


@runtime_checkable
class ExperimentTaskView(Protocol):
    """Structural contract for a task carrying the experiment reference field.

    Any object exposing ``gt_code_wo_comments`` satisfies it. Kept a Protocol
    (not a subclass of ``HumanEvalTask``) so the generic dr-code task type is
    never widened with an experiment field.
    """

    @property
    def gt_code_wo_comments(self) -> str: ...


def compression_reference_key(task_identity: str) -> CompressionReferenceKey:
    """The generic Compression Reference Key naming one task's reference.

    ``task_identity`` is the dr-code Task Identity Hash. The returned key is a
    plain namespaced dr-code key; it carries no dataset-field knowledge.
    """

    return CompressionReferenceKey(
        namespace=COMPRESSION_REFERENCE_NAMESPACE,
        name=task_identity,
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
    task_identity: str,
    task: ExperimentTaskView,
) -> tuple[CompressionReferenceKey, CompressionReferenceArtifact]:
    """The ``(key, artifact)`` binding for one task's compression reference."""

    return (
        compression_reference_key(task_identity),
        select_compression_reference(task),
    )


def build_resolver(
    bindings: Mapping[str, ExperimentTaskView],
) -> CompressionReferenceResolver:
    """Build a generic dr-code resolver over ``{task_identity: task}``.

    Each task's exact ``gt_code_wo_comments`` bytes become the resolved
    artifact for its generic key. The resulting resolver is an ordinary
    dr-code resolver with no dataset-field knowledge.
    """

    mapping = {
        compression_reference_key(task_identity): select_compression_reference(
            task
        )
        for task_identity, task in bindings.items()
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
