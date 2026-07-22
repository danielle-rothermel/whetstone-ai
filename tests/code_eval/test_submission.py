"""Code Generation producer role + Submission Text boundary.

Proves the exact decoder Generation string is projected into native
``TextArtifact.text`` with no duplicate type, artifact, schema, or identity:
the boundary returns an ordinary dr-code ``TextArtifact`` and introduces no
new artifact class.
"""

from __future__ import annotations

import ast
import inspect

import dr_code.trace as trace_pkg
from dr_code.trace import TextArtifact

import whetstone.code_eval.submission as submission_module
from whetstone.code_eval import submission_text, submission_text_artifact

from .support import generation


def test_submission_text_is_native_text_artifact() -> None:
    gen = generation(text="def f():\n    return 1\n")
    artifact = submission_text_artifact(gen)
    # The boundary yields the native dr-code TextArtifact type exactly.
    assert type(artifact) is TextArtifact
    assert artifact.text == gen.text


def test_submission_text_is_byte_exact() -> None:
    # Whitespace, unicode, trailing newlines are preserved verbatim — the
    # Submission Text is the exact decoder Generation string.
    text = "  def f():\r\n\treturn 'π'  \n\n"
    gen = generation(text=text)
    assert submission_text(gen) == text
    assert submission_text_artifact(gen).text == text
    assert submission_text_artifact(gen).text.encode("utf-8") == (
        text.encode("utf-8")
    )


def test_no_new_artifact_class_defined() -> None:
    # The module defines no new artifact/type/schema: it must not declare any
    # class at all (Submission Text is a *role*, not a type). It reuses the
    # dr-code TextArtifact.
    module_classes = [
        name
        for name, obj in inspect.getmembers(submission_module, inspect.isclass)
        if obj.__module__ == submission_module.__name__
    ]
    assert module_classes == []


def test_no_subclass_of_text_artifact_anywhere() -> None:
    # No Submission Text subtype of TextArtifact was introduced.
    source = inspect.getsource(submission_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not isinstance(node, ast.ClassDef), (
            "submission boundary must define no class"
        )


def test_uses_released_text_artifact() -> None:
    # The projected artifact is the same class dr-code.trace exports; no
    # duplicate identity/schema is introduced (TextArtifact has a fixed kind).
    assert submission_module.TextArtifact is trace_pkg.TextArtifact
    artifact = submission_text_artifact(generation(text="x = 1\n"))
    assert artifact.kind == "text"
