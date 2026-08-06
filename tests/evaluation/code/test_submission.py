from __future__ import annotations

import ast
import inspect

import dr_code.trace as trace_pkg
from dr_code.trace import TextArtifact

import whetstone.evaluation.code.submission as submission_module
from whetstone.evaluation.code import submission_text, submission_text_artifact

from .support import generation


def test_submission_text_is_native_text_artifact() -> None:
    gen = generation(text="def f():\n    return 1\n")
    artifact = submission_text_artifact(gen)
    assert type(artifact) is TextArtifact
    assert artifact.text == gen.text


def test_submission_text_is_byte_exact() -> None:
    text = "  def f():\r\n\treturn 'π'  \n\n"
    gen = generation(text=text)
    assert submission_text(gen) == text
    assert submission_text_artifact(gen).text == text
    assert submission_text_artifact(gen).text.encode("utf-8") == (
        text.encode("utf-8")
    )


def test_no_new_artifact_class_defined() -> None:
    module_classes = [
        name
        for name, obj in inspect.getmembers(submission_module, inspect.isclass)
        if obj.__module__ == submission_module.__name__
    ]
    assert module_classes == []


def test_no_subclass_of_text_artifact_anywhere() -> None:
    source = inspect.getsource(submission_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not isinstance(node, ast.ClassDef), (
            "submission boundary must define no class"
        )


def test_uses_released_text_artifact() -> None:
    assert submission_module.TextArtifact is trace_pkg.TextArtifact
    artifact = submission_text_artifact(generation(text="x = 1\n"))
    assert artifact.kind == "text"
