from __future__ import annotations

from dr_code.trace import TextArtifact

from whetstone.provider.classification import Generation


def submission_text_artifact(generation: Generation) -> TextArtifact:
    """Project a Code Generation into the Submission Text boundary role.

    Returns the native dr-code ``TextArtifact`` whose ``text`` is exactly the
    Code Generation string. No new type, schema, or identity is introduced;
    the returned value is an ordinary ``TextArtifact`` (Submission Text is a
    role carried *by* ``TextArtifact.text``, not a subtype of it).
    """

    return TextArtifact(text=generation.text)


def submission_text(generation: Generation) -> str:
    """The exact Submission Text string of a Code Generation.

    Byte-for-byte identical to ``generation.text``; provided so callers that
    only need the string do not have to reach through the artifact.
    """

    return generation.text


__all__ = [
    "submission_text",
    "submission_text_artifact",
]
