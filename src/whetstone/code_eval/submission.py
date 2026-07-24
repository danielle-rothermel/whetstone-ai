"""Code Generation producer role and the Submission Text boundary.

Two Whetstone roles project onto one native dr-code type:

* **Code Generation** — a
  :class:`~whetstone.provider.classification.Generation`
  read in Whetstone's *code-generation producer/lifecycle* role: its text is
  intended for the code-evaluation pipeline but does **not** yet assert valid
  Python source. Code Generation is Whetstone-owned producer semantics; dr-code
  never learns it.
* **Submission Text** — the shared *boundary role* of the exact decoder Code
  Generation string carried as native ``dr_code.trace.TextArtifact.text`` into
  preprocessing. It is a role, not a type: it creates **no** duplicate type,
  artifact, schema, or identity. The dr-code kernel receives a plain
  ``TextArtifact`` and cannot tell it apart from any other text artifact.

The projection is exact: the Submission Text is byte-for-byte the Code
Generation string (no normalization, trimming, or re-encoding). Preprocessing,
not this boundary, decides candidate validity.
"""

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
