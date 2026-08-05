"""Environment-owned HumanEval input transformations.

These transformations define the frozen input arms used by the direct D1
family.  They are pure and independent of orchestration, persistence, and
provider transport.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dr_code.humaneval import HumanEvalTask

NAME_ONLY_WRAPPER = "Write a Python function named {name}."
DEFAULT_RENAME_TOKEN = "target_fxn"
DIRECT_PROMPT_INSTRUCTION = (
    "Write a complete, correct Python implementation for the following. "
    "Output only Python code."
)
DIRECT_ARMS: tuple[str, ...] = (
    "direct_original",
    "direct_docstring",
    "direct_signature",
    "direct_name",
    "direct_renamed",
)


@dataclass(frozen=True, slots=True)
class PromptParts:
    """The stable slices of one canonical HumanEval prompt."""

    original: str
    docstring: str
    signature: str
    name_only: str
    entry_point: str


def rename_identifier(text: str, old: str, new: str) -> str:
    """Replace every whole-identifier occurrence of ``old`` with ``new``."""
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])"
    return re.sub(pattern, new, text)


def split_prompt(prompt: str, entry_point: str) -> PromptParts:
    """Split a HumanEval prompt into its direct-generation input forms."""
    lines = prompt.split("\n")
    signature_end: int | None = None
    for index, line in enumerate(lines):
        if not (
            line.lstrip().startswith("def ") and f"{entry_point}(" in line
        ):
            continue
        depth = 0
        cursor = index
        while cursor < len(lines):
            depth += lines[cursor].count("(") - lines[cursor].count(")")
            if lines[cursor].rstrip().endswith(":") and depth <= 0:
                signature_end = cursor
                break
            cursor += 1
        break
    signature = (
        "\n".join(lines[: signature_end + 1])
        if signature_end is not None
        else prompt
    )
    match = re.search(r"(\"\"\"|''')(.*?)(\1)", prompt, re.DOTALL)
    docstring = match.group(2).strip() if match else ""
    return PromptParts(
        original=prompt,
        docstring=docstring,
        signature=signature,
        name_only=NAME_ONLY_WRAPPER.format(name=entry_point),
        entry_point=entry_point,
    )


def direct_body(
    arm: str,
    parts: PromptParts,
    *,
    rename_token: str = DEFAULT_RENAME_TOKEN,
) -> str:
    """Return the prompt-slice body for one direct input arm."""
    if arm == "direct_original":
        return parts.original
    if arm == "direct_docstring":
        return parts.docstring
    if arm == "direct_signature":
        return parts.signature
    if arm == "direct_name":
        return parts.name_only
    if arm == "direct_renamed":
        return rename_identifier(
            parts.original,
            parts.entry_point,
            rename_token,
        )
    raise ValueError(f"unknown direct arm {arm!r}")


def direct_prompt(
    arm: str,
    parts: PromptParts,
    *,
    rename_token: str = DEFAULT_RENAME_TOKEN,
) -> str:
    """Compose the generation prompt for one direct input arm."""
    body = direct_body(arm, parts, rename_token=rename_token)
    return f"{DIRECT_PROMPT_INSTRUCTION}\n{body}"


def renamed_task(
    task: HumanEvalTask,
    *,
    old: str,
    new: str,
) -> HumanEvalTask:
    """Return a HumanEval task whose entry point is renamed everywhere."""

    def _rename(text: str | None) -> str:
        return rename_identifier(text, old, new) if text else (text or "")

    return HumanEvalTask(
        task_id=task.task_id,
        prompt=_rename(task.prompt),
        canonical_solution=_rename(task.canonical_solution),
        entry_point=new,
        test=_rename(task.test),
    )


__all__ = [
    "DEFAULT_RENAME_TOKEN",
    "DIRECT_ARMS",
    "DIRECT_PROMPT_INSTRUCTION",
    "NAME_ONLY_WRAPPER",
    "PromptParts",
    "direct_body",
    "direct_prompt",
    "rename_identifier",
    "renamed_task",
    "split_prompt",
]
