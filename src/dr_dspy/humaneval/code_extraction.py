from __future__ import annotations

import ast
import re
import textwrap
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from dr_dspy.humaneval.import_inference import infer_necessary_imports

DEFAULT_TAB_WIDTH = 4
BLANK_RUN_RE = re.compile(r"\n{3,}")
ANCHOR_RE = re.compile(
    r"^(?:def |async def |class |import |from |@|if __name__)"
)
BLOCK_CODE_LIKE_RE = re.compile(
    r"^(?:\s+\S|"
    r"def |async def |class |import |from |@|if |for |while |with |try |"
    r"except|else|elif|return |raise |pass\b|continue\b|break\b|"
    r"#|"
    r"[a-zA-Z_]\w*\s*=)"
)
FENCE_LINE_RE = re.compile(
    r"^[ \t]*(?P<fence>```|~~~)(?P<tag>[A-Za-z0-9_+\-]*)[ \t]*$"
)
MARKDOWN_WRAPPER_RE = re.compile(
    r"^[ \t]*(?:>+[ \t]?|\d+[.)][ \t]?|[*+\-][ \t])"
)
RETURN_LINE_RE = re.compile(r"^\s*return(?:\b|$)")


class PythonSourceValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse_ok: bool
    parse_error: str | None
    compile_ok: bool
    compile_error: str | None


def apply_cleaning(
    gen_str: str,
    apply_dedent: bool = False,
) -> list[str]:
    normalized = _normalize_text(gen_str)
    if not normalized:
        return []

    blocks = _candidate_blocks(normalized.split("\n"))
    candidates = _extract_code_candidate_blocks(blocks)
    if not candidates:
        candidates = _extract_code_candidate_blocks(
            [_strip_markdown_wrappers(block) for block in blocks]
        )

    cleaned: list[str] = []
    for candidate in candidates:
        candidate_lines = _strip_external_fences(candidate)
        candidate_text = "\n".join(candidate_lines)
        if apply_dedent:
            candidate_text = textwrap.dedent(candidate_text)
        cleaned.extend(
            infer_necessary_imports(
                "\n".join(
                    _drop_after_last_return(split_candidate.split("\n"))
                )
            )
            for split_candidate in _drop_if_name(candidate_text.split("\n"))
        )
    return cleaned


def validate_python_source(source: str) -> PythonSourceValidation:
    try:
        ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        parse_ok = False
        parse_error = f"{type(exc).__name__}: {exc}"
    else:
        parse_ok = True
        parse_error = None

    try:
        compile(source, "<candidate>", "exec")
    except (SyntaxError, ValueError) as exc:
        compile_ok = False
        compile_error = f"{type(exc).__name__}: {exc}"
    else:
        compile_ok = True
        compile_error = None

    return PythonSourceValidation(
        parse_ok=parse_ok,
        parse_error=parse_error,
        compile_ok=compile_ok,
        compile_error=compile_error,
    )


def extract_dspy_code(pred: Any, *, field_name: str = "code") -> str:
    """Pull Python source out of a DSPy prediction field."""
    code_field = getattr(pred, field_name, None)
    if code_field is None:
        return ""
    inner = getattr(code_field, "code", None)
    if isinstance(inner, str):
        return inner
    if isinstance(code_field, str):
        return code_field
    try:
        as_str = str(code_field)
    except Exception:
        return ""
    if as_str.startswith("code="):
        try:
            return as_str.split("=", 1)[1].strip().strip("'\"")
        except Exception:
            return as_str
    return as_str


def _normalize_text(raw: str, tab_width: int = DEFAULT_TAB_WIDTH) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.expandtabs(tab_width)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RUN_RE.sub("\n\n", text)
    return text.strip("\n")


def _candidate_blocks(lines: Sequence[str]) -> list[list[str]]:
    unfenced, fenced = _split_by_fences(lines)
    if fenced:
        return fenced
    return unfenced[:1]


def _split_by_fences(
    lines: Sequence[str],
) -> tuple[list[list[str]], list[list[str]]]:
    unfenced: list[list[str]] = []
    fenced: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    fence_marker: str | None = None

    for line in lines:
        marker = _fence_marker(line)
        if marker is not None and (
            fence_marker is None or marker == fence_marker
        ):
            _append_nonempty(fenced if in_fence else unfenced, current)
            current = []
            in_fence = not in_fence
            fence_marker = marker if in_fence else None
            continue
        current.append(line)

    _append_nonempty(fenced if in_fence else unfenced, current)
    return unfenced, fenced


def _fence_marker(line: str) -> str | None:
    match = FENCE_LINE_RE.match(line)
    if match is None:
        return None
    return match.group("fence")


def _append_nonempty(blocks: list[list[str]], lines: list[str]) -> None:
    if lines:
        blocks.append(lines)


def _extract_code_candidate_blocks(
    blocks: Iterable[Sequence[str]],
) -> list[list[str]]:
    candidates: list[list[str]] = []
    for block in blocks:
        candidates.extend(_function_patterns(block))
    return candidates


def _function_patterns(lines: Sequence[str]) -> list[list[str]]:
    if _is_code_like_block(lines):
        return [list(lines)]

    candidates: list[list[str]] = []
    prefix: list[str] = []
    for index, line in enumerate(lines):
        if not ANCHOR_RE.match(line):
            prefix.append(line)
            continue

        if prefix and _is_code_like_block(prefix):
            candidates.append(prefix)

        remaining = list(lines[index:])
        if _is_code_like_block(remaining) or not candidates:
            candidates.append(remaining)
            break
    return candidates


def _is_code_like_block(lines: Sequence[str]) -> bool:
    first_line = _first_line(lines)
    if first_line is None or not first_line.strip():
        return True
    return bool(BLOCK_CODE_LIKE_RE.match(first_line))


def _first_line(lines: Sequence[str]) -> str | None:
    return lines[0] if lines else None


def _strip_external_fences(lines: Sequence[str]) -> list[str]:
    stripped = list(lines)
    if stripped and _fence_marker(stripped[0]) is not None:
        stripped = stripped[1:]
    if stripped and _fence_marker(stripped[-1]) is not None:
        stripped = stripped[:-1]
    return stripped


def _strip_markdown_wrappers(lines: Sequence[str]) -> list[str]:
    return [MARKDOWN_WRAPPER_RE.sub("", line, count=1) for line in lines]


def _drop_if_name(lines: Sequence[str]) -> list[str]:
    split_lines = [line for line in lines if "if __name__" in line]
    if not split_lines:
        return ["\n".join(lines)]

    remaining = "\n".join(lines)
    splits: list[str] = []
    for split_line in split_lines:
        before, *after = remaining.split(split_line)
        splits.append(before)
        if after:
            remaining = "\n".join(after)
    return splits


def _drop_after_last_return(lines: Sequence[str]) -> list[str]:
    for index in range(len(lines) - 1, -1, -1):
        if RETURN_LINE_RE.match(lines[index]):
            return list(lines[: index + 1])
    return list(lines)
