from __future__ import annotations

import ast
import re
from typing import Final

IMPORT_ALIAS_MAP: Final[dict[str, str]] = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "torch": "import torch",
    "nn": "import torch.nn as nn",
    "F": "import torch.nn.functional as F",
    "Path": "from pathlib import Path",
    "re": "import re",
    "os": "import os",
    "sys": "import sys",
    "math": "import math",
    "json": "import json",
    "defaultdict": "from collections import defaultdict",
    "Counter": "from collections import Counter",
    "deque": "from collections import deque",
    "Enum": "from enum import Enum",
    "StrEnum": "from enum import StrEnum",
    "IntEnum": "from enum import IntEnum",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "itertools": "import itertools",
    "functools": "import functools",
    "reduce": "from functools import reduce",
    "lru_cache": "from functools import lru_cache",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Tuple": "from typing import Tuple",
    "Set": "from typing import Set",
    "Optional": "from typing import Optional",
    "Union": "from typing import Union",
    "Any": "from typing import Any",
}

IMPORT_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(import |from )")
TRAILING_JUNK_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*(?:#|//|--|/\*).*$"
)


def infer_necessary_imports(source: str) -> str:
    repaired, _changed = _repair_import_lines(source)
    inferred = _infer_missing_imports(repaired)
    return _dedup_imports(inferred)


def _parse_or_none(text: str) -> ast.AST | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _repair_import_line(line: str) -> str | None:
    candidate = TRAILING_JUNK_RE.sub("", line)
    candidate = candidate.rstrip().rstrip(",")
    if _parse_or_none(candidate) is not None:
        return candidate

    opens = candidate.count("(")
    closes = candidate.count(")")
    if opens <= closes:
        return None

    closed_candidate = candidate + (")" * (opens - closes))
    if _parse_or_none(closed_candidate) is not None:
        return closed_candidate
    return None


def _repair_import_lines(source: str) -> tuple[str, bool]:
    changed = False
    lines: list[str] = []
    for line in source.splitlines():
        if IMPORT_LINE_RE.match(line) and _parse_or_none(line) is None:
            fixed = _repair_import_line(line)
            if fixed is not None:
                lines.append(fixed)
            changed = True
            continue
        lines.append(line)
    return "\n".join(lines), changed


def _infer_missing_imports(source: str) -> str:
    tree = _parse_or_none(source)
    if tree is None:
        return source

    referenced = _collect_referenced_names(tree)
    bound = _collect_bound_names(tree)
    imports = [
        import_statement
        for name, import_statement in IMPORT_ALIAS_MAP.items()
        if name in referenced and name not in bound
    ]
    if not imports:
        return source
    return "\n".join(imports) + "\n" + source


def _collect_referenced_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            value = node
            while isinstance(value, ast.Attribute):
                value = value.value
            if isinstance(value, ast.Name):
                names.add(value.id)
    return names


def _collect_bound_names(tree: ast.AST) -> set[str]:
    if not isinstance(tree, ast.Module):
        return set()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            names.add(node.name)
    return names


def _dedup_imports(source: str) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for line in source.splitlines():
        if IMPORT_LINE_RE.match(line):
            key = line.strip()
            if key in seen:
                continue
            seen.add(key)
        lines.append(line)
    return "\n".join(lines)
